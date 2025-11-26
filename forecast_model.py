"""
System Resource Forecast Model Trainer

This module is responsible for training a Machine Learning model to forecast 
future system resource usage (CPU and Memory) based on current process metrics.

It utilizes a Gradient Boosting Regressor wrapped in a MultiOutput strategy 
to predict multiple targets simultaneously. The pipeline includes:
1. Data loading and aggregation.
2. Synthetic data augmentation to handle edge cases.
3. Feature engineering and preprocessing (One-Hot Encoding).
4. Model training and serialization.

Dependencies:
    - pandas
    - numpy
    - joblib
    - scikit-learn
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Configuration Constants
DATA_FILE = 'process_metrics.csv'
MODEL_FILE = 'forecast_model.joblib'

def train_forecast_model():
    """
    Trains and persists the resource forecasting model.

    The function performs the following steps:
    1. Loads raw process metrics from CSV.
    2. Aggregates data by timestamp to create system-level snapshots.
    3. Extracts features of the most resource-intensive process per snapshot.
    4. Generates synthetic training data to ensure model robustness against 
       high-load scenarios not present in the historical data.
    5. Trains a Gradient Boosting pipeline.
    6. Saves the trained model to disk.

    Returns:
        None
    """
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found. Aborting training.")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    # ---------------------------------------------------------
    # Feature Engineering: Aggregation & Top Process Extraction
    # ---------------------------------------------------------

    def get_top_process_metrics(x):
        """
        Helper function to identify and extract metrics for the 
        process consuming the most CPU within a specific timestamp group.
        
        Args:
            x (pd.DataFrame): A subset of the dataframe for a single timestamp.
            
        Returns:
            pd.Series: Extracted features for the top process.
        """
        # Handle edge case where no processes are active or logged
        if x.empty or x['cpu_percent'].max() == 0:
            return pd.Series({
                'top_process': 'idle',
                'top_process_cpu': 0,
                'top_process_memory': 0,
                'top_process_threads': 0,
                'top_process_read_speed': 0,
                'top_process_write_speed': 0,
                'hist_avg_cpu': 0,
                'hist_max_cpu': 0,
                'hist_avg_mem': 0,
                'hist_max_mem': 0
            })
        
        # Identify index of the process with max CPU
        top_proc = x.loc[x['cpu_percent'].idxmax()]
        
        # NOTE: In a production environment, 'hist' metrics should come from 
        # a time-series database. Here, we approximate them for training purposes.
        return pd.Series({
            'top_process': top_proc['name'],
            'top_process_cpu': top_proc['cpu_percent'],
            'top_process_memory': top_proc['memory_percent'],
            'top_process_threads': top_proc['num_threads'],
            'top_process_read_speed': top_proc.get('read_speed', 0),
            'top_process_write_speed': top_proc.get('write_speed', 0),
            'hist_avg_cpu': top_proc['cpu_percent'] * 0.8, # Simulated heuristic
            'hist_max_cpu': top_proc['cpu_percent'] * 1.2, # Simulated heuristic
            'hist_avg_mem': top_proc['memory_percent'] * 0.9,
            'hist_max_mem': top_proc['memory_percent'] * 1.1
        })

    # Group raw data by timestamp to create a single training example per time slice
    print("Processing historical data...")
    system_df = df.groupby('timestamp').apply(
        lambda x: pd.concat([
            pd.Series({
                'system_cpu': x['system_cpu'].iloc[0],
                'system_memory': x['system_memory'].iloc[0],
                'cpu_percent': x['cpu_percent'].sum(),
                'memory_percent': x['memory_percent'].sum(),
                'num_threads': x['num_threads'].sum()
            }),
            get_top_process_metrics(x)
        ])
    ).reset_index()

    system_df = system_df.sort_values('timestamp')

    # ---------------------------------------------------------
    # Data Augmentation: Synthetic Data Generation
    # ---------------------------------------------------------
    # Real data often lacks extreme scenarios (100% CPU load, memory leaks).
    # We inject synthetic records to teach the model how to react to these events.
    
    print("Generating synthetic data for robustness...")
    synthetic_records = []
    avg_threads = system_df['num_threads'].mean() if not system_df.empty else 100
    
    # Process signatures to simulate
    high_load_processes = ['stress_test.exe', 'rendering_engine.exe', 'compiler.exe', 'chrome.exe', 'python.exe']
    low_load_processes = ['idle', 'system_daemon.exe', 'notepad.exe']

    for load in range(5, 101, 5):
        # Scenario 1: General High Load (Balanced CPU/Mem)
        proc_name = np.random.choice(high_load_processes) if load > 50 else np.random.choice(low_load_processes)
        
        synthetic_records.append({
            'system_cpu': load,
            'system_memory': load,
            'cpu_percent': load * 2, 
            'memory_percent': load,
            'num_threads': avg_threads * (1 + load/100),
            'top_process': proc_name,
            'top_process_cpu': load * 0.8,
            'top_process_memory': load * 0.5,
            'top_process_threads': 20 if load > 50 else 5,
            'top_process_read_speed': 1024 * 1024 if load > 70 else 0,
            'top_process_write_speed': 1024 * 1024 if load > 70 else 0,
            'hist_avg_cpu': load * 0.6,
            'hist_max_cpu': load * 0.9,
            'hist_avg_mem': load * 0.4,
            'hist_max_mem': load * 0.6,
            'target_cpu': load,       # Expect load to persist/stabilize
            'target_memory': load
        })
        
        # Scenario 2: CPU Bound (High CPU, Low Memory) - e.g., Crypto mining or calculations
        synthetic_records.append({
            'system_cpu': load,
            'system_memory': 30, 
            'cpu_percent': load * 2,
            'memory_percent': 30,
            'num_threads': avg_threads,
            'top_process': 'cpu_burner.exe',
            'top_process_cpu': load * 0.9,
            'top_process_memory': 10,
            'top_process_threads': 4,
            'top_process_read_speed': 0,
            'top_process_write_speed': 0,
            'hist_avg_cpu': load * 0.7,
            'hist_max_cpu': load * 1.0,
            'hist_avg_mem': 8,
            'hist_max_mem': 12,
            'target_cpu': load,
            'target_memory': 30
        })

        # Scenario 3: Memory Bound (Low CPU, High Memory) - e.g., Memory Leaks
        synthetic_records.append({
            'system_cpu': 10,
            'system_memory': load, 
            'cpu_percent': 10,
            'memory_percent': load,
            'num_threads': avg_threads,
            'top_process': 'memory_leak.exe',
            'top_process_cpu': 5,
            'top_process_memory': load * 0.8,
            'top_process_threads': 10,
            'top_process_read_speed': 0,
            'top_process_write_speed': 0,
            'hist_avg_cpu': 4,
            'hist_max_cpu': 6,
            'hist_avg_mem': load * 0.7,
            'hist_max_mem': load * 0.9,
            'target_cpu': 10,
            'target_memory': load
        })

    synthetic_df = pd.DataFrame(synthetic_records)
    
    # ---------------------------------------------------------
    # Target Generation & Dataset Splitting
    # ---------------------------------------------------------

    # Shift columns to create supervised learning targets (predicting t+1 based on t)
    system_df['target_cpu'] = system_df['system_cpu'].shift(-1)
    system_df['target_memory'] = system_df['system_memory'].shift(-1)
    
    # Drop the last row (NaN target due to shift)
    system_df = system_df.dropna()
    
    # Merge real historical data with synthetic data
    final_df = pd.concat([system_df, synthetic_df], ignore_index=True)
    
    features = [
        'system_cpu', 'system_memory', 'cpu_percent', 'memory_percent', 'num_threads', 
        'top_process', 'top_process_cpu', 'top_process_memory', 'top_process_threads',
        'top_process_read_speed', 'top_process_write_speed',
        'hist_avg_cpu', 'hist_max_cpu', 'hist_avg_mem', 'hist_max_mem'
    ]
    
    X = final_df[features]
    y = final_df[['target_cpu', 'target_memory']]
    
    print(f"Training on {len(X)} records...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # Model Pipeline Construction
    # ---------------------------------------------------------

    categorical_features = ['top_process']
    numeric_features = [f for f in features if f not in categorical_features]

    # Preprocessing: Handle categorical 'top_process' name via OneHotEncoding
    # 'passthrough' keeps numeric features as-is
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Model: Gradient Boosting Regressor (GBR)
    # Wrapped in MultiOutputRegressor to predict both CPU and Memory simultaneously
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
        ))
    ])

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    print(f"Model successfully trained and saved to {MODEL_FILE}")

def predict_next_state(
    current_system_cpu, current_system_mem, current_total_proc_cpu, 
    current_total_proc_mem, current_total_threads, top_process_name, 
    top_proc_cpu, top_proc_mem, top_proc_threads, top_proc_read, 
    top_proc_write, hist_avg_cpu, hist_max_cpu, hist_avg_mem, hist_max_mem
):
    """
    Infers the next system state based on the current system snapshot.

    Args:
        current_system_cpu (float): Current total System CPU usage %.
        current_system_mem (float): Current total System Memory usage %.
        current_total_proc_cpu (float): Sum of all tracked processes CPU.
        current_total_proc_mem (float): Sum of all tracked processes Memory.
        current_total_threads (int): Total thread count.
        top_process_name (str): Name of the highest load process.
        top_proc_cpu (float): CPU usage of the top process.
        top_proc_mem (float): Memory usage of the top process.
        top_proc_threads (int): Thread count of the top process.
        top_proc_read (float): Read speed (bytes/s) of top process.
        top_proc_write (float): Write speed (bytes/s) of top process.
        hist_avg_cpu (float): Historical average CPU for this process.
        hist_max_cpu (float): Historical max CPU for this process.
        hist_avg_mem (float): Historical average Mem for this process.
        hist_max_mem (float): Historical max Mem for this process.

    Returns:
        tuple: (predicted_cpu, predicted_memory) or (None, None) on error.
    """
    try:
        model = joblib.load(MODEL_FILE)
        
        # Construct input dataframe matching training feature set
        input_data = pd.DataFrame([{
            'system_cpu': current_system_cpu,
            'system_memory': current_system_mem,
            'cpu_percent': current_total_proc_cpu,
            'memory_percent': current_total_proc_mem,
            'num_threads': current_total_threads,
            'top_process': top_process_name,
            'top_process_cpu': top_proc_cpu,
            'top_process_memory': top_proc_mem,
            'top_process_threads': top_proc_threads,
            'top_process_read_speed': top_proc_read,
            'top_process_write_speed': top_proc_write,
            'hist_avg_cpu': hist_avg_cpu,
            'hist_max_cpu': hist_max_cpu,
            'hist_avg_mem': hist_avg_mem,
            'hist_max_mem': hist_max_mem
        }])
        
        prediction = model.predict(input_data)
        
        # Extract predictions (MultiOutputRegressor returns an array of arrays)
        pred_cpu = prediction[0][0]
        pred_mem = prediction[0][1]
        
        return pred_cpu, pred_mem
        
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first.")
        return None, None
    except Exception as e:
        print(f"Prediction runtime error: {e}")
        return None, None

if __name__ == "__main__":
    train_forecast_model()