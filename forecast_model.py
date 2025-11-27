import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

DATA_FILE = 'process_metrics.csv'
MODEL_FILE = 'forecast_model.joblib'

def train_forecast_model():
    if not os.path.exists(DATA_FILE):
        return

    df = pd.read_csv(DATA_FILE)

    system_df = df.groupby('timestamp').agg({
        'system_cpu': 'first',
        'system_memory': 'first',
        'cpu_percent': 'sum',
        'memory_percent': 'sum',
        'num_threads': 'sum'
    }).reset_index()

    system_df = system_df.sort_values('timestamp')

    synthetic_records = []
    avg_threads = system_df['num_threads'].mean() if not system_df.empty else 100
    
    for load in range(5, 101, 5):
        synthetic_records.append({
            'system_cpu': load,
            'system_memory': load,
            'cpu_percent': load * 2, 
            'memory_percent': load,
            'num_threads': avg_threads * (1 + load/100),
            'target_cpu': load,
            'target_memory': load
        })
        
        synthetic_records.append({
            'system_cpu': load,
            'system_memory': 30, 
            'cpu_percent': load * 2,
            'memory_percent': 30,
            'num_threads': avg_threads,
            'target_cpu': load,
            'target_memory': 30
        })

        synthetic_records.append({
            'system_cpu': 10,
            'system_memory': load, 
            'cpu_percent': 10,
            'memory_percent': load,
            'num_threads': avg_threads,
            'target_cpu': 10,
            'target_memory': load
        })

    for load in range(10, 90, 10):
        synthetic_records.append({
            'system_cpu': load,
            'system_memory': 40,
            'cpu_percent': load * 2,
            'memory_percent': 40,
            'num_threads': avg_threads,
            'target_cpu': load + 10,
            'target_memory': 40
        })
        
        synthetic_records.append({
            'system_cpu': 20,
            'system_memory': load,
            'cpu_percent': 20,
            'memory_percent': load,
            'num_threads': avg_threads,
            'target_cpu': 20,
            'target_memory': load + 5
        })

    synthetic_df = pd.DataFrame(synthetic_records)
    
    system_df['target_cpu'] = system_df['system_cpu'].shift(-1)
    system_df['target_memory'] = system_df['system_memory'].shift(-1)
    system_df = system_df.dropna()
    
    final_df = pd.concat([system_df, synthetic_df], ignore_index=True)
    
    features = ['system_cpu', 'system_memory', 'cpu_percent', 'memory_percent', 'num_threads']
    X = final_df[features]
    y = final_df[['target_cpu', 'target_memory']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)

def predict_next_state(current_system_cpu, current_system_mem, current_total_proc_cpu, current_total_proc_mem, current_total_threads):
    try:
        model = joblib.load(MODEL_FILE)
        
        input_data = np.array([[
            current_system_cpu, 
            current_system_mem, 
            current_total_proc_cpu, 
            current_total_proc_mem, 
            current_total_threads
        ]])
        
        prediction = model.predict(input_data)
        pred_cpu = prediction[0][0]
        pred_mem = prediction[0][1]
        
        return pred_cpu, pred_mem
    except Exception:
        return None, None

if __name__ == "__main__":
    train_forecast_model()
