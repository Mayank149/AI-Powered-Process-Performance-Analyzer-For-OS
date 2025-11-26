"""
Real-time System Anomaly Detection Service

This module runs a continuous loop to monitor system processes and detect 
anomalies using an Isolation Forest model. It operates in two modes:
1. Training Mode: If no model exists, it trains on historical data.
2. Inference Mode: Loads the model and scores live process metrics.

The script utilizes `psutil` for high-performance system introspection and
`scikit-learn` for unsupervised anomaly detection.

Dependencies:
    - psutil: For process-level metric collection.
    - scikit-learn: For the Isolation Forest algorithm.
    - pandas/numpy: For data manipulation.
"""

import pandas as pd
import numpy as np
import joblib
import time
import psutil
import warnings
import os
from sklearn.ensemble import IsolationForest

# Suppress sklearn warnings about feature names (common in production loops)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
Model_File = 'isolation_forest_model.joblib'
Data_File = 'process_metrics.csv'

# Features used specifically for the ML Model (Numeric only)
MODEL_FEATURES = [
    'cpu_percent', 'memory_percent', 'num_threads', 
    'system_cpu', 'system_memory', 
    'cpu_user_time', 'cpu_system_time', 
    'mem_rss', 'mem_vms', 
    'read_speed', 'write_speed'
]

# Full feature set including metadata for logging/identification
REQUIRED_FEATURES = MODEL_FEATURES + ['name', 'pid', 'timestamp', 'username', 'exe', 'cmdline']

# ---------------------------------------------------------
# Model Bootstrap / Initialization
# ---------------------------------------------------------
# Logic: Load existing model to save time. If missing, train a new one 
# using available CSV data. If neither exists, abort.

if os.path.exists(Model_File):
    print(f"Loading existing model from {Model_File}...")
    model = joblib.load(Model_File)
else:
    print("Model not found. Attempting to train on historical data...")
    try:
        if not os.path.exists(Data_File):
            print("Error: No training data found. Exiting.")
            exit()
            
        raw_data = pd.read_csv(Data_File)
        # Ensure data is numeric and clean before training
        data_features = raw_data[MODEL_FEATURES].apply(pd.to_numeric, errors='coerce').dropna()
        
        # Isolation Forest Setup:
        # n_estimators=200: Sufficient trees for stable variance
        # contamination=0.05: We estimate ~5% of data might be anomalous
        model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        model.fit(data_features)
        
        joblib.dump(model, Model_File)
        print("Model trained and saved successfully.")
    except Exception as e:
        print(f"Critical error during initialization: {e}")
        exit()

Loaded_Model = model

# ---------------------------------------------------------
# Data Collection Service
# ---------------------------------------------------------

def collect_data(system_cpu_percent, system_memory_percent):
    """
    Captures a snapshot of all currently running processes.

    Uses psutil.oneshot() to maximize performance by minimizing 
    system calls when retrieving multiple attributes for a single process.

    Args:
        system_cpu_percent (float): Global system CPU usage.
        system_memory_percent (float): Global system Memory usage.

    Returns:
        pd.DataFrame: A structured DataFrame of current process metrics.
    """
    records = []
    
    # Iterate over all running processes
    for process in psutil.process_iter(['pid', 'name', 'username']):
        try:
            metrics = process.info
            metrics['exe'] = ""
            
            # Optimization: 'oneshot' context manager retrieves all info in fewer syscalls
            try:
                with process.oneshot():
                    metrics['cpu_percent'] = process.cpu_percent(interval=None)
                    metrics['memory_percent'] = process.memory_percent()
                    metrics['num_threads'] = process.num_threads()
                    
                    cpu_times = process.cpu_times()
                    metrics['cpu_user_time'] = cpu_times.user
                    metrics['cpu_system_time'] = cpu_times.system
                    
                    mem_info = process.memory_info()
                    metrics['mem_rss'] = mem_info.rss  # Resident Set Size (Physical mem)
                    metrics['mem_vms'] = mem_info.vms  # Virtual Memory Size
                    
                    # Security/Permissions handling: 
                    # Some system processes (like Antivirus or Kernel tasks) block access.
                    try:
                        metrics['exe'] = process.exe()
                    except (psutil.AccessDenied, psutil.ZombieProcess):
                        metrics['exe'] = ""

                    try:
                        io_counters = process.io_counters()
                        metrics['read_speed'] = io_counters.read_bytes
                        metrics['write_speed'] = io_counters.write_bytes
                    except:
                        # IO counters might not be available for all platforms/processes
                        metrics['read_speed'] = 0
                        metrics['write_speed'] = 0

                    try:
                        cmd = process.cmdline()
                        metrics['cmdline'] = " ".join(cmd) if cmd else ""
                    except:
                        metrics['cmdline'] = ""

            except (psutil.AccessDenied, psutil.ZombieProcess):
                # Fallback for processes that become inaccessible during iteration
                metrics['cpu_percent'] = 0
                metrics['memory_percent'] = 0
                metrics['num_threads'] = 0
                metrics['cpu_user_time'] = 0
                metrics['cpu_system_time'] = 0
                metrics['mem_rss'] = 0
                metrics['mem_vms'] = 0
                metrics['read_speed'] = 0
                metrics['write_speed'] = 0
                metrics['cmdline'] = ""
            
            # Inject global context into the process record
            metrics['system_cpu'] = system_cpu_percent
            metrics['system_memory'] = system_memory_percent
            metrics['timestamp'] = time.time()
            
            if not metrics.get('username'):
                 metrics['username'] = 'unknown'

            records.append(metrics)

        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            # Process died between iteration and inspection
            continue
        except Exception:
            continue

    df = pd.DataFrame(records)
    
    # Ensure DataFrame schema consistency (fill missing cols with 0)
    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            df[col] = 0
            
    return df[REQUIRED_FEATURES]

# ---------------------------------------------------------
# Inference Logic
# ---------------------------------------------------------

def live_detection_cycle(model):
    """
    Performs a single cycle of data collection, preprocessing, and anomaly scoring.

    Args:
        model (IsolationForest): The trained sklearn model.

    Returns:
        pd.DataFrame: The dataframe containing metrics and anomaly scores, 
                      or None if no data was collected.
    """
    # 1. Capture Global System State
    system_cpu = psutil.cpu_percent(interval=None)
    system_mem = psutil.virtual_memory().percent

    # 2. Collect Process-Level Data
    raw_df = collect_data(system_cpu, system_mem)
    
    if raw_df.empty:
        return None

    # 3. Preprocess for ML (Numeric only, handle NaNs)
    data_for_ml = raw_df[MODEL_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    if data_for_ml.empty:
        return None

    try:
        # 4. Predict Anomalies
        # label: -1 (anomaly), 1 (normal)
        raw_df['anomaly_label'] = model.predict(data_for_ml)
        
        # score: Negative scores indicate anomalies. Lower is more anomalous.
        raw_df['anomaly_score'] = model.decision_function(data_for_ml)
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None
    
    # 5. Filter for Critical Anomalies
    # We use a threshold of -0.25 to reduce false positives.
    # Standard IsolationForest cutoff is 0, but -0.25 selects only 'deep' outliers.
    anomalies = raw_df[(raw_df['anomaly_label'] == -1) & (raw_df['anomaly_score'] < -0.25)]

    if not anomalies.empty:
        print(f"\n[ALERT] {len(anomalies)} Critical Anomalies Detected:")
        critical = anomalies.sort_values(by='anomaly_score').head(5)
        print(critical[['name', 'pid', 'cpu_percent', 'memory_percent', 'anomaly_score']].to_string(index=False))

    return raw_df

# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Starting Live Anomaly Detection Service...")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            # Execute detection cycle
            live_detection_cycle(Loaded_Model)
            
            # Wait before next cycle to prevent CPU saturation by the monitoring tool itself
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nStopping service...")
        pass