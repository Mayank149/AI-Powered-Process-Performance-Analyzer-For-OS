import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
import time
import psutil
import warnings
import os 

warnings.filterwarnings("ignore") 

MODEL_FILE = 'isolation_forest_model.joblib'
DATA_FILE = 'process_metrics.csv'

ML_FEATURES = [
    'cpu_percent', 'memory_percent', 'num_threads', 
    'system_cpu', 'system_memory', 
    'cpu_user_time', 'cpu_system_time', 
    'mem_rss', 'mem_vms', 
    'read_speed', 'write_speed'
]
REQUIRED_FEATURES = ML_FEATURES + ['name', 'pid', 'timestamp']
ANOMALY_THRESHOLD = -0.29  # Fixed threshold: only scores below this are flagged

print("--- Starting Local Model Verification ---")

PROCESS_CACHE = {}

def collect_metrics(system_cpu_percent, system_memory_percent):
    """Siphons real-time metrics from the local OS, with error tolerance (macOS fix)."""
    global PROCESS_CACHE
    records = []
    current_pids = set(psutil.pids())
    
    # Clean up dead processes from cache
    for pid in list(PROCESS_CACHE.keys()):
        if pid not in current_pids:
            del PROCESS_CACHE[pid]

    for pid in current_pids:
        try:
            if pid not in PROCESS_CACHE:
                PROCESS_CACHE[pid] = psutil.Process(pid)
                # Initialize cpu_percent (returns 0.0 first time)
                PROCESS_CACHE[pid].cpu_percent(interval=None)
            
            process = PROCESS_CACHE[pid]
            
            # Check if process is still running
            if not process.is_running():
                del PROCESS_CACHE[pid]
                continue

            metrics = process.as_dict(attrs=['pid', 'name', 'memory_percent', 'num_threads', 'name'])
            
            # Non-blocking call. Returns 0.0 on first call (handled above), valid value on subsequent calls
            metrics['cpu_percent'] = process.cpu_percent(interval=None)

            cpu_times = process.cpu_times()
            mem_info = process.memory_info()
            
            try:
                io_counters = process.io_counters()
                metrics['read_speed'] = io_counters.read_bytes
                metrics['write_speed'] = io_counters.write_bytes
            except (AttributeError, psutil.AccessDenied):
                metrics['read_speed'] = 0
                metrics['write_speed'] = 0
            
            metrics['cpu_user_time'] = cpu_times.user
            metrics['cpu_system_time'] = cpu_times.system
            metrics['mem_rss'] = mem_info.rss
            metrics['mem_vms'] = mem_info.vms
            metrics['system_cpu'] = system_cpu_percent
            metrics['system_memory'] = system_memory_percent
            
            metrics['timestamp'] = time.time()
            records.append(metrics)
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError, ValueError):
            if pid in PROCESS_CACHE:
                del PROCESS_CACHE[pid]
            continue
            
    df = pd.DataFrame(records)
    return df[[col for col in REQUIRED_FEATURES if col in df.columns]]

# INFERENCE_MODEL is initialized within ensure_model_and_data
INFERENCE_MODEL = None 

def live_detection_cycle(model):
    system_cpu_initial = psutil.cpu_percent(interval=None) 
    system_mem = psutil.virtual_memory().percent
    
    time.sleep(0.2) 
    
    system_cpu = psutil.cpu_percent(interval=None) 
    
    raw_df = collect_metrics(system_cpu, system_mem)
    if raw_df.empty: return None

    data_for_ml = raw_df[ML_FEATURES].copy()
    data_for_ml = data_for_ml.apply(pd.to_numeric, errors='coerce').dropna(subset=ML_FEATURES)
    if data_for_ml.empty: return None
        
    # raw_df.loc[data_for_ml.index, 'anomaly_label'] = model.predict(data_for_ml[ML_FEATURES])
    raw_df.loc[data_for_ml.index, 'anomaly_score'] = model.decision_function(data_for_ml[ML_FEATURES])
    
    # Apply manual threshold
    raw_df['anomaly_label'] = raw_df['anomaly_score'].apply(lambda x: -1 if x < ANOMALY_THRESHOLD else 1)
    
    anomalies = raw_df[raw_df['anomaly_label'] == -1]
    
    if not anomalies.empty:
        critical_anomalies = anomalies.sort_values(by='anomaly_score').head(5)
        
        print("\n--- IMMEDIATE BOTTLENECK DETECTED (Top 5) ---")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print(critical_anomalies[['name', 'pid', 'cpu_percent', 'memory_percent', 'anomaly_score']])
        
    return raw_df

def ensure_model_and_data():
    """Ensures model and data file exist. Creates dummy data if needed."""
    global INFERENCE_MODEL
    
    if not os.path.exists(DATA_FILE):
        print(f"Creating dummy {DATA_FILE} for initialization...")
        # Create a dummy CSV with headers and some random data to allow model to fit
        dummy_data = []
        for _ in range(50):
            dummy_data.append({
                'cpu_percent': np.random.uniform(0, 100),
                'memory_percent': np.random.uniform(0, 100),
                'num_threads': np.random.randint(1, 100),
                'system_cpu': np.random.uniform(0, 100),
                'system_memory': np.random.uniform(0, 100),
                'cpu_user_time': np.random.uniform(0, 10),
                'cpu_system_time': np.random.uniform(0, 10),
                'mem_rss': np.random.randint(1000, 1000000),
                'mem_vms': np.random.randint(1000, 1000000),
                'read_speed': np.random.randint(0, 1000),
                'write_speed': np.random.randint(0, 1000),
                'name': 'dummy_process',
                'pid': 1234,
                'timestamp': time.time()
            })
        pd.DataFrame(dummy_data).to_csv(DATA_FILE, index=False)

    if not os.path.exists(MODEL_FILE):
        print("Model not found. Training on available data...")
        try:
            data_raw_final = pd.read_csv(DATA_FILE)
            # Ensure we have enough data
            if len(data_raw_final) < 10:
                 print("Not enough data to train. Appending dummy data.")
            
            data_features_final = data_raw_final[ML_FEATURES].apply(pd.to_numeric, errors='coerce').dropna()
            
            final_model = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
            final_model.fit(data_features_final)
            
            joblib.dump(final_model, MODEL_FILE)
            print("Training complete. Model saved locally.")
        except Exception as e:
            print(f"Error training model: {e}")
            return None

    try:
        INFERENCE_MODEL = joblib.load(MODEL_FILE)
    except Exception as e:
        print(f"Error loading model: {e}")
        INFERENCE_MODEL = None

# Initialize on import
ensure_model_and_data()

if __name__ == "__main__":
    print("\nInitiating continuous monitoring on your local system. Verify the model's power.")
    print("Press Ctrl+C in your terminal to cease operation.")

    monitoring_cycles = 0
    try:
        while True:
            if INFERENCE_MODEL is None:
                print("Model not loaded. Cannot perform live detection. Exiting.")
                break
            cycle_result = live_detection_cycle(INFERENCE_MODEL)
            monitoring_cycles += 1
            time.sleep(3) 

    except KeyboardInterrupt:
        print(f"\nMonitoring ceased after {monitoring_cycles} cycles.")