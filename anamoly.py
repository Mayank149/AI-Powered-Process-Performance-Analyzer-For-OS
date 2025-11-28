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

print("--- Starting Local Model Verification ---")

if os.path.exists(MODEL_FILE):
    print("Model already exists. Loading trained intelligence...")
else:
    
    print("Model not found. Initiating MANDATORY training and saving...")
    try:
        
        data_raw_final = pd.read_csv(DATA_FILE)
        data_features_final = data_raw_final[ML_FEATURES].apply(pd.to_numeric, errors='coerce').dropna()

       
        final_model = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
        final_model.fit(data_features_final)
        
       
        joblib.dump(final_model, MODEL_FILE)
        print("Training complete. Model saved locally.")
    except FileNotFoundError:
       
        print(f"\n[CRITICAL FAILURE]: Cannot find {DATA_FILE}. Place the CSV file for the initial run.")
        exit()




def collect_metrics(system_cpu_percent, system_memory_percent):
    records = []
    for pid in psutil.pids():
        try:
            process = psutil.Process(pid)
            metrics = process.as_dict(attrs=['pid', 'name', 'memory_percent', 'num_threads', 'name'])
            metrics['cpu_percent'] = process.cpu_percent(interval=0.1)

            
           
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
           
            continue
            
    df = pd.DataFrame(records)
    wanted_columns = REQUIRED_FEATURES
    valid_columns = []
    for col in wanted_columns:
        if col in df.columns:
            valid_columns.append(col)
    return df[valid_columns]



INFERENCE_MODEL = joblib.load(MODEL_FILE)

def live_detection_cycle(model):
    
    system_cpu_initial = psutil.cpu_percent(interval=None) 
    system_mem = psutil.virtual_memory().percent

    
    time.sleep(0.1) 
    
    
    system_cpu = psutil.cpu_percent(interval=None) 
    
    raw_df = collect_metrics(system_cpu, system_mem)
    if raw_df.empty: return None

    
    data_for_ml = raw_df[ML_FEATURES].copy()
    data_for_ml = data_for_ml.apply(pd.to_numeric, errors='coerce').dropna(subset=ML_FEATURES)
    if data_for_ml.empty: return None
        
    
    raw_df.loc[data_for_ml.index, 'anomaly_label'] = model.predict(data_for_ml[ML_FEATURES])
    raw_df.loc[data_for_ml.index, 'anomaly_score'] = model.decision_function(data_for_ml[ML_FEATURES])
    
   
    anomalies = raw_df[raw_df['anomaly_label'] == -1]
    
    if not anomalies.empty:
        critical_anomalies = anomalies.sort_values(by='anomaly_score').head(5)
        
        print("\n--- IMMEDIATE BOTTLENECK DETECTED (Top 5) ---")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print(critical_anomalies[['name', 'pid', 'cpu_percent', 'memory_percent', 'anomaly_score']])
        
    return raw_df



print("\nInitiating continuous monitoring on your local system. Verify the model's power.")
print("Press Ctrl+C in your terminal to cease operation.")

monitoring_cycles = 0
try:
    while True:
        cycle_result = live_detection_cycle(INFERENCE_MODEL)
        monitoring_cycles += 1
        time.sleep(3) 

except KeyboardInterrupt:
    print(f"\nMonitoring ceased after {monitoring_cycles} cycles.")
