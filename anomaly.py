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