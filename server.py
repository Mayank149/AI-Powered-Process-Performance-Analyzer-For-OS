from flask import Flask, jsonify, send_from_directory, request
import anomaly
import time
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

app = Flask(__name__, static_folder='.')

# Store logs in memory
anomaly_logs = []
MAX_LOGS = 1000

# Store system history
system_history = {
    "cpu": [],
    "memory": []
}
# Store forecast history
forecast_history = {
    "cpu": [],
    "memory": []
}
DATA_FILE = 'process_metrics.csv'
MODEL_FILE = 'forecast_model.joblib'
MAX_HISTORY = 30 # Keep last 30 points (e.g. 60 seconds if 2s interval)

def train_forecast_model():
    if not os.path.exists(DATA_FILE):
        print("Data file not found, skipping training")
        return

    print("Training forecast model...")
    df = pd.read_csv(DATA_FILE)

    # Check if required columns exist
    required_columns = ['timestamp', 'system_cpu', 'system_memory', 'cpu_percent', 'memory_percent', 'num_threads']
    if not all(col in df.columns for col in required_columns):
        print("Data file missing required columns")
        return

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
    print("Forecast model trained and saved.")

def predict_next_state(current_system_cpu, current_system_mem, current_total_proc_cpu, current_total_proc_mem, current_total_threads):
    try:
        if not os.path.exists(MODEL_FILE):
            return None, None
            
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
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/live-process-data')
def live_process_data():
    global anomaly_logs, system_history, forecast_history
    
    # Use the model from anomaly.py
    if anomaly.INFERENCE_MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    df = anomaly.live_detection_cycle(anomaly.INFERENCE_MODEL)
    
    # Update system history
    import psutil
    current_cpu = psutil.cpu_percent()
    current_mem = psutil.virtual_memory().percent
    
    # print(f"DEBUG: CPU={current_cpu}, Mem={current_mem}") # Debug print
    
    system_history["cpu"].append(current_cpu)
    system_history["memory"].append(current_mem)
    # print(f"DEBUG: History len={len(system_history['cpu'])}") # Debug print
    
    if len(system_history["cpu"]) > MAX_HISTORY:
        system_history["cpu"].pop(0)
        system_history["memory"].pop(0)

    # Forecast next state
    if df is not None and not df.empty:
        total_proc_cpu = df['cpu_percent'].sum()
        total_proc_mem = df['memory_percent'].sum()
        total_threads = df['num_threads'].sum()
        
        pred_cpu, pred_mem = predict_next_state(current_cpu, current_mem, total_proc_cpu, total_proc_mem, total_threads)
        
        if pred_cpu is not None:
            forecast_history["cpu"].append(pred_cpu)
            forecast_history["memory"].append(pred_mem)
            
            if len(forecast_history["cpu"]) > MAX_HISTORY:
                forecast_history["cpu"].pop(0)
                forecast_history["memory"].pop(0)
    
    if df is None or df.empty:
        return jsonify([])

    # Process anomalies for logging
    anomalies = df[df['anomaly_label'] == -1]
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    for _, row in anomalies.iterrows():
        log_entry = {
            "timestamp": timestamp,
            "pid": int(row['pid']),
            "name": row['name'],
            "anomaly_score": float(row['anomaly_score']),
            "details": f"CPU: {row['cpu_percent']}%, Mem: {row['memory_percent']}%"
        }
        anomaly_logs.insert(0, log_entry) # Prepend
    
    # Trim logs
    if len(anomaly_logs) > MAX_LOGS:
        anomaly_logs = anomaly_logs[:MAX_LOGS]

    # Filter out System Idle Process (PID 0) just in case
    print(f"DEBUG: Before filter, count={len(df)}")
    if 'pid' in df.columns:
        df['pid'] = pd.to_numeric(df['pid'], errors='coerce')
        df = df[df['pid'] != 0]
    
    if 'name' in df.columns:
        df = df[df['name'] != 'System Idle Process']
    # print(f"DEBUG: After filter, count={len(df)}")

    # Convert to list of dicts for JSON
    # Handle NaN values for JSON compliance
    df = df.fillna(0)
    return jsonify(df.to_dict(orient='records'))

@app.route('/log-stream')
def log_stream():
    return jsonify(anomaly_logs)

@app.route('/clear-logs', methods=['POST'])
def clear_logs():
    global anomaly_logs
    anomaly_logs = []
    return jsonify({"status": "cleared"})

@app.route('/forecast')
def real_time_stats():
    # Renamed logic, keeping endpoint name to avoid breaking frontend fetch
    return jsonify({
        "realtime": system_history,
        "forecast": forecast_history
    })

if __name__ == '__main__':
    if not os.path.exists(MODEL_FILE):
        train_forecast_model()
    app.run(debug=True, port=5000)
