from flask import Flask, jsonify, send_from_directory, request
import anomaly
import time
import pandas as pd
import numpy as np
import os

app = Flask(__name__, static_folder='.')

# Store logs in memory
anomaly_logs = []
MAX_LOGS = 1000

# Store system history
system_history = {
    "cpu": [],
    "memory": []
}
MAX_HISTORY = 30 # Keep last 30 points (e.g. 60 seconds if 2s interval)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/live-process-data')
def live_process_data():
    global anomaly_logs, system_history
    
    # Use the model from anomaly.py
    if anomaly.INFERENCE_MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    df = anomaly.live_detection_cycle(anomaly.INFERENCE_MODEL)
    
    # Update system history
    import psutil
    current_cpu = psutil.cpu_percent()
    current_mem = psutil.virtual_memory().percent
    
    print(f"DEBUG: CPU={current_cpu}, Mem={current_mem}") # Debug print
    
    system_history["cpu"].append(current_cpu)
    system_history["memory"].append(current_mem)
    print(f"DEBUG: History len={len(system_history['cpu'])}") # Debug print
    
    if len(system_history["cpu"]) > MAX_HISTORY:
        system_history["cpu"].pop(0)
        system_history["memory"].pop(0)
    
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
    print(f"DEBUG: After filter, count={len(df)}")

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
    return jsonify(system_history)

if __name__ == '__main__':
    app.run(debug=False, port=5000)
