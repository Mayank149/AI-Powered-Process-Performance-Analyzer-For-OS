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

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/live-process-data')
def live_process_data():
    global anomaly_logs
    
    # Use the model from anomaly.py
    if anomaly.INFERENCE_MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    df = anomaly.live_detection_cycle(anomaly.INFERENCE_MODEL)
    
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
def forecast():
    # Mock forecast data for now (or simple projection)
    # In a real app, this would use a time-series model
    # Returning 15 seconds of predicted data points (every 5s)
    
    future_steps = 3 # 15 seconds / 5 seconds
    
    # Get current system stats as baseline
    import psutil
    current_cpu = psutil.cpu_percent()
    current_mem = psutil.virtual_memory().percent
    
    forecast_data = {
        "cpu": [],
        "memory": []
    }
    
    # Generate some slightly noisy projection
    for i in range(future_steps):
        forecast_data["cpu"].append(min(100, max(0, current_cpu + np.random.normal(0, 5))))
        forecast_data["memory"].append(min(100, max(0, current_mem + np.random.normal(0, 2))))
        
    return jsonify(forecast_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
