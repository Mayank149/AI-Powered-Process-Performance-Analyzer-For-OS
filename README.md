# AI Powered Performance Analyzer

## Overview

The **AI Powered Performance Analyzer** is a sophisticated system monitoring tool designed to track real-time system metrics, detect process anomalies using machine learning, and forecast future resource usage. Built with a robust Python backend and a modern, responsive frontend, it provides deep insights into system performance.

## Key Features

-   **Real-Time Monitoring**: Live tracking of CPU, Memory, Threads, and Disk I/O for all active processes.
-   **AI-Driven Anomaly Detection**: Utilizes an **Isolation Forest** model to identify suspicious or abnormal process behavior in real-time.
-   **Predictive Forecasting**: Implements **Random Forest Regressors** to predict future system resource consumption (CPU & Memory).
-   **Interactive Dashboard**: A dark-themed, responsive web interface featuring:
    -   Live charts for current and forecasted usage.
    -   Searchable and filterable process table (User vs. System processes).
    -   Real-time anomaly log stream.
-   **Data Export**: Ability to export anomaly logs to CSV for further analysis.

## 🛠️ Tech Stack

-   **Backend**: Python, Flask
-   **System Interaction**: Psutil
-   **Machine Learning**: Scikit-learn (Isolation Forest, Random Forest), Pandas, NumPy, Joblib
-   **Frontend**: HTML5, CSS3 (Custom Properties, Glassmorphism), JavaScript (Vanilla)

## ⚙️ Installation & Usage

### Prerequisites

-   Python 3.8 or higher
-   pip (Python Package Manager)

### Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Dependencies**
    ```bash
    pip install flask pandas numpy scikit-learn psutil joblib
    ```

3.  **Run the Application**
    ```bash
    python server.py
    ```

4.  **Access the Dashboard**
    Open your web browser and navigate to:
    `http://localhost:5000`

## 👥 Teammates

This project was developed by:

-   **Mayank Bansal**
-   **Surya**
-   **Lukshya**
