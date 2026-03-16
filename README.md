# 🚀 Deep Learning Portfolio Allocation App (Euréka 2024 First Prize)

## 📌 Project Overview
This repository contains the production-ready Streamlit web application for my First Prize-winning research at the 26th National Euréka Scientific Research Competition. 

This interactive application allows users to dynamically optimize equity portfolios using a custom **Long Short-Term Memory (LSTM)** neural network combined with algorithmic trading strategies (SMA Crossover). It serves as the deployment frontend, bridging the gap between rigorous quantitative research and real-world systematic trading execution.

*(For the mathematical methodology, data preprocessing, and exploratory data analysis, please visit the companion [Research Repository](#) - insert link here).*

## ✨ Key Features & Functionality
The application is structured as a multipage Streamlit dashboard, offering two primary execution modes:
* **Custom Data Injection (CSV Mode):** Users can upload their own historical OHLCV datasets to test the model on out-of-sample or alternative asset classes.
* **Real-time API Integration (Date Mode):** The app dynamically fetches live market data from the Ho Chi Minh City Stock Exchange (HOSE) using the `vnstock` API based on user-defined date ranges.
* **Algorithmic Backtesting:** Integrates the `backtrader` framework to simulate and evaluate a customized Simple Moving Average (SMA20/SMA50) crossover strategy on the selected assets.
* **Interactive Visualization:** Leverages `Plotly` to generate real-time Treemap charts, visualizing the precise percentage weight allocations for the optimal portfolio.

## 🗂️ Repository Structure
```text
LSTM-Portfolio-WebApp/
│
├── Main_app.py                # Main entry point and homepage routing
├── pages/
│   ├── 1_input-csv.py         # Subpage logic for CSV data upload & processing
│   └── 2_input-date.py        # Subpage logic for real-time VNStock data fetching
│
├── requirements.txt           # Python dependencies
└── README.md
```

## 🚀 Technologies Used
* **Frontend Dashboard:** Streamlit
* **Deep Learning Engine:** TensorFlow, Keras, SciKeras
* **Algorithmic Trading System:** Backtrader
* **Data Engineering & Financial API:** Pandas, NumPy, VNStock
* **Interactive Visualization:** Plotly (Treemaps), Matplotlib, Seaborn

## 📸 Application Demo
*(Update later: Insert a GIF or screenshot of the Streamlit dashboard demonstrating the Treemap portfolio allocation and Backtrader results)*

## ⚙️ How to Run Locally

1. Clone this repository:
```bash
git clone https://github.com/HuyLe3011/LSTM-Portfolio-WebApp.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application from the root directory:
```bash
streamlit run Main_app.py
```

## ⚠️ Known Limitations & Future Work

While this application successfully bridges the gap between deep learning research and portfolio execution, it was built as a proof-of-concept. Future production-grade iterations will address the following technical and market execution constraints:

* **Compute Constraints & Architecture:** Currently, the deep learning inference and `backtrader` simulations run synchronously on the Streamlit server. This can lead to latency during heavy computations. Future versions will decouple the frontend from the backend, moving the deep learning engine to a dedicated microservice (e.g., FastAPI with Celery/Redis task queues) to handle concurrent user requests efficiently.
* **Market Friction & Execution Slippage:** The current algorithmic backtesting module simulates ideal execution conditions. In reality, Vietnamese equity markets exhibit liquidity constraints, bid-ask spreads, and transaction fees. Future updates will inject strict transaction cost models, tax penalties, and slippage parameters into the `backtrader` engine to reflect true net profitability.
* **Data Feed Dependency:** The real-time mode relies on the free `vnstock` API, which is excellent for research but may be subject to rate limits or latency in a high-frequency trading environment. Scaling this app for institutional use would require integration with premium, low-latency market data providers (e.g., FiinPro or Bloomberg Data License).
