Energy Demand Forecasting System

Live Application
https://energy-demand-forecasting-systemapp.streamlit.app/

Overview
This project presents a production-grade, end-to-end energy demand forecasting system designed to simulate real-world deployment scenarios. It integrates data preprocessing, machine learning modeling, API development, and an interactive dashboard into a unified pipeline.

The system leverages engineered time-series features and an XGBoost regression model to deliver highly accurate short-term electricity demand predictions.

Business Problem
Accurate forecasting of energy demand is essential for modern energy systems. Poor predictions can lead to overproduction, energy waste, increased operational costs, and grid instability.

This project addresses these challenges by providing a scalable and deployable forecasting solution.

Key Features
- End-to-end machine learning pipeline
- Advanced time-series feature engineering
- High-performance XGBoost model
- FastAPI backend for real-time inference
- Interactive Streamlit dashboard
- Multi-step forecasting (24–72 hours)
- Prediction logging and monitoring
- Docker containerization for reproducibility

Model Performance
MAE: 310.24
RMSE: 418.04
MAPE: 0.99%

The model effectively captures short-term fluctuations and weekly seasonality in energy demand.

System Architecture
The system consists of the following components:
- Data Processing Layer: Feature engineering (lag, rolling stats, calendar features)
- Modeling Layer: XGBoost regression model
- API Layer: FastAPI for serving predictions
- UI Layer: Streamlit dashboard
- Deployment Layer: Docker containerization

Dashboard Modules
1. Single Forecast: Real-time predictions from user inputs
2. Model Performance: Metrics and visualization
3. Future Forecast: Multi-step predictions
4. Prediction Logs: Monitoring outputs

Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, FastAPI, Streamlit, Docker

Business Impact
- Enhances demand forecasting accuracy
- Improves resource allocation
- Reduces operational costs
- Supports real-time decision-making

Future Enhancements
- Cloud deployment (AWS/GCP/Azure)
- Automated retraining pipeline
- Data drift detection
- CI/CD integration
- Advanced deep learning models

Author
Prince Appiah
Ph.D. in Data Science

