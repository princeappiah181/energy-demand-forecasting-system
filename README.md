# Energy Demand Forecasting System

An end-to-end energy demand forecasting system built with machine learning, FastAPI, Streamlit, and Docker.

This project predicts electricity demand using engineered time-series features and an XGBoost forecasting model. It includes data preprocessing, model training, API deployment, an interactive dashboard, and containerization for reproducible execution.

## Project Overview

This system was designed to demonstrate how machine learning can be applied in practice for energy forecasting and operational decision-making.

The application allows users to:
- evaluate model performance using standard forecasting metrics
- visualize actual vs predicted energy demand
- generate future demand forecasts
- test single-demand scenarios through interactive inputs
- monitor predictions through logged outputs

## Key Features

- End-to-end forecasting pipeline
- Time-series feature engineering with lag and rolling statistics
- XGBoost regression model for demand prediction
- FastAPI backend for serving predictions
- Streamlit dashboard for interactive exploration
- Dockerized application for portability and deployment
- Prediction logging for basic monitoring

## Model Performance

The baseline XGBoost model achieved strong forecasting performance on the PJM hourly energy dataset:

- **MAE:** 310.24
- **RMSE:** 418.04
- **MAPE:** 0.99%

These results indicate that the model captures short-term and weekly energy demand patterns effectively.

## Dashboard Sections

### 1. Single Forecast
Users can manually input feature values and receive an estimated energy demand prediction.

### 2. Saved Model Evaluation
Displays:
- MAE
- RMSE
- MAPE
- actual vs predicted energy demand plot

### 3. Future Forecast
Generates multi-step forecasts for future demand over selectable horizons such as 24, 48, or 72 hours.

### 4. Prediction Logs
Tracks recent model outputs to simulate real-world monitoring.

## Project Structure


energy-demand-forecasting-system/ <br>
│ <br>
├── app/ <br>
│   ├── api.py  <br>
│   └── dashboard.py <br>
│ <br>
├── src/ <br>
│   ├── data_preprocessing.py <br>
│   └── train.py <br>
│ <br>
├── Dockerfile <br>
├── requirements.txt <br>
├── .dockerignore <br>
├── .gitignore <br>
└── README.md <br>


## Workflow

### 1. Data Preprocessing
The raw PJM hourly energy data is cleaned, sorted, and transformed into a forecasting-ready dataset with:

- Calendar-based features
- Lag features
- Rolling mean and rolling standard deviation features

---

### 2. Model Training
An XGBoost regressor is trained using a time-based train/test split to avoid data leakage.

---

### 3. Evaluation
The trained model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

---

### 4. Deployment
The model is deployed using:

- **FastAPI** for prediction endpoints  
- **Streamlit** for dashboard interaction  
- **Docker** for containerized execution  

### Installation

Clone the repository:

git clone https://github.com/princeappiah181/energy-demand-forecasting-system.git
cd energy-demand-forecasting-system

### Install dependencies:

pip install -r requirements.txt

Run Locally

Train the model

python src/train.py

### Start the API

uvicorn app.api:app --reload

### Start the dashboard

streamlit run app/dashboard.py

### Run with Docker

Build the image:

docker build -t energy-forecast-app .

### Run the container:

docker run -p 8000:8000 -p 8501:8501 energy-forecast-app

Then open:

API docs: http://localhost:8000/docs

Dashboard: http://localhost:8501 (http://localhost:8501/)

### Tech Stack

Python
Pandas
NumPy
Scikit-learn
XGBoost
FastAPI
Streamlit
Docker
Matplotlib

### Business Value

Energy demand forecasting supports:
load planning
resource allocation
operational efficiency
scenario analysis

This project demonstrates the ability to move from raw time-series data to a deployable machine learning application.

### Future Improvements

Potential next steps include:
cloud deployment
automated retraining
drift detection
model versioning
CI/CD integration
richer monitoring dashboards

### Author
Prince Appiah <br>
Ph.D. in Data Science

