from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
import os
from typing import List


app = FastAPI(title="Energy Forecast API")

model = xgb.XGBRegressor()
model.load_model("models/xgboost_model.json")

DATA_PATH = "data/processed/pjm_energy_processed.csv"


class SinglePredictionInput(BaseModel):
    hour: int
    day_of_week: int
    day_of_month: int
    month: int
    quarter: int
    year: int
    is_weekend: int
    lag_1: float
    lag_24: float
    lag_168: float
    rolling_mean_24: float
    rolling_std_24: float
    rolling_mean_168: float
    rolling_std_168: float


class ForecastRequest(BaseModel):
    horizon: int = 24


def log_prediction(prediction: float):
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/predictions.csv"
    file_exists = os.path.exists(log_file)

    with open(log_file, "a") as f:
        if not file_exists:
            f.write("timestamp,prediction_MW\n")
        f.write(f"{datetime.now().isoformat()},{float(prediction)}\n")


def load_processed_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"], index_col="Datetime")
    return df.sort_index()


def build_feature_row(
    ts: pd.Timestamp,
    lag_1: float,
    lag_24: float,
    lag_168: float,
    history_values: List[float]
) -> dict:
    rolling_24_series = history_values[-24:]
    rolling_168_series = history_values[-168:]

    return {
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "day_of_month": ts.day,
        "month": ts.month,
        "quarter": ts.quarter,
        "year": ts.year,
        "is_weekend": int(ts.dayofweek in [5, 6]),
        "lag_1": lag_1,
        "lag_24": lag_24,
        "lag_168": lag_168,
        "rolling_mean_24": float(pd.Series(rolling_24_series).mean()),
        "rolling_std_24": float(pd.Series(rolling_24_series).std(ddof=1)),
        "rolling_mean_168": float(pd.Series(rolling_168_series).mean()),
        "rolling_std_168": float(pd.Series(rolling_168_series).std(ddof=1)),
    }


@app.get("/")
def home():
    return {"message": "Energy Forecast API is running"}


@app.post("/predict")
def predict(data: SinglePredictionInput):
    try:
        df = pd.DataFrame([data.dict()])

        expected_features = [
            "hour", "day_of_week", "day_of_month", "month",
            "quarter", "year", "is_weekend",
            "lag_1", "lag_24", "lag_168",
            "rolling_mean_24", "rolling_std_24",
            "rolling_mean_168", "rolling_std_168"
        ]

        df = df[expected_features]
        prediction = float(model.predict(df)[0])
        log_prediction(prediction)

        return {"prediction_MW": prediction}

    except Exception as e:
        return {"error": str(e)}


@app.post("/forecast_future")
def forecast_future(request: ForecastRequest):
    try:
        horizon = int(request.horizon)
        if horizon < 1 or horizon > 168:
            return {"error": "horizon must be between 1 and 168 hours"}

        df = load_processed_data()
        history_values = df["PJME_MW"].tolist()

        if len(history_values) < 168:
            return {"error": "Not enough history to generate forecast."}

        last_timestamp = df.index[-1]
        forecasts = []

        for step in range(1, horizon + 1):
            future_ts = last_timestamp + timedelta(hours=step)

            lag_1 = history_values[-1]
            lag_24 = history_values[-24]
            lag_168 = history_values[-168]

            row = build_feature_row(
                ts=future_ts,
                lag_1=lag_1,
                lag_24=lag_24,
                lag_168=lag_168,
                history_values=history_values
            )

            feature_df = pd.DataFrame([row])
            pred = float(model.predict(feature_df)[0])

            history_values.append(pred)
            forecasts.append({
                "Datetime": future_ts.isoformat(),
                "predicted_MW": pred
            })

            log_prediction(pred)

        return {"forecast": forecasts}

    except Exception as e:
        return {"error": str(e)}