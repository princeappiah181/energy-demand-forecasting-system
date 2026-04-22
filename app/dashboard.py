import streamlit as st
import requests
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Demand Forecast Dashboard", layout="wide")

API_BASE = "http://localhost:8000"


st.title("Energy Demand Forecast Dashboard")

st.markdown("""
This dashboard presents an end-to-end energy demand forecasting system that combines machine learning with an interactive interface.

It enables users to explore model performance, generate future forecasts, and simulate real-world energy demand scenarios in a simple and intuitive way.

- **Saved Model Evaluation** displays model performance using key metrics (MAE, RMSE, MAPE) along with a comparison of actual vs predicted energy demand.
- **Future Forecast** generates multi-step predictions (e.g., next 24–72 hours) to support planning and decision-making.
- **Single Forecast** allows users to input custom conditions and instantly estimate energy demand.
- **Prediction Logs** track recent model outputs to simulate real-world monitoring and usage.

This system demonstrates how machine learning models can be deployed in practice to support energy planning, resource allocation, and operational decision-making.
""")
st.caption("Model Version: XGBoost | Last Trained: April 2026")


st.caption("Adjust the inputs below to simulate different energy demand scenarios.")
with st.expander("What do these inputs mean?"):
    st.markdown("""
- **Hour**: Time of the day (0–23).
- **Day of Week**: 0 = Monday, 6 = Sunday.
- **Lag 1**: Energy demand from the previous hour.
- **Lag 24**: Demand at the same hour yesterday.
- **Lag 168**: Demand at the same hour last week.
- **Rolling Mean (24)**: Average demand over the past 24 hours.
- **Rolling Std (24)**: Variability in demand over the past 24 hours.
- **Rolling Mean (168)**: Weekly average demand.
- **Rolling Std (168)**: Weekly variability.
    
These features help the model capture daily and weekly patterns in energy usage.
""")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Single Forecast",
    "Model Performance",
    "Future Forecast",
    "Prediction Logs"
])

# -------------------------------------------------
# TAB 1: SINGLE FORECAST
# -------------------------------------------------
with tab1:
    st.subheader("Single Energy Demand Prediction")

    with st.form("single_forecast_form"):
        col1, col2 = st.columns(2)

        with col1:
            hour = st.number_input("Hour", min_value=0, max_value=23, value=10)
            day_of_week = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)
            day_of_month = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
            month = st.number_input("Month", min_value=1, max_value=12, value=6)
            quarter = st.number_input("Quarter", min_value=1, max_value=4, value=2)
            year = st.number_input("Year", min_value=2000, max_value=2100, value=2016)
            is_weekend = st.selectbox("Is Weekend", options=[0, 1], index=0)

        with col2:
            lag_1 = st.number_input("Lag 1", value=30000.0)
            lag_24 = st.number_input("Lag 24", value=29000.0)
            lag_168 = st.number_input("Lag 168", value=28000.0)
            rolling_mean_24 = st.number_input("Rolling Mean 24", value=29500.0)
            rolling_std_24 = st.number_input("Rolling Std 24", value=1200.0)
            rolling_mean_168 = st.number_input("Rolling Mean 168", value=28500.0)
            rolling_std_168 = st.number_input("Rolling Std 168", value=1500.0)

        submitted = st.form_submit_button("Get Forecast")

    if submitted:
        payload = {
            "hour": hour,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "month": month,
            "quarter": quarter,
            "year": year,
            "is_weekend": is_weekend,
            "lag_1": lag_1,
            "lag_24": lag_24,
            "lag_168": lag_168,
            "rolling_mean_24": rolling_mean_24,
            "rolling_std_24": rolling_std_24,
            "rolling_mean_168": rolling_mean_168,
            "rolling_std_168": rolling_std_168
        }

        try:
            response = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
            result = response.json()

            if response.status_code == 200 and "prediction_MW" in result:
                st.success(f"Predicted Energy Demand: {result['prediction_MW']:.2f} MW")
            else:
                st.error(result.get("error", f"Unexpected response: {result}"))
        except Exception as e:
            st.error(f"Request failed: {e}")

# -------------------------------------------------
# TAB 2: MODEL PERFORMANCE
# -------------------------------------------------
with tab2:
    st.subheader("Saved Model Evaluation on Test Data")
    st.info("These metrics and the actual-vs-predicted plot are saved from the last training run, so they stay the same until the model is retrained.")

    metrics_path = Path("artifacts/metrics.json")
    predictions_path = Path("artifacts/test_predictions.csv")

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", metrics["MAE"])
        c2.metric("RMSE", metrics["RMSE"])
        c3.metric("MAPE (%)", metrics["MAPE"])
    else:
        st.warning("metrics.json not found. Run train.py first.")

    if predictions_path.exists():
        pred_df = pd.read_csv(predictions_path, parse_dates=["Datetime"])

        st.markdown("### Actual vs Predicted Values")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(pred_df["Datetime"][:500], pred_df["actual_MW"][:500], label="Actual")
        ax.plot(pred_df["Datetime"][:500], pred_df["predicted_MW"][:500], label="Predicted")
        ax.set_title("Actual vs Predicted Energy Demand")
        ax.set_xlabel("Time")
        ax.set_ylabel("MW")
        ax.legend()
        plt.xticks(rotation=30)
        st.pyplot(fig)

        with st.expander("View saved prediction samples"):
            st.dataframe(pred_df.head(20), use_container_width=True)
    else:
        st.warning("test_predictions.csv not found. Run train.py first.")

# -------------------------------------------------
# TAB 3: FUTURE FORECAST
# -------------------------------------------------
with tab3:
    st.subheader("Forecast Future Demand")

    horizon = st.selectbox("Forecast Horizon (hours)", [24, 48, 72], index=0)

    if st.button("Generate Future Forecast"):
        try:
            response = requests.post(
                f"{API_BASE}/forecast_future",
                json={"horizon": horizon},
                timeout=30
            )
            result = response.json()

            if response.status_code == 200 and "forecast" in result:
                forecast_df = pd.DataFrame(result["forecast"])
                forecast_df["Datetime"] = pd.to_datetime(forecast_df["Datetime"])

                st.success(f"Generated {horizon}-hour forecast successfully.")

                fig2, ax2 = plt.subplots(figsize=(12, 5))
                ax2.plot(forecast_df["Datetime"], forecast_df["predicted_MW"])
                ax2.set_title(f"Future Energy Demand Forecast ({horizon} Hours)")
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Predicted MW")
                plt.xticks(rotation=30)
                st.pyplot(fig2)

                st.dataframe(forecast_df, use_container_width=True)
            else:
                st.error(result.get("error", f"Unexpected response: {result}"))
        except Exception as e:
            st.error(f"Forecast request failed: {e}")

# -------------------------------------------------
# TAB 4: PREDICTION LOGS
# -------------------------------------------------
with tab4:
    st.subheader("Prediction Logs")

    logs_path = Path("logs/predictions.csv")

    if logs_path.exists():
        logs_df = pd.read_csv(logs_path, parse_dates=["timestamp"])

        st.markdown("### Recent Logged Predictions")
        st.dataframe(logs_df.tail(20), use_container_width=True)

        if len(logs_df) > 1:
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            ax3.plot(logs_df["timestamp"], logs_df["prediction_MW"])
            ax3.set_title("Logged Predictions Over Time")
            ax3.set_xlabel("Timestamp")
            ax3.set_ylabel("Predicted MW")
            plt.xticks(rotation=30)
            st.pyplot(fig3)
    else:
        st.warning("No prediction logs found yet. Make a few predictions first.")