import json
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Energy Demand Forecast Dashboard", layout="wide")


# -------------------------------------------------
# Paths
# -------------------------------------------------
MODEL_PATH = Path("models/xgboost_model.json")
PROCESSED_DATA_PATH = Path("data/processed/pjm_energy_processed.csv")
METRICS_PATH = Path("artifacts/metrics.json")
TEST_PREDICTIONS_PATH = Path("artifacts/test_predictions.csv")
LOGS_PATH = Path("logs/predictions.csv")

FEATURE_ORDER = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "rolling_std_168",
]


# -------------------------------------------------
# Data / model loaders
# -------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_data
def load_processed_data():
    if not PROCESSED_DATA_PATH.exists():
        return None
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Datetime"], index_col="Datetime")
    return df.sort_index()


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_test_predictions():
    if not TEST_PREDICTIONS_PATH.exists():
        return None
    return pd.read_csv(TEST_PREDICTIONS_PATH, parse_dates=["Datetime"])


# -------------------------------------------------
# Core helpers
# -------------------------------------------------
def predict_single(model, payload: dict) -> float:
    df = pd.DataFrame([payload])[FEATURE_ORDER]
    prediction = float(model.predict(df)[0])
    return prediction


def build_feature_row(ts: pd.Timestamp, history_values: list[float]) -> dict:
    rolling_24 = history_values[-24:]
    rolling_168 = history_values[-168:]

    return {
        "hour": int(ts.hour),
        "day_of_week": int(ts.dayofweek),
        "day_of_month": int(ts.day),
        "month": int(ts.month),
        "quarter": int(ts.quarter),
        "year": int(ts.year),
        "is_weekend": int(ts.dayofweek in [5, 6]),
        "lag_1": float(history_values[-1]),
        "lag_24": float(history_values[-24]),
        "lag_168": float(history_values[-168]),
        "rolling_mean_24": float(pd.Series(rolling_24).mean()),
        "rolling_std_24": float(pd.Series(rolling_24).std(ddof=1)),
        "rolling_mean_168": float(pd.Series(rolling_168).mean()),
        "rolling_std_168": float(pd.Series(rolling_168).std(ddof=1)),
    }


def forecast_future(model, df_history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    history_values = df_history["PJME_MW"].tolist()
    last_timestamp = df_history.index[-1]
    forecast_rows = []

    for step in range(1, horizon + 1):
        future_ts = last_timestamp + timedelta(hours=step)
        row = build_feature_row(future_ts, history_values)
        pred = predict_single(model, row)
        history_values.append(pred)

        forecast_rows.append(
            {
                "Datetime": future_ts,
                "predicted_MW": pred,
            }
        )

    return pd.DataFrame(forecast_rows)


def compute_residual_std(pred_df: pd.DataFrame | None) -> float:
    if pred_df is None or pred_df.empty:
        return 0.0
    residuals = pred_df["actual_MW"] - pred_df["predicted_MW"]
    return float(residuals.std(ddof=1))


def log_prediction(prediction: float, mode: str, horizon: int | None = None) -> None:
    LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp.now(),
                "mode": mode,
                "horizon": horizon,
                "prediction_MW": prediction,
            }
        ]
    )

    if LOGS_PATH.exists():
        row.to_csv(LOGS_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(LOGS_PATH, index=False)


def get_latest_feature_defaults(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {
            "hour": 10,
            "day_of_week": 2,
            "day_of_month": 15,
            "month": 6,
            "quarter": 2,
            "year": 2016,
            "is_weekend": 0,
            "lag_1": 30000.0,
            "lag_24": 29000.0,
            "lag_168": 28000.0,
            "rolling_mean_24": 29500.0,
            "rolling_std_24": 1200.0,
            "rolling_mean_168": 28500.0,
            "rolling_std_168": 1500.0,
        }

    latest = df.iloc[-1]
    ts = df.index[-1]
    return {
        "hour": int(latest["hour"]),
        "day_of_week": int(latest["day_of_week"]),
        "day_of_month": int(latest["day_of_month"]),
        "month": int(latest["month"]),
        "quarter": int(latest["quarter"]),
        "year": int(latest["year"]),
        "is_weekend": int(latest["is_weekend"]),
        "lag_1": float(latest["lag_1"]),
        "lag_24": float(latest["lag_24"]),
        "lag_168": float(latest["lag_168"]),
        "rolling_mean_24": float(latest["rolling_mean_24"]),
        "rolling_std_24": float(latest["rolling_std_24"]),
        "rolling_mean_168": float(latest["rolling_mean_168"]),
        "rolling_std_168": float(latest["rolling_std_168"]),
        "reference_timestamp": ts,
    }


def scenario_presets(latest_defaults: dict) -> dict:
    return {
        "Weekday Morning Peak": {
            **latest_defaults,
            "hour": 8,
            "day_of_week": 1,
            "is_weekend": 0,
            "lag_1": max(latest_defaults["lag_1"], latest_defaults["rolling_mean_24"] * 1.03),
        },
        "Weekend Low Demand": {
            **latest_defaults,
            "hour": 5,
            "day_of_week": 6,
            "is_weekend": 1,
            "lag_1": latest_defaults["rolling_mean_24"] * 0.88,
            "lag_24": latest_defaults["rolling_mean_24"] * 0.90,
        },
        "Summer High Load": {
            **latest_defaults,
            "month": 7,
            "quarter": 3,
            "hour": 16,
            "is_weekend": 0,
            "lag_1": latest_defaults["rolling_mean_24"] * 1.08,
            "lag_24": latest_defaults["rolling_mean_24"] * 1.07,
        },
        "Winter Evening Peak": {
            **latest_defaults,
            "month": 1,
            "quarter": 1,
            "hour": 19,
            "is_weekend": 0,
            "lag_1": latest_defaults["rolling_mean_24"] * 1.05,
            "lag_24": latest_defaults["rolling_mean_24"] * 1.02,
        },
    }


def initialize_state(defaults: dict) -> None:
    for key, value in defaults.items():
        if key == "reference_timestamp":
            continue
        if key not in st.session_state:
            st.session_state[key] = value


def apply_values_to_state(values: dict) -> None:
    for key, value in values.items():
        if key in FEATURE_ORDER:
            st.session_state[key] = value


def feature_importance_df(model) -> pd.DataFrame:
    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    fi = pd.DataFrame(
        {
            "feature": FEATURE_ORDER,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return fi


# -------------------------------------------------
# Load resources
# -------------------------------------------------
model = load_model()
processed_df = load_processed_data()
metrics = load_metrics()
pred_df = load_test_predictions()
residual_std = compute_residual_std(pred_df)

latest_defaults = get_latest_feature_defaults(processed_df)
initialize_state(latest_defaults)
presets = scenario_presets(latest_defaults)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("Energy Demand Forecast Dashboard")

st.markdown(
    """
This dashboard presents an end-to-end energy demand forecasting system that combines machine learning with an interactive interface.

It enables users to explore model performance, generate future forecasts, and simulate real-world energy demand scenarios in a simple and intuitive way.

- **Saved Model Evaluation** displays model performance using key metrics and compares actual vs predicted energy demand.
- **Future Forecast** generates multi-step predictions to support planning and decision-making.
- **Single Forecast** allows users to input custom conditions and instantly estimate energy demand.
- **Prediction Logs** track recent model outputs to simulate real-world monitoring.
- **Methodology** explains the modeling workflow, features, and deployment design.

This system demonstrates how machine learning models can be deployed in practice to support energy planning, resource allocation, and operational decision-making.
"""
)

st.caption("Adjust the inputs below to simulate different energy demand scenarios or use quick presets.")

# -------------------------------------------------
# Top summary cards
# -------------------------------------------------
summary_cols = st.columns(4)

if processed_df is not None and not processed_df.empty:
    latest_actual = float(processed_df["PJME_MW"].iloc[-1])
    summary_cols[0].metric("Latest Observed Demand (MW)", f"{latest_actual:,.0f}")

    if model is not None:
        next_hour_payload = build_feature_row(processed_df.index[-1] + timedelta(hours=1), processed_df["PJME_MW"].tolist())
        next_hour_pred = predict_single(model, next_hour_payload)
        summary_cols[1].metric("Next-Hour Forecast (MW)", f"{next_hour_pred:,.0f}")

        forecast_24 = forecast_future(model, processed_df, 24)
        avg_24 = float(forecast_24["predicted_MW"].mean())
        peak_24 = float(forecast_24["predicted_MW"].max())
        summary_cols[2].metric("24-Hour Avg Forecast (MW)", f"{avg_24:,.0f}")
        summary_cols[3].metric("24-Hour Peak Forecast (MW)", f"{peak_24:,.0f}")
    else:
        summary_cols[1].metric("Next-Hour Forecast (MW)", "N/A")
        summary_cols[2].metric("24-Hour Avg Forecast (MW)", "N/A")
        summary_cols[3].metric("24-Hour Peak Forecast (MW)", "N/A")
else:
    summary_cols[0].metric("Latest Observed Demand (MW)", "N/A")
    summary_cols[1].metric("Next-Hour Forecast (MW)", "N/A")
    summary_cols[2].metric("24-Hour Avg Forecast (MW)", "N/A")
    summary_cols[3].metric("24-Hour Peak Forecast (MW)", "N/A")

with st.expander("What do these inputs mean?"):
    st.markdown(
        """
- **Hour**: Time of day (0–23).
- **Day of Week**: 0 = Monday, 6 = Sunday.
- **Lag 1**: Energy demand from the previous hour.
- **Lag 24**: Demand at the same hour yesterday.
- **Lag 168**: Demand at the same hour last week.
- **Rolling Mean (24)**: Average demand over the past 24 hours.
- **Rolling Std (24)**: Variability in demand over the past 24 hours.
- **Rolling Mean (168)**: Weekly average demand.
- **Rolling Std (168)**: Weekly variability.

These features help the model capture daily and weekly patterns in energy usage.
"""
    )

st.divider()

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Single Forecast",
        "Saved Model Evaluation",
        "Future Forecast",
        "Prediction Logs",
        "Methodology",
    ]
)

# -------------------------------------------------
# Tab 1: Single Forecast
# -------------------------------------------------
with tab1:
    st.subheader("Single Energy Demand Prediction")

    preset_cols = st.columns(5)
    if preset_cols[0].button("Use Latest Values"):
        apply_values_to_state(latest_defaults)
        st.rerun()

    preset_names = list(presets.keys())
    for idx, preset_name in enumerate(preset_names, start=1):
        if preset_cols[idx].button(preset_name):
            apply_values_to_state(presets[preset_name])
            st.rerun()

    if "reference_timestamp" in latest_defaults:
        st.caption(f"Latest reference row loaded from: {latest_defaults['reference_timestamp']}")

    with st.form("single_forecast_form"):
        col1, col2 = st.columns(2)

        with col1:
            hour = st.number_input("Hour", min_value=0, max_value=23, key="hour")
            day_of_week = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, key="day_of_week")
            day_of_month = st.number_input("Day of Month", min_value=1, max_value=31, key="day_of_month")
            month = st.number_input("Month", min_value=1, max_value=12, key="month")
            quarter = st.number_input("Quarter", min_value=1, max_value=4, key="quarter")
            year = st.number_input("Year", min_value=2000, max_value=2100, key="year")
            is_weekend = st.selectbox("Is Weekend", options=[0, 1], key="is_weekend")

        with col2:
            lag_1 = st.number_input("Lag 1", min_value=0.0, key="lag_1")
            lag_24 = st.number_input("Lag 24", min_value=0.0, key="lag_24")
            lag_168 = st.number_input("Lag 168", min_value=0.0, key="lag_168")
            rolling_mean_24 = st.number_input("Rolling Mean 24", min_value=0.0, key="rolling_mean_24")
            rolling_std_24 = st.number_input("Rolling Std 24", min_value=0.0, key="rolling_std_24")
            rolling_mean_168 = st.number_input("Rolling Mean 168", min_value=0.0, key="rolling_mean_168")
            rolling_std_168 = st.number_input("Rolling Std 168", min_value=0.0, key="rolling_std_168")

        submitted = st.form_submit_button("Get Forecast")

    if submitted:
        if model is None:
            st.error("Model file not found. Add models/xgboost_model.json to the repository.")
        else:
            payload = {
                "hour": int(hour),
                "day_of_week": int(day_of_week),
                "day_of_month": int(day_of_month),
                "month": int(month),
                "quarter": int(quarter),
                "year": int(year),
                "is_weekend": int(is_weekend),
                "lag_1": float(lag_1),
                "lag_24": float(lag_24),
                "lag_168": float(lag_168),
                "rolling_mean_24": float(rolling_mean_24),
                "rolling_std_24": float(rolling_std_24),
                "rolling_mean_168": float(rolling_mean_168),
                "rolling_std_168": float(rolling_std_168),
            }

            prediction = predict_single(model, payload)
            log_prediction(prediction, mode="single_forecast")

            result_cols = st.columns(3)
            result_cols[0].metric("Predicted Demand (MW)", f"{prediction:,.2f}")
            result_cols[1].metric(
                "Difference vs Lag 1",
                f"{prediction - payload['lag_1']:,.2f}",
            )
            result_cols[2].metric(
                "Difference vs 24h Mean",
                f"{prediction - payload['rolling_mean_24']:,.2f}",
            )

            st.success("Forecast generated successfully.")

# -------------------------------------------------
# Tab 2: Saved Model Evaluation
# -------------------------------------------------
with tab2:
    st.subheader("Saved Model Evaluation on Test Data")
    st.info("These metrics and plots come from the most recent training run and remain unchanged until the model is retrained.")

    if metrics is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", metrics.get("MAE", "N/A"))
        c2.metric("RMSE", metrics.get("RMSE", "N/A"))
        c3.metric("MAPE (%)", metrics.get("MAPE", "N/A"))
    else:
        st.warning("metrics.json not found. Run src/train.py and save artifacts.")

    if pred_df is not None and not pred_df.empty:
        view_window = st.selectbox(
            "Select evaluation window",
            options=["Last 168 points", "Last 500 points", "All points"],
            index=1,
        )

        if view_window == "Last 168 points":
            plot_df = pred_df.tail(168)
        elif view_window == "Last 500 points":
            plot_df = pred_df.tail(500)
        else:
            plot_df = pred_df.copy()

        st.markdown("### Actual vs Predicted")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(plot_df["Datetime"], plot_df["actual_MW"], label="Actual")
        ax1.plot(plot_df["Datetime"], plot_df["predicted_MW"], label="Predicted")
        ax1.set_title("Actual vs Predicted Energy Demand")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("MW")
        ax1.legend()
        plt.xticks(rotation=30)
        st.pyplot(fig1)

        residuals = pred_df["actual_MW"] - pred_df["predicted_MW"]

        eval_col1, eval_col2 = st.columns(2)

        with eval_col1:
            st.markdown("### Residuals Over Time")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(pred_df["Datetime"], residuals)
            ax2.axhline(0, linestyle="--")
            ax2.set_title("Residuals (Actual - Predicted)")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("MW")
            plt.xticks(rotation=30)
            st.pyplot(fig2)

        with eval_col2:
            st.markdown("### Predicted vs Actual Scatter")
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            ax3.scatter(pred_df["actual_MW"], pred_df["predicted_MW"], alpha=0.5)
            min_val = min(pred_df["actual_MW"].min(), pred_df["predicted_MW"].min())
            max_val = max(pred_df["actual_MW"].max(), pred_df["predicted_MW"].max())
            ax3.plot([min_val, max_val], [min_val, max_val], linestyle="--")
            ax3.set_xlabel("Actual MW")
            ax3.set_ylabel("Predicted MW")
            ax3.set_title("Predicted vs Actual")
            st.pyplot(fig3)

        st.markdown("### Error Distribution")
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.hist(residuals, bins=30)
        ax4.set_title("Residual Distribution")
        ax4.set_xlabel("Residual (MW)")
        ax4.set_ylabel("Frequency")
        st.pyplot(fig4)

        with st.expander("View saved prediction samples"):
            st.dataframe(pred_df.head(20))

        csv_eval = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Evaluation Predictions CSV",
            data=csv_eval,
            file_name="test_predictions.csv",
            mime="text/csv",
        )
    else:
        st.warning("test_predictions.csv not found. Run src/train.py and save artifacts.")

    st.markdown("### Feature Importance")
    fi_df = feature_importance_df(model)
    if not fi_df.empty:
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        top_fi = fi_df.head(10).sort_values("importance")
        ax5.barh(top_fi["feature"], top_fi["importance"])
        ax5.set_title("Top 10 Feature Importances")
        ax5.set_xlabel("Importance")
        st.pyplot(fig5)
        st.dataframe(fi_df)
    else:
        st.warning("Feature importances unavailable. Ensure the model is loaded successfully.")

    st.markdown("### Model Metadata")
    meta_rows = []
    if processed_df is not None and not processed_df.empty:
        meta_rows.append(("Dataset Start", str(processed_df.index.min())))
        meta_rows.append(("Dataset End", str(processed_df.index.max())))
        meta_rows.append(("Rows", f"{len(processed_df):,}"))
    meta_rows.extend(
        [
            ("Model Type", "XGBoost Regressor"),
            ("Validation Strategy", "Time-based train/test split"),
            ("Primary Features", "Calendar, lag, and rolling statistics"),
        ]
    )
    meta_df = pd.DataFrame(meta_rows, columns=["Field", "Value"])
    st.dataframe(meta_df)

# -------------------------------------------------
# Tab 3: Future Forecast
# -------------------------------------------------
with tab3:
    st.subheader("Future Energy Demand Forecast")

    if model is None:
        st.error("Model file not found. Add models/xgboost_model.json to the repository.")
    elif processed_df is None or processed_df.empty:
        st.warning("Processed dataset not found. Add data/processed/pjm_energy_processed.csv to the repository.")
    else:
        horizon = st.selectbox("Forecast Horizon (hours)", [24, 48, 72], index=0)
        history_window = st.selectbox("Historical context to display", [48, 72, 168], index=1)

        if st.button("Generate Future Forecast"):
            forecast_df = forecast_future(model, processed_df, horizon)

            if not forecast_df.empty:
                for value in forecast_df["predicted_MW"]:
                    log_prediction(float(value), mode="future_forecast", horizon=horizon)

                historical_df = processed_df.tail(history_window).reset_index()[["Datetime", "PJME_MW"]]
                historical_df = historical_df.rename(columns={"PJME_MW": "actual_MW"})

                if residual_std > 0:
                    forecast_df["lower_bound"] = forecast_df["predicted_MW"] - 1.96 * residual_std
                    forecast_df["upper_bound"] = forecast_df["predicted_MW"] + 1.96 * residual_std
                else:
                    forecast_df["lower_bound"] = forecast_df["predicted_MW"]
                    forecast_df["upper_bound"] = forecast_df["predicted_MW"]

                st.success(f"Generated {horizon}-hour forecast successfully.")

                summary_a, summary_b, summary_c = st.columns(3)
                summary_a.metric("Forecast Average (MW)", f"{forecast_df['predicted_MW'].mean():,.0f}")
                summary_b.metric("Forecast Peak (MW)", f"{forecast_df['predicted_MW'].max():,.0f}")
                summary_c.metric("Forecast Minimum (MW)", f"{forecast_df['predicted_MW'].min():,.0f}")

                st.markdown("### Historical + Future Forecast")
                fig6, ax6 = plt.subplots(figsize=(12, 5))
                ax6.plot(historical_df["Datetime"], historical_df["actual_MW"], label="Historical Actual")
                ax6.plot(forecast_df["Datetime"], forecast_df["predicted_MW"], label="Forecast")
                ax6.fill_between(
                    forecast_df["Datetime"],
                    forecast_df["lower_bound"],
                    forecast_df["upper_bound"],
                    alpha=0.2,
                    label="Approx. uncertainty band",
                )
                ax6.set_title("Historical Demand and Future Forecast")
                ax6.set_xlabel("Time")
                ax6.set_ylabel("MW")
                ax6.legend()
                plt.xticks(rotation=30)
                st.pyplot(fig6)

                st.markdown("### Forecast Table")
                st.dataframe(forecast_df)

                forecast_csv = forecast_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Forecast CSV",
                    data=forecast_csv,
                    file_name=f"future_forecast_{horizon}h.csv",
                    mime="text/csv",
                )

# -------------------------------------------------
# Tab 4: Prediction Logs
# -------------------------------------------------
with tab4:
    st.subheader("Prediction Logs and Monitoring")

    if LOGS_PATH.exists():
        logs_df = pd.read_csv(LOGS_PATH, parse_dates=["timestamp"], on_bad_lines="skip")
        
        

        st.markdown("### Recent Logged Predictions")
        st.dataframe(logs_df.tail(25))

        if not logs_df.empty:
            log_col1, log_col2 = st.columns(2)

            with log_col1:
                fig7, ax7 = plt.subplots(figsize=(10, 4))
                ax7.plot(logs_df["timestamp"], logs_df["prediction_MW"])
                ax7.set_title("Logged Predictions Over Time")
                ax7.set_xlabel("Timestamp")
                ax7.set_ylabel("Predicted MW")
                plt.xticks(rotation=30)
                st.pyplot(fig7)

            with log_col2:
                mode_counts = logs_df["mode"].fillna("unknown").value_counts()
                fig8, ax8 = plt.subplots(figsize=(6, 4))
                ax8.bar(mode_counts.index, mode_counts.values)
                ax8.set_title("Prediction Volume by Mode")
                ax8.set_xlabel("Mode")
                ax8.set_ylabel("Count")
                plt.xticks(rotation=20)
                st.pyplot(fig8)

        logs_csv = logs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Logs CSV",
            data=logs_csv,
            file_name="prediction_logs.csv",
            mime="text/csv",
        )
    else:
        st.warning("No prediction logs found yet. Make a few predictions first.")

# -------------------------------------------------
# Tab 5: Methodology
# -------------------------------------------------
with tab5:
    st.subheader("Methodology and Design")

    st.markdown(
        """
### Problem Framing
This application forecasts electricity demand using time-series feature engineering and machine learning. The goal is to support planning, load management, and operational decision-making.

### Data Preparation
The workflow starts with cleaned hourly demand data. The preprocessing pipeline creates:
- calendar-based features such as hour, weekday, month, and quarter
- lag features capturing recent and seasonal demand patterns
- rolling mean and rolling standard deviation features to capture short-term and weekly behavior

### Modeling
The forecasting model is an **XGBoost Regressor** trained using a **time-based split** to avoid leakage and preserve chronological structure.

### Evaluation
The app reports standard forecasting metrics:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

### Deployment Design
This project was built with a production-style structure:
- model training and preprocessing scripts
- interactive dashboard
- deployable app design
- prediction logging for simple monitoring

### Business Value
Energy demand forecasting can support:
- load planning
- resource allocation
- scenario analysis
- operational monitoring
"""
    )