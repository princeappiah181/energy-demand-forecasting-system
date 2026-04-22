import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import numpy as np
import json


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, parse_dates=["Datetime"], index_col="Datetime")


def train_test_split_time(df: pd.DataFrame, split_ratio: float = 0.8):
    split_index = int(len(df) * split_ratio)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test


def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=["PJME_MW"])
    y = df["PJME_MW"]
    return X, y


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


def save_artifacts(y_test, y_pred, mae, rmse, mape):
    Path("artifacts").mkdir(exist_ok=True)

    metrics = {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE": round(float(mape), 2)
    }

    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    pred_df = pd.DataFrame({
        "Datetime": y_test.index,
        "actual_MW": y_test.values,
        "predicted_MW": y_pred
    })
    pred_df.to_csv("artifacts/test_predictions.csv", index=False)

    print("Saved artifacts/metrics.json")
    print("Saved artifacts/test_predictions.csv")


def main():
    data_path = Path("data/processed/pjm_energy_processed.csv")
    if not data_path.exists():
        print("Processed data not found. Run preprocessing first.")
        return

    df = load_data(data_path)

    train, test = train_test_split_time(df)
    print(f"Train size: {train.shape}")
    print(f"Test size: {test.shape}")

    X_train, y_train = prepare_features(train)
    X_test, y_test = prepare_features(test)

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae, rmse, mape = evaluate(y_test, y_pred)

    print("\nModel Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")

    Path("models").mkdir(exist_ok=True)
    model.save_model("models/xgboost_model.json")
    print("\nModel saved to models/xgboost_model.json")

    save_artifacts(y_test, y_pred, mae, rmse, mape)


if __name__ == "__main__":
    main()