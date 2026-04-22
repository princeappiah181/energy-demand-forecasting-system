import pandas as pd
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the PJM energy consumption dataset.
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the time series data.
    """
    # Convert Datetime column to datetime type
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # Sort by time
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Remove duplicate timestamps if any
    df = df.drop_duplicates(subset=["Datetime"])

    # Set Datetime as index
    df = df.set_index("Datetime")

    # Create time-based features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)

    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag and rolling features for time series forecasting.
    """

    # Lag features (previous values)
    df["lag_1"] = df["PJME_MW"].shift(1)
    df["lag_24"] = df["PJME_MW"].shift(24)     # same hour previous day
    df["lag_168"] = df["PJME_MW"].shift(168)   # same hour previous week

    # Rolling statistics
    df["rolling_mean_24"] = df["PJME_MW"].rolling(window=24).mean()
    df["rolling_std_24"] = df["PJME_MW"].rolling(window=24).std()

    df["rolling_mean_168"] = df["PJME_MW"].rolling(window=168).mean()
    df["rolling_std_168"] = df["PJME_MW"].rolling(window=168).std()

    return df




def inspect_processed_data(df: pd.DataFrame) -> None:
    """
    Print summary of processed data.
    """
    print("Processed data preview:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nIndex range:")
    print(f"Start: {df.index.min()}")
    print(f"End:   {df.index.max()}")

    print("\nShape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed data to CSV.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"\nProcessed data saved to: {output_path}")


def main():
    input_path = Path("data/raw/PJME_hourly.csv")
    output_path = "data/processed/pjm_energy_processed.csv"

    if not input_path.exists():
        print(f"File not found: {input_path}")
        print("Please place your PJM dataset inside data/raw/")
        return
  
    df = load_data(input_path)
    df_processed = preprocess_data(df)

    # NEW STEP
    df_processed = create_lag_features(df_processed)

    # Drop rows with NaN caused by lagging
    df_processed = df_processed.dropna()

    inspect_processed_data(df_processed)
    save_processed_data(df_processed, output_path)


if __name__ == "__main__":
    main()