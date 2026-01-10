import pandas as pd

def detect_poor_sleep(df:pd.DataFrame, threshold=5):
    return df[df["sleep_hours"] < threshold]

def detect_high_resting_hr(df: pd.DataFrame, window=7):
    df = df.sort_values("date")

    rolling_avg = df["resting_heart_rate"].rolling(window, min_periods=window).mean()
    rolling_std = df["resting_heart_rate"].rolling(window, min_periods=window).std()

    anomalies = df[
        df["resting_heart_rate"] > rolling_avg + 1.5 * rolling_std
    ].copy()

    anomalies["reason"] = (
        "Resting heart rate significantly above recent baseline"
    )

    return anomalies
