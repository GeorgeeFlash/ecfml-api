from __future__ import annotations

import pandas as pd


def ensure_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError("Dataset must include a timestamp column")

    result = df.copy()
    result[column] = pd.to_datetime(result[column], errors="coerce")
    result = result.dropna(subset=[column])
    result = result.sort_values(column)
    result = result.set_index(column)
    return result


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["hour"] = result.index.hour
    result["dayofweek"] = result.index.dayofweek
    result["month"] = result.index.month
    result["is_weekend"] = (result.index.dayofweek >= 5).astype(int)
    return result


def add_lag_features(
    df: pd.DataFrame,
    target: str = "consumption_kwh",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    if lags is None:
        lags = [1, 24, 168]

    result = df.copy()
    for lag in lags:
        result[f"lag_{lag}"] = result[target].shift(lag)

    result["rolling_24h"] = result[target].rolling(24).mean()
    result["rolling_7d"] = result[target].rolling(168).mean()
    return result


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if "consumption_kwh" not in df.columns:
        raise ValueError("Dataset must include consumption_kwh column")

    result = ensure_timestamp(df)
    result = add_time_features(result)
    result = add_lag_features(result)

    feature_columns = [
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_24h",
        "rolling_7d",
    ]

    for weather_col in ("temperature", "humidity", "rainfall"):
        if weather_col in result.columns:
            feature_columns.append(weather_col)

    result = result.dropna(subset=feature_columns + ["consumption_kwh"])
    X = result[feature_columns]
    y = result["consumption_kwh"]
    return X, y, feature_columns
