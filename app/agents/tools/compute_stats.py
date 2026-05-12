import pandas as pd


def compute_stats(series: pd.Series) -> dict:
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
    }
