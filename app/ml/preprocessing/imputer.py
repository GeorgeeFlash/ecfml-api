from __future__ import annotations

import pandas as pd


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = result.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        median = result[col].median()
        result[col] = result[col].fillna(median)

    non_numeric = [c for c in result.columns if c not in numeric_cols]
    for col in non_numeric:
        result[col] = result[col].fillna(method="ffill").fillna(method="bfill")

    return result
