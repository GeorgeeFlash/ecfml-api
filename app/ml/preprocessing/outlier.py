from __future__ import annotations

import pandas as pd


def clip_outliers(
    df: pd.DataFrame,
    column: str,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    if column not in df.columns:
        return df

    result = df.copy()
    lower = result[column].quantile(lower_quantile)
    upper = result[column].quantile(upper_quantile)
    result[column] = result[column].clip(lower=lower, upper=upper)
    return result
