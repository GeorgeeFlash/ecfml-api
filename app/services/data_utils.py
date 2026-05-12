from __future__ import annotations

import io
import os
from urllib.parse import urlparse

import httpx
import pandas as pd


def _read_bytes(data: bytes, suffix: str) -> pd.DataFrame:
    if suffix.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(data))
    if suffix.endswith(".xlsx") or suffix.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def _read_file(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_dataset(file_url: str) -> pd.DataFrame:
    parsed = urlparse(file_url)
    if parsed.scheme in ("http", "https"):
        with httpx.Client(timeout=30.0) as client:
            response = client.get(file_url)
            response.raise_for_status()
            suffix = os.path.basename(parsed.path).lower()
            return _read_bytes(response.content, suffix)

    if parsed.scheme == "file":
        return _read_file(parsed.path)

    return _read_file(file_url)


def normalize_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError("Dataset must include a timestamp column")

    result = df.copy()
    result[column] = pd.to_datetime(result[column], errors="coerce")
    result = result.dropna(subset=[column])
    result = result.sort_values(column)
    result = result.set_index(column)
    return result


def load_processed_frame(path: str) -> pd.DataFrame:
    df = _read_file(path)
    if "timestamp" in df.columns:
        df = normalize_timestamp(df, "timestamp")
    return df
