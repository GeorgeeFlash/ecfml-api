from __future__ import annotations

import pandas as pd

from app.ml.evaluation import compute_metrics
from app.ml.models import get_feature_importance
from app.ml.persistence import load_sklearn_model
from app.schemas.model import ModelEvaluateRequest, ModelEvaluateResponse


def evaluate_model(request: ModelEvaluateRequest) -> ModelEvaluateResponse:
    if not request.model_file_path or not request.processed_file_path:
        raise ValueError("model_file_path and processed_file_path are required")

    model = load_sklearn_model(request.model_file_path)
    df = pd.read_parquet(request.processed_file_path)
    feature_columns = [
        col for col in df.columns if col not in ("target", "split", "timestamp")
    ]
    test_df = df[df["split"] == "test"]

    X_test = test_df[feature_columns]
    y_test = test_df["target"].to_numpy()
    timestamps = test_df.get("timestamp")

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    timestamp_values = (
        timestamps.tolist() if timestamps is not None else [None] * len(y_test)
    )

    actual = [
        {"timestamp": str(ts) if timestamps is not None else None, "value": float(val)}
        for ts, val in zip(timestamp_values, y_test.tolist())
    ]
    predicted = [
        {"timestamp": str(ts) if timestamps is not None else None, "value": float(val)}
        for ts, val in zip(timestamp_values, y_pred.tolist())
    ]

    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = get_feature_importance(model, feature_columns)

    return ModelEvaluateResponse(
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        mape=metrics["mape"],
        r2=metrics["r2"],
        test_set_size=len(y_test),
        actual=actual,
        predicted=predicted,
        feature_importance=feature_importance,
    )
