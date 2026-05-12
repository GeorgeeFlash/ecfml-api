from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from app.config import settings
from app.ml.models import get_feature_importance, train_rf, train_svr
from app.ml.persistence import save_sklearn_model
from app.schemas.model import ModelTrainRequest, ModelTrainResponse, RFHyperparams, SVRHyperparams
from app.utils.job_store import job_store


def _default_processed_path(preprocess_job_id: str) -> str:
    return str(Path(settings.DATA_DIR) / "processed" / f"{preprocess_job_id}.parquet")


def _load_processed(processed_file_path: str):
    df = pd.read_parquet(processed_file_path)
    feature_columns = [
        col for col in df.columns if col not in ("target", "split", "timestamp")
    ]
    train_df = df[df["split"] == "train"]

    X_train = train_df[feature_columns]
    y_train = train_df["target"]
    return X_train, y_train, feature_columns


def train_model(request: ModelTrainRequest) -> ModelTrainResponse:
    job_store.init_job(request.job_id, status="RUNNING")

    processed_path = request.processed_file_path or _default_processed_path(
        request.preprocess_job_id
    )
    if not os.path.exists(processed_path):
        job_store.update_job(request.job_id, status="FAILED", error="Processed file not found")
        raise FileNotFoundError("Processed file not found")

    X_train, y_train, feature_columns = _load_processed(processed_path)

    model_file_path = None
    training_time = None
    feature_importance = None

    if request.model_type.value == "RANDOM_FOREST":
        params = RFHyperparams(**(request.hyperparams or {}))
        model, training_time = train_rf(X_train, y_train, params, request.job_id, job_store)
        model_file_path = save_sklearn_model(model, request.job_id)
        feature_importance = get_feature_importance(model, feature_columns)
    else:
        params = SVRHyperparams(**(request.hyperparams or {}))
        model, training_time = train_svr(X_train, y_train, params, request.job_id, job_store)
        model_file_path = save_sklearn_model(model, request.job_id)

    job_store.update_job(
        request.job_id,
        status="COMPLETE",
        progress=100.0,
        meta={
            "model_file_path": model_file_path,
            "processed_file_path": processed_path,
            "model_type": request.model_type.value,
            "feature_names": feature_columns,
            "feature_importance": feature_importance,
        },
    )

    return ModelTrainResponse(
        job_id=request.job_id,
        status="COMPLETE",
        model_file_path=model_file_path,
        scaler_file_path=None,
        training_time_secs=training_time,
        error=None,
    )


def get_training_status(job_id: str) -> ModelTrainResponse:
    job = job_store.get_job(job_id)
    if not job:
        return ModelTrainResponse(job_id=job_id, status="FAILED", error="Job not found")

    meta = job.get("meta", {})
    return ModelTrainResponse(
        job_id=job_id,
        status=job.get("status", "PENDING"),
        model_file_path=meta.get("model_file_path"),
        scaler_file_path=None,
        training_time_secs=None,
        error=job.get("error"),
    )
