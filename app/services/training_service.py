import asyncio
import os
from pathlib import Path

import pandas as pd

from app.config import settings
from app.database import get_session
from app.ml.models import get_feature_importance, train_rf, train_svr
from app.ml.persistence import save_sklearn_model
from app.models.db import TrainedModel
from app.schemas.model import ModelTrainRequest, ModelTrainResponse, RFHyperparams, SVRHyperparams
from app.utils.job_store import job_store
from sqlmodel import Session


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


def run_training_task(request: ModelTrainRequest, user_id: str, model_id: str):
    job_store.init_job(request.job_id, status="RUNNING")

    try:
        processed_path = request.processed_file_path or _default_processed_path(
            request.preprocess_job_id
        )
        if not os.path.exists(processed_path):
            error_msg = "Processed file not found"
            job_store.update_job(request.job_id, status="FAILED", error=error_msg)
            with next(get_session()) as session:
                db_model = session.get(TrainedModel, model_id)
                if db_model:
                    db_model.status = "FAILED"
                    db_model.error = error_msg
                    session.add(db_model)
                    session.commit()
            return

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

        # Update Job Store (for compatibility/streaming)
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

        # Update Database
        with next(get_session()) as session:
            db_model = session.get(TrainedModel, model_id)
            if db_model:
                db_model.status = "COMPLETE"
                db_model.model_file_path = model_file_path
                db_model.feature_importance = feature_importance
                db_model.training_time_secs = training_time
                session.add(db_model)
                session.commit()

    except Exception as e:
        error_msg = str(e)
        job_store.update_job(request.job_id, status="FAILED", error=error_msg)
        with next(get_session()) as session:
            db_model = session.get(TrainedModel, model_id)
            if db_model:
                db_model.status = "FAILED"
                db_model.error = error_msg
                session.add(db_model)
                session.commit()


def get_training_status(job_id: str, session: Session | None = None, user_id: str | None = None) -> ModelTrainResponse:
    # Try DB first
    if session:
        from sqlmodel import select
        statement = select(TrainedModel).where(TrainedModel.job_id == job_id)
        if user_id:
            statement = statement.where(TrainedModel.user_id == user_id)
        db_model = session.exec(statement).first()
        if db_model:
            return ModelTrainResponse(
                job_id=job_id,
                status=db_model.status,
                model_file_path=db_model.model_file_path,
                scaler_file_path=None,
                training_time_secs=db_model.training_time_secs,
                error=db_model.error,
            )

    # Fallback to Job Store
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
