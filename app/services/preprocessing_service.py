from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config import settings
from app.database import get_session
from app.ml.preprocessing import build_feature_frame, clip_outliers, impute_missing, time_split
from app.models.db import PreprocessingJob
from app.schemas.preprocessing import PreprocessingRunRequest, PreprocessingStatusResponse, SplitConfig
from app.services.data_utils import load_dataset
from app.utils.job_store import job_store
from sqlmodel import Session, select


def _merge_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in weather.columns:
        return df
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce")
    return pd.merge(df, weather, on="timestamp", how="left")


def run_preprocessing_task(request: PreprocessingRunRequest, user_id: str, db_job_id: str):
    job_store.init_job(request.job_id, status="RUNNING")

    try:
        splits = request.splits or SplitConfig()
        processed_dir = Path(settings.DATA_DIR) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        df = load_dataset(request.dataset_url)
        if "timestamp" not in df.columns or "consumption_kwh" not in df.columns:
            error_msg = "Dataset must include timestamp and consumption_kwh"
            job_store.update_job(request.job_id, status="FAILED", error=error_msg)
            with next(get_session()) as session:
                db_job = session.get(PreprocessingJob, db_job_id)
                if db_job:
                    db_job.status = "FAILED"
                    db_job.error = error_msg
                    session.add(db_job)
                    session.commit()
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        if request.weather_url:
            weather = load_dataset(request.weather_url)
            df = _merge_weather(df, weather)

        df = impute_missing(df)
        df = clip_outliers(df, "consumption_kwh")

        X, y, feature_columns = build_feature_frame(df)
        X_train, X_val, X_test, y_train, y_val, y_test = time_split(
            X, y, splits.train, splits.val, splits.test
        )

        processed_path = processed_dir / f"{request.job_id}.parquet"
        processed_df = pd.concat(
            [
                X_train.assign(split="train", target=y_train, timestamp=X_train.index),
                X_val.assign(split="val", target=y_val, timestamp=X_val.index),
                X_test.assign(split="test", target=y_test, timestamp=X_test.index),
            ]
        )
        processed_df.to_parquet(processed_path, index=False)

        summary = {
            "feature_columns": feature_columns,
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
            "processed_file_path": str(processed_path),
        }

        # Update Job Store
        job_store.update_job(request.job_id, status="COMPLETE", progress=100.0, meta=summary)

        # Update Database
        with next(get_session()) as session:
            db_job = session.get(PreprocessingJob, db_job_id)
            if db_job:
                db_job.status = "COMPLETE"
                db_job.processed_file_path = str(processed_path)
                db_job.result_summary = summary
                session.add(db_job)
                session.commit()

    except Exception as e:
        error_msg = str(e)
        job_store.update_job(request.job_id, status="FAILED", error=error_msg)
        with next(get_session()) as session:
            db_job = session.get(PreprocessingJob, db_job_id)
            if db_job:
                db_job.status = "FAILED"
                db_job.error = error_msg
                session.add(db_job)
                session.commit()


def get_preprocessing_status(job_id: str, session: Session | None = None, user_id: str | None = None) -> PreprocessingStatusResponse:
    # Try DB first
    if session:
        statement = select(PreprocessingJob).where(PreprocessingJob.job_id == job_id)
        if user_id:
            statement = statement.where(PreprocessingJob.user_id == user_id)
        db_job = session.exec(statement).first()
        if db_job:
            return PreprocessingStatusResponse(
                job_id=job_id,
                status=db_job.status,
                progress=100.0 if db_job.status == "COMPLETE" else 0.0,
                error=db_job.error,
                processed_file_path=db_job.processed_file_path,
                result_summary=db_job.result_summary,
                eda_charts=None,
            )

    # Fallback to Job Store
    job = job_store.get_job(job_id)
    if not job:
        return PreprocessingStatusResponse(job_id=job_id, status="PENDING")

    meta = job.get("meta", {})
    return PreprocessingStatusResponse(
        job_id=job_id,
        status=job.get("status", "PENDING"),
        progress=job.get("progress"),
        error=job.get("error"),
        processed_file_path=meta.get("processed_file_path"),
        result_summary=meta or None,
        eda_charts=None,
    )
