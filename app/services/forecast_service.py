from __future__ import annotations

import asyncio
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.config import settings
from app.database import get_session
from app.ml.persistence import load_sklearn_model
from app.models.db import Dataset, Forecast
from app.schemas.common import EngineType, Resolution
from app.schemas.forecast import ForecastRequest, ForecastResponse
from app.utils.job_store import job_store
from sqlmodel import Session, select

_agent_queues: dict[str, asyncio.Queue] = {}
_agent_results: dict[str, dict] = {}


def _default_processed_path(preprocess_job_id: str) -> str:
    return str(Path(settings.DATA_DIR) / "processed" / f"{preprocess_job_id}.parquet")


def _resolve_processed_path(request: ForecastRequest) -> str:
    if request.processed_file_path:
        return request.processed_file_path
    return _default_processed_path(request.preprocess_job_id)


def _resolve_model_path(request: ForecastRequest) -> str:
    if request.model_override:
        return request.model_override

    if request.model_run_id:
        job = job_store.get_job(request.model_run_id)
        if job:
            return job.get("meta", {}).get("model_file_path", "")
    return ""


def _step_for_resolution(resolution: Resolution) -> tuple[timedelta, int]:
    if resolution == Resolution.HOURLY:
        return timedelta(hours=1), 24
    if resolution == Resolution.DAILY:
        return timedelta(days=1), 1
    return timedelta(weeks=1), 1


def _generate_future_timestamps(
    start: datetime, horizon_days: int, resolution: Resolution
) -> list[datetime]:
    step, per_day = _step_for_resolution(resolution)
    if resolution == Resolution.WEEKLY:
        steps = max(1, math.ceil(horizon_days / 7))
    else:
        steps = max(1, horizon_days * per_day)
    return [start + (step * i) for i in range(steps)]


def _build_feature_row(
    ts: datetime,
    history: list[float],
    weather_defaults: dict[str, float],
    feature_columns: list[str],
) -> dict:
    lag_1 = history[-1]
    lag_24 = history[-24] if len(history) >= 24 else history[-1]
    lag_168 = history[-168] if len(history) >= 168 else history[-1]
    rolling_24h = sum(history[-24:]) / min(len(history), 24)
    rolling_7d = sum(history[-168:]) / min(len(history), 168)

    row = {
        "hour": ts.hour,
        "dayofweek": ts.weekday(),
        "month": ts.month,
        "is_weekend": 1 if ts.weekday() >= 5 else 0,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "lag_168": lag_168,
        "rolling_24h": rolling_24h,
        "rolling_7d": rolling_7d,
    }

    for key in ("temperature", "humidity", "rainfall"):
        if key in feature_columns:
            row[key] = weather_defaults.get(key, 0.0)

    return row


def _forecast_with_model(
    model,
    df: pd.DataFrame,
    start_date: str,
    horizon_days: int,
    resolution: Resolution,
) -> list[dict]:
    if "timestamp" not in df.columns:
        raise ValueError("Processed dataset must include timestamp")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    history = df["target"].astype(float).tolist()
    last_timestamp = df["timestamp"].max()
    start = pd.to_datetime(start_date)

    step, _ = _step_for_resolution(resolution)
    if start <= last_timestamp:
        start = last_timestamp + step

    feature_columns = [
        col for col in df.columns if col not in ("target", "split", "timestamp")
    ]

    weather_defaults = {
        key: float(df[key].iloc[-1])
        for key in ("temperature", "humidity", "rainfall")
        if key in df.columns
    }

    timestamps = _generate_future_timestamps(start, horizon_days, resolution)
    predictions = []

    for ts in timestamps:
        row = _build_feature_row(ts, history, weather_defaults, feature_columns)
        X = pd.DataFrame([row], columns=feature_columns)
        value = float(model.predict(X)[0])
        if value < 0:
            value = 0.0
        history.append(value)
        predictions.append({"timestamp": ts.isoformat(), "value": value})

    return predictions


async def start_agent_run(
    app_state, request: ForecastRequest, processed_path: str, user_id: str, forecast_id: str
) -> str:
    agent_run_id = forecast_id  # Use forecast_id as agent_run_id for consistency
    queue: asyncio.Queue = asyncio.Queue()
    _agent_queues[agent_run_id] = queue

    async def _run():
        try:
            graph = app_state.agent_graph
            state = {
                "dataset_path": processed_path,
                "forecast_params": {
                    "start_date": request.start_date,
                    "horizon_days": request.horizon_days,
                    "resolution": request.resolution.value,
                },
                "agent_run_id": agent_run_id,
                "revision_count": 0,
            }

            final_state = None
            async for part in graph.astream(
                state, version="v2", stream_mode=["custom", "values"]
            ):
                if part.get("type") == "custom":
                    await queue.put(part.get("data"))
                if part.get("type") == "values":
                    final_state = part.get("data")

            result = {
                "predictions": final_state.get("predictions", []) if final_state else [],
                "reasoning": final_state.get("reasoning") if final_state else None,
                "confidence": None,
            }
            _agent_results[agent_run_id] = result
            
            # Persist to database
            with next(get_session()) as session:
                db_forecast = session.get(Forecast, forecast_id)
                if db_forecast:
                    db_forecast.status = "COMPLETE"
                    db_forecast.predictions = result["predictions"]
                    db_forecast.reasoning = result["reasoning"]
                    session.add(db_forecast)
                    session.commit()

            await queue.put({"type": "complete", "data": result})
        except Exception as exc:
            # Update status to FAILED in DB
            try:
                with next(get_session()) as session:
                    db_forecast = session.get(Forecast, forecast_id)
                    if db_forecast:
                        db_forecast.status = "FAILED"
                        session.add(db_forecast)
                        session.commit()
            except Exception as db_exc:
                print(f"Failed to update forecast status: {db_exc}")
                
            await queue.put({"type": "error", "error": str(exc)})
        finally:
            await queue.put(None)

    asyncio.create_task(_run())
    return agent_run_id


def get_agent_queue(agent_run_id: str) -> asyncio.Queue | None:
    return _agent_queues.get(agent_run_id)


def get_agent_result(agent_run_id: str) -> dict | None:
    return _agent_results.get(agent_run_id)


async def create_forecast(
    app_state, request: ForecastRequest, session: Session, user_id: str
) -> ForecastResponse:
    processed_path = _resolve_processed_path(request)
    if not os.path.exists(processed_path):
        raise FileNotFoundError("Processed file not found")

    # Create initial forecast record
    db_forecast = Forecast(
        user_id=user_id,
        engine=request.engine.value,
        start_date=request.start_date,
        horizon_days=request.horizon_days,
        resolution=request.resolution.value,
        status="PENDING",
    )
    session.add(db_forecast)
    session.commit()
    session.refresh(db_forecast)

    if request.engine in (EngineType.RF, EngineType.SVR):
        try:
            model_path = _resolve_model_path(request)
            if not model_path:
                raise ValueError("Model file path required for RF/SVR forecasts")
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model file not found")

            model = load_sklearn_model(model_path)
            df = pd.read_parquet(processed_path)
            predictions = _forecast_with_model(
                model,
                df,
                request.start_date,
                request.horizon_days,
                request.resolution,
            )
            
            # Update record with results
            db_forecast.status = "COMPLETE"
            db_forecast.predictions = predictions
            session.add(db_forecast)
            session.commit()
            
            return ForecastResponse(
                forecast_id=db_forecast.id,
                engine=request.engine,
                predictions=predictions,
            )
        except Exception as e:
            db_forecast.status = "FAILED"
            session.add(db_forecast)
            session.commit()
            raise e

    agent_run_id = await start_agent_run(
        app_state, request, processed_path, user_id, db_forecast.id
    )
    return ForecastResponse(
        forecast_id=db_forecast.id,
        engine=request.engine,
        predictions=[],
        agent_run_id=agent_run_id,
    )
