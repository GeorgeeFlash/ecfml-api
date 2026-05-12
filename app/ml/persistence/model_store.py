from __future__ import annotations

import os

import joblib

from app.config import settings


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_sklearn_model(model, model_run_id: str) -> str:
    _ensure_dir(settings.MODELS_DIR)
    path = os.path.join(settings.MODELS_DIR, f"{model_run_id}.joblib")
    joblib.dump(model, path)
    return path


def load_sklearn_model(path: str):
    return joblib.load(path)


def save_scaler(scaler, model_run_id: str) -> str:
    _ensure_dir(settings.MODELS_DIR)
    path = os.path.join(settings.MODELS_DIR, f"{model_run_id}_scaler.joblib")
    joblib.dump(scaler, path)
    return path


def load_scaler(path: str):
    return joblib.load(path)
