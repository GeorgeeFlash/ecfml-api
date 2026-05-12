from __future__ import annotations

import time

from sklearn.ensemble import RandomForestRegressor

from app.schemas.model import RFHyperparams


def train_rf(X_train, y_train, params: RFHyperparams, job_id: str, job_store: dict):
    start = time.time()
    job_store.update_job(job_id, status="RUNNING")

    model = RandomForestRegressor(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        min_samples_split=params.min_samples_split,
        min_samples_leaf=params.min_samples_leaf,
        random_state=params.random_state,
        n_jobs=params.n_jobs,
    )
    model.fit(X_train, y_train)

    job_store.update_job(job_id, status="COMPLETE", progress=100.0)
    return model, time.time() - start


def get_feature_importance(model: RandomForestRegressor, feature_names: list[str]) -> list[dict]:
    if not hasattr(model, "feature_importances_"):
        return []

    ranked = [
        {"feature": name, "importance": round(float(imp), 6)}
        for name, imp in zip(feature_names, model.feature_importances_)
    ]
    ranked.sort(key=lambda item: item["importance"], reverse=True)
    return ranked[:15]
