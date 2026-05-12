from __future__ import annotations

import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.svm import SVR

from app.schemas.model import SVRHyperparams


def train_svr(X_train, y_train, params: SVRHyperparams, job_id: str, job_store: dict):
    start = time.time()
    job_store.update_job(job_id, status="RUNNING")

    pipeline = Pipeline(
        [
            ("scaler", SKStandardScaler()),
            (
                "svr",
                SVR(
                    C=params.c,
                    epsilon=params.epsilon,
                    kernel=params.kernel,
                    gamma=params.gamma,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    job_store.update_job(job_id, status="COMPLETE", progress=100.0)
    return pipeline, time.time() - start
