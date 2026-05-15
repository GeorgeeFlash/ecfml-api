from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scaler=None) -> dict:
    if scaler is not None:
        zeros = np.zeros((len(y_true), scaler.n_features_in_))
        zeros[:, 0] = y_true
        y_true = scaler.inverse_transform(zeros)[:, 0]

        zeros = np.zeros((len(y_pred), scaler.n_features_in_))
        zeros[:, 0] = y_pred
        y_pred = scaler.inverse_transform(zeros)[:, 0]

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r2": float(r2),
    }
