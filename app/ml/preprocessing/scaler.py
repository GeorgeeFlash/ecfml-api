from __future__ import annotations

from sklearn.preprocessing import StandardScaler


def fit_scaler(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled


def apply_scaler(scaler: StandardScaler, X):
    return scaler.transform(X)
