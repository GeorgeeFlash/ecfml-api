from app.ml.models import train_rf
from app.schemas.model import RFHyperparams
from app.utils.job_store import job_store


def test_train_rf_runs():
    import pandas as pd

    X = pd.DataFrame({"f": [1, 2, 3, 4]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    model, duration = train_rf(X, y, RFHyperparams(n_estimators=10), "job", job_store)
    assert model is not None
    assert duration >= 0
