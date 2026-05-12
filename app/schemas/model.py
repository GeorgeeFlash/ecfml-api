from pydantic import BaseModel, Field

from app.schemas.common import JobStatus, ModelType


class RFHyperparams(BaseModel):
    n_estimators: int = Field(300, ge=10, le=2000)
    max_depth: int | None = None
    min_samples_split: int = Field(2, ge=2, le=50)
    min_samples_leaf: int = Field(1, ge=1, le=50)
    random_state: int = 42
    n_jobs: int = -1


class SVRHyperparams(BaseModel):
    c: float = Field(10.0, ge=0.01, le=1000.0)
    epsilon: float = Field(0.1, ge=0.0, le=10.0)
    kernel: str = "rbf"
    gamma: str = "scale"


class ModelTrainRequest(BaseModel):
    job_id: str
    preprocess_job_id: str
    model_type: ModelType
    hyperparams: dict | None = None
    processed_file_path: str | None = None


class ModelTrainResponse(BaseModel):
    job_id: str
    status: JobStatus
    model_file_path: str | None = None
    scaler_file_path: str | None = None
    training_time_secs: float | None = None
    error: str | None = None


class ModelEvaluateRequest(BaseModel):
    model_run_id: str | None = None
    model_file_path: str | None = None
    processed_file_path: str | None = None


class ModelEvaluateResponse(BaseModel):
    rmse: float
    mae: float
    mape: float
    r2: float
    test_set_size: int
    actual: list[dict]
    predicted: list[dict]
    feature_importance: list[dict] | None = None
