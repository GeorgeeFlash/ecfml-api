from pydantic import BaseModel, Field

from app.schemas.common import JobStatus


class SplitConfig(BaseModel):
    train: float = Field(0.7, ge=0.0, le=1.0)
    val: float = Field(0.15, ge=0.0, le=1.0)
    test: float = Field(0.15, ge=0.0, le=1.0)


class PreprocessingRunRequest(BaseModel):
    job_id: str
    dataset_id: str
    dataset_url: str
    weather_url: str | None = None
    splits: SplitConfig | None = None


class PreprocessingStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float | None = None
    error: str | None = None
    processed_file_path: str | None = None
    result_summary: dict | None = None
    eda_charts: dict | None = None
