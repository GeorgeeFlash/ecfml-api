from pydantic import BaseModel

from app.schemas.common import EngineType, Resolution


class ForecastRequest(BaseModel):
    engine: EngineType
    preprocess_job_id: str
    model_run_id: str | None = None
    start_date: str
    horizon_days: int
    resolution: Resolution
    model_override: str | None = None
    processed_file_path: str | None = None


class ForecastResponse(BaseModel):
    forecast_id: str
    engine: EngineType
    predictions: list[dict]
    agent_reasoning: str | None = None
    confidence: str | None = None
    agent_run_id: str | None = None
