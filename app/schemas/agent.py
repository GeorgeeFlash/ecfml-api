from pydantic import BaseModel, Field


class NodeEvent(BaseModel):
    type: str
    node: str | None = None
    message: str | None = None
    data: dict | None = None


class ForecastOutput(BaseModel):
    predictions: list[dict] = Field(default_factory=list)
    reasoning: str
    confidence: str
