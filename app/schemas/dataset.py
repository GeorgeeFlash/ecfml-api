from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.schemas.common import ValidationStatus


class DatasetValidationRequest(BaseModel):
    file_url: str
    required_columns: list[str] | None = None


class DatasetValidationReport(BaseModel):
    status: ValidationStatus
    missing_columns: list[str] = []
    columns: list[str] = []
    row_count: int | None = None
    warnings: list[str] = []
    details: dict | None = None


class DatasetPreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict]
    row_count: int


class DatasetCreate(BaseModel):
    name: str
    file_url: str
    id: Optional[str] = None
    uploadthing_key: Optional[str] = None


class DatasetRead(DatasetCreate):
    id: str # Ensure id is present in read model
    user_id: str
    validation_status: str
    created_at: datetime
    row_count: Optional[int] = None


class WeatherDatasetCreate(BaseModel):
    dataset_id: str
    file_url: str
    id: Optional[str] = None
    uploadthing_key: Optional[str] = None
