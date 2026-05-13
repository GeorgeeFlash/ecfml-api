import uuid
from datetime import datetime
from typing import Optional
from sqlmodel import Field, Relationship, SQLModel, JSON, Column

class Dataset(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(index=True)
    name: str
    file_url: str
    uploadthing_key: Optional[str] = None
    row_count: Optional[int] = None
    validation_status: str = Field(default="PENDING")
    validation_report: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: Optional[datetime] = None

    weather_datasets: list["WeatherDataset"] = Relationship(back_populates="dataset")

class WeatherDataset(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(index=True)
    dataset_id: str = Field(foreign_key="dataset.id")
    file_url: str
    uploadthing_key: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    dataset: Dataset = Relationship(back_populates="weather_datasets")

class Forecast(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(index=True)
    dataset_id: Optional[str] = None
    engine: str
    status: str = Field(default="PENDING")
    predictions: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    reasoning: Optional[str] = None
    confidence: Optional[str] = None
    start_date: str
    horizon_days: int
    resolution: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TrainedModel(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(index=True)
    name: str
    model_type: str
    job_id: str = Field(index=True)
    status: str = Field(default="PENDING")
    model_file_path: Optional[str] = None
    feature_importance: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    training_time_secs: Optional[float] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class PreprocessingJob(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    user_id: str = Field(index=True)
    dataset_id: str
    job_id: str = Field(index=True) # External/Frontend job ID
    status: str = Field(default="PENDING")
    processed_file_path: Optional[str] = None
    result_summary: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
