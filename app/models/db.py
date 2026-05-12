from datetime import datetime
from typing import Optional
from sqlmodel import Field, Relationship, SQLModel, JSON, Column

class Dataset(SQLModel, table=True):
    id: str = Field(primary_key=True)
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
    id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    dataset_id: str = Field(foreign_key="dataset.id")
    file_url: str
    uploadthing_key: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    dataset: Dataset = Relationship(back_populates="weather_datasets")
