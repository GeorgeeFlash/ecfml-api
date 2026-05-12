from sqlmodel import Session, select
from datetime import datetime

from app.schemas.common import ValidationStatus
from app.schemas.dataset import (
    DatasetCreate,
    DatasetPreviewResponse,
    DatasetRead,
    DatasetValidationReport,
    WeatherDatasetCreate,
)
from app.services.data_utils import load_dataset
from app.models.db import Dataset, WeatherDataset

DEFAULT_REQUIRED_COLUMNS = ["timestamp", "consumption_kwh"]


def validate_dataset(
    file_url: str, required_columns: list[str] | None = None
) -> DatasetValidationReport:
    required = required_columns or DEFAULT_REQUIRED_COLUMNS
    df = load_dataset(file_url)
    columns = list(df.columns)

    missing = [col for col in required if col not in columns]
    status = ValidationStatus.VALID if not missing else ValidationStatus.INVALID
    warnings = []
    if status == ValidationStatus.VALID and len(df) < 100:
        status = ValidationStatus.WARNING
        warnings.append("Dataset has fewer than 100 rows.")

    return DatasetValidationReport(
        status=status,
        missing_columns=missing,
        columns=columns,
        row_count=len(df),
        warnings=warnings,
        details=None,
    )


def preview_dataset(file_url: str, rows: int = 100) -> DatasetPreviewResponse:
    df = load_dataset(file_url)
    subset = df.head(rows)
    return DatasetPreviewResponse(
        columns=list(subset.columns),
        rows=subset.to_dict(orient="records"),
        row_count=len(df),
    )


def create_dataset(session: Session, data: DatasetCreate, user_id: str) -> Dataset:
    db_dataset = Dataset(
        id=data.id,
        user_id=user_id,
        name=data.name,
        file_url=data.file_url,
        uploadthing_key=data.uploadthing_key,
    )
    # Proactively validate and update row count
    try:
        report = validate_dataset(data.file_url)
        db_dataset.validation_status = report.status.value
        db_dataset.row_count = report.row_count
        db_dataset.validation_report = report.model_dump()
    except Exception:
        db_dataset.validation_status = ValidationStatus.INVALID.value

    session.add(db_dataset)
    session.commit()
    session.refresh(db_dataset)
    return db_dataset


def list_datasets(session: Session, user_id: str) -> list[Dataset]:
    statement = select(Dataset).where(Dataset.user_id == user_id, Dataset.deleted_at == None)
    return session.exec(statement).all()


def delete_dataset(session: Session, dataset_id: str, user_id: str):
    statement = select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    db_dataset = session.exec(statement).first()
    if db_dataset:
        db_dataset.deleted_at = datetime.utcnow()
        session.add(db_dataset)
        session.commit()
    return db_dataset


def create_weather_dataset(session: Session, data: WeatherDatasetCreate, user_id: str) -> WeatherDataset:
    db_weather = WeatherDataset(
        id=data.id,
        user_id=user_id,
        dataset_id=data.dataset_id,
        file_url=data.file_url,
        uploadthing_key=data.uploadthing_key,
    )
    session.add(db_weather)
    session.commit()
    session.refresh(db_weather)
    return db_weather


def list_weather_datasets(session: Session, user_id: str) -> list[WeatherDataset]:
    statement = select(WeatherDataset).where(WeatherDataset.user_id == user_id)
    return session.exec(statement).all()
