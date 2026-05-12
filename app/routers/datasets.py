from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from app.database import get_session
from app.dependencies import get_current_user
from app.schemas.dataset import (
    DatasetCreate,
    DatasetPreviewResponse,
    DatasetRead,
    DatasetValidationReport,
    DatasetValidationRequest,
    WeatherDatasetCreate,
)
from app.services.dataset_service import (
    create_dataset,
    create_weather_dataset,
    delete_dataset,
    list_datasets,
    list_weather_datasets,
    preview_dataset,
    validate_dataset,
)

router = APIRouter(tags=["datasets"])


@router.get("/datasets", response_model=list[DatasetRead])
async def list_datasets_route(
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    return list_datasets(session, user["sub"])


@router.post("/datasets", response_model=DatasetRead)
async def create_dataset_route(
    body: DatasetCreate,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    return create_dataset(session, body, user["sub"])


@router.post("/datasets/{dataset_id}/validate", response_model=DatasetValidationReport)
async def validate_dataset_route(
    dataset_id: str,
    body: DatasetValidationRequest,
    user: dict = Depends(get_current_user),
):
    return validate_dataset(body.file_url, body.required_columns)


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def preview_dataset_route(
    dataset_id: str,
    file_url: str = Query(..., description="Dataset URL"),
    rows: int = Query(100, ge=1, le=500),
    user: dict = Depends(get_current_user),
):
    return preview_dataset(file_url=file_url, rows=rows)


@router.delete("/datasets/{dataset_id}")
async def delete_dataset_route(
    dataset_id: str,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    db_dataset = delete_dataset(session, dataset_id, user["sub"])
    if not db_dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"status": "deleted"}


@router.get("/datasets/weather")
async def list_weather_datasets_route(
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    return list_weather_datasets(session, user["sub"])


@router.post("/datasets/weather")
async def create_weather_dataset_route(
    body: WeatherDatasetCreate,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    return create_weather_dataset(session, body, user["sub"])
