from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session
import logging

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
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["datasets"])


@router.get("/datasets", response_model=list[DatasetRead])
async def list_datasets_route(
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} listing datasets")
        return list_datasets(session, user["sub"])
    except Exception as e:
        logger.error(f"Error listing datasets for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list datasets"
        )


@router.post("/datasets", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def create_dataset_route(
    body: DatasetCreate,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} creating dataset: {body.name}")
        return create_dataset(session, body, user["sub"])
    except Exception as e:
        logger.error(f"Error creating dataset for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create dataset: {str(e)}"
        )


@router.post("/datasets/{dataset_id}/validate", response_model=DatasetValidationReport)
async def validate_dataset_route(
    dataset_id: str,
    body: DatasetValidationRequest,
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} validating dataset: {dataset_id}")
        return validate_dataset(body.file_url, body.required_columns)
    except Exception as e:
        logger.error(f"Error validating dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to validate dataset: {str(e)}"
        )


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def preview_dataset_route(
    dataset_id: str,
    file_url: str = Query(..., description="Dataset URL"),
    rows: int = Query(100, ge=1, le=500),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} previewing dataset: {dataset_id}")
        return preview_dataset(file_url=file_url, rows=rows)
    except Exception as e:
        logger.error(f"Error previewing dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to preview dataset: {str(e)}"
        )


@router.delete("/datasets/{dataset_id}")
async def delete_dataset_route(
    dataset_id: str,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} deleting dataset: {dataset_id}")
        db_dataset = delete_dataset(session, dataset_id, user["sub"])
        if not db_dataset:
            logger.warning(f"Dataset {dataset_id} not found for deletion by user {user['sub']}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete dataset"
        )


@router.get("/datasets/weather")
async def list_weather_datasets_route(
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} listing weather datasets")
        return list_weather_datasets(session, user["sub"])
    except Exception as e:
        logger.error(f"Error listing weather datasets for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list weather datasets"
        )


@router.post("/datasets/weather", status_code=status.HTTP_201_CREATED)
async def create_weather_dataset_route(
    body: WeatherDatasetCreate,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} creating weather dataset for dataset: {body.dataset_id}")
        return create_weather_dataset(session, body, user["sub"])
    except Exception as e:
        logger.error(f"Error creating weather dataset for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create weather dataset: {str(e)}"
        )
