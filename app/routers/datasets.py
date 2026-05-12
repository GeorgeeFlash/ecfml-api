from fastapi import APIRouter, Depends, Query

from app.dependencies import get_current_user
from app.schemas.dataset import DatasetPreviewResponse, DatasetValidationReport, DatasetValidationRequest
from app.services.dataset_service import preview_dataset, validate_dataset

router = APIRouter(tags=["datasets"])


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
