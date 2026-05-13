from fastapi import APIRouter, Depends, HTTPException, status
import logging

from app.dependencies import get_current_user
from app.schemas.preprocessing import PreprocessingRunRequest, PreprocessingStatusResponse
from app.services.preprocessing_service import get_preprocessing_status, run_preprocessing
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["preprocessing"])


@router.post("/preprocessing/run", response_model=PreprocessingStatusResponse, status_code=status.HTTP_201_CREATED)
async def run_preprocessing_route(
    body: PreprocessingRunRequest,
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} starting preprocessing for dataset: {body.dataset_id}")
        return run_preprocessing(body)
    except Exception as e:
        logger.error(f"Error starting preprocessing for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to start preprocessing: {str(e)}"
        )


@router.get("/preprocessing/{job_id}/status", response_model=PreprocessingStatusResponse)
async def preprocessing_status_route(
    job_id: str,
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} checking preprocessing status for job: {job_id}")
        return get_preprocessing_status(job_id)
    except Exception as e:
        logger.error(f"Error checking preprocessing status for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or error retrieving status"
        )
