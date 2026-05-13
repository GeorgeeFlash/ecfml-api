from fastapi import APIRouter, Depends, HTTPException, status
import logging

from app.dependencies import get_current_user
from app.schemas.model import ModelEvaluateRequest, ModelEvaluateResponse, ModelTrainRequest, ModelTrainResponse
from app.services.evaluation_service import evaluate_model
from app.services.training_service import get_training_status, train_model
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["models"])


@router.post("/models/train", response_model=ModelTrainResponse, status_code=status.HTTP_201_CREATED)
async def train_model_route(
    body: ModelTrainRequest,
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} starting model training: {body.model_type}")
        return train_model(body)
    except Exception as e:
        logger.error(f"Error starting model training for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to start model training: {str(e)}"
        )


@router.post("/models/tune")
async def tune_model_route(user: dict = Depends(get_current_user)):
    logger.warning(f"User {user['sub']} attempted to access unimplemented tuning endpoint")
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Hyperparameter tuning not implemented")


@router.get("/models/jobs/{job_id}/status", response_model=ModelTrainResponse)
async def model_job_status_route(job_id: str, user: dict = Depends(get_current_user)):
    try:
        logger.info(f"User {user['sub']} checking training status for job: {job_id}")
        return get_training_status(job_id)
    except Exception as e:
        logger.error(f"Error checking training status for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or error retrieving status"
        )


@router.post("/models/{model_id}/evaluate", response_model=ModelEvaluateResponse)
async def evaluate_model_route(
    model_id: str,
    body: ModelEvaluateRequest,
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} evaluating model: {model_id}")
        if not body.model_file_path:
            body.model_file_path = model_id
        return evaluate_model(body)
    except Exception as e:
        logger.error(f"Error evaluating model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to evaluate model: {str(e)}"
        )
