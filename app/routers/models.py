from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlmodel import Session, select
import logging
import uuid

from app.database import get_session
from app.dependencies import get_current_user
from app.models.db import TrainedModel
from app.schemas.model import ModelEvaluateRequest, ModelEvaluateResponse, ModelTrainRequest, ModelTrainResponse
from app.services.evaluation_service import evaluate_model
from app.services.training_service import get_training_status, run_training_task
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelTrainResponse])
async def list_models_route(
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} listing models")
        statement = select(TrainedModel).where(TrainedModel.user_id == user["sub"])
        db_models = session.exec(statement).all()
        return [
            ModelTrainResponse(
                job_id=m.job_id,
                status=m.status,
                model_file_path=m.model_file_path,
                scaler_file_path=None,
                training_time_secs=m.training_time_secs,
                error=m.error,
            )
            for m in db_models
        ]
    except Exception as e:
        logger.error(f"Error listing models for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list models"
        )


@router.post("/models/train", response_model=ModelTrainResponse, status_code=status.HTTP_201_CREATED)
async def train_model_route(
    body: ModelTrainRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} starting model training: {body.model_type}")
        
        # Create DB record
        model_id = str(uuid.uuid4())
        db_model = TrainedModel(
            id=model_id,
            user_id=user["sub"],
            name=body.model_type.value, # Default name
            model_type=body.model_type.value,
            job_id=body.job_id,
            status="RUNNING"
        )
        session.add(db_model)
        session.commit()
        
        # Start background task
        background_tasks.add_task(run_training_task, body, user["sub"], model_id)
        
        return ModelTrainResponse(
            job_id=body.job_id,
            status="RUNNING",
            model_file_path=None,
            scaler_file_path=None,
            training_time_secs=None,
            error=None
        )
    except Exception as e:
        logger.error(f"Error starting model training for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to start model training: {str(e)}"
        )


@router.get("/models/jobs/{job_id}/status", response_model=ModelTrainResponse)
async def model_job_status_route(
    job_id: str,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"User {user['sub']} checking training status for job: {job_id}")
        return get_training_status(job_id, session, user["sub"])
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
        request = body.model_copy(update={"model_file_path": body.model_file_path or model_id})
        return evaluate_model(request)
    except Exception as e:
        logger.error(f"Error evaluating model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to evaluate model: {str(e)}"
        )
