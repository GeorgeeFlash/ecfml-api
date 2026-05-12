from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_current_user
from app.schemas.model import ModelEvaluateRequest, ModelEvaluateResponse, ModelTrainRequest, ModelTrainResponse
from app.services.evaluation_service import evaluate_model
from app.services.training_service import get_training_status, train_model

router = APIRouter(tags=["models"])


@router.post("/models/train", response_model=ModelTrainResponse)
async def train_model_route(
    body: ModelTrainRequest,
    user: dict = Depends(get_current_user),
):
    return train_model(body)


@router.post("/models/tune")
async def tune_model_route(user: dict = Depends(get_current_user)):
    raise HTTPException(status_code=501, detail="Hyperparameter tuning not implemented")


@router.get("/models/jobs/{job_id}/status", response_model=ModelTrainResponse)
async def model_job_status_route(job_id: str, user: dict = Depends(get_current_user)):
    return get_training_status(job_id)


@router.post("/models/{model_id}/evaluate", response_model=ModelEvaluateResponse)
async def evaluate_model_route(
    model_id: str,
    body: ModelEvaluateRequest,
    user: dict = Depends(get_current_user),
):
    if not body.model_file_path:
        body.model_file_path = model_id
    return evaluate_model(body)
