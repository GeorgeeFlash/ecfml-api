from fastapi import APIRouter, Depends

from app.dependencies import get_current_user
from app.schemas.preprocessing import PreprocessingRunRequest, PreprocessingStatusResponse
from app.services.preprocessing_service import get_preprocessing_status, run_preprocessing

router = APIRouter(tags=["preprocessing"])


@router.post("/preprocessing/run", response_model=PreprocessingStatusResponse)
async def run_preprocessing_route(
    body: PreprocessingRunRequest,
    user: dict = Depends(get_current_user),
):
    return run_preprocessing(body)


@router.get("/preprocessing/{job_id}/status", response_model=PreprocessingStatusResponse)
async def preprocessing_status_route(
    job_id: str,
    user: dict = Depends(get_current_user),
):
    return get_preprocessing_status(job_id)
