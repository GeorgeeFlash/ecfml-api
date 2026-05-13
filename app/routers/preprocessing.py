from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlmodel import Session, select
import logging
import uuid

from app.database import get_session
from app.dependencies import get_current_user
from app.models.db import PreprocessingJob
from app.schemas.preprocessing import PreprocessingRunRequest, PreprocessingStatusResponse
from app.services.preprocessing_service import get_preprocessing_status, run_preprocessing_task
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["preprocessing"])


@router.get("/preprocessing/jobs", response_model=list[PreprocessingStatusResponse])
async def list_preprocessing_jobs_route(
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} listing preprocessing jobs")
        statement = select(PreprocessingJob).where(PreprocessingJob.user_id == user["sub"])
        db_jobs = session.exec(statement).all()
        return [
            PreprocessingStatusResponse(
                job_id=j.job_id,
                status=j.status,
                progress=100.0 if j.status == "COMPLETE" else 0.0,
                error=j.error,
                processed_file_path=j.processed_file_path,
                result_summary=j.result_summary,
                eda_charts=None,
            )
            for j in db_jobs
        ]
    except Exception as e:
        logger.error(f"Error listing preprocessing jobs for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list preprocessing jobs"
        )


@router.post("/preprocessing/run", response_model=PreprocessingStatusResponse, status_code=status.HTTP_201_CREATED)
async def run_preprocessing_route(
    body: PreprocessingRunRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} starting preprocessing for dataset: {body.dataset_id}")
        
        # Create DB record
        db_job_id = str(uuid.uuid4())
        db_job = PreprocessingJob(
            id=db_job_id,
            user_id=user["sub"],
            dataset_id=body.dataset_id,
            job_id=body.job_id,
            status="RUNNING"
        )
        session.add(db_job)
        session.commit()
        
        # Start background task
        background_tasks.add_task(run_preprocessing_task, body, user["sub"], db_job_id)
        
        return PreprocessingStatusResponse(
            job_id=body.job_id,
            status="RUNNING",
            progress=0.0,
            error=None,
            processed_file_path=None,
            result_summary=None,
            eda_charts=None,
        )
    except Exception as e:
        logger.error(f"Error starting preprocessing for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to start preprocessing: {str(e)}"
        )


@router.get("/preprocessing/{job_id}/status", response_model=PreprocessingStatusResponse)
async def preprocessing_status_route(
    job_id: str,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} checking preprocessing status for job: {job_id}")
        return get_preprocessing_status(job_id, session, user["sub"])
    except Exception as e:
        logger.error(f"Error checking preprocessing status for job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found or error retrieving status"
        )
