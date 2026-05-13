import json
from typing import AsyncIterator
from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.database import get_session
from app.dependencies import get_current_user
from app.models.db import Forecast
from app.schemas.forecast import ForecastRequest, ForecastResponse, ForecastRead
from app.services.forecast_service import (
    create_forecast,
    get_agent_queue,
    get_agent_result,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["forecast"])


@router.get("/forecasts", response_model=list[ForecastRead])
async def list_forecasts_route(
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} listing forecasts")
        statement = select(Forecast).where(Forecast.user_id == user["sub"])
        return session.exec(statement).all()
    except Exception as e:
        logger.error(f"Error listing forecasts for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list forecasts",
        )


@router.post(
    "/forecast", response_model=ForecastResponse, status_code=status.HTTP_201_CREATED
)
async def create_forecast_route(
    body: ForecastRequest,
    request: Request,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} initiating forecast")
        return await create_forecast(request.app.state, body, session, user["sub"])
    except Exception as e:
        logger.error(f"Error creating forecast for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create forecast: {str(e)}",
        )


@router.get("/forecast/stream/{agent_run_id}")
async def stream_agent_forecast(
    agent_run_id: str,
    request: Request,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    # Verify ownership
    statement = select(Forecast).where(
        Forecast.id == agent_run_id, Forecast.user_id == user["sub"]
    )
    db_forecast = session.exec(statement).first()
    if not db_forecast:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast not found")

    async def event_stream() -> AsyncIterator[str]:
        try:
            queue = get_agent_queue(agent_run_id)
            if queue is None:
                logger.warning(f"Agent run {agent_run_id} not found for streaming")
                yield f"data: {json.dumps({'type': 'error', 'error': 'Unknown agent run'})}\n\n"
                return

            logger.info(f"User {user['sub']} streaming agent forecast: {agent_run_id}")
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Error in event stream for agent {agent_run_id}: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'error': 'Stream interrupted'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/agents/{agent_run_id}/status")
async def agent_status_route(
    agent_run_id: str,
    session: Session = Depends(get_session),
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} checking status for agent run: {agent_run_id}")
        
        # Verify ownership
        statement = select(Forecast).where(
            Forecast.id == agent_run_id, Forecast.user_id == user["sub"]
        )
        db_forecast = session.exec(statement).first()
        if not db_forecast:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Forecast not found")

        result = get_agent_result(agent_run_id)
        if not result:
            # Fallback to DB if memory cache is gone
            if db_forecast.status == "COMPLETE":
                return {
                    "status": "COMPLETE",
                    "result": {
                        "predictions": db_forecast.predictions,
                        "reasoning": db_forecast.reasoning,
                        "confidence": db_forecast.confidence
                    }
                }
            return {"status": db_forecast.status}
            
        return {"status": "COMPLETE", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking status for agent {agent_run_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check agent status",
        )
