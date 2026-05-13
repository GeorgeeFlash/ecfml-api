import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import StreamingResponse

from app.dependencies import get_current_user
from app.schemas.forecast import ForecastRequest, ForecastResponse
from app.services.forecast_service import create_forecast, get_agent_queue, get_agent_result
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["forecast"])


@router.post("/forecast", response_model=ForecastResponse, status_code=status.HTTP_201_CREATED)
async def create_forecast_route(
    body: ForecastRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    try:
        logger.info(f"User {user['sub']} initiating forecast")
        return await create_forecast(request.app.state, body)
    except Exception as e:
        logger.error(f"Error creating forecast for user {user['sub']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create forecast: {str(e)}"
        )


@router.get("/forecast/stream/{agent_run_id}")
async def stream_agent_forecast(
    agent_run_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
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
async def agent_status_route(agent_run_id: str, user: dict = Depends(get_current_user)):
    try:
        logger.info(f"User {user['sub']} checking status for agent run: {agent_run_id}")
        result = get_agent_result(agent_run_id)
        if not result:
            return {"status": "PENDING"}
        return {"status": "COMPLETE", "result": result}
    except Exception as e:
        logger.error(f"Error checking status for agent {agent_run_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check agent status"
        )
