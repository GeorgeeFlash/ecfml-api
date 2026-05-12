import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.dependencies import get_current_user
from app.schemas.forecast import ForecastRequest, ForecastResponse
from app.services.forecast_service import create_forecast, get_agent_queue, get_agent_result

router = APIRouter(tags=["forecast"])


@router.post("/forecast", response_model=ForecastResponse)
async def create_forecast_route(
    body: ForecastRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    return await create_forecast(request.app.state, body)


@router.get("/forecast/stream/{agent_run_id}")
async def stream_agent_forecast(
    agent_run_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
):
    async def event_stream() -> AsyncIterator[str]:
        queue = get_agent_queue(agent_run_id)
        if queue is None:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Unknown agent run'})}\n\n"
            return

        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/agents/{agent_run_id}/status")
async def agent_status_route(agent_run_id: str, user: dict = Depends(get_current_user)):
    result = get_agent_result(agent_run_id)
    if not result:
        return {"status": "PENDING"}
    return {"status": "COMPLETE", "result": result}
