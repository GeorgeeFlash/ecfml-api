from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.agents.graph import compile_graph
from app.config import settings
from app.database import create_db_and_tables
from app.routers import datasets, forecast, models, preprocessing

logger = logging.getLogger(__name__)

compiled_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_graph
    # Create DB tables
    create_db_and_tables()
    
    compiled_graph = compile_graph()
    app.state.agent_graph = compiled_graph
    yield


app = FastAPI(title="ECFML API", version="2.0.0", lifespan=lifespan)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    logger.error(f"Request body: {await request.body()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router, prefix="/api/v1")
app.include_router(preprocessing.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(forecast.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}
