from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.graph import compile_graph
from app.config import settings
from app.database import create_db_and_tables
from app.routers import datasets, forecast, models, preprocessing

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
