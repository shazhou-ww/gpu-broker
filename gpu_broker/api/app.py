"""FastAPI application factory."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gpu_broker import __version__
from gpu_broker.db import init_db
from gpu_broker.api.routes import status, models, tasks
from gpu_broker.config import DB_PATH, MODELS_DIR, OUTPUTS_DIR
from gpu_broker.engine.engine import InferenceEngine
from gpu_broker.scheduler.scheduler import TaskScheduler
from gpu_broker.models.manager import ModelManager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info(f"Starting GPU Broker v{__version__}")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Initialize components
    model_manager = ModelManager(DB_PATH, MODELS_DIR)
    engine = InferenceEngine(OUTPUTS_DIR)
    scheduler = TaskScheduler(DB_PATH, engine, model_manager)
    
    # Store in app state
    app.state.model_manager = model_manager
    app.state.engine = engine
    app.state.scheduler = scheduler
    
    # Start scheduler worker
    await scheduler.start()
    logger.info("Scheduler worker started")
    
    yield
    
    # Cleanup
    logger.info("Shutting down GPU Broker")
    await scheduler.stop()
    engine.unload_model()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="GPU Broker",
        description="GPU inference task broker with model management and scheduling",
        version=__version__,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routers
    app.include_router(status.router)
    app.include_router(models.router)
    app.include_router(tasks.router)
    
    return app
