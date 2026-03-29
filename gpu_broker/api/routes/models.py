"""Model management endpoints."""
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from gpu_broker.api.schemas import (
    ModelListResponse, ModelInfo, ModelPullRequest
)
from gpu_broker.config import DB_PATH, MODELS_DIR
from gpu_broker.models.manager import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/models", tags=["models"])

# Initialize ModelManager
model_manager = ModelManager(DB_PATH, MODELS_DIR)


@router.get("", response_model=ModelListResponse)
async def list_models():
    """List all available models."""
    try:
        models = model_manager.list()
        model_infos = [ModelInfo(**m) for m in models]
        return ModelListResponse(models=model_infos, count=len(model_infos))
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get model information by ID."""
    try:
        model = model_manager.get(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return ModelInfo(**model)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _pull_model_background(source: str, repo_id: str = None, 
                           url: str = None, filename: str = None):
    """Background task to pull a model."""
    try:
        model_manager.pull(source=source, repo_id=repo_id, url=url, filename=filename)
        logger.info(f"Successfully pulled model from {source}")
    except Exception as e:
        logger.error(f"Failed to pull model: {e}")


@router.post("/pull", status_code=202)
async def pull_model(request: ModelPullRequest, background_tasks: BackgroundTasks):
    """Trigger model download (async, returns 202 immediately).
    
    Returns 202 Accepted and starts download in background.
    """
    try:
        # Validate request
        if request.source == 'huggingface' and not request.repo_id:
            raise HTTPException(
                status_code=400, 
                detail="repo_id is required for HuggingFace models"
            )
        if request.source == 'civitai' and not request.url:
            raise HTTPException(
                status_code=400,
                detail="url is required for Civitai models"
            )
        
        # Start download in background
        background_tasks.add_task(
            _pull_model_background,
            source=request.source,
            repo_id=request.repo_id,
            url=request.url,
            filename=request.filename
        )
        
        return {
            "status": "accepted",
            "message": f"Model download started from {request.source}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initiate model pull: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{model_id}", status_code=200)
async def delete_model(model_id: str):
    """Delete a model."""
    try:
        success = model_manager.delete(model_id)
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        return {"status": "deleted", "model_id": model_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
