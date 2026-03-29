"""Model management endpoints."""
from fastapi import APIRouter

from gpu_broker.api.schemas import ModelListResponse, ModelInfo

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
async def list_models():
    """List all available models."""
    # Stub implementation
    return ModelListResponse(models=[], total=0)


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get model information by ID."""
    # Stub implementation - will be implemented in later milestones
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Model not found")


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model."""
    # Stub implementation
    from fastapi import HTTPException
    raise HTTPException(status_code=501, detail="Not implemented yet")
