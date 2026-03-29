"""Status endpoint."""
import logging
from typing import Optional
from fastapi import APIRouter, Request

from gpu_broker import __version__
from gpu_broker.api.schemas import StatusResponse, GPUInfo, LoadedModelInfo, QueueStats

router = APIRouter(prefix="/v1", tags=["status"])
logger = logging.getLogger(__name__)

# Try to import torch for GPU info
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_gpu_info() -> Optional[GPUInfo]:
    """Get GPU information using torch.cuda."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None
    
    try:
        # Get first GPU info
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        # Memory info in MB
        total_mb = props.total_memory // (1024 * 1024)
        allocated_mb = torch.cuda.memory_allocated(device) // (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(device) // (1024 * 1024)
        free_mb = total_mb - reserved_mb
        
        # CUDA version
        cuda_version = torch.version.cuda or "unknown"
        
        return GPUInfo(
            name=props.name,
            vram_total_mb=total_mb,
            vram_used_mb=reserved_mb,
            vram_free_mb=free_mb,
            cuda_version=cuda_version
        )
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
        return None


@router.post("/unload")
async def unload_model(request: Request):
    """Unload current model and free VRAM."""
    engine = request.app.state.engine
    model_id = engine.loaded_model_id
    if model_id is None:
        return {"status": "ok", "message": "No model loaded"}
    engine.unload_model()
    return {"status": "ok", "message": f"Model {model_id} unloaded, VRAM freed"}


@router.get("/status", response_model=StatusResponse)
async def get_status(request: Request):
    """Get server status with GPU, loaded model, and queue info."""
    # Get GPU info
    gpu_info = get_gpu_info()
    
    # Get loaded model info
    loaded_model_info = None
    try:
        engine = request.app.state.engine
        if engine.loaded_model_id:
            # TODO: Track loaded_at timestamp in engine
            loaded_model_info = LoadedModelInfo(
                id=engine.loaded_model_id,
                loaded_at="unknown"  # We can add this to engine later
            )
    except Exception as e:
        logger.warning(f"Failed to get loaded model info: {e}")
    
    # Get queue stats
    queue_stats = None
    try:
        scheduler = request.app.state.scheduler
        stats = await scheduler.get_queue_stats()
        queue_stats = QueueStats(**stats)
    except Exception as e:
        logger.warning(f"Failed to get queue stats: {e}")
    
    return StatusResponse(
        status="ok",
        version=__version__,
        gpu=gpu_info,
        loaded_model=loaded_model_info,
        queue=queue_stats
    )
