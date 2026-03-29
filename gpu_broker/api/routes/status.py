"""Status endpoint."""
from fastapi import APIRouter

from gpu_broker import __version__
from gpu_broker.api.schemas import StatusResponse

router = APIRouter(prefix="/v1", tags=["status"])


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get server status."""
    return StatusResponse(
        status="ok",
        version=__version__
    )
