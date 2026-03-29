"""Task management endpoints."""
from fastapi import APIRouter

from gpu_broker.api.schemas import (
    TaskListResponse,
    TaskInfo,
    TaskSubmitRequest,
    TaskSubmitResponse
)

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


@router.get("", response_model=TaskListResponse)
async def list_tasks():
    """List all tasks."""
    # Stub implementation
    return TaskListResponse(tasks=[], total=0)


@router.post("", response_model=TaskSubmitResponse)
async def submit_task(request: TaskSubmitRequest):
    """Submit a new task."""
    # Stub implementation
    from fastapi import HTTPException
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str):
    """Get task information by ID."""
    # Stub implementation
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Task not found")


@router.delete("/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a task."""
    # Stub implementation
    from fastapi import HTTPException
    raise HTTPException(status_code=501, detail="Not implemented yet")
