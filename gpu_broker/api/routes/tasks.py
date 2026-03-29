"""Task management endpoints."""
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pathlib import Path

from gpu_broker.api.schemas import (
    TaskListResponse,
    TaskInfo,
    TaskSubmitRequest,
    TaskSubmitResponse
)

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])
logger = logging.getLogger(__name__)


@router.post("", response_model=TaskSubmitResponse, status_code=202)
async def submit_task(request: TaskSubmitRequest, req: Request):
    """Submit a new task."""
    try:
        scheduler = req.app.state.scheduler
        
        # Merge input fields and params into a single params dict for the engine
        # input carries task-type-specific data (e.g. prompt, negative_prompt)
        # params carries generic tuning knobs (e.g. width, height, steps)
        params = dict(request.params) if request.params else {}
        if request.input:
            params.update(request.input)
        
        task_id = await scheduler.submit(
            task_type=request.type,
            model_id=request.model,
            params=params
        )
        
        return TaskSubmitResponse(task_id=task_id, status="pending")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to submit task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str, req: Request):
    """Get task information by ID."""
    scheduler = req.app.state.scheduler
    
    task = await scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskInfo(**task)


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    req: Request,
    status: str = None,
    model_id: str = None,
    limit: int = 20,
    offset: int = 0
):
    """List all tasks with optional filters."""
    scheduler = req.app.state.scheduler
    
    tasks, total = await scheduler.list_tasks(
        status=status,
        model_id=model_id,
        limit=limit,
        offset=offset
    )
    
    # Convert to TaskInfo models
    task_infos = [TaskInfo(**task) for task in tasks]
    
    return TaskListResponse(tasks=task_infos, count=total)


@router.delete("/{task_id}")
async def cancel_task(task_id: str, req: Request):
    """Cancel a pending task."""
    scheduler = req.app.state.scheduler
    
    success = await scheduler.cancel_task(task_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Task not found or not in pending state"
        )
    
    return {"message": "Task cancelled successfully"}


@router.get("/{task_id}/image")
async def get_task_image(task_id: str, req: Request):
    """Get the generated image for a completed task."""
    scheduler = req.app.state.scheduler
    
    task = await scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Task is {task['status']}, not completed"
        )
    
    if not task['result_path']:
        raise HTTPException(status_code=404, detail="Image not found")
    
    result_path = Path(task['result_path'])
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    return FileResponse(
        result_path,
        media_type="image/png",
        filename=result_path.name
    )
