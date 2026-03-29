"""Pydantic schemas for API request/response models."""
from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str
    version: str


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    name: str
    type: str
    repo_id: str
    local_path: Optional[str] = None
    status: str
    size_bytes: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class ModelListResponse(BaseModel):
    """Response model for model list."""
    models: list[ModelInfo] = Field(default_factory=list)
    total: int = 0


class TaskInfo(BaseModel):
    """Task information."""
    id: str
    model_id: str
    type: str
    status: str
    priority: int = 0
    params: dict[str, Any]
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskListResponse(BaseModel):
    """Response model for task list."""
    tasks: list[TaskInfo] = Field(default_factory=list)
    total: int = 0


class TaskSubmitRequest(BaseModel):
    """Request model for task submission."""
    model_id: str
    type: str
    params: dict[str, Any]
    priority: int = 0


class TaskSubmitResponse(BaseModel):
    """Response model for task submission."""
    task_id: str
    status: str
