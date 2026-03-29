"""Pydantic schemas for API request/response models."""
from typing import Optional, Any, Literal
from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str
    version: str


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    name: str
    source: Literal['huggingface', 'civitai', 'local']
    source_url: Optional[str] = None
    path: str
    format: Literal['diffusers', 'safetensors']
    size_bytes: int = 0
    type: Literal['checkpoint'] = 'checkpoint'
    trigger_words: Optional[str] = None
    pulled_at: str
    loaded: bool = False  # Runtime status, not from DB


class ModelListResponse(BaseModel):
    """Response model for model list."""
    models: list[ModelInfo] = Field(default_factory=list)
    count: int = 0


class ModelPullRequest(BaseModel):
    """Request model for pulling a model."""
    source: Literal['huggingface', 'civitai']
    repo_id: Optional[str] = None  # For HuggingFace
    url: Optional[str] = None      # For Civitai
    filename: Optional[str] = None # For Civitai


class Txt2ImgParams(BaseModel):
    """Text-to-image generation parameters."""
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: Optional[int] = None


class TaskSubmitRequest(BaseModel):
    """Request model for task submission."""
    type: Literal['txt2img'] = 'txt2img'
    model_id: str
    params: Txt2ImgParams


class TaskInfo(BaseModel):
    """Task information."""
    id: str
    type: Literal['txt2img']
    model_id: str
    params: str  # JSON string from DB
    status: Literal['pending', 'running', 'completed', 'failed', 'cancelled']
    result_path: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TaskListResponse(BaseModel):
    """Response model for task list."""
    tasks: list[TaskInfo] = Field(default_factory=list)
    count: int = 0


class TaskSubmitResponse(BaseModel):
    """Response model for task submission."""
    task_id: str
    status: str
