"""Pydantic schemas for API request/response models."""
from typing import Optional, Any, Literal
from pydantic import BaseModel, Field


class GPUInfo(BaseModel):
    """GPU information."""
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    cuda_version: str


class LoadedModelInfo(BaseModel):
    """Loaded model information."""
    id: str
    loaded_at: str


class QueueStats(BaseModel):
    """Queue statistics."""
    pending: int
    running: int
    completed_today: int


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str
    version: str
    gpu: Optional[GPUInfo] = None
    loaded_model: Optional[LoadedModelInfo] = None
    queue: Optional[QueueStats] = None


class ModelInfo(BaseModel):
    """Model information."""
    id: str             # SHA256 short ID (12 chars)
    sha256: str = ''    # Full SHA256 hash
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


class ModelDownloadRequest(BaseModel):
    """Request model for downloading a model by URL (auto-detect source)."""
    url: str


class ModelPullRequest(BaseModel):
    """Legacy request model for pulling a model. Kept for backward compatibility."""
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
    """Request model for task submission (new JSON format).
    
    The 'model' field accepts a short ID, full SHA256 hash, or model name.
    resolve_id() will find the matching model.
    """
    type: Literal['txt2img'] = 'txt2img'
    model: str               # Short ID, full hash, or name (resolve_id will find it)
    prompt: str
    negative_prompt: Optional[str] = ""
    params: Optional[dict] = Field(default_factory=dict)  # width, height, steps, etc.


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
