from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from app.models.model_entity import ModelStatus


class ModelCreate(BaseModel):
    name: str = Field(..., max_length=255)
    family: str = Field(..., max_length=128)
    runtime_type: str = Field(..., max_length=64)
    version: str = Field(..., max_length=64)
    artifact_uri: Optional[str] = None
    base_model: Optional[str] = None
    compatibility: Optional[dict] = None
    metrics: Optional[dict] = None
    tags: Optional[list[str]] = None
    description: Optional[str] = None
    source_task_id: Optional[int] = None


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = None
    artifact_uri: Optional[str] = None
    compatibility: Optional[dict] = None
    metrics: Optional[dict] = None
    tags: Optional[list[str]] = None
    description: Optional[str] = None


class ModelResponse(BaseModel):
    id: int
    name: str
    family: str
    runtime_type: str
    version: str
    artifact_uri: Optional[str] = None
    base_model: Optional[str] = None
    compatibility: Optional[dict] = None
    metrics: Optional[dict] = None
    tags: Optional[list[str]] = None
    status: ModelStatus
    description: Optional[str] = None
    source_task_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ModelListResponse(BaseModel):
    total: int
    items: list[ModelResponse]
