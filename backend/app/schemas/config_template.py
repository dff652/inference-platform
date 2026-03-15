from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class ConfigCreate(BaseModel):
    name: str = Field(..., max_length=255)
    algorithm_id: Optional[int] = None
    algorithm_name: Optional[str] = None
    default_params: Optional[dict] = None
    validation_schema: Optional[dict] = None
    env_profile: Optional[dict] = None
    resource_profile: Optional[str] = None
    enabled: bool = True


class ConfigUpdate(BaseModel):
    name: Optional[str] = None
    default_params: Optional[dict] = None
    validation_schema: Optional[dict] = None
    env_profile: Optional[dict] = None
    resource_profile: Optional[str] = None
    enabled: Optional[bool] = None


class ConfigResponse(BaseModel):
    id: int
    name: str
    algorithm_id: Optional[int] = None
    algorithm_name: Optional[str] = None
    default_params: Optional[dict] = None
    validation_schema: Optional[dict] = None
    env_profile: Optional[dict] = None
    resource_profile: Optional[str] = None
    enabled: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConfigListResponse(BaseModel):
    total: int
    items: list[ConfigResponse]
