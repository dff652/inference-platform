from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from app.models.data_source import DataSourceType, DataSourceStatus


class DataSourceCreate(BaseModel):
    name: str = Field(..., max_length=255)
    type: DataSourceType
    connection_config: Optional[dict] = None
    schema_mapping: Optional[dict] = None
    auth_mode: Optional[str] = None


class DataSourceUpdate(BaseModel):
    name: Optional[str] = None
    connection_config: Optional[dict] = None
    schema_mapping: Optional[dict] = None
    auth_mode: Optional[str] = None


class DataSourceResponse(BaseModel):
    id: int
    name: str
    type: DataSourceType
    connection_config: Optional[dict] = None
    schema_mapping: Optional[dict] = None
    auth_mode: Optional[str] = None
    status: DataSourceStatus
    last_check_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DataSourceListResponse(BaseModel):
    total: int
    items: list[DataSourceResponse]


class DataSourceTestResult(BaseModel):
    success: bool
    message: str
    latency_ms: Optional[float] = None
