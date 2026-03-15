from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from app.models.inference_task import TaskStatus, TaskType


class TaskCreate(BaseModel):
    task_name: str = Field(..., max_length=255)
    task_type: TaskType = TaskType.SINGLE
    algorithm_id: Optional[int] = None
    algorithm_name: Optional[str] = None
    model_id: Optional[int] = None
    model_version: Optional[str] = None
    data_source_id: Optional[int] = None
    input_snapshot: Optional[dict] = None
    parameter_snapshot: Optional[dict] = None
    resource_profile: Optional[str] = None
    executor_type: str = "cli"
    priority: int = 0
    submitter: Optional[str] = None


class TaskUpdate(BaseModel):
    task_name: Optional[str] = None
    parameter_snapshot: Optional[dict] = None
    resource_profile: Optional[str] = None
    priority: Optional[int] = None


class TaskResponse(BaseModel):
    id: int
    task_name: str
    task_type: TaskType
    status: TaskStatus
    algorithm_id: Optional[int] = None
    algorithm_name: Optional[str] = None
    model_id: Optional[int] = None
    model_version: Optional[str] = None
    data_source_id: Optional[int] = None
    input_snapshot: Optional[dict] = None
    parameter_snapshot: Optional[dict] = None
    resource_profile: Optional[str] = None
    executor_type: str
    priority: int
    submitter: Optional[str] = None
    celery_task_id: Optional[str] = None
    worker_node: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_summary: Optional[dict] = None
    log_ref: Optional[str] = None

    model_config = {"from_attributes": True}


class TaskListResponse(BaseModel):
    total: int
    items: list[TaskResponse]


class TaskStatusStats(BaseModel):
    draft: int = 0
    pending: int = 0
    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    timeout: int = 0
