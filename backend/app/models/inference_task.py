import enum
from datetime import datetime
from sqlalchemy import String, Text, Integer, Float, Enum, DateTime, JSON, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class TaskStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskType(str, enum.Enum):
    SINGLE = "single"
    BATCH = "batch"


class InferenceTask(Base):
    __tablename__ = "inference_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_name: Mapped[str] = mapped_column(String(255), nullable=False)
    task_type: Mapped[TaskType] = mapped_column(Enum(TaskType), default=TaskType.SINGLE)
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus), default=TaskStatus.DRAFT, index=True)

    # Algorithm & Model
    algorithm_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    algorithm_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Data source
    data_source_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Snapshots - frozen at submission time
    input_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    parameter_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Resource & executor
    resource_profile: Mapped[str | None] = mapped_column(String(64), nullable=True)
    executor_type: Mapped[str] = mapped_column(String(64), default="cli")

    # Execution info
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    worker_node: Mapped[str | None] = mapped_column(String(128), nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)

    # Submitter
    submitter: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Result
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    log_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
