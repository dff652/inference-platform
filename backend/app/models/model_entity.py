import enum
from datetime import datetime
from sqlalchemy import String, Text, Integer, Enum, DateTime, JSON, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ModelStatus(str, enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DISABLED = "disabled"


class ModelEntity(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    family: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    runtime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)

    artifact_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)
    base_model: Mapped[str | None] = mapped_column(String(255), nullable=True)

    compatibility: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)

    status: Mapped[ModelStatus] = mapped_column(Enum(ModelStatus), default=ModelStatus.ACTIVE, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    source_task_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
