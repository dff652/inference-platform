from datetime import datetime
from sqlalchemy import String, Integer, Boolean, DateTime, JSON, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class InferenceConfigTemplate(Base):
    __tablename__ = "inference_config_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    algorithm_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    algorithm_name: Mapped[str | None] = mapped_column(String(128), nullable=True)

    default_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    validation_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    env_profile: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    resource_profile: Mapped[str | None] = mapped_column(String(64), nullable=True)

    enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
