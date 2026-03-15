from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class InferenceResultIndex(Base):
    __tablename__ = "inference_result_indexes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    point_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    point_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    method: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)

    result_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metrics_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    segments_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    score_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    segment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
