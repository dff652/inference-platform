import enum
from datetime import datetime
from sqlalchemy import String, Integer, Enum, DateTime, JSON, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DataSourceType(str, enum.Enum):
    IOTDB = "iotdb"
    CSV = "csv"
    FILE_SYSTEM = "file_system"
    OBJECT_STORAGE = "object_storage"


class DataSourceStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    type: Mapped[DataSourceType] = mapped_column(Enum(DataSourceType), nullable=False)

    connection_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    schema_mapping: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    auth_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)

    status: Mapped[DataSourceStatus] = mapped_column(
        Enum(DataSourceStatus), default=DataSourceStatus.ACTIVE, index=True
    )
    last_check_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
