from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # backend/
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "inference_platform.db"


class Settings(BaseSettings):
    APP_NAME: str = "Inference Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{DB_PATH}"
    DATABASE_SYNC_URL: str = f"sqlite:///{DB_PATH}"

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # Old executor config
    OLD_PROJECT_PATH: str = "/home/dff652/TS-anomaly-detection/ts-iteration-loop"
    OLD_EXECUTOR_SCRIPT: str = "services/inference/run.py"
    OLD_PYTHON_PATH: str = "/home/dff652/miniconda3/envs/ts/bin/python"

    # Data paths
    DATA_ROOT: str = "/home/share/data"
    DATA_INFERENCE_DIR: str = "/home/share/data/inference"
    MODEL_ARTIFACT_DIR: str = "/home/share/models"

    # API
    API_PREFIX: str = "/api/v1"
    API_PORT: int = 8100

    # Task defaults
    TASK_TIMEOUT_SECONDS: int = 3600
    MAX_CONCURRENT_TASKS: int = 4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
