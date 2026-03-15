import asyncio
import logging
from celery import Celery
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "inference_platform",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_soft_time_limit=settings.TASK_TIMEOUT_SECONDS,
    task_time_limit=settings.TASK_TIMEOUT_SECONDS + 60,
    worker_concurrency=settings.MAX_CONCURRENT_TASKS,
    worker_prefetch_multiplier=1,
)


def _run_async(coro):
    """Run an async function in sync context for Celery tasks."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="inference.run")
def run_inference_task(self, task_id: int):
    """Celery task: execute an inference job via the executor adapter."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.core.config import settings
    from app.models.inference_task import InferenceTask, TaskStatus
    from app.adapters.executor_adapter import CLIExecutorAdapter, ExecutionRequest

    engine = create_engine(settings.DATABASE_SYNC_URL)

    with Session(engine) as db:
        task = db.get(InferenceTask, task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return {"error": "Task not found"}

        # Transition to RUNNING
        task.status = TaskStatus.RUNNING
        from datetime import datetime
        task.started_at = datetime.utcnow()
        task.celery_task_id = self.request.id
        db.commit()

    # Build execution request from task snapshots
    params = task.parameter_snapshot or {}
    input_snap = task.input_snapshot or {}

    output_dir = str(Path(settings.DATA_INFERENCE_DIR) / str(task_id))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    request = ExecutionRequest(
        task_id=task_id,
        method=task.algorithm_name or params.get("method", "chatts"),
        input_files=input_snap.get("files", []),
        output_dir=output_dir,
        model_path=params.get("model_path"),
        lora_adapter_path=params.get("lora_adapter_path"),
        load_in_4bit=params.get("load_in_4bit", False),
        n_downsample=params.get("n_downsample", 5000),
        extra_args=params.get("extra_args", {}),
    )

    adapter = CLIExecutorAdapter()
    result = _run_async(adapter.execute(request))

    # Update task with results
    with Session(engine) as db:
        task = db.get(InferenceTask, task_id)
        task.completed_at = datetime.utcnow()

        if result.success:
            task.status = TaskStatus.COMPLETED
            task.result_summary = {
                "result_files": result.result_files,
                "annotation_count": len(result.annotations),
            }
        else:
            task.status = TaskStatus.FAILED
            task.error_message = result.stderr[:2000] if result.stderr else "Unknown error"

        task.log_ref = str(Path(output_dir) / "execution.log")
        db.commit()

    # Save execution log
    log_path = Path(output_dir) / "execution.log"
    with open(log_path, "w") as f:
        f.write(f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}\n")

    logger.info(f"Task {task_id} completed: success={result.success}")
    return {"task_id": task_id, "success": result.success}
