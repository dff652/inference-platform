import asyncio
import json
import logging
from celery import Celery
from datetime import datetime
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


def _get_sync_session():
    """Create a synchronous database session for Celery tasks."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    engine = create_engine(settings.DATABASE_SYNC_URL)
    return Session(engine)


def _parse_metrics_json(metrics_path: str) -> dict | None:
    """Parse a metrics.json file and return summary data."""
    try:
        with open(metrics_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            summary = data.get("summary", data)
            return {
                "score_avg": summary.get("score_avg"),
                "score_max": summary.get("score_max"),
                "segment_count": summary.get("segment_count"),
                "method": data.get("method"),
                "point_name": data.get("point_name"),
                "task_id": data.get("task_id"),
            }
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to parse metrics: {metrics_path}: {e}")
    return None


def _write_result_index(db_session, task_id: int, output_dir: str, method: str):
    """Parse result files and write InferenceResultIndex records."""
    from app.models.result_index import InferenceResultIndex

    output_path = Path(output_dir)
    metrics_files = list(output_path.rglob("*_metrics.json"))

    for metrics_path in metrics_files:
        parsed = _parse_metrics_json(str(metrics_path))
        if not parsed:
            continue

        # Find corresponding segments and result CSV
        base = str(metrics_path).replace("_metrics.json", "")
        segments_path = f"{base}_segments.json"
        result_csv = f"{base}.csv"

        index = InferenceResultIndex(
            task_id=task_id,
            point_name=parsed.get("point_name"),
            method=parsed.get("method") or method,
            result_path=result_csv if Path(result_csv).exists() else None,
            metrics_path=str(metrics_path),
            segments_path=segments_path if Path(segments_path).exists() else None,
            score_avg=parsed.get("score_avg"),
            score_max=parsed.get("score_max"),
            segment_count=parsed.get("segment_count"),
        )
        db_session.add(index)

    db_session.flush()


@celery_app.task(bind=True, name="inference.run")
def run_inference_task(self, task_id: int):
    """Celery task: execute an inference job via the executor adapter."""
    from app.models.inference_task import InferenceTask, TaskStatus
    from app.adapters.executor_adapter import CLIExecutorAdapter, ExecutionRequest

    # --- Phase 1: Load task data and transition PENDING → QUEUED → RUNNING ---
    with _get_sync_session() as db:
        task = db.get(InferenceTask, task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return {"error": "Task not found"}

        # Read snapshots while session is open
        algorithm_name = task.algorithm_name
        params = dict(task.parameter_snapshot or {})
        input_snap = dict(task.input_snapshot or {})

        # PENDING → QUEUED
        task.status = TaskStatus.QUEUED
        task.celery_task_id = self.request.id
        db.commit()

        # QUEUED → RUNNING
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        db.commit()

    # --- Phase 2: Execute inference ---
    output_dir = str(Path(settings.DATA_INFERENCE_DIR) / str(task_id))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    method = algorithm_name or params.get("method", "chatts")

    request = ExecutionRequest(
        task_id=task_id,
        method=method,
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

    # --- Phase 3: Save execution log ---
    log_path = Path(output_dir) / "execution.log"
    with open(log_path, "w") as f:
        f.write(f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}\n")

    # --- Phase 4: Update task status and write result index ---
    with _get_sync_session() as db:
        task = db.get(InferenceTask, task_id)
        task.completed_at = datetime.utcnow()
        task.log_ref = str(log_path)

        if result.success:
            task.status = TaskStatus.COMPLETED

            # Collect all result files (JSON + CSV)
            output_path = Path(output_dir)
            all_result_files = [
                str(f) for f in output_path.rglob("*")
                if f.is_file() and f.suffix in (".json", ".csv")
            ]

            # Build result summary from metrics
            metrics_files = list(output_path.rglob("*_metrics.json"))
            metrics_summary = None
            if metrics_files:
                metrics_summary = _parse_metrics_json(str(metrics_files[0]))

            task.result_summary = {
                "result_files": all_result_files,
                "annotation_count": len(result.annotations),
                "metrics": metrics_summary,
            }

            # Write InferenceResultIndex records
            _write_result_index(db, task_id, output_dir, method)
        else:
            task.status = TaskStatus.FAILED
            task.error_message = result.stderr[:2000] if result.stderr else "Unknown error"

        db.commit()

    logger.info(f"Task {task_id} finished: success={result.success}")
    return {"task_id": task_id, "success": result.success}
