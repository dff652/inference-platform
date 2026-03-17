import asyncio
import json
import logging
from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded, Terminated
from datetime import datetime, timezone
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
    beat_schedule={
        "sweep-stuck-tasks": {
            "task": "inference.sweep_stuck_tasks",
            "schedule": 60.0,  # every 60 seconds
        },
    },
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


def _run_direct(task_id, method, input_snap, output_dir, params):
    """Execute a CPU algorithm directly via the dispatcher (no subprocess)."""
    from app.algorithms.dispatcher import run as dispatcher_run
    from app.adapters.executor_adapter import ExecutionResult

    input_files = input_snap.get("files", [])
    if not input_files:
        return ExecutionResult(
            success=False, return_code=1,
            stdout="", stderr="No input files provided",
        )

    all_result_files = []
    errors = []
    stdout_parts = []

    threshold = params.get("threshold")
    n_downsample = params.get("n_downsample", 5000)
    extra_args = {k: v for k, v in params.items()
                  if k not in ("method", "threshold", "n_downsample")}

    for input_file in input_files:
        logger.info(f"Task {task_id}: running {method} on {input_file}")
        result = dispatcher_run(
            method=method,
            input_file=input_file,
            output_dir=output_dir,
            task_id=task_id,
            n_downsample=n_downsample,
            threshold=threshold,
            extra_args=extra_args,
        )
        stdout_parts.append(f"[{input_file}] success={result['success']}")
        if result["success"]:
            all_result_files.extend(result["result_files"])
        else:
            errors.append(f"[{input_file}] {result['error']}")

    success = len(errors) == 0
    return ExecutionResult(
        success=success,
        return_code=0 if success else 1,
        stdout="\n".join(stdout_parts),
        stderr="\n".join(errors),
        result_files=all_result_files,
        annotations=[],
    )


def _run_vllm(task_id, method, input_snap, output_dir, params):
    """Execute a GPU algorithm via vLLM API (async, run in event loop)."""
    from app.algorithms.dispatcher import run_vllm
    from app.adapters.executor_adapter import ExecutionResult

    input_files = input_snap.get("files", [])
    if not input_files:
        return ExecutionResult(
            success=False, return_code=1,
            stdout="", stderr="No input files provided",
        )

    all_result_files = []
    errors = []
    stdout_parts = []

    n_downsample = params.get("n_downsample", 5000)
    extra_args = {k: v for k, v in params.items()
                  if k not in ("method", "n_downsample")}

    for input_file in input_files:
        logger.info(f"Task {task_id}: running {method} (vLLM) on {input_file}")
        result = _run_async(run_vllm(
            method=method,
            input_file=input_file,
            output_dir=output_dir,
            task_id=task_id,
            n_downsample=n_downsample,
            extra_args=extra_args,
        ))
        stdout_parts.append(f"[{input_file}] success={result['success']}")
        if result["success"]:
            all_result_files.extend(result["result_files"])
        else:
            errors.append(f"[{input_file}] {result['error']}")

    success = len(errors) == 0
    return ExecutionResult(
        success=success,
        return_code=0 if success else 1,
        stdout="\n".join(stdout_parts),
        stderr="\n".join(errors),
        result_files=all_result_files,
        annotations=[],
    )


def _run_subprocess(celery_task, task_id, method, input_snap, output_dir, params):
    """Execute an inference job via the old project's subprocess adapter."""
    from app.adapters.executor_adapter import CLIExecutorAdapter, ExecutionRequest

    input_files = input_snap.get("files", [])
    adapter = CLIExecutorAdapter()

    request = ExecutionRequest(
        task_id=task_id,
        method=method,
        input_files=input_files,
        output_dir=output_dir,
        model_path=params.get("model_path"),
        lora_adapter_path=params.get("lora_adapter_path"),
        load_in_4bit=params.get("load_in_4bit", "auto"),
        n_downsample=params.get("n_downsample", 5000),
        device=params.get("device"),
        extra_args={k: v for k, v in params.items()
                    if k not in ("method", "model_path", "lora_adapter_path",
                                 "load_in_4bit", "n_downsample", "device")},
    )

    try:
        return _run_async(adapter.execute(request))
    except SoftTimeLimitExceeded:
        logger.warning(f"Task {task_id} hit soft time limit")
        _run_async(adapter.cancel(task_id))
        from app.adapters.executor_adapter import ExecutionResult
        return ExecutionResult(
            success=False, return_code=-1,
            stdout="", stderr=f"Task exceeded time limit ({settings.TASK_TIMEOUT_SECONDS}s)",
        )
    except Terminated:
        logger.info(f"Task {task_id} was terminated (cancelled)")
        from app.adapters.executor_adapter import ExecutionResult
        return ExecutionResult(
            success=False, return_code=-1,
            stdout="", stderr="Task was cancelled",
        )


@celery_app.task(bind=True, name="inference.run")
def run_inference_task(self, task_id: int):
    """Celery task: execute an inference job via direct call or subprocess."""
    from app.models.inference_task import InferenceTask, TaskStatus

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

        # PENDING → QUEUED → RUNNING (atomic commit)
        task.status = TaskStatus.RUNNING
        task.celery_task_id = self.request.id
        task.started_at = datetime.now(timezone.utc)
        db.commit()

    # --- Phase 2: Execute inference ---
    output_dir = str(Path(settings.DATA_INFERENCE_DIR) / str(task_id))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    method = algorithm_name or params.get("method", "chatts")

    # Route: CPU → direct Python; GPU (vLLM) → async API call; fallback → subprocess
    from app.algorithms.dispatcher import is_direct_method, is_vllm_method
    if is_direct_method(method):
        result = _run_direct(task_id, method, input_snap, output_dir, params)
    elif is_vllm_method(method):
        result = _run_vllm(task_id, method, input_snap, output_dir, params)
    else:
        result = _run_subprocess(self, task_id, method, input_snap, output_dir, params)

    # --- Phase 3: Save execution log ---
    log_path = Path(output_dir) / "execution.log"
    with open(log_path, "w") as f:
        f.write(f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}\n")

    # --- Phase 4: Update task status and write result index ---
    with _get_sync_session() as db:
        task = db.get(InferenceTask, task_id)
        task.completed_at = datetime.now(timezone.utc)
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


@celery_app.task(name="inference.sweep_stuck_tasks")
def sweep_stuck_tasks():
    """Periodic task: detect tasks stuck in RUNNING/PENDING beyond timeout and mark as TIMEOUT."""
    from sqlalchemy import select
    from app.models.inference_task import InferenceTask, TaskStatus

    timeout_seconds = settings.TASK_TIMEOUT_SECONDS
    now = datetime.now(timezone.utc)

    with _get_sync_session() as db:
        # Find RUNNING tasks that exceeded timeout
        stmt = select(InferenceTask).where(
            InferenceTask.status.in_([TaskStatus.RUNNING, TaskStatus.PENDING, TaskStatus.QUEUED]),
            InferenceTask.started_at.isnot(None),
        )
        result = db.execute(stmt)
        tasks = result.scalars().all()

        timed_out = 0
        for task in tasks:
            elapsed = (now - task.started_at.replace(tzinfo=timezone.utc)).total_seconds()
            if elapsed > timeout_seconds + 120:  # grace period of 2 minutes beyond soft limit
                task.status = TaskStatus.TIMEOUT
                task.completed_at = now
                task.error_message = f"Task timed out after {int(elapsed)}s (limit: {timeout_seconds}s)"
                timed_out += 1
                logger.warning(f"Sweep: task {task.id} timed out ({int(elapsed)}s)")

        if timed_out:
            db.commit()
            logger.info(f"Sweep: marked {timed_out} task(s) as TIMEOUT")

    return {"timed_out": timed_out}
