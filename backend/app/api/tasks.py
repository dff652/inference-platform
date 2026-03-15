from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.inference_task import InferenceTask, TaskStatus
from app.models.result_index import InferenceResultIndex
from app.schemas.inference_task import (
    TaskCreate,
    TaskUpdate,
    TaskResponse,
    TaskListResponse,
    TaskStatusStats,
)
from app.services.task_service import TaskService
from app.services.state_machine import InvalidTransitionError

router = APIRouter(prefix="/inference/tasks", tags=["Inference Tasks"])


@router.post("", response_model=TaskResponse, status_code=201)
async def create_task(data: TaskCreate, db: AsyncSession = Depends(get_db)):
    task = await TaskService.create_task(db, data)
    return task


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    status: TaskStatus | None = None,
    submitter: str | None = None,
    algorithm_name: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    items, total = await TaskService.list_tasks(db, status, submitter, algorithm_name, offset, limit)
    return TaskListResponse(total=total, items=items)


@router.get("/stats", response_model=TaskStatusStats)
async def get_task_stats(db: AsyncSession = Depends(get_db)):
    return await TaskService.get_status_stats(db)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int, db: AsyncSession = Depends(get_db)):
    task = await TaskService.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(task_id: int, data: TaskUpdate, db: AsyncSession = Depends(get_db)):
    task = await TaskService.update_task(db, task_id, data)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/{task_id}/submit", response_model=TaskResponse)
async def submit_task(task_id: int, db: AsyncSession = Depends(get_db)):
    """Submit a draft task for execution."""
    from app.core.celery_app import run_inference_task

    try:
        task = await TaskService.transition(db, task_id, TaskStatus.PENDING)
    except (ValueError, InvalidTransitionError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Commit PENDING status first so Celery worker can read it
    await db.commit()

    # Dispatch to Celery
    celery_result = run_inference_task.delay(task_id)
    task.celery_task_id = celery_result.id
    await db.commit()

    await db.refresh(task)
    return task


@router.post("/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(task_id: int, db: AsyncSession = Depends(get_db)):
    try:
        task = await TaskService.cancel_task(db, task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return task


@router.post("/{task_id}/retry", response_model=TaskResponse)
async def retry_task(task_id: int, db: AsyncSession = Depends(get_db)):
    """Retry a failed/timeout task: reset to PENDING and dispatch to Celery."""
    from app.core.celery_app import run_inference_task

    try:
        task = await TaskService.retry_task(db, task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Commit PENDING status then dispatch
    await db.commit()
    celery_result = run_inference_task.delay(task_id)
    task.celery_task_id = celery_result.id
    await db.commit()

    await db.refresh(task)
    return task


@router.get("/{task_id}/results")
async def get_task_results(task_id: int, db: AsyncSession = Depends(get_db)):
    """Get inference result details (metrics, segments, file list)."""
    task = await db.get(InferenceTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Query result index records
    result = await db.execute(
        select(InferenceResultIndex).where(InferenceResultIndex.task_id == task_id)
    )
    indexes = result.scalars().all()

    # Read segments data from files
    results = []
    for idx in indexes:
        entry = {
            "id": idx.id,
            "point_name": idx.point_name,
            "method": idx.method,
            "score_avg": idx.score_avg,
            "score_max": idx.score_max,
            "segment_count": idx.segment_count,
            "result_path": idx.result_path,
            "metrics_path": idx.metrics_path,
            "segments_path": idx.segments_path,
        }
        # Load segments if file exists
        if idx.segments_path and Path(idx.segments_path).exists():
            import json
            try:
                with open(idx.segments_path) as f:
                    entry["segments"] = json.load(f)
            except (json.JSONDecodeError, OSError):
                entry["segments"] = None
        results.append(entry)

    return {
        "task_id": task_id,
        "status": task.status.value,
        "result_summary": task.result_summary,
        "results": results,
    }


@router.get("/{task_id}/logs", response_class=PlainTextResponse)
async def get_task_logs(task_id: int, db: AsyncSession = Depends(get_db)):
    """Get execution log for a task."""
    task = await db.get(InferenceTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not task.log_ref:
        raise HTTPException(status_code=404, detail="No logs available for this task")

    log_path = Path(task.log_ref)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {task.log_ref}")

    return log_path.read_text(encoding="utf-8", errors="replace")
