from datetime import datetime
from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.inference_task import InferenceTask, TaskStatus
from app.schemas.inference_task import TaskCreate, TaskUpdate, TaskStatusStats
from app.services.state_machine import validate_transition, can_cancel, can_retry


class TaskService:

    @staticmethod
    async def create_task(db: AsyncSession, data: TaskCreate) -> InferenceTask:
        task = InferenceTask(
            task_name=data.task_name,
            task_type=data.task_type,
            status=TaskStatus.DRAFT,
            algorithm_id=data.algorithm_id,
            algorithm_name=data.algorithm_name,
            model_id=data.model_id,
            model_version=data.model_version,
            data_source_id=data.data_source_id,
            input_snapshot=data.input_snapshot,
            parameter_snapshot=data.parameter_snapshot,
            resource_profile=data.resource_profile,
            executor_type=data.executor_type,
            priority=data.priority,
            submitter=data.submitter,
        )
        db.add(task)
        await db.flush()
        await db.refresh(task)
        return task

    @staticmethod
    async def get_task(db: AsyncSession, task_id: int) -> InferenceTask | None:
        return await db.get(InferenceTask, task_id)

    @staticmethod
    async def list_tasks(
        db: AsyncSession,
        status: TaskStatus | None = None,
        submitter: str | None = None,
        algorithm_name: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[InferenceTask], int]:
        query = select(InferenceTask)
        count_query = select(sa_func.count(InferenceTask.id))

        if status:
            query = query.where(InferenceTask.status == status)
            count_query = count_query.where(InferenceTask.status == status)
        if submitter:
            query = query.where(InferenceTask.submitter == submitter)
            count_query = count_query.where(InferenceTask.submitter == submitter)
        if algorithm_name:
            query = query.where(InferenceTask.algorithm_name == algorithm_name)
            count_query = count_query.where(InferenceTask.algorithm_name == algorithm_name)

        total = (await db.execute(count_query)).scalar() or 0
        query = query.order_by(InferenceTask.created_at.desc()).offset(offset).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all()), total

    @staticmethod
    async def update_task(db: AsyncSession, task_id: int, data: TaskUpdate) -> InferenceTask | None:
        task = await db.get(InferenceTask, task_id)
        if not task:
            return None
        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(task, key, value)
        await db.flush()
        await db.refresh(task)
        return task

    @staticmethod
    async def transition(db: AsyncSession, task_id: int, target_status: TaskStatus) -> InferenceTask:
        task = await db.get(InferenceTask, task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        validate_transition(task.status, target_status)
        task.status = target_status

        if target_status == TaskStatus.RUNNING:
            task.started_at = datetime.utcnow()
        elif target_status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT):
            task.completed_at = datetime.utcnow()

        await db.flush()
        await db.refresh(task)
        return task

    @staticmethod
    async def cancel_task(db: AsyncSession, task_id: int) -> InferenceTask:
        task = await db.get(InferenceTask, task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        if not can_cancel(task.status):
            raise ValueError(f"Task {task_id} cannot be cancelled in status {task.status.value}")

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        await db.flush()
        await db.refresh(task)
        return task

    @staticmethod
    async def retry_task(db: AsyncSession, task_id: int) -> InferenceTask:
        task = await db.get(InferenceTask, task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        if not can_retry(task.status):
            raise ValueError(f"Task {task_id} cannot be retried in status {task.status.value}")

        task.status = TaskStatus.PENDING
        task.started_at = None
        task.completed_at = None
        task.error_message = None
        await db.flush()
        await db.refresh(task)
        return task

    @staticmethod
    async def get_status_stats(db: AsyncSession) -> TaskStatusStats:
        stats = TaskStatusStats()
        for status in TaskStatus:
            count_q = select(sa_func.count(InferenceTask.id)).where(InferenceTask.status == status)
            count = (await db.execute(count_q)).scalar() or 0
            setattr(stats, status.value, count)
        return stats
