from app.models.inference_task import TaskStatus

# Valid state transitions: current_status -> set of allowed next statuses
TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.DRAFT: {TaskStatus.PENDING, TaskStatus.CANCELLED},
    TaskStatus.PENDING: {TaskStatus.QUEUED, TaskStatus.CANCELLED},
    TaskStatus.QUEUED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: {TaskStatus.PENDING},  # retry
    TaskStatus.CANCELLED: set(),
    TaskStatus.TIMEOUT: {TaskStatus.PENDING},  # retry
}


class InvalidTransitionError(Exception):
    def __init__(self, current: TaskStatus, target: TaskStatus):
        self.current = current
        self.target = target
        super().__init__(f"Invalid transition: {current.value} -> {target.value}")


def validate_transition(current: TaskStatus, target: TaskStatus) -> None:
    allowed = TRANSITIONS.get(current, set())
    if target not in allowed:
        raise InvalidTransitionError(current, target)


def can_cancel(status: TaskStatus) -> bool:
    return TaskStatus.CANCELLED in TRANSITIONS.get(status, set())


def can_retry(status: TaskStatus) -> bool:
    return TaskStatus.PENDING in TRANSITIONS.get(status, set())
