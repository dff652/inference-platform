import asyncio
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRequest:
    task_id: int
    method: str  # chatts, qwen, adtk_hbos, ensemble
    input_files: list[str]
    output_dir: str
    model_path: str | None = None
    lora_adapter_path: str | None = None
    load_in_4bit: bool = False
    n_downsample: int = 5000
    extra_args: dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    success: bool
    return_code: int
    stdout: str
    stderr: str
    result_files: list[str] = field(default_factory=list)
    annotations: list[dict] = field(default_factory=list)


class CLIExecutorAdapter:
    """Adapter that calls the old project's run.py via CLI subprocess."""

    def __init__(self):
        self.project_path = Path(settings.OLD_PROJECT_PATH)
        self.executor_script = self.project_path / settings.OLD_EXECUTOR_SCRIPT
        self.python_path = settings.OLD_PYTHON_PATH

    def _build_command(self, request: ExecutionRequest) -> list[str]:
        cmd = [
            self.python_path,
            str(self.executor_script),
            "--method", request.method,
            "--n_downsample", str(request.n_downsample),
            "--data_path", request.output_dir,
            "--task-id", str(request.task_id),
        ]

        # Handle single or batch input
        for input_file in request.input_files:
            cmd.extend(["--input", input_file])

        if request.model_path:
            cmd.extend(["--chatts_model_path", request.model_path])
        if request.lora_adapter_path:
            cmd.extend(["--chatts_lora_adapter_path", request.lora_adapter_path])
        if request.load_in_4bit:
            cmd.append("--chatts_load_in_4bit")

        for key, value in request.extra_args.items():
            cmd.extend([f"--{key}", str(value)])

        return cmd

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        cmd = self._build_command(request)
        logger.info(f"Executing inference task {request.task_id}: {' '.join(cmd)}")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_path)
        env["PYTHONUNBUFFERED"] = "1"

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_path),
                env=env,
            )
            stdout, stderr = await process.communicate()
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            result_files = self._collect_result_files(request.output_dir)
            annotations = self._parse_annotations(result_files)

            return ExecutionResult(
                success=process.returncode == 0,
                return_code=process.returncode or 0,
                stdout=stdout_str,
                stderr=stderr_str,
                result_files=result_files,
                annotations=annotations,
            )
        except Exception as e:
            logger.exception(f"Executor failed for task {request.task_id}")
            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
            )

    def _collect_result_files(self, output_dir: str) -> list[str]:
        output_path = Path(output_dir)
        if not output_path.exists():
            return []
        return [str(f) for f in output_path.rglob("*.json")]

    def _parse_annotations(self, result_files: list[str]) -> list[dict]:
        annotations = []
        for file_path in result_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "annotations" in data:
                    annotations.extend(data["annotations"])
                elif isinstance(data, list):
                    annotations.extend(data)
            except (json.JSONDecodeError, OSError):
                continue
        return annotations

    async def cancel(self, task_id: int) -> bool:
        logger.info(f"Cancel requested for task {task_id}")
        # TODO: implement process tracking and SIGTERM
        return True
