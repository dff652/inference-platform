"""
Performance tracking for inference tasks.

Records per-task timing breakdown, data statistics, GPU/system metrics,
and detection results. Stores as JSON for API access (future Performance tab).

Migrated from: ts-iteration-loop/services/inference/perf_logger.py
Adapted for: inference platform (removed file-based logging, added task_id)
"""
import functools
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional GPU monitoring
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# Optional system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class PerfMetrics:
    """Performance metrics for a single inference task."""

    # Task info
    task_id: int = 0
    point_name: str = ""
    start_time: str = ""
    end_time: str = ""
    process_id: int = 0

    # Method info
    method: str = ""
    downsampler: str = ""
    device: str = ""  # cpu / cuda:0 / cuda:1

    # Data statistics
    raw_data_length: int = 0
    downsampled_data_length: int = 0
    downsample_ratio: float = 0.0
    sampling_rate_hz: float = 0.0

    # Timing breakdown (seconds)
    total_time: float = 0.0
    data_read_time: float = 0.0
    downsample_time: float = 0.0
    preprocess_time: float = 0.0
    model_inference_time: float = 0.0
    stl_decompose_time: float = 0.0
    anomaly_detect_time: float = 0.0
    postprocess_time: float = 0.0
    save_time: float = 0.0

    # GPU metrics (if applicable)
    gpu_id: int = -1
    gpu_name: str = ""
    gpu_utilization_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_temperature_c: float = 0.0
    gpu_power_w: float = 0.0

    # System metrics
    cpu_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_percent: float = 0.0

    # Detection results
    anomaly_count: int = 0
    anomaly_ratio: float = 0.0
    segment_count: int = 0

    # Data quality
    is_step_data: bool = False
    is_noisy_data: bool = False
    data_type: List[str] = field(default_factory=list)

    # Status
    status: str = "success"
    error_message: str = ""
    timestamp: str = ""


class PerfLogger:
    """Singleton performance logger with GPU/system monitoring."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, json_log_dir: Optional[str] = None):
        if PerfLogger._initialized:
            return

        if json_log_dir is None:
            # Default: backend/data/perf_logs/
            from app.core.config import DATA_DIR
            json_log_dir = str(DATA_DIR / "perf_logs")

        self.log_dir = Path(json_log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pynvml
        self._nvml_initialized = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception as e:
                logger.debug(f"pynvml init failed: {e}")

        PerfLogger._initialized = True
        logger.info(
            f"PerfLogger initialized (gpu={self._nvml_initialized}, "
            f"psutil={PSUTIL_AVAILABLE})"
        )

    def get_gpu_metrics(self, device: str = "cuda:0") -> Dict[str, Any]:
        """Snapshot current GPU metrics."""
        metrics = {
            "gpu_id": -1, "gpu_name": "",
            "gpu_utilization_percent": 0.0,
            "gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0,
            "gpu_memory_percent": 0.0,
            "gpu_temperature_c": 0.0, "gpu_power_w": 0.0,
        }
        if not self._nvml_initialized:
            return metrics
        try:
            if device.startswith("cuda:"):
                gpu_id = int(device.split(":")[1])
            elif device == "cuda":
                gpu_id = 0
            else:
                return metrics

            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            metrics["gpu_id"] = gpu_id
            name = pynvml.nvmlDeviceGetName(handle)
            metrics["gpu_name"] = name.decode("utf-8") if isinstance(name, bytes) else name

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["gpu_utilization_percent"] = util.gpu

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["gpu_memory_used_mb"] = mem.used / 1024 / 1024
            metrics["gpu_memory_total_mb"] = mem.total / 1024 / 1024
            metrics["gpu_memory_percent"] = (mem.used / mem.total) * 100

            try:
                metrics["gpu_temperature_c"] = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass
            try:
                metrics["gpu_power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"GPU metrics failed: {e}")
        return metrics

    def get_system_metrics(self) -> Dict[str, Any]:
        """Snapshot current CPU/memory metrics."""
        metrics = {"cpu_percent": 0.0, "memory_used_gb": 0.0, "memory_percent": 0.0}
        if not PSUTIL_AVAILABLE:
            return metrics
        try:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            metrics["memory_used_gb"] = round(mem.used / 1024**3, 2)
            metrics["memory_percent"] = mem.percent
        except Exception as e:
            logger.debug(f"System metrics failed: {e}")
        return metrics

    def log_metrics(self, metrics: PerfMetrics):
        """Persist a PerfMetrics record as JSON (one file per task)."""
        metrics.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics.process_id = os.getpid()

        record = asdict(metrics)
        filename = f"task_{metrics.task_id}_{metrics.method}.json"
        filepath = self.log_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to write perf log: {e}")

        logger.info(
            f"[PERF] task={metrics.task_id} method={metrics.method} "
            f"total={metrics.total_time:.2f}s "
            f"detect={metrics.anomaly_detect_time:.2f}s "
            f"anomalies={metrics.anomaly_count}"
        )

    def load_task_metrics(self, task_id: int) -> List[dict]:
        """Load all perf records for a given task_id."""
        results = []
        for f in self.log_dir.glob(f"task_{task_id}_*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    results.append(json.load(fh))
            except Exception:
                pass
        return results


class Timer:
    """Context manager for timing code blocks.

    Usage:
        with Timer("inference") as t:
            result = model.predict(data)
        metrics.model_inference_time = t.elapsed
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._end = time.perf_counter()
        self.elapsed = self._end - self._start


def get_perf_logger(json_log_dir: Optional[str] = None) -> PerfLogger:
    """Get or create the singleton PerfLogger instance."""
    return PerfLogger(json_log_dir=json_log_dir)


def track_time(metric_name: str):
    """Decorator that returns (result, elapsed_seconds)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            return result, elapsed
        return wrapper
    return decorator
