"""
Synchronous predict API for external product integration.

Unlike the async task API (POST /inference/tasks + submit),
this endpoint runs inference inline and returns results directly.
Supports both CPU algorithms and GPU methods (via vLLM).
"""
import json
import tempfile
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.algorithms.dispatcher import (
    is_direct_method, is_vllm_method,
    run as dispatcher_run, run_vllm,
    DIRECT_METHODS, VLLM_METHODS, ALL_SUPPORTED_METHODS,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Predict (Sync)"])


class PredictRequest(BaseModel):
    method: str = Field(..., description="Algorithm name: iforest, ensemble, wavelet, adtk_hbos, stl_wavelet, chatts, qwen")
    input_files: list[str] = Field(..., min_length=1, description="Paths to input CSV files")
    threshold: float | None = Field(None, description="Detection threshold (CPU algorithms, default varies)")
    n_downsample: int = Field(5000, ge=100, description="Downsample target size")
    extra_args: dict = Field(default_factory=dict, description="Additional algorithm parameters (prompt_template, max_tokens, etc.)")


class SegmentResult(BaseModel):
    start: int
    end: int
    score: float
    length: int
    raw_p: float | None = None
    left: float | None = None
    right: float | None = None
    label: str | None = None
    detail: str | None = None


class FileResult(BaseModel):
    input_file: str
    success: bool
    error: str | None = None
    result_files: list[str] = []
    metrics: dict | None = None
    segments: list[SegmentResult] = []


class PredictResponse(BaseModel):
    success: bool
    method: str
    results: list[FileResult]


@router.get("/methods")
async def list_methods():
    """List available prediction methods."""
    from app.adapters.vllm_backend import check_vllm_health

    # Check vLLM availability
    vllm_status = {}
    for m in VLLM_METHODS:
        vllm_status[m] = await check_vllm_health(m)

    return {
        "cpu_methods": sorted(DIRECT_METHODS),
        "gpu_methods": {m: vllm_status[m] for m in sorted(VLLM_METHODS)},
    }


@router.post("", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Run inference synchronously and return results.

    Supports both CPU algorithms (iforest, ensemble, wavelet, adtk_hbos,
    stl_wavelet) and GPU methods (chatts, qwen) via vLLM.
    """
    if req.method not in ALL_SUPPORTED_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method '{req.method}'. "
                   f"Supported: {sorted(ALL_SUPPORTED_METHODS)}",
        )

    # Validate input files exist
    for f in req.input_files:
        if not Path(f).is_file():
            raise HTTPException(status_code=400, detail=f"Input file not found: {f}")

    output_dir = tempfile.mkdtemp(prefix="predict_")
    results = []
    any_success = False

    for input_file in req.input_files:
        if is_direct_method(req.method):
            # CPU method — synchronous
            raw = dispatcher_run(
                method=req.method,
                input_file=input_file,
                output_dir=output_dir,
                task_id=0,
                n_downsample=req.n_downsample,
                threshold=req.threshold,
                extra_args=req.extra_args,
            )
        elif is_vllm_method(req.method):
            # GPU method — async vLLM call
            raw = await run_vllm(
                method=req.method,
                input_file=input_file,
                output_dir=output_dir,
                task_id=0,
                n_downsample=req.n_downsample,
                extra_args=req.extra_args,
            )
        else:
            raw = {"success": False, "result_files": [], "error": f"Unsupported method: {req.method}"}

        file_result = FileResult(
            input_file=input_file,
            success=raw["success"],
            error=raw.get("error"),
            result_files=raw.get("result_files", []),
        )

        if raw["success"]:
            any_success = True
            for rf in raw.get("result_files", []):
                if rf.endswith("_metrics.json"):
                    try:
                        with open(rf) as fh:
                            file_result.metrics = json.load(fh)
                    except (json.JSONDecodeError, OSError):
                        pass
                elif rf.endswith("_segments.json"):
                    try:
                        with open(rf) as fh:
                            raw_segments = json.load(fh)
                        file_result.segments = [
                            SegmentResult.model_validate(seg)
                            for seg in raw_segments
                        ]
                    except ValidationError:
                        logger.exception("Invalid segment payload in %s", rf)
                    except (json.JSONDecodeError, OSError):
                        pass

        results.append(file_result)

    return PredictResponse(
        success=any_success,
        method=req.method,
        results=results,
    )
