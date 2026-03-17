"""
Unified algorithm dispatcher for anomaly detection.

- CPU methods: direct Python calls (no subprocess)
- GPU methods (chatts, qwen): call vLLM serving API
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from . import iforest, ensemble, wavelet, adtk_hbos
from .helpers import create_outlier_mask, split_continuous_outliers, range_split_outliers
from .preprocessing import adaptive_downsample, min_max_scaling, stl_decompose
from .helpers import nsigma_find_anomaly_indices, reconstruct_residuals
from .lb_eval import evaluate_segments, avg_score

logger = logging.getLogger(__name__)

# Methods handled by direct Python calls (no subprocess needed)
DIRECT_METHODS = {
    "ensemble", "piecewise_linear",
    "wavelet",
    "iforest", "isolation_forest",
    "adtk_hbos",
    "stl_wavelet",
}

# GPU methods handled by vLLM
VLLM_METHODS = {"chatts", "qwen"}

# All methods that the platform can handle directly (no subprocess)
ALL_SUPPORTED_METHODS = DIRECT_METHODS | VLLM_METHODS


def is_direct_method(method: str) -> bool:
    """Check if a method can be executed directly (no subprocess)."""
    return method in DIRECT_METHODS


def is_vllm_method(method: str) -> bool:
    """Check if a method is served by vLLM."""
    return method in VLLM_METHODS


def run(
    method: str,
    input_file: str,
    output_dir: str,
    task_id: int,
    n_downsample: int = 5000,
    threshold: float | None = None,
    extra_args: dict | None = None,
) -> dict:
    """
    Execute a CPU anomaly detection algorithm directly.

    Args:
        method: algorithm name
        input_file: path to input CSV
        output_dir: directory for output files
        task_id: task ID for file naming
        n_downsample: number of points after downsampling
        threshold: detection threshold (method-specific default if None)
        extra_args: additional algorithm parameters

    Returns:
        dict with keys: success, result_files, error
    """
    import json
    from datetime import datetime, timezone

    extra_args = extra_args or {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Read input CSV
        df = pd.read_csv(input_file, parse_dates=['Time'], index_col='Time')
        if df.empty:
            return {"success": False, "result_files": [], "error": "Empty input data"}

        point_name = df.columns[0]
        raw_series = df[[point_name]].copy()

        # Downsample
        downsampled_data, ts, position_index = adaptive_downsample(
            raw_series[point_name], downsampler='m4',
            sample_param=None, min_threshold=n_downsample,
        )
        ds_df = pd.DataFrame(downsampled_data, index=ts, columns=[point_name])
        ds_df = ds_df[~ds_df.index.duplicated(keep='first')]

        # Dispatch to algorithm
        cli_method = _normalize_method(method)

        if cli_method == 'adtk_hbos':
            result_mask, pos_idx = adtk_hbos.detect(
                raw_series,
                downsampler='m4',
                sample_param=0.1,
                bin_nums=extra_args.get('bin_nums', 20),
                min_threshold=extra_args.get('min_threshold', 200000),
            )
            # adtk_hbos returns its own mask on raw_data
            outlier_mask = result_mask[point_name].astype(int).values
            anomaly_indices = np.where(outlier_mask == 1)[0]
            # Use raw position index
            position_index = pos_idx

        else:
            # Methods that work on downsampled data
            th = threshold
            if cli_method == 'iforest':
                th = th or 0.01
                anomaly_indices = iforest.detect(ds_df[point_name], contamination=th)
            elif cli_method in ('ensemble', 'piecewise_linear'):
                th = th or 1.5
                anomaly_indices = ensemble.detect(ds_df, threshold=th)
            elif cli_method == 'wavelet':
                th = th or 5
                anomaly_indices = wavelet.detect(ds_df, threshold=th)
            elif cli_method == 'stl_wavelet':
                th = th or 5
                anomaly_indices = wavelet.detect(ds_df, threshold=th)
            else:
                return {"success": False, "result_files": [], "error": f"Unknown method: {method}"}

            outlier_mask = np.zeros(len(ds_df), dtype=int)
            if len(anomaly_indices) > 0:
                valid_idx = anomaly_indices[anomaly_indices < len(ds_df)]
                outlier_mask[valid_idx] = 1

        # Build and evaluate segments (KS-test + boundary scoring)
        raw_values = ds_df[point_name].values.astype(float)
        segments = evaluate_segments(raw_values, anomaly_indices)

        # Compute metrics
        scores = [s["score"] for s in segments] if segments else [0]
        metrics = {
            "version": 1,
            "summary": {
                "score_avg": round(float(avg_score(scores)), 2) if scores else 0,
                "score_max": float(np.max(scores)) if scores else 0,
                "score_min": float(np.min(scores)) if scores else 0,
                "segment_count": len(segments),
            },
            "method": cli_method,
            "task_id": str(task_id),
            "point_name": point_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write output files
        method_dir = output_path / "global" / cli_method
        method_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        base_name = f"{cli_method}_m4_{point_name}_{date_str}"

        # Result CSV
        result_csv_path = method_dir / f"{base_name}.csv"
        _write_result_csv(ds_df, point_name, outlier_mask, position_index, result_csv_path)

        # Metrics JSON
        metrics_path = method_dir / f"{base_name}_metrics.json"
        metrics["result_csv"] = str(result_csv_path)
        metrics["result_path"] = str(result_csv_path)
        metrics["metrics_path"] = str(metrics_path)
        metrics["segments_path"] = str(method_dir / f"{base_name}_segments.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # Segments JSON
        segments_path = method_dir / f"{base_name}_segments.json"
        with open(segments_path, 'w') as f:
            json.dump(segments, f, indent=2)

        result_files = [str(result_csv_path), str(metrics_path), str(segments_path)]

        return {"success": True, "result_files": result_files, "error": None}

    except Exception as e:
        logger.exception(f"Algorithm {method} failed for task {task_id}")
        return {"success": False, "result_files": [], "error": str(e)}


def _normalize_method(method: str) -> str:
    """Normalize method names to internal names."""
    mapping = {
        "ensemble": "piecewise_linear",
        "isolation_forest": "iforest",
    }
    return mapping.get(method, method)


def _write_result_csv(ds_df, point_name, outlier_mask, position_index, path):
    """Write result CSV in the standard format."""
    result = ds_df[[point_name]].copy()
    result['outlier_mask'] = outlier_mask[:len(result)]

    # Global/local mask (simplified: same as outlier for now)
    result['global_mask'] = result['outlier_mask']
    result['local_mask'] = 0
    result['global_mask_cluster'] = result['global_mask']
    result['local_mask_cluster'] = 0

    if position_index is not None and len(position_index) == len(result):
        result['orig_pos'] = position_index[:len(result)]
    else:
        result['orig_pos'] = range(len(result))

    result.index.name = 'Time'
    result.to_csv(path)


# ---------- GPU (vLLM) methods ----------

async def run_vllm(
    method: str,
    input_file: str,
    output_dir: str,
    task_id: int,
    n_downsample: int = 5000,
    extra_args: dict | None = None,
) -> dict:
    """
    Execute a GPU anomaly detection algorithm via vLLM API.

    Reads CSV, downsamples, calls vLLM, parses anomalies,
    and writes output files in the same format as CPU methods.

    Returns:
        dict with keys: success, result_files, error
    """
    import json
    from datetime import datetime, timezone
    from app.adapters.vllm_backend import (
        call_chatts_vllm, call_qwen_vllm, VLLMResult,
    )

    extra_args = extra_args or {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Read input CSV
        df = pd.read_csv(input_file, parse_dates=['Time'], index_col='Time')
        if df.empty:
            return {"success": False, "result_files": [], "error": "Empty input data"}

        point_name = df.columns[0]
        raw_series = df[[point_name]].copy()

        # Downsample
        ds_target = extra_args.get("n_downsample", n_downsample)
        if method == "chatts":
            ds_target = min(ds_target, 768)  # ChatTS default context

        downsampled_data, ts, position_index = adaptive_downsample(
            raw_series[point_name], downsampler='m4',
            sample_param=None, min_threshold=ds_target,
        )
        ds_df = pd.DataFrame(downsampled_data, index=ts, columns=[point_name])
        ds_df = ds_df[~ds_df.index.duplicated(keep='first')]
        values = ds_df[point_name].values.astype(float)

        # Call vLLM
        if method == "chatts":
            prompt_name = extra_args.get("prompt_template", "default")
            prompt_tpl = _load_chatts_prompt(prompt_name)
            vllm_result: VLLMResult = await call_chatts_vllm(
                values,
                prompt_template=prompt_tpl,
                max_tokens=extra_args.get("max_tokens"),
            )
        elif method == "qwen":
            vllm_result: VLLMResult = await call_qwen_vllm(
                values,
                max_tokens=extra_args.get("max_tokens"),
            )
        else:
            return {"success": False, "result_files": [], "error": f"Unknown vLLM method: {method}"}

        if not vllm_result.success:
            return {"success": False, "result_files": [], "error": vllm_result.error}

        # Convert vLLM anomalies to outlier mask
        outlier_mask = np.zeros(len(ds_df), dtype=int)
        label_map = {}  # start -> {label, detail} from model output

        for anom in vllm_result.anomalies:
            # ChatTS uses "range", Qwen uses "interval"
            interval = anom.get("range") or anom.get("interval")
            if not interval or len(interval) != 2:
                continue
            start, end = int(interval[0]), int(interval[1])
            start = max(0, min(start, len(ds_df) - 1))
            end = max(start, min(end, len(ds_df) - 1))
            outlier_mask[start:end + 1] = 1
            label_map[start] = {
                "label": anom.get("label") or anom.get("type", ""),
                "detail": anom.get("detail") or anom.get("reason", ""),
            }

        # Evaluate segments with KS-test scoring (same as CPU methods)
        anomaly_indices = np.where(outlier_mask > 0)[0]
        segments = evaluate_segments(values, anomaly_indices)

        # Attach model-provided labels to evaluated segments
        for seg in segments:
            info = label_map.get(seg["start"], {})
            seg["label"] = info.get("label", "")
            seg["detail"] = info.get("detail", "")

        # Compute summary metrics
        scores = [s["score"] for s in segments] if segments else [0]
        metrics = {
            "version": 1,
            "summary": {
                "score_avg": round(float(avg_score(scores)), 2) if scores else 0,
                "score_max": float(np.max(scores)) if scores else 0,
                "score_min": float(np.min(scores)) if scores else 0,
                "segment_count": len(segments),
            },
            "method": method,
            "task_id": str(task_id),
            "point_name": point_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "raw_output": vllm_result.raw_text[:2000],
        }

        # Write output files (same structure as CPU methods)
        method_dir = output_path / "global" / method
        method_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        base_name = f"{method}_m4_{point_name}_{date_str}"

        result_csv_path = method_dir / f"{base_name}.csv"
        _write_result_csv(ds_df, point_name, outlier_mask, position_index, result_csv_path)

        metrics_path = method_dir / f"{base_name}_metrics.json"
        metrics["result_csv"] = str(result_csv_path)
        metrics["result_path"] = str(result_csv_path)
        metrics["metrics_path"] = str(metrics_path)
        metrics["segments_path"] = str(method_dir / f"{base_name}_segments.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        segments_path = method_dir / f"{base_name}_segments.json"
        with open(segments_path, 'w') as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)

        result_files = [str(result_csv_path), str(metrics_path), str(segments_path)]
        return {"success": True, "result_files": result_files, "error": None}

    except Exception as e:
        logger.exception(f"vLLM algorithm {method} failed for task {task_id}")
        return {"success": False, "result_files": [], "error": str(e)}


def _load_chatts_prompt(name: str) -> str:
    """Load a ChatTS prompt template by name."""
    import json
    prompts_file = Path(__file__).parent / "configs" / "chatts_prompts.json"
    try:
        with open(prompts_file) as f:
            templates = json.load(f)
        if name in templates:
            return templates[name]["template"]
    except (json.JSONDecodeError, OSError, KeyError):
        logger.warning(f"Failed to load prompt template '{name}', using default")
    # Inline fallback
    from app.adapters.vllm_backend import CHATTS_PROMPT_DEFAULT
    return CHATTS_PROMPT_DEFAULT
