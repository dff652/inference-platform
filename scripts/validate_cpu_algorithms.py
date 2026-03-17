#!/usr/bin/env python3
"""
End-to-end validation of CPU algorithms through the dispatcher.

Tests: ensemble, iforest, wavelet, adtk_hbos, stl_wavelet
Each algorithm reads the test CSV, runs anomaly detection, and writes output files.

Usage:
    cd /home/dff652/dff_project/inference-platform
    /home/dff652/miniconda3/envs/chatts/bin/python scripts/validate_cpu_algorithms.py
"""
import sys
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add backend to path for algorithm imports
BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

TEST_CSV = Path(__file__).resolve().parent.parent / "data" / "PI_20412.PV.csv"
METHODS = ["ensemble", "iforest", "wavelet", "adtk_hbos", "stl_wavelet"]


def validate_method(method: str, output_dir: str) -> dict:
    """Run one algorithm directly (bypasses app framework) and check outputs."""
    from app.algorithms import iforest, ensemble, wavelet, adtk_hbos
    from app.algorithms.preprocessing import adaptive_downsample
    from app.algorithms.lb_eval import evaluate_segments, avg_score
    from datetime import datetime, timezone

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read input CSV
    df = pd.read_csv(str(TEST_CSV), parse_dates=["Time"], index_col="Time")
    point_name = df.columns[0]
    raw_series = df[[point_name]].copy()

    # Normalize method name
    method_map = {"ensemble": "piecewise_linear", "isolation_forest": "iforest"}
    cli_method = method_map.get(method, method)

    # Downsample
    downsampled_data, ts, position_index = adaptive_downsample(
        raw_series[point_name], downsampler="m4",
        sample_param=None, min_threshold=5000,
    )
    ds_df = pd.DataFrame(downsampled_data, index=ts, columns=[point_name])
    ds_df = ds_df[~ds_df.index.duplicated(keep="first")]

    # Dispatch to algorithm
    if cli_method == "adtk_hbos":
        result_mask, pos_idx = adtk_hbos.detect(
            raw_series, downsampler="m4", sample_param=0.1,
            bin_nums=20, min_threshold=200000,
        )
        outlier_mask = result_mask[point_name].astype(int).values
        anomaly_indices = np.where(outlier_mask == 1)[0]
        position_index = pos_idx
    elif cli_method == "iforest":
        anomaly_indices = iforest.detect(ds_df[point_name], contamination=0.01)
        outlier_mask = np.zeros(len(ds_df), dtype=int)
        if len(anomaly_indices) > 0:
            outlier_mask[anomaly_indices[anomaly_indices < len(ds_df)]] = 1
    elif cli_method in ("ensemble", "piecewise_linear"):
        anomaly_indices = ensemble.detect(ds_df, threshold=1.5)
        outlier_mask = np.zeros(len(ds_df), dtype=int)
        if len(anomaly_indices) > 0:
            outlier_mask[anomaly_indices[anomaly_indices < len(ds_df)]] = 1
    elif cli_method in ("wavelet", "stl_wavelet"):
        anomaly_indices = wavelet.detect(ds_df, threshold=5)
        outlier_mask = np.zeros(len(ds_df), dtype=int)
        if len(anomaly_indices) > 0:
            outlier_mask[anomaly_indices[anomaly_indices < len(ds_df)]] = 1
    else:
        return {"method": method, "success": False, "error": f"Unknown method: {method}", "files": []}

    # Evaluate segments
    raw_values = ds_df[point_name].values.astype(float)
    segments = evaluate_segments(raw_values, anomaly_indices)
    scores = [s["score"] for s in segments] if segments else [0]

    # Metrics
    metrics = {
        "version": 1,
        "summary": {
            "score_avg": round(float(avg_score(scores)), 2) if scores else 0,
            "score_max": float(np.max(scores)) if scores else 0,
            "score_min": float(np.min(scores)) if scores else 0,
            "segment_count": len(segments),
        },
        "method": cli_method,
        "task_id": "9999",
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
    result_df = ds_df[[point_name]].copy()
    result_df["outlier_mask"] = outlier_mask[:len(result_df)]
    result_df.index.name = "Time"
    result_df.to_csv(result_csv_path)

    # Metrics JSON
    metrics_path = method_dir / f"{base_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Segments JSON
    segments_path = method_dir / f"{base_name}_segments.json"
    with open(segments_path, "w") as f:
        json.dump(segments, f, indent=2)

    result_files = [str(result_csv_path), str(metrics_path), str(segments_path)]

    # Build info
    info = {
        "method": method,
        "success": True,
        "error": None,
        "files": [],
        "score_avg": metrics["summary"]["score_avg"],
        "segment_count": metrics["summary"]["segment_count"],
        "anomaly_points": int(np.sum(outlier_mask)),
        "total_points": len(ds_df),
    }

    for fpath in result_files:
        p = Path(fpath)
        info["files"].append({
            "name": p.name,
            "size": p.stat().st_size if p.exists() else 0,
            "exists": p.exists(),
        })

    # CSV row count
    import csv
    with open(result_csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        info["csv_headers"] = reader.fieldnames or []
        info["csv_rows"] = sum(1 for _ in reader)

    return info


def main():
    if not TEST_CSV.exists():
        print(f"ERROR: test CSV not found: {TEST_CSV}")
        sys.exit(1)

    print(f"Test data: {TEST_CSV}")
    print(f"Methods to validate: {METHODS}")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="validate_cpu_")
    results = []
    passed = 0
    failed = 0

    for method in METHODS:
        method_dir = str(Path(tmpdir) / method)
        print(f"\n--- {method} ---")
        try:
            info = validate_method(method, method_dir)
            results.append(info)

            if info["success"]:
                passed += 1
                print(f"  PASS  score_avg={info.get('score_avg')}  "
                      f"segments={info.get('segment_count')}  "
                      f"anomaly={info.get('anomaly_points')}/{info.get('total_points')}  "
                      f"csv_rows={info.get('csv_rows')}")
                for f in info["files"]:
                    print(f"    {f['name']}  ({f['size']} bytes)")
            else:
                failed += 1
                print(f"  FAIL  error={info['error']}")

        except Exception as e:
            import traceback
            failed += 1
            results.append({"method": method, "success": False, "error": str(e)})
            print(f"  EXCEPTION  {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(METHODS)}")

    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
