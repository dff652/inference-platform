"""
Offline evaluation metrics for anomaly detection.

Computes Precision, Recall, F1, MCC, MAR, FAR, and more
by comparing detected anomaly intervals against ground-truth labels.

This module is for **offline** evaluation against labeled data.
For online segment scoring (no labels), see lb_eval.py.

Migrated from: ts-iteration-loop/services/inference/evaluation/eval_metrics.py
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binary label conversion
# ---------------------------------------------------------------------------

def intervals_to_binary_labels(intervals, length: int) -> np.ndarray:
    """Convert a list of [start, end] intervals to a binary label array."""
    binary = np.zeros(length, dtype=int)
    if intervals is None:
        return binary
    for interval in intervals:
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            start, end = int(interval[0]), int(interval[1])
            binary[start:end + 1] = 1
    return binary


# ---------------------------------------------------------------------------
# Harmonic weight (anomaly-ratio aware)
# ---------------------------------------------------------------------------

def get_harmonic_weight(anomaly_ratio: float):
    """Determine MAR/FAR penalty weights based on anomaly ratio.

    Low anomaly ratio → heavier penalty for missed alarms (MAR).
    """
    if anomaly_ratio < 0.01:
        return 0.8, 0.2
    elif anomaly_ratio < 0.05:
        return 0.7, 0.3
    return 0.6, 0.4


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def calculate_combined_metrics(
    true_intervals,
    detected_intervals,
    timeseries_length: int,
    consider_anomaly_nums: bool = True,
) -> dict:
    """Compute 20+ evaluation metrics from interval-based anomaly labels.

    Args:
        true_intervals: list of [start, end] ground-truth anomaly intervals
        detected_intervals: list of [start, end] detected anomaly intervals
        timeseries_length: total length of the time series
        consider_anomaly_nums: use anomaly-ratio-based weighting for harmonic accuracy

    Returns:
        dict with TP, TN, FP, FN, Precision, Recall, F1, F2, F0.5,
        Balanced Accuracy, MCC, G-Mean, Kappa, Jaccard, Dice, Overlap,
        Accuracy, FAR, MAR, Harmonic Accuracy.
    """
    true_bin = intervals_to_binary_labels(true_intervals, timeseries_length)
    det_bin = intervals_to_binary_labels(detected_intervals, timeseries_length)

    TP = int(np.sum((true_bin == 1) & (det_bin == 1)))
    TN = int(np.sum((true_bin == 0) & (det_bin == 0)))
    FP = int(np.sum((true_bin == 0) & (det_bin == 1)))
    FN = int(np.sum((true_bin == 1) & (det_bin == 0)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    def f_beta(beta):
        if (precision + recall) > 0:
            return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        return 0

    f2 = f_beta(2)
    f05 = f_beta(0.5)

    balanced_acc = (recall + specificity) / 2

    # MCC (Matthews Correlation Coefficient)
    num = TP * TN - FP * FN
    den = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    mcc = num / den if den > 0 else 0

    g_mean = np.sqrt(precision * recall) if (precision * recall) >= 0 else 0

    # Kappa
    po = (TP + TN) / timeseries_length
    pe = ((TP + FN) * (TP + FP) + (TN + FP) * (TN + FN)) / (timeseries_length**2)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

    jaccard = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    overlap = TP / min(TP + FN, TP + FP) if min(TP + FN, TP + FP) > 0 else 0
    accuracy = (TP + TN) / timeseries_length

    far = FP / (FP + TN) if (FP + TN) > 0 else 0
    mar = FN / (FN + TP) if (FN + TP) > 0 else 0

    if far == 1 or mar == 1:
        harmonic_acc = 0
    else:
        if consider_anomaly_nums:
            anomaly_ratio = (TP + FN) / timeseries_length if timeseries_length > 0 else 0
            mar_w, far_w = get_harmonic_weight(anomaly_ratio)
        else:
            mar_w, far_w = 0.7, 0.3
        harmonic_acc = 1 - (mar_w * mar + far_w * far)

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'Precision': round(precision, 3),
        'Recall': round(recall, 3),
        'Specificity': round(specificity, 3),
        'NPV': round(npv, 3),
        'F1 Score': round(f1, 3),
        'F2 Score': round(f2, 3),
        'F0.5 Score': round(f05, 3),
        'Balanced Accuracy': round(balanced_acc, 3),
        'MCC': round(mcc, 3),
        'G-Mean': round(g_mean, 3),
        'Kappa': round(kappa, 3),
        'Jaccard Index': round(jaccard, 3),
        'Dice Coefficient': round(dice, 3),
        'Overlap Coefficient': round(overlap, 3),
        'Accuracy': round(accuracy, 3),
        'FAR': round(far, 3),
        'MAR': round(mar, 3),
        'Harmonic Accuracy': round(harmonic_acc, 3),
    }


# ---------------------------------------------------------------------------
# Ground truth / detection result loaders
# ---------------------------------------------------------------------------

def load_ground_truth(truth_file: str) -> Optional[list]:
    """Load ground truth anomaly intervals from annotation JSON."""
    if not os.path.exists(truth_file):
        logger.warning(f"Ground truth file not found: {truth_file}")
        return None
    try:
        with open(truth_file, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        items = json.loads(gt[0]["conversations"][1]["value"])["detected_anomalies"]
        return [info["interval"] for info in items]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error(f"Failed to parse ground truth: {e}")
        return None


def load_detection_results(predict_file: str) -> Optional[dict]:
    """Load detection results JSON (point_name -> intervals mapping)."""
    if not os.path.exists(predict_file):
        logger.warning(f"Detection results file not found: {predict_file}")
        return None
    with open(predict_file, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def batch_evaluate(
    truth_dir: str,
    data_path: str,
    point_names: list,
    detect_results: dict,
    output_file: Optional[str] = None,
) -> list:
    """Evaluate multiple sensor points against ground truth.

    Args:
        truth_dir: directory containing *_annotations.json files
        data_path: directory containing *_ds.csv data files
        point_names: list of sensor point names to evaluate
        detect_results: dict mapping point_name -> detected intervals
        output_file: optional CSV output path for results

    Returns:
        list of dicts with point_name and Harmonic Accuracy, or empty list.
    """
    all_results = []
    for i, pname in enumerate(point_names, 1):
        logger.info(f"[{i}/{len(point_names)}] Evaluating: {pname}")
        truth_file = os.path.join(truth_dir, f"{pname}_ds.csv_annotations.json")
        true_intervals = load_ground_truth(truth_file)
        detected = detect_results.get(pname)

        csv_file = os.path.join(data_path, f"{pname}_ds.csv")
        if not os.path.exists(csv_file):
            logger.warning(f"Data file not found: {csv_file}")
            continue
        csv_data = pd.read_csv(csv_file)

        if csv_data.empty or not true_intervals or detected is None:
            logger.warning(f"Skipping {pname}: missing data/labels/predictions")
            continue

        result = calculate_combined_metrics(true_intervals, detected, len(csv_data))
        result['point_name'] = pname
        all_results.append(result)

    if output_file and all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"Results saved to: {output_file}")

    return all_results
