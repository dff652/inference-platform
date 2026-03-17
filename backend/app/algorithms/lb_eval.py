"""
Segment-level evaluation for anomaly detection results.

Scores each detected anomaly segment by comparing its statistical distribution
against the normal (non-anomalous) data using the Kolmogorov-Smirnov test,
combined with boundary difference analysis.

Ported from: ts-iteration-loop/services/inference/evaluation/lb_eval.py
"""
import math
import logging

import numpy as np
from scipy.stats import ks_2samp

from .helpers import split_continuous_outliers

logger = logging.getLogger(__name__)


def evaluate_segments(raw_data: np.ndarray, anomaly_indices: np.ndarray, merge_gap: int = 5) -> list[dict]:
    """
    Evaluate anomaly segments using KS-test and boundary difference scoring.

    Args:
        raw_data: full time series values (1-D array)
        anomaly_indices: integer indices of detected anomaly points
        merge_gap: merge adjacent segments with gap <= this value

    Returns:
        list of segment dicts with keys:
            start, end, length, score, raw_p, left, right
    """
    if len(anomaly_indices) == 0:
        return []

    raw_data = np.asarray(raw_data, dtype=float)
    anomaly_indices = np.asarray(anomaly_indices, dtype=int)

    # Split into contiguous groups, then merge nearby ones
    groups = split_continuous_outliers(sorted(anomaly_indices))
    groups = _merge_close_segments(groups, max_gap=merge_gap)

    if not groups:
        return []

    # Normal data: everything NOT in any anomaly index
    normal_mask = np.ones(len(raw_data), dtype=bool)
    normal_mask[anomaly_indices.astype(int)] = False
    normal_data = raw_data[normal_mask]
    normal_indices = np.arange(len(raw_data))[normal_mask]

    if len(normal_data) < 2:
        # Not enough normal data for statistical comparison
        return _fallback_segments(groups, raw_data)

    normal_std = float(np.std(normal_data)) + 1e-6
    n_total = len(raw_data)

    segments = []
    for group in groups:
        group = [int(x) for x in group]
        start, end = group[0], group[-1]
        length = end - start + 1

        # KS test: compare (data without this segment) vs (normal data)
        seg_mask = np.ones(len(raw_data), dtype=bool)
        seg_mask[group] = False
        data_without_seg = raw_data[seg_mask]

        try:
            _, ks_pvalue = ks_2samp(data_without_seg, normal_data)
        except Exception:
            ks_pvalue = 1.0

        # Boundary difference: compare segment values vs left/right context
        ctx_len = max(1, int(0.03 * n_total))
        pos_start = int(np.searchsorted(normal_indices, start))
        pos_end = int(np.searchsorted(normal_indices, end))

        left_ctx = normal_data[max(0, pos_start - ctx_len):pos_start]
        right_ctx = normal_data[pos_end:min(len(normal_data), pos_end + ctx_len)]
        seg_values = raw_data[group]

        left_diff = _boundary_diff(left_ctx, seg_values, normal_std)
        right_diff = _boundary_diff(right_ctx, seg_values, normal_std)

        # Composite score
        score = _compute_score(left_diff, right_diff, ks_pvalue)

        segments.append({
            "start": start,
            "end": end,
            "length": length,
            "score": round(float(score), 2),
            "raw_p": round(float(ks_pvalue), 6),
            "left": round(float(left_diff), 4),
            "right": round(float(right_diff), 4),
        })

    return segments


def avg_score(scores, p: int = 5) -> float:
    """Compute p-norm average of scores (emphasizes larger values)."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return 0.0
    n = len(scores)
    return float(np.linalg.norm(scores, ord=p) / n ** (1 / p))


# ---------- Internal helpers ----------

def _boundary_diff(context: np.ndarray, seg_values: np.ndarray, normal_std: float) -> float:
    """Compute normalized boundary difference between context and segment."""
    if len(context) == 0 or len(seg_values) == 0:
        return 0.0
    ctx_mean = float(np.mean(context))
    diff = max(
        abs(ctx_mean - float(np.max(seg_values))),
        abs(ctx_mean - float(np.min(seg_values))),
    )
    return diff / normal_std


def _lr_sc(x: float) -> float:
    """Logistic-like score transform."""
    try:
        return math.exp(-math.exp(2 - x))
    except (OverflowError, ValueError):
        return 0.0


def _w_sc(x: float) -> float:
    """Weight function for combining boundary and p-value scores."""
    try:
        inner = 1 + 25 * x - 25 * x ** 2
        if inner <= 0 or x == -1:
            return float('nan')
        return 1 / 5 * (math.log(inner) / (1 + 5 * x) - 1) + 0.5 * x
    except (ValueError, ZeroDivisionError):
        return float('nan')


def _compute_score(left_diff: float, right_diff: float, ks_pvalue: float) -> float:
    """
    Compute composite segment score from boundary diffs and KS p-value.

    Score range: 0 ~ 100 (higher = more anomalous).
    """
    # Handle NaN from empty boundaries
    if math.isnan(left_diff):
        left_diff = right_diff
    if math.isnan(right_diff):
        right_diff = left_diff

    avg_lr = (left_diff + right_diff) / 2
    p0 = -math.log(ks_pvalue + 1e-12)

    sc_lr = _lr_sc(avg_lr)
    sc_p0 = _lr_sc(p0)

    w = _w_sc(sc_lr)
    if math.isnan(w):
        w = 0.5

    score_adj = (100 - 1e-3) * (sc_lr * (1 - w) + sc_p0 * w)

    # Softplus normalization: maps to ~0-100 smoothly
    score = 10 * math.log1p(math.exp(score_adj / 10))
    return max(0.0, min(100.0, score))


def _merge_close_segments(segments: list[list], max_gap: int = 5) -> list[list]:
    """Merge adjacent segments with gap <= max_gap."""
    if len(segments) <= 1:
        return segments

    merged = []
    current = list(segments[0])

    for i in range(1, len(segments)):
        nxt = segments[i]
        gap = nxt[0] - current[-1]
        if gap <= max_gap:
            current.extend(nxt)
        else:
            merged.append(current)
            current = list(nxt)

    merged.append(current)
    return merged


def _fallback_segments(groups: list[list], raw_data: np.ndarray) -> list[dict]:
    """Simple fallback when there's not enough normal data for KS test."""
    segments = []
    for group in groups:
        group = [int(x) for x in group]
        start, end = group[0], group[-1]
        segments.append({
            "start": start,
            "end": end,
            "length": end - start + 1,
            "score": 0.0,
            "raw_p": 1.0,
            "left": 0.0,
            "right": 0.0,
        })
    return segments
