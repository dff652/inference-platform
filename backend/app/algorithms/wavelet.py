"""
Wavelet decomposition anomaly detection.

Migrated from: ts-iteration-loop/services/inference/wavelet.py + run.py
Uses wavelet detail coefficients + N-sigma detection.
"""
import numpy as np

from .helpers import reconstruct_residuals, nsigma_find_anomaly_indices
from .preprocessing import min_max_scaling


def detect(data, threshold=5, merge_ratio=0.01, wavelet='db1', level=3):
    """
    Detect anomalies using wavelet decomposition residuals.

    Args:
        data: pd.Series or pd.DataFrame (single column)
        threshold: N-sigma threshold for detection
        merge_ratio: merge distance as ratio of data length
        wavelet: wavelet family name
        level: decomposition level

    Returns:
        anomaly_indices: np.ndarray of anomaly point indices
    """
    if hasattr(data, 'iloc') and hasattr(data, 'columns'):
        values = data.iloc[:, 0].values.ravel()
    elif hasattr(data, 'values'):
        values = data.values.ravel()
    else:
        values = np.asarray(data).ravel()

    data_length = len(values)
    merge_len = int(merge_ratio * data_length) if merge_ratio < 1 else int(merge_ratio)

    scaled = min_max_scaling(values)
    residuals = reconstruct_residuals(scaled, wavelet=wavelet, level=level)
    return nsigma_find_anomaly_indices(residuals, threshold, merge_len)
