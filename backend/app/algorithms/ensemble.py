"""
Ensemble (piecewise linear) anomaly detection.

Migrated from: ts-iteration-loop/services/inference/ensemble.py + run.py
Uses Savitzky-Golay filter + polynomial regression + IQR detection.
"""
import numpy as np
from scipy.signal import savgol_filter

from .helpers import piecewise_linear, iqr_find_anomaly_indices
from .preprocessing import min_max_scaling


def detect(data, threshold=1.5, merge_ratio=0.01):
    """
    Detect anomalies using piecewise linear regression + IQR.

    Args:
        data: pd.Series or pd.DataFrame (single column)
        threshold: IQR multiplier for outlier detection
        merge_ratio: merge distance as ratio of data length

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

    # Savitzky-Golay smoothing
    window_length = min(601, data_length if data_length % 2 == 1 else data_length - 1)
    polyorder = min(3, window_length - 1)
    pre_data = savgol_filter(values, window_length=window_length, polyorder=polyorder)

    # Polynomial regression
    result = piecewise_linear(pre_data)
    if result is None or len(result) != 2 or result[0] is None:
        return np.array([])

    _, residuals = result
    return iqr_find_anomaly_indices(residuals, threshold, merge_len)
