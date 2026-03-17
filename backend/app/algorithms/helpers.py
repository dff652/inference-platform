"""
Shared helper functions for anomaly detection algorithms.

Migrated from: ts-iteration-loop/services/inference/ensemble.py, wavelet.py, morphological.py
"""
import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import label
from scipy.signal import savgol_filter


# --- Mask and index utilities ---

def create_outlier_mask(df, anomaly_indices):
    """Create a boolean mask marking anomaly indices as True."""
    outlier_mask = np.full_like(df, False)
    for idx in anomaly_indices:
        outlier_mask[idx] = True
    df_copy = df.copy()
    df_copy.loc[:] = outlier_mask
    return df_copy


def split_continuous_outliers(outlier_indices, min_size=1, gp=1):
    """Split outlier indices into groups of continuous segments."""
    split_indices = []
    current_split = []
    for i in range(len(outlier_indices)):
        if not current_split or outlier_indices[i] == current_split[-1] + gp:
            current_split.append(outlier_indices[i])
        else:
            if len(current_split) >= min_size:
                split_indices.append(current_split)
            current_split = [outlier_indices[i]]
    if len(current_split) >= min_size:
        split_indices.append(current_split)
    return split_indices


def aggregate_anomalies(anomaly_mask, threshold):
    """Aggregate nearby anomaly clusters in time domain."""
    labeled_array, num_features = label(anomaly_mask)
    aggregated_mask = np.zeros_like(anomaly_mask)
    for i in range(1, num_features + 1):
        region_indices = np.where(labeled_array == i)[0]
        if i < num_features:
            next_region_indices = np.where(labeled_array == i + 1)[0]
            if next_region_indices[0] - region_indices[-1] < threshold:
                aggregated_mask[region_indices[0]:next_region_indices[-1] + 1] = 1
            else:
                aggregated_mask[region_indices] = 1
        else:
            aggregated_mask[region_indices] = 1
    return aggregated_mask


def range_split_outliers(data, outliers, range_th=0.1):
    """Split outliers into global vs local based on range ratio."""
    global_data = np.copy(data)
    global_range = np.max(np.abs(global_data)) - np.min(np.abs(global_data))
    local_outliers = []
    global_outliers = []
    for outlier in outliers:
        local_data = global_data[outlier]
        local_range = np.max(np.abs(local_data)) - np.min(np.abs(local_data))
        range_ratio = local_range / global_range if global_range > 0 else 0
        if range_ratio >= range_th:
            global_outliers.extend(outlier)
        else:
            local_outliers.extend(outlier)
    return np.array(global_outliers), np.array(local_outliers)


# --- Detection primitives ---

def iqr_find_anomaly_indices(data, th=1.5, N=600):
    """IQR-based anomaly detection with nearby merging."""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - th * IQR
    upper_bound = Q3 + th * IQR
    anomalies = np.where((data < lower_bound) | (data > upper_bound))[0]
    return _merge_nearby_anomalies(anomalies, N)


def nsigma_find_anomaly_indices(data, th=3, N=600):
    """N-sigma anomaly detection with nearby merging."""
    mean = np.mean(data)
    std = np.std(data)
    lower_bound = mean - th * std
    upper_bound = mean + th * std
    anomalies = np.where((data < lower_bound) | (data > upper_bound))[0]
    return _merge_nearby_anomalies(anomalies, N)


def standardized_find_anomaly_indices(data, th=10, N=600):
    """Standardized deviation anomaly detection."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.array([])
    standardized_data = (data - mean) / std
    anomalies = np.where((standardized_data < -th) | (standardized_data > th))[0]
    return _merge_nearby_anomalies(anomalies, N)


def _merge_nearby_anomalies(anomalies, N):
    """Merge anomaly points that are within N distance of each other."""
    if len(anomalies) == 0:
        return np.array([])
    anomaly_indices = []
    for i in range(len(anomalies)):
        start = anomalies[i]
        if i < len(anomalies) - 1:
            next_anomaly = anomalies[i + 1]
            if next_anomaly - start <= N:
                anomaly_indices.extend(list(range(start, next_anomaly + 1)))
            else:
                anomaly_indices.append(start)
        else:
            anomaly_indices.append(start)
    return np.array(sorted(set(anomaly_indices)))


# --- Signal processing primitives ---

def reconstruct_residuals(data, wavelet='db1', level=3):
    """Reconstruct residuals using wavelet detail coefficients."""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    cA, *cD = coeffs
    return pywt.waverec([None] + cD, wavelet)


def morphological_gradient(data, kernel_size=6):
    """Compute morphological gradient (dilation - erosion)."""
    pad_width = kernel_size // 2
    padded = np.pad(data, pad_width, mode='edge')
    dilated = np.array([np.max(padded[i:i + kernel_size]) for i in range(len(data))])
    eroded = np.array([np.min(padded[i:i + kernel_size]) for i in range(len(data))])
    return dilated - eroded


def piecewise_linear(data):
    """Piecewise polynomial regression for trend fitting."""
    try:
        from numpy.exceptions import RankWarning  # numpy >= 2.0
    except ImportError:
        from numpy.polynomial.polyutils import RankWarning  # numpy < 2.0
    import warnings
    warnings.filterwarnings('error', category=RankWarning)

    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    data = data.fillna(0)
    if np.any(np.isinf(data)):
        data = data.replace([np.inf, -np.inf], 0)

    n = len(data)
    trim_size = max(1, int(0.1 * n))

    try:
        coefficients_start = np.polyfit(np.arange(trim_size), data[:trim_size], 2, rcond=1e-10)
        coefficients_end = np.polyfit(np.arange(n - trim_size, n), data[-trim_size:], 2, rcond=1e-10)
    except np.linalg.LinAlgError:
        return None, None

    fitted_curve = None
    for degree in range(12, 3, -1):
        try:
            coefficients = np.polyfit(np.arange(n), data, degree)
            fitted_curve = np.polyval(coefficients, np.arange(n))
            break
        except (RankWarning, np.linalg.LinAlgError):
            continue

    if fitted_curve is None:
        return None, None

    fitted_curve[:trim_size] = np.polyval(coefficients_start, np.arange(trim_size))
    fitted_curve[n - trim_size:] = np.polyval(coefficients_end, np.arange(n - trim_size, n))

    residual = data - fitted_curve
    return fitted_curve, residual
