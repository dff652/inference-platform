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


# --- Wavelet advanced processing ---
# Migrated from: ts-iteration-loop/services/inference/wavelet.py


def extract_outlier_features(data, outliers):
    """Extract statistical features for each anomaly segment.

    Returns an (N, 2) array with [mean_ratio, range_ratio] per segment,
    used by cluster_based_outlier_split for KMeans/DBSCAN classification.
    """
    features = []
    if hasattr(data, 'values'):
        data_array = data.iloc[:, 0].values if data.ndim > 1 and data.shape[1] > 0 else np.array(data).ravel()
    else:
        data_array = np.array(data).ravel()

    global_mean = np.mean(data_array)
    global_range = np.max(data_array) - np.min(data_array)

    for seg in outliers:
        if len(seg) == 0:
            continue
        local_data = data_array[seg]
        mean_val = np.mean(local_data)
        range_val = np.max(local_data) - np.min(local_data)
        mean_ratio = mean_val / global_mean if global_mean != 0 else 0
        range_ratio = range_val / global_range if global_range != 0 else 0
        features.append([mean_ratio, range_ratio])
    return np.array(features) if features else np.empty((0, 2))


def cluster_based_outlier_split(data, outliers, n_clusters=2, method='kmeans',
                                random_state=42):
    """Split outliers into global/local via clustering (KMeans/DBSCAN/Hierarchical).

    Falls back to range_split_outliers if clustering fails or too few segments.
    """
    if len(outliers) < 2:
        if len(outliers) == 1:
            return np.array(outliers[0]), np.array([])
        return np.array([]), np.array([])

    try:
        features = extract_outlier_features(data, outliers)
        if features.shape[0] < n_clusters:
            return range_split_outliers(data, outliers, range_th=0.1)

        from sklearn.preprocessing import StandardScaler
        features_scaled = StandardScaler().fit_transform(features)

        if method == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state,
                               n_init=10)
        elif method == 'dbscan':
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(
                n_neighbors=min(3, len(features_scaled))).fit(features_scaled)
            distances, _ = nbrs.kneighbors(features_scaled)
            eps = np.percentile(distances[:, -1], 75)
            clusterer = DBSCAN(eps=eps, min_samples=1)
        elif method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        labels = clusterer.fit_predict(features_scaled)
        local_outliers, global_outliers = [], []
        for outlier_seg, lbl in zip(outliers, labels):
            if lbl == 0 or lbl == -1:
                global_outliers.extend(outlier_seg)
            else:
                local_outliers.extend(outlier_seg)
        return np.array(global_outliers), np.array(local_outliers)
    except Exception:
        return range_split_outliers(data, outliers, range_th=0.1)


def adaptive_outlier_split(data, outliers, method='auto', **kwargs):
    """Auto-select outlier split method: threshold for <3 segments, cluster otherwise."""
    if method == 'auto':
        if len(outliers) < 3:
            method = 'threshold'
        else:
            features = extract_outlier_features(data, outliers)
            method = 'cluster' if features.shape[0] >= 3 else 'threshold'

    if method == 'threshold':
        return range_split_outliers(data, outliers, kwargs.get('range_th', 0.1))
    return cluster_based_outlier_split(
        data, outliers,
        n_clusters=kwargs.get('n_clusters', 2),
        method=kwargs.get('cluster_method', 'kmeans'),
        random_state=kwargs.get('random_state', 42),
    )


def cv_sort_local_outlier(signal, outliers, th=0.5):
    """Sort and filter local outliers by coefficient of variation.

    Keeps top (1-th) fraction of segments ranked by CV (descending).
    Returns (cv_values, merged_indices).
    """
    cv_values = []
    cv_indices = []
    for outlier in outliers:
        local_part = signal[outlier]
        if len(local_part) > 1:
            mean_part = np.mean(local_part)
            cv_part = np.std(local_part) / mean_part if mean_part != 0 else np.std(local_part)
        else:
            cv_part = 0
        cv_values.append(cv_part)
        cv_indices.append(outlier)

    sorted_pairs = sorted(zip(cv_values, cv_indices), reverse=True)
    sorted_cv, sorted_idx = zip(*sorted_pairs) if sorted_pairs else ([], [])
    num_select = max(1, int(len(sorted_cv) * (1 - th)))
    selected = sorted_pairs[:num_select]
    sel_cv, sel_idx = zip(*selected) if selected else ([], [])
    merged = sorted(set(item for idx in sel_idx for item in idx))
    return list(sel_cv), merged


def refine_local_outliers(signal, outliers):
    """Refine local outlier boundaries by sliding-window STD maximization.

    For each outlier segment, expands search region by segment length in
    both directions, then finds the window position with maximum STD.
    """
    refined = []
    for outlier in outliers:
        start_idx = max(outlier[0] - len(outlier), 0)
        end_idx = min(outlier[-1] + len(outlier), len(signal))

        right_parts = signal[start_idx:outlier[0]]
        left_parts = signal[outlier[-1]:end_idx]
        local_parts = signal[outlier]

        right_std = np.std(right_parts) if len(right_parts) > 1 else 0
        left_std = np.std(left_parts) if len(left_parts) > 1 else 0

        if len(local_parts) == 1:
            combined = np.concatenate([left_parts, local_parts, right_parts])
            combined_start = start_idx
        elif not np.isnan(left_std) and (np.isnan(right_std) or left_std > right_std):
            combined = np.concatenate([left_parts, local_parts])
            combined_start = start_idx
        elif not np.isnan(right_std):
            combined = np.concatenate([local_parts, right_parts])
            combined_start = start_idx
        else:
            combined = local_parts
            combined_start = outlier[0]

        window_size = len(local_parts)
        max_std = 0
        max_seg = []
        for i in range(len(combined) - window_size + 1):
            window = combined[i:i + window_size]
            if len(window) > 1:
                w_std = np.std(window)
                if w_std > max_std:
                    max_std = w_std
                    max_seg = list(range(combined_start + i,
                                         combined_start + i + window_size))
        if max_seg:
            refined.extend(max_seg)
    return np.array(refined)


def combine_local_outliers(signal, outliers):
    """Bidirectional iterative expansion of local outlier regions.

    Expands each segment left and right (up to 10 steps) as long as
    STD keeps increasing, capturing the full extent of the anomaly.
    """
    refined = []
    for outlier in outliers:
        start_idx = max(outlier[0] - len(outlier), 0)
        end_idx = min(outlier[-1] + len(outlier), len(signal))

        left_parts = signal[outlier[-1]:end_idx]
        local_parts = signal[outlier]

        left_std = np.std(left_parts) if len(left_parts) > 1 else 0
        right_parts = signal[start_idx:outlier[0]]
        right_std = np.std(right_parts) if len(right_parts) > 1 else 0

        # Expand left
        extend_left = 0
        prev_std = left_std
        while start_idx > 0 and extend_left < 10:
            start_idx = max(start_idx - len(local_parts), 0)
            left_parts = signal[start_idx:outlier[0]]
            cur_std = np.std(left_parts) if len(left_parts) > 1 else 0
            if cur_std > prev_std:
                prev_std = cur_std
                extend_left += 1
            else:
                break

        # Expand right
        extend_right = 0
        prev_std = right_std
        while end_idx < len(signal) and extend_right < 10:
            end_idx = min(end_idx + len(local_parts), len(signal))
            right_parts = signal[outlier[-1]:end_idx]
            cur_std = np.std(right_parts) if len(right_parts) > 1 else 0
            if cur_std > prev_std:
                prev_std = cur_std
                extend_right += 1
            else:
                break

        refined.extend(list(range(start_idx, end_idx)))
    return np.array(refined)


def exclude_indices(data, indices):
    """Return indices of data NOT in the given index list."""
    if len(indices) == 0:
        return np.arange(data.shape[0])
    idx_arr = np.array(indices, dtype=int)
    mask_arr = np.ones(data.shape[0], dtype=bool)
    mask_arr[idx_arr] = False
    return np.where(mask_arr)[0]


def fit_and_replace_outliers(data, outliers, degree=2):
    """Replace outlier segments with polynomial-fitted values."""
    from numpy.polynomial.polynomial import Polynomial
    continuous_groups = split_continuous_outliers(outliers)
    data_fitted = np.copy(data)
    for group in continuous_groups:
        x_values = np.array(group)
        y_values = data[x_values]
        if len(x_values) > 1:
            p = Polynomial.fit(x_values, y_values, degree)
            data_fitted[x_values] = p(x_values)
    return data_fitted


# --- Piecewise regression ---

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
