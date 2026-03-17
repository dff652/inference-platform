"""
Data preprocessing utilities for time series anomaly detection.

Migrated from: ts-iteration-loop/services/inference/signal_processing.py
Removed: IoTDB dependencies, GPU STL (hastl), matplotlib, logging to file.
"""
import logging
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.seasonal import STL
from tsdownsample import M4Downsampler, MinMaxLTTBDownsampler

logger = logging.getLogger(__name__)


def check_time_continuity(data, discontinuity_threshold=None):
    """Check time series continuity and find missing timestamps."""
    ts = 'ts'
    time_index = pd.DataFrame(columns=[ts], index=data.index)
    time_index[ts] = data.index
    interval = (time_index[ts] - time_index[ts].shift(1))
    interval_seconds = interval.dt.total_seconds()

    if discontinuity_threshold is None or discontinuity_threshold == '':
        if len(interval_seconds) > 1:
            discontinuity_threshold = interval_seconds.mode()[0]
        else:
            return pd.DataFrame({
                'missing_ratio': [0],
                'missing_timestamps_count': [0]
            }).transpose(), pd.Series(False, index=data.index)
    else:
        discontinuity_threshold = int(discontinuity_threshold)

    freq = f'{int(discontinuity_threshold)}s'
    start_time = time_index[ts].min()
    end_time = time_index[ts].max()

    full_timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
    missing_timestamps = full_timestamps.difference(time_index[ts])

    continuity = pd.Series(False, index=full_timestamps)
    continuity[missing_timestamps] = True

    missing_count = continuity.sum()
    missing_ratio = missing_count / len(data)

    result_df = pd.DataFrame({
        'missing_ratio': [missing_ratio],
        'missing_timestamps_count': [missing_count]
    }).transpose()

    return result_df, continuity


def get_fulldata(data, col_name):
    """Fill missing timestamps with forward/backward fill."""
    result_df, continuity = check_time_continuity(data)
    df = data.copy()
    missing_timestamps = continuity[continuity > 0].index
    full_time_index = df.index.append(pd.DatetimeIndex(missing_timestamps)).unique()
    df = df.reindex(full_time_index)
    df.sort_index(inplace=True)
    df[col_name] = df[col_name].bfill()
    df[col_name] = df[col_name].ffill()
    return df


def ts_downsample(data, downsampler='m4', n_out=100000):
    """Downsample time series, returns (data, timestamps, position_index)."""
    if downsampler == 'm4':
        s_ds = M4Downsampler().downsample(data, n_out=n_out)
    elif downsampler == 'minmax':
        s_ds = MinMaxLTTBDownsampler().downsample(data, n_out=n_out)
    else:
        raise ValueError(f"Unknown downsampler: {downsampler}")

    downsampled_data = data.iloc[s_ds]
    downsampled_time = data.index[s_ds]
    position_index = np.asarray(s_ds, dtype=np.int64)
    return downsampled_data, downsampled_time, position_index


def adaptive_downsample(data, downsampler='m4', sample_param=0.1, min_threshold=1000):
    """Adaptive downsampling with ratio or fixed count."""
    if isinstance(data, pd.DataFrame):
        col_name = data.columns[0]
        series_data = data[col_name].copy()
    else:
        series_data = data.copy()

    data_length = len(series_data)

    if data_length < min_threshold or downsampler is None or (isinstance(downsampler, str) and downsampler.lower() == 'none'):
        position_index = np.arange(data_length, dtype=np.int64)
        if isinstance(data, pd.DataFrame):
            return data.copy(), data.index.copy(), position_index
        return series_data.copy(), series_data.index.copy(), position_index

    if sample_param is None or sample_param > 1:
        n_out = min_threshold
    elif 0 < sample_param <= 1:
        n_out = int(data_length * sample_param)
    else:
        n_out = min_threshold

    n_out = min(n_out, data_length)

    if isinstance(downsampler, str) and downsampler.lower() == 'm4':
        n_out = n_out + (4 - n_out % 4) if n_out % 4 != 0 else n_out

    return ts_downsample(series_data, downsampler, n_out)


def stl_decompose(data, period=60):
    """STL decomposition (CPU only)."""
    res = STL(data, period=period).fit()
    return res.trend, res.seasonal, res.resid


def min_max_scaling(data):
    """Scale data to [0, 1]."""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return data
    return (data - min_val) / (max_val - min_val)


def dead_value_detection(data, duration_threshold=3600, distinct_threshold=10):
    """Detect long constant-value segments."""
    columns_to_check = data.columns
    result_dfs = []
    constant_indices = {}

    for col in columns_to_check:
        value_len = data[col].nunique()
        if value_len < distinct_threshold:
            temp_df = pd.DataFrame({
                'constant_rate': [0.0],
                'constant_count': [0]
            }).transpose()
            result_dfs.append(temp_df)
            constant_indices[col] = np.zeros(len(data), dtype=bool)
            continue

        df = data.copy()
        df['Group'] = (df[col] != df[col].shift()).cumsum()
        df['t'] = df.index
        result = df.groupby('Group').agg(Start=('t', 'first'), End=('t', 'last'), Value=(col, 'first'))
        result['Duration'] = (result['End'] - result['Start']).dt.total_seconds()
        filtered_result = result[result['Duration'] > duration_threshold]

        mask = df['Group'].isin(filtered_result.index).values
        count_constant = mask.sum()
        data_quality_rate = count_constant / len(data)

        temp_df = pd.DataFrame({
            'constant_rate': [data_quality_rate],
            'constant_count': [count_constant]
        }).transpose()
        result_dfs.append(temp_df)
        constant_indices[col] = mask

    result_df = pd.concat(result_dfs, axis=1)
    result_df.columns = columns_to_check
    constant_indices = pd.DataFrame(constant_indices, index=data.index)
    constant_data = data.loc[constant_indices.any(axis=1), columns_to_check]

    return result_df, constant_indices, constant_data


def get_true_indices(data):
    """Get natural indices where value is True."""
    df_reset = data.reset_index(drop=True)
    true_rows = df_reset[df_reset[df_reset.columns[0]] == True]
    return true_rows.index.tolist()


def find_constant_segments(data):
    """Find segments of repeated values."""
    if not data:
        return [], 0
    left = 0
    right = 1
    segments = []
    points = 0
    n = len(data)
    while right < n:
        if data[right] != data[left]:
            left += 1
            right += 1
        else:
            while right < n and data[right] == data[left]:
                right += 1
            if right - left > 5:
                segments.append((left, right - 1))
                points += (right - left)
            left = right
            right = left + 1
    return segments, round(points / n, 3) if n > 0 else 0


def is_step_by_distribution(data, th=0.01, bins=100, tn=5):
    hist, _ = np.histogram(data, bins=bins)
    tp = np.argsort(hist)[-tn:][::-1]
    trh = np.max(hist) * th
    pairs = [(min(u, v), max(u, v)) for u, v in combinations(tp, 2)]
    for left, right in pairs:
        if (not np.any(hist[left + 1:right] < trh)) and (min(hist[left], hist[right]) > trh):
            return False
        elif (sum(x > trh for x in hist) / len(hist) > 0.25):
            return False
    return True


def is_step_data(ts_data, zero_threshold=0.8):
    """Detect if data is step/staircase type."""
    values = ts_data.values.tolist()
    _, constant_ratio = find_constant_segments(values)
    if constant_ratio > zero_threshold:
        return is_step_by_distribution(values)
    return False


def calculate_sampling_rate(data):
    """Calculate median sampling rate in Hz."""
    if len(data) > 1000:
        data = data.iloc[:1000]
    intervals = (data.index[1:] - data.index[:-1]).total_seconds()
    median_interval = np.median(intervals)
    return 1.0 / median_interval if median_interval > 0 else 0


# ---------------------------------------------------------------------------
# Data quality / noise detection
# Migrated from: ts-iteration-loop/services/inference/signal_processing.py
# ---------------------------------------------------------------------------

def psd_noise_analysis(signal, fs=1.0, noise_freq_threshold=0.1,
                       nperseg=256, noverlap=None):
    """Power spectral density based noise analysis (Welch method)."""
    from scipy.signal import welch
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, psd = welch(signal, fs, nperseg=nperseg,
                       noverlap=noverlap, window='hann')
    df = freqs[1] - freqs[0]
    total_e = np.sum(psd) * df
    noise_e = np.sum(psd[freqs >= noise_freq_threshold]) * df
    ratio = noise_e / total_e if total_e else 0.0
    return ratio, freqs, psd


def wavelet_noise_analysis(signal, wavelet='db4', level=5):
    """Wavelet energy ratio based noise analysis."""
    import pywt
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    energy = [np.sum(c**2) for c in coeffs]
    total = sum(energy)
    detail = sum(energy[1:])
    ratio = detail / total if total else 0.0
    recon = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]],
                         wavelet)
    return ratio, recon, energy


def noise_analysis(signal, fs=1.0, nperseg=256, wavelet='db4', level=5,
                   threshold=0.5):
    """Dual-method noise detection (PSD + wavelet energy)."""
    noise_freq_threshold = fs / 10
    signal = signal - np.mean(signal)
    psd_ratio, _, _ = psd_noise_analysis(signal, fs, noise_freq_threshold,
                                         nperseg)
    wavelet_ratio, _, _ = wavelet_noise_analysis(signal, wavelet, level)
    combined_ratio = (psd_ratio + wavelet_ratio) / 2
    if combined_ratio > threshold:
        return "noise"
    return None


def detect_noise_data(ts_data):
    """Detect high-noise data using Hilbert transform + phase STL + DB1 wavelet.

    Returns 'noise' if detected, empty string otherwise.
    """
    import pywt
    from scipy.signal import hilbert
    from scipy.ndimage import gaussian_filter1d

    try:
        data_filtered = gaussian_filter1d(ts_data, sigma=3)
        analytic_signal = hilbert(data_filtered)
        data_phase = np.angle(analytic_signal)
        res = STL(data_phase, period=60).fit()
        coeffs = pywt.wavedec(res.resid + res.seasonal, 'db1', level=3)
        cA, *cD = coeffs
        db1 = pywt.waverec([None] + cD, 'db1')

        def normalize(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        def nsigma_indices(df_arr, window_size=600):
            df_inner = pd.DataFrame(df_arr, columns=['data_db1'])
            global_mean = df_inner['data_db1'].mean()
            global_std = df_inner['data_db1'].std()
            if global_mean != 0:
                df_inner['rolling_mean'] = df_inner['data_db1'].rolling(
                    window=window_size, center=True).mean()
                df_inner['rolling_std'] = df_inner['data_db1'].rolling(
                    window=window_size, center=True).std()
                df_inner['rolling_mean'] = df_inner['rolling_mean'].ffill().bfill()
                df_inner['rolling_std'] = df_inner['rolling_std'].ffill().bfill()
                if ((df_inner['rolling_mean'] >= global_mean * 0.9) &
                    (df_inner['rolling_mean'] <= global_mean * 1.1) &
                    (df_inner['rolling_std'] >= global_std * 0.9) &
                    (df_inner['rolling_std'] <= global_std * 1.1)).all():
                    return 1
            return None

        try:
            if np.max(db1) != np.min(db1):
                n = nsigma_indices(normalize(db1))
                if n == 1:
                    return "noise"
        except Exception:
            pass
    except Exception:
        pass
    return ""


def is_noisy_data(ts_data, fs=1.0, nperseg=256, wavelet='db4', level=5,
                  threshold=0.45):
    """Comprehensive noise detection combining PSD + wavelet + Hilbert methods.

    Returns True if data is classified as noise.
    """
    noise_result = noise_analysis(ts_data, fs, nperseg, wavelet, level,
                                  threshold)
    values = ts_data.values.ravel() if hasattr(ts_data, 'values') else ts_data
    high_noise_result = detect_noise_data(values)
    return (noise_result == "noise" or high_noise_result == "noise")


# ---------------------------------------------------------------------------
# Variance filter (post-processing)
# Migrated from: ts-iteration-loop/services/inference/signal_processing.py
# ---------------------------------------------------------------------------

def _calculate_group_median_variance(df, column):
    """Calculate median of rolling variance for each group."""
    group_median_variances = []
    for group, data_g in df.groupby('group'):
        data_g = data_g.sort_index()
        window_size = max(int(len(data_g) * 0.05), 1)
        rolling_variance = data_g[column].rolling(window=window_size).var()
        median_variance = rolling_variance.median()
        group_median_variances.append({
            'group': group, 'median_variance': median_variance
        })
    return pd.DataFrame(group_median_variances)


def variance_filter(data, mask='global_mask', method='mean', threshold=0.05):
    """Filter anomaly segments by variance and mean deviation.

    Groups data by mask boundaries, computes per-group statistics,
    and marks groups whose mean/median deviates significantly from overall.
    Rolls back if change ratio > 30%.

    Args:
        data: DataFrame with value column + mask column
        mask: name of the mask column
        method: 'mean' or 'median'
        threshold: relative deviation threshold (fraction of overall metric)

    Returns:
        DataFrame with updated mask column
    """
    df = data.copy()
    column = df.columns[0]
    df['group'] = (df[mask] != df[mask].shift()).cumsum()
    df_nonoutlier = df[df[mask] == 0]

    group_variances = _calculate_group_median_variance(df_nonoutlier, column)
    if group_variances.empty:
        logger.warning("variance_filter: group_variances empty, skipping")
        return data

    group_variances.columns = ['group', 'variance']

    if method == 'mean':
        group_metric = df_nonoutlier.groupby('group')[column].apply(
            lambda x: x[(x.quantile(0.05) < x) & (x < x.quantile(0.95))].mean()
        ).reset_index(name='mean')
        overall_metric = df_nonoutlier[
            (df_nonoutlier[column] > df_nonoutlier[column].quantile(0.05)) &
            (df_nonoutlier[column] < df_nonoutlier[column].quantile(0.95))
        ][column].mean()
    else:
        group_metric = df_nonoutlier.groupby('group')[column].apply(
            lambda x: x[(x.quantile(0.05) < x) & (x < x.quantile(0.95))].median()
        ).reset_index(name='median')
        overall_metric = df_nonoutlier[
            (df_nonoutlier[column] > df_nonoutlier[column].quantile(0.05)) &
            (df_nonoutlier[column] < df_nonoutlier[column].quantile(0.95))
        ][column].median()

    if group_metric.isnull().values.any():
        overall_metric = df_nonoutlier[column].mean()
        group_metric = df_nonoutlier.groupby('group')[column].mean().reset_index()

    group_metric.columns = ['group', 'group_metric']
    group_variances = group_variances.merge(group_metric, on='group', how='left')
    group_variances['metric_deviation'] = abs(
        group_variances['group_metric'] - overall_metric)
    deviation_threshold = threshold * abs(overall_metric)
    group_variances['is_anomaly'] = (
        group_variances['metric_deviation'] > deviation_threshold)

    group_num = group_variances[group_variances['is_anomaly']]['group'].values
    for group in group_num:
        df.loc[df['group'] == group, mask] = 1

    change_ratio = np.array(df[mask].sum() - data[mask].sum()) / len(data)
    if change_ratio > 0.3:
        return data.iloc[:, :3]
    else:
        return df.iloc[:, :-1]
