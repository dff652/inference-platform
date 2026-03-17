"""
ADTK + HBOS anomaly detection (jump, mean-shift, dead-value).

Migrated from: ts-iteration-loop/services/inference/jm_detect.py
The most complex CPU algorithm — combines HBOS density scoring,
ADTK multi-scale drift detection, mean-shift histogram analysis,
and dead-value detection.
"""
import numpy as np
import pandas as pd
from kneed import KneeLocator
from adtk.data import validate_series
from adtk.transformer import DoubleRollingAggregate

from .helpers import create_outlier_mask, split_continuous_outliers
from .preprocessing import (
    is_step_data, adaptive_downsample, get_fulldata,
    calculate_sampling_rate, dead_value_detection, get_true_indices,
)


class MeanShiftDetect:
    """Histogram-based mean-shift detection."""

    def __init__(self, bin_nums=20):
        self.bin_nums = bin_nums

    def detect(self, data):
        hist, edges = np.histogram(data, bins=self.bin_nums)
        bin_indices = np.digitize(data, edges) - 1
        bin_indices[bin_indices == -1] = 0
        bin_indices[bin_indices == len(edges) - 1] = len(edges) - 2

        hist = list(hist)
        if len(hist) == 2:
            return list(np.where(bin_indices == np.argmin(hist))[0])

        main_peak_idx = np.argmax(hist)
        max_bin_center = (edges[main_peak_idx] + edges[main_peak_idx + 1]) / 2
        upper, lower = 1.2 * max_bin_center, 0.8 * max_bin_center

        left_border, right_border = main_peak_idx, main_peak_idx

        i = main_peak_idx - 1
        while i > 0:
            if hist[i] > 0.009 * max(hist):
                i -= 1
            else:
                left_border = i
                break

        j = main_peak_idx + 1
        while j < len(hist):
            if hist[j] > 0.009 * max(hist):
                j += 1
            else:
                right_border = j
                break

        gap_indices = []
        if left_border in [0, main_peak_idx] and right_border not in [main_peak_idx, len(hist) - 1]:
            gap_indices = list(range(right_border, len(hist)))
        if left_border not in [0, main_peak_idx] and right_border in [main_peak_idx, len(hist) - 1]:
            gap_indices = list(range(left_border + 1))
        if left_border not in [0, main_peak_idx] and right_border not in [main_peak_idx, len(hist) - 1]:
            gap_indices = list(range(left_border + 1)) + list(range(right_border, len(hist)))

        left_sum = np.sum([hist[i] for i in gap_indices if i < main_peak_idx])
        right_sum = np.sum([hist[i] for i in gap_indices if i > main_peak_idx])

        if left_sum > hist[main_peak_idx] or right_sum > hist[main_peak_idx]:
            revised_gap_indices = [main_peak_idx]
        elif left_sum / hist[main_peak_idx] > 0.85:
            revised_gap_indices = [i for i in gap_indices if i > main_peak_idx]
        elif right_sum / hist[main_peak_idx] > 0.85:
            revised_gap_indices = [i for i in gap_indices if i < main_peak_idx]
        else:
            revised_gap_indices = gap_indices

        outlier_indices = []
        for bin_idx in revised_gap_indices:
            indices = np.where(bin_indices == bin_idx)[0]
            outlier_indices.extend(indices)

        mean_shift_indices = []
        outlier_indices.sort()
        groups = split_continuous_outliers(outlier_indices)
        for group in groups:
            group_data = data.iloc[group]
            if len(group) > 500 and ((np.mean(group_data) > upper) or (np.mean(group_data) < lower)):
                mean_shift_indices.extend(group)
        return mean_shift_indices

    def detect_adapt(self, data):
        hist, edges = np.histogram(data, bins=self.bin_nums)
        bin_indices = np.digitize(data, edges) - 1
        bin_indices[bin_indices == -1] = 0
        bin_indices[bin_indices == len(edges) - 1] = len(edges) - 2

        hist = list(hist)
        if len(hist) == 2:
            return list(np.where(bin_indices == np.argmin(hist))[0])

        main_peak_idx = np.argmax(hist)
        max_bin_center = (edges[main_peak_idx] + edges[main_peak_idx + 1]) / 2
        upper, lower = 1.2 * max_bin_center, 0.8 * max_bin_center

        left_border, right_border = main_peak_idx, main_peak_idx
        lgh = np.log1p(np.array(hist))
        th = np.expm1(self._knpt(lgh, 0))

        i = main_peak_idx - 1
        while i > 0:
            if hist[i] > th:
                i -= 1
            else:
                left_border = i
                break

        j = main_peak_idx + 1
        while j < len(hist):
            if hist[j] > th:
                j += 1
            else:
                right_border = j
                break

        gap_indices = []
        if left_border in [0, main_peak_idx] and right_border not in [main_peak_idx, len(hist) - 1]:
            gap_indices = list(range(right_border, len(hist)))
        if left_border not in [0, main_peak_idx] and right_border in [main_peak_idx, len(hist) - 1]:
            gap_indices = list(range(left_border + 1))
        if left_border not in [0, main_peak_idx] and right_border not in [main_peak_idx, len(hist) - 1]:
            gap_indices = list(range(left_border + 1)) + list(range(right_border, len(hist)))

        left_sum = np.sum([hist[i] for i in gap_indices if i < main_peak_idx])
        right_sum = np.sum([hist[i] for i in gap_indices if i > main_peak_idx])

        if left_sum > hist[main_peak_idx] or right_sum > hist[main_peak_idx]:
            revised_gap_indices = [main_peak_idx]
        elif left_sum / hist[main_peak_idx] > 0.85:
            revised_gap_indices = [i for i in gap_indices if i > main_peak_idx]
        elif right_sum / hist[main_peak_idx] > 0.85:
            revised_gap_indices = [i for i in gap_indices if i < main_peak_idx]
        else:
            revised_gap_indices = gap_indices

        outlier_indices = []
        for bin_idx in revised_gap_indices:
            indices = np.where(bin_indices == bin_idx)[0]
            outlier_indices.extend(indices)

        mean_shift_indices = []
        outlier_indices.sort()
        groups = split_continuous_outliers(outlier_indices)
        for group in groups:
            group_data = data.iloc[group]
            if len(group) > 500 and ((np.mean(group_data) > upper) or (np.mean(group_data) < lower)):
                mean_shift_indices.extend(group)
        return mean_shift_indices

    def detect_patch(self, data):
        tn = 5
        hist, bin_edges = np.histogram(data, bins=self.bin_nums)
        bin_indices = np.digitize(data, bin_edges) - 1
        bin_indices[bin_indices == -1] = 0
        bin_indices[bin_indices == len(bin_edges) - 1] = len(bin_edges) - 2

        main_peak_idx = np.argmax(hist)
        max_bin_center = (bin_edges[main_peak_idx] + bin_edges[main_peak_idx + 1]) / 2
        upper, lower = 1.2 * max_bin_center, 0.8 * max_bin_center

        lgh = np.log1p(np.array(hist))
        trh = np.expm1(self._knpt(lgh, 0))
        th_idx = [np.where(hist == x)[0][0] for x in hist if x > trh]
        ht = self._idx_groups(th_idx)

        avgh = np.array([sum([hist[xx] for xx in cl]) for cl in ht])
        mh = np.argmax(avgh)
        sf = [y for i, li in enumerate(ht) if i != mh for y in li]

        filtered_indices = []
        anomaly_indices = np.where(np.isin(bin_indices, sf))[0]
        anomaly_indices.sort()
        groups = split_continuous_outliers(anomaly_indices)
        for group in groups:
            group_data = data.iloc[group]
            if len(group) > 500 and ((np.mean(group_data) > upper) or (np.mean(group_data) < lower)):
                filtered_indices.extend(group)
        return filtered_indices

    def _knpt(self, scores, mth='auto'):
        sorted_scores = np.sort(scores)
        indices = np.arange(len(sorted_scores))
        kn = KneeLocator(indices, sorted_scores, curve='convex', direction='increasing')
        kne = kn.knee
        if mth == 'auto' and kne:
            return sorted_scores[kne]
        kne2 = self._get_knee2(sorted_scores, indices)
        return sorted_scores[kne2]

    @staticmethod
    def _get_knee2(y1, x1):
        AA = y1[0] - y1[-1]
        BB = x1[-1]
        CC = -x1[-1] * y1[0]
        dist = (AA * x1 + BB * y1 + CC) / np.sqrt(AA ** 2 + BB ** 2)
        return x1[np.argmax(np.abs(dist))]

    @staticmethod
    def _idx_groups(li):
        if not li:
            return []
        groups, group = [], [li[0]]
        for i, j in zip(li, li[1:]):
            if j == i + 1:
                group.append(j)
            else:
                groups.append(group)
                group = [j]
        return groups + [group]


class AnomalyBordersDetector:
    """HBOS + ADTK drift detection for jump boundaries."""

    def __init__(self, ratio=None):
        self.ratio = ratio

    def _nsigma(self, data, th=3):
        clean_data = data[data != 0]
        if len(clean_data) == 0:
            return []
        mean = np.mean(clean_data)
        std = np.std(clean_data)
        lower_bound = mean - th * std
        upper_bound = mean + th * std
        return list(np.where((data < lower_bound) | (data > upper_bound))[0])

    def _adtk_drift_detect(self, data):
        ts = validate_series(data)
        s_fast = DoubleRollingAggregate(agg="mean", window=(100, 3), diff="l1").transform(ts).fillna(0)
        s_mid = DoubleRollingAggregate(agg="mean", window=(3000, 100), diff="l1").transform(ts).fillna(0)
        s_slow = DoubleRollingAggregate(agg="mean", window=(100000, 2000), diff="l1").transform(ts).fillna(0)
        fast_a = self._nsigma(s_fast, th=5)
        mid_a = self._nsigma(s_mid, th=5)
        slow_a = self._nsigma(s_slow, th=5)
        return list(set(fast_a + mid_a + slow_a))

    def _hbos(self, data, bins=50):
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_width = bin_edges[1] - bin_edges[0]
        densities = hist / (len(data) * bin_width)
        densities = np.clip(densities, 1e-6, None)
        scores = np.zeros_like(data, dtype=float)
        bin_indices = np.digitize(data, bin_edges) - 1
        bin_indices[bin_indices == -1] = 0
        bin_indices[bin_indices == len(bin_edges) - 1] = len(bin_edges) - 2
        for bin_idx in range(bins):
            indices = np.where(bin_indices == bin_idx)[0]
            scores[indices] = -np.log(densities[bin_idx])
        drops = np.diff(scores)
        idx = np.argmax(drops)
        threshold = scores[idx + 1]
        return np.sort(np.where(scores > threshold)[0])

    def _get_adapted_ratio(self, values):
        chunks = 200
        global_change = max(values) - min(values)
        if global_change == 0:
            return 0.1
        split_points = np.linspace(0, len(values), chunks + 1, dtype=int)
        change_ratio_list = []
        for i in range(len(split_points) - 1):
            sv = np.array(values)[split_points[i]:split_points[i + 1]]
            change_ratio_list.append((max(sv) - min(sv)) / global_change)
        change_ratio_list.sort()
        indices = np.arange(len(change_ratio_list))
        kn = KneeLocator(indices, change_ratio_list, curve='convex', direction='increasing')
        kne = kn.knee
        if kne:
            return change_ratio_list[kne]
        arr = np.array(change_ratio_list)
        return arr[np.argmax(arr[1:] / (arr[:-1] + 0.001))]

    def detect(self, data):
        border_hbos = self._hbos(data.values.tolist())
        border_adtk = self._adtk_drift_detect(data)

        jump_indices = []
        merged = sorted(set(list(border_hbos) + border_adtk))
        groups = split_continuous_outliers(merged)

        if not self.ratio and len(np.unique(data)) > 1:
            self.ratio = self._get_adapted_ratio(data)

        global_change = max(data) - min(data)
        if global_change == 0:
            return []

        for group in groups:
            if len(group) == 1:
                rng = data.iloc[max(0, group[0] - 10):group[0] + 10]
                local_change = max(rng) - min(rng)
            else:
                local_change = max(data.iloc[group]) - min(data.iloc[group])
            if local_change / global_change >= self.ratio:
                jump_indices.extend(group)
        return jump_indices


def _process_data(raw_data, col, sampler, sr, min_nums):
    """Downsample and prepare data for detection."""
    full_data = get_fulldata(raw_data, col)
    downsampled_data, ts, position_index = adaptive_downsample(
        raw_data[col], downsampler=sampler, sample_param=sr, min_threshold=min_nums
    )
    ds_data = pd.DataFrame(downsampled_data, index=ts)
    ds_data = ds_data[~ds_data.index.duplicated(keep='first')]
    return ds_data, position_index


def detect(data, downsampler='m4', sample_param=0.1, bin_nums=20,
           min_threshold=200000, ratio=None):
    """
    Full ADTK+HBOS detection pipeline: jump + mean-shift + dead-value.

    Args:
        data: pd.DataFrame with DatetimeIndex, single value column
        downsampler: 'm4', 'minmax', or 'none'
        sample_param: downsample ratio (0-1) or fixed count (>1)
        bin_nums: histogram bins for mean-shift detection
        min_threshold: minimum data length before downsampling
        ratio: jump filter ratio, None for auto-adaptive

    Returns:
        raw_data: DataFrame with boolean mask column
        position_index: downsample position mapping
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data = data.copy()
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError(f"Index must be DatetimeIndex. Error: {e}")

    raw_data = data.copy()
    step_type = is_step_data(data)
    point_name = data.columns[0]
    processed_data, position_index = _process_data(data, point_name, downsampler, sample_param, min_threshold)

    if point_name not in processed_data.columns:
        if processed_data.shape[1] == 1:
            processed_data.columns = [point_name]
        else:
            processed_data[point_name] = processed_data.iloc[:, 0]

    global_indices = np.array([], dtype=int)
    if not step_type:
        abd = AnomalyBordersDetector(ratio=ratio)
        msd = MeanShiftDetect(bin_nums=bin_nums)
        jump_anomalies = abd.detect(processed_data[point_name])
        mean_shift = msd.detect(processed_data[point_name])
        if not mean_shift:
            mean_shift = msd.detect_adapt(processed_data[point_name])
            if not mean_shift:
                mean_shift = msd.detect_patch(processed_data[point_name])
        merged_indices = list(set(list(jump_anomalies) + mean_shift))

        _, dead_df, _ = dead_value_detection(processed_data, 72000)
        dead_per = dead_df.sum() / len(dead_df)
        dead_per = dead_per.values[0]
        if dead_per < 0.7:
            natural_indices = get_true_indices(dead_df)
            global_indices = np.unique(merged_indices + natural_indices)
        else:
            global_indices = merged_indices

    global_mask = create_outlier_mask(processed_data, global_indices)
    processed_data['global_mask'] = global_mask
    processed_data['Group'] = (processed_data['global_mask'] != processed_data['global_mask'].shift()).cumsum()

    grouped = processed_data[processed_data['global_mask'] == 1].groupby('Group')

    raw_data[point_name] = False
    for name, group in grouped:
        start_time = group.index.min()
        end_time = group.index.max()
        raw_data.loc[(raw_data.index >= start_time) & (raw_data.index <= end_time), point_name] = True

    return raw_data, position_index
