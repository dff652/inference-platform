"""
Isolation Forest anomaly detection.

Migrated from: ts-iteration-loop/services/inference/isolation_forest.py
"""
import numpy as np
from sklearn.ensemble import IsolationForest

from .helpers import _merge_nearby_anomalies


def detect(data, contamination=0.01, merge_distance=1200):
    """
    Detect anomalies using Isolation Forest.

    Args:
        data: pd.Series or 1-d array of values
        contamination: expected proportion of outliers (0, 0.5]
        merge_distance: max gap to merge nearby anomalies

    Returns:
        anomaly_indices: np.ndarray of anomaly point indices
    """
    values = data.values.reshape(-1, 1) if hasattr(data, 'values') else np.asarray(data).reshape(-1, 1)
    clf = IsolationForest(contamination=contamination)
    labels = clf.fit_predict(values)
    anomalies = np.where(labels == -1)[0]
    return _merge_nearby_anomalies(anomalies, merge_distance)
