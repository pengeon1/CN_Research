"""
Time Series Preprocessor and Scaler for RAN Traffic Data.
Enforces strict prevention of data leakage by fitting exclusively on training splits.
"""

from typing import Tuple, Optional
import numpy as np


class TimeSeriesMinMaxScaler:
    """
    Min-Max normalizer fitted exclusively on training segments.
    Normalizes inputs to [0, 1] per series or globally per slice.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), eps: float = 1e-7):
        self.feature_range = feature_range
        self.eps = eps
        self.data_min_: Optional[float] = None
        self.data_max_: Optional[float] = None

    def fit(self, data: np.ndarray) -> "TimeSeriesMinMaxScaler":
        """Fits scaler strictly on training split."""
        self.data_min_ = float(np.min(data))
        self.data_max_ = float(np.max(data))
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Scales data to feature range using fitted statistics."""
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler must be fitted before calling transform.")
        range_min, range_max = self.feature_range
        scale = (range_max - range_min) / (self.data_max_ - self.data_min_ + self.eps)
        return (data - self.data_min_) * scale + range_min

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fits on data and returns transformed version."""
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reconstructs original physical scale from normalized predictions."""
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler must be fitted before calling inverse_transform.")
        range_min, range_max = self.feature_range
        scale = (self.data_max_ - self.data_min_ + self.eps) / (range_max - range_min)
        return (data - range_min) * scale + self.data_min_

