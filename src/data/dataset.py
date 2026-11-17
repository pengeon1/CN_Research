"""
PyTorch Dataset and Sliding Window Builder for RAN Traffic Forecasting.
Implements observation window H and prediction window W sliding windows.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.preprocessor import TimeSeriesMinMaxScaler


class SlidingWindowDataset(Dataset):
    """
    Constructs sequential input-target pairs from 1D time series.
    Input x: shape (H, 1) historical observation window
    Target y: shape (W, 1) or scalar future prediction window
    """

    def __init__(self, x_windows: np.ndarray, y_windows: np.ndarray):
        self.x = torch.from_numpy(x_windows).float()
        self.y = torch.from_numpy(y_windows).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def create_sliding_windows(
    series: np.ndarray,
    observation_window: int = 5,
    prediction_window: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts sliding window tensors from a 1D sequence.
    Returns:
      x: shape (num_samples, observation_window, 1)
      y: shape (num_samples, prediction_window)
    """
    n = len(series)
    num_samples = n - observation_window - prediction_window + 1
    if num_samples <= 0:
        raise ValueError(
            f"Series length {n} too short for H={observation_window} and W={prediction_window}"
        )

    x = np.zeros((num_samples, observation_window, 1), dtype=np.float32)
    y = np.zeros((num_samples, prediction_window), dtype=np.float32)

    for i in range(num_samples):
        x[i, :, 0] = series[i : i + observation_window]
        y[i, :] = series[i + observation_window : i + observation_window + prediction_window]

    return x, y


def build_client_dataloaders(
    series: np.ndarray,
    observation_window: int = 5,
    prediction_window: int = 1,
    train_split: float = 0.8,
    val_split_within_train: float = 0.15,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Preprocesses a single base station time series:
    1. Chronological train/test split (80% train, 20% test).
    2. Carves validation set strictly from end of train split.
    3. Normalizes using scaler fitted strictly on train split.
    4. Creates PyTorch DataLoaders.
    """
    total_len = len(series)
    train_end = int(total_len * train_split)

    raw_train = series[:train_end]
    raw_test = series[train_end:]

    # Carve validation set from train
    val_len = int(len(raw_train) * val_split_within_train)
    pure_train = raw_train[:-val_len]
    pure_val = raw_train[-val_len:]

    # Fit scaler strictly on training data
    scaler = TimeSeriesMinMaxScaler().fit(pure_train)
    scaled_train = scaler.transform(pure_train)
    scaled_val = scaler.transform(pure_val)
    scaled_test = scaler.transform(raw_test)

    # Build windows
    x_train, y_train = create_sliding_windows(scaled_train, observation_window, prediction_window)
    x_val, y_val = create_sliding_windows(scaled_val, observation_window, prediction_window)
    x_test, y_test = create_sliding_windows(scaled_test, observation_window, prediction_window)

    train_ds = SlidingWindowDataset(x_train, y_train)
    val_ds = SlidingWindowDataset(x_val, y_val)
    test_ds = SlidingWindowDataset(x_test, y_test)

    return {
        "scaler": scaler,
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        "x_test_raw": x_test,
        "y_test_raw": y_test,
        "num_train_samples": len(train_ds),
        "num_test_samples": len(test_ds),
    }

