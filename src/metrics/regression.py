"""
Regression accuracy metrics: RMSE, MAE, MAPE.
"""

from typing import Dict
import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-4) -> float:
    """Mean Absolute Percentage Error (percentage format, e.g. 12.5%)."""
    denominator = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def compute_all_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes full suite of regression metrics."""
    return {
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
        "MAPE": compute_mape(y_true, y_pred),
    }

