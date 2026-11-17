"""
Unit tests for data generation, preprocessing, and windowing.
"""

import numpy as np
import pytest
from src.data.generator import CellularTrafficGenerator
from src.data.preprocessor import TimeSeriesMinMaxScaler
from src.data.dataset import create_sliding_windows, build_client_dataloaders


def test_cellular_traffic_generator():
    """Verify generator output dimensions and statistical characteristics."""
    gen = CellularTrafficGenerator(
        num_base_stations=10,  # fast test on 10 stations
        slices=["Facebook", "YouTube", "Google", "Instagram"],
        duration_days=2,       # 2 days = 288 slots
        granularity_minutes=10,
        seed=42,
    )
    traffic = gen.generate_all_traffic()

    assert set(traffic.keys()) == {"Facebook", "YouTube", "Google", "Instagram"}
    for sname, matrix in traffic.items():
        assert matrix.shape == (10, 288)
        assert np.all(matrix >= 0.0), f"Negative traffic values in {sname}"
        assert not np.isnan(matrix).any(), f"NaNs detected in {sname}"


def test_minmax_scaler():
    """Verify scaler fits and normalizes correctly to [0, 1]."""
    data = np.array([10.0, 20.0, 50.0, 100.0])
    scaler = TimeSeriesMinMaxScaler()
    scaled = scaler.fit_transform(data)

    assert pytest.approx(scaled.min(), abs=1e-5) == 0.0
    assert pytest.approx(scaled.max(), abs=1e-5) == 1.0

    inversed = scaler.inverse_transform(scaled)
    assert np.allclose(data, inversed, atol=1e-4)


def test_sliding_window_dimensions():
    """Verify sliding window shape logic."""
    series = np.arange(100, dtype=np.float32)
    H, W = 5, 2
    x, y = create_sliding_windows(series, observation_window=H, prediction_window=W)

    expected_samples = 100 - H - W + 1
    assert x.shape == (expected_samples, H, 1)
    assert y.shape == (expected_samples, W)

    # First window check: x[0] = [0, 1, 2, 3, 4], y[0] = [5, 6]
    assert np.allclose(x[0, :, 0], np.array([0, 1, 2, 3, 4]))
    assert np.allclose(y[0, :], np.array([5, 6]))


def test_client_dataloaders_split():
    """Verify train/val/test splits and dataloader batch shapes."""
    series = np.sin(np.linspace(0, 50, 500)).astype(np.float32) + 2.0
    result = build_client_dataloaders(
        series,
        observation_window=5,
        prediction_window=1,
        train_split=0.8,
        val_split_within_train=0.15,
        batch_size=16,
    )

    assert result["num_train_samples"] > 0
    assert result["num_test_samples"] > 0
    assert result["train_loader"] is not None
    assert result["test_loader"] is not None

    batch_x, batch_y = next(iter(result["train_loader"]))
    assert batch_x.shape[1] == 5
    assert batch_x.shape[2] == 1
    assert batch_y.shape[1] == 1

