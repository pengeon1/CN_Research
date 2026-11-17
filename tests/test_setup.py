"""
Unit and CI smoke test for Phase 0 setup.
Verifies config integrity, tensor operations, FL client/server initialization,
and mathematical aggregation correctness.
"""

import os
import yaml
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.fl.client import FLClient
from src.fl.server import FLServer
from src.fl.simulator import FLSimulator


class DummyLSTM(nn.Module):
    """Minimal LSTM module for testing."""

    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def test_config_loading():
    """Verify default.yaml exists and contains required parameters."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    assert os.path.exists(config_path), "configs/default.yaml not found"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg["data"]["num_base_stations"] == 57
    assert cfg["data"]["slices"] == ["Facebook", "YouTube", "Google", "Instagram"]
    assert cfg["fl"]["total_clients"] == 228
    assert cfg["model"]["hidden_dim"] == 64
    assert cfg["fl"]["rounds"] == 20
    assert cfg["training"]["learning_rate"] == 0.001
    assert cfg["training"]["batch_size"] == 16


def test_fl_aggregation_mathematics():
    """Verify exact mathematical behavior of unweighted and weighted aggregation."""
    model = DummyLSTM(input_dim=1, hidden_dim=4, output_dim=1)
    server_unweighted = FLServer(global_model=model, aggregation="unweighted")
    server_weighted = FLServer(global_model=model, aggregation="weighted")

    # Create dummy client updates
    w1 = {k: torch.ones_like(v) * 2.0 for k, v in model.state_dict().items()}
    w2 = {k: torch.ones_like(v) * 4.0 for k, v in model.state_dict().items()}

    updates = [
        {"client_id": "c1", "weights": w1, "num_samples": 10},
        {"client_id": "c2", "weights": w2, "num_samples": 30},
    ]

    # Unweighted: (2.0 + 4.0) / 2 = 3.0
    agg_unweighted = server_unweighted.aggregate(updates)
    for v in agg_unweighted.values():
        assert torch.allclose(v, torch.tensor(3.0)), "Unweighted aggregation mismatch"

    # Weighted: (2.0 * 10 + 4.0 * 30) / 40 = (20 + 120) / 40 = 140 / 40 = 3.5
    agg_weighted = server_weighted.aggregate(updates)
    for v in agg_weighted.values():
        assert torch.allclose(v, torch.tensor(3.5)), "Weighted aggregation mismatch"


def test_fl_simulator_dry_run():
    """Verify that FLSimulator executes a 2-round dry run deterministically."""
    model = DummyLSTM(input_dim=1, hidden_dim=8, output_dim=1)
    server = FLServer(global_model=model, aggregation="unweighted")

    # Create 2 dummy clients with small synthetic datasets
    clients = {}
    for i in range(2):
        x = torch.randn(20, 5, 1)
        y = torch.randn(20, 1)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)
        c = FLClient(
            client_id=f"client_{i}",
            slice_name=f"slice_{i}",
            base_station_id=i,
            model=model,
            train_loader=loader,
            val_loader=loader,
            test_loader=loader,
            local_epochs=2,
            lr=0.01,
            mu=0.1,
            variant="true_fedprox",
        )
        clients[c.client_id] = c

    sim = FLSimulator(server=server, clients=clients, rounds=2, fraction=1.0, seed=42)
    results = sim.run(verbose=False)

    assert results["total_rounds"] == 2
    assert len(results["history"]) == 2
    assert "final_test_rmse" in results
    assert results["final_test_rmse"] >= 0.0


def test_communication_cost_formula():
    """Verify calculation of communication cost model: 2 * C * N * Omega * psi."""
    model = DummyLSTM(input_dim=1, hidden_dim=8, output_dim=1)
    server = FLServer(global_model=model)

    params = server.count_parameters()
    assert params > 0

    cost_mb = server.compute_communication_cost(
        rounds=20,
        num_clients=228,
        participation_fraction=1.0,
        precision_bits=32,
    )
    expected_bytes = 2.0 * 20 * 228 * 1.0 * (params * 32 / 8.0)
    expected_mb = expected_bytes / (1024.0 * 1024.0)
    assert abs(cost_mb - expected_mb) < 1e-6

