"""
Experiment Runner for Baselines: DLSTM and Centralized LSTM.
Saves comprehensive results (RMSE, MAE, MAPE, SLA violations) across slices and seeds.
"""

from typing import Dict, Any, List
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.generator import CellularTrafficGenerator
from src.data.dataset import build_client_dataloaders
from src.models.lstm import TrafficLSTM
from src.models.centralized_lstm import CentralizedLSTMTrainer
from src.metrics.regression import compute_all_regression_metrics
from src.metrics.sla import evaluate_sla_across_margins


def train_dlstm_agent(
    train_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int = 1,
    hidden_dim: int = 64,
    output_dim: int = 1,
    lr: float = 0.001,
    epochs: int = 20,  # CPU friendly
    device: str = "cpu",
) -> Dict[str, Any]:
    """Trains an isolated, non-collaborative DLSTM agent."""
    model = TrafficLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            pred = model(x_b)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_b.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    metrics = compute_all_regression_metrics(targets, preds)
    sla_metrics = evaluate_sla_across_margins(targets, preds)

    return {
        "metrics": metrics,
        "sla": sla_metrics,
        "predictions": preds.tolist(),
        "targets": targets.tolist(),
    }


def run_all_baselines(
    num_stations: int = 57,
    duration_days: int = 10,
    seeds: List[int] = [42, 1337, 2024],
    epochs: int = 20,
    results_dir: str = "results",
) -> Dict[str, Any]:
    """Runs DLSTM and Centralized LSTM across multiple random seeds."""
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    for seed in seeds:
        print(f"\n================ Running Baselines with Seed {seed} ================")
        torch.manual_seed(seed)
        np.random.seed(seed)

        gen = CellularTrafficGenerator(
            num_base_stations=num_stations,
            duration_days=duration_days,
            seed=seed,
        )
        traffic_data = gen.generate_all_traffic()
        slices = list(traffic_data.keys())

        seed_results = {"DLSTM": {}, "Centralized": {}}

        # 1. Run DLSTM for all slices
        for sname in slices:
            print(f"--> [Seed {seed}] Training DLSTM for slice {sname} across {num_stations} stations...")
            slice_matrix = traffic_data[sname]
            station_rmses = []
            station_maes = []
            station_mapes = []
            all_targets = []
            all_preds = []

            for b in range(num_stations):
                series = slice_matrix[b]
                dl_data = build_client_dataloaders(
                    series,
                    observation_window=5,
                    prediction_window=1,
                    train_split=0.8,
                    val_split_within_train=0.15,
                    batch_size=16,
                )
                res = train_dlstm_agent(
                    dl_data["train_loader"],
                    dl_data["test_loader"],
                    epochs=epochs,
                )
                station_rmses.append(res["metrics"]["RMSE"])
                station_maes.append(res["metrics"]["MAE"])
                station_mapes.append(res["metrics"]["MAPE"])
                all_targets.extend(res["targets"])
                all_preds.extend(res["predictions"])

            targets_arr = np.array(all_targets)
            preds_arr = np.array(all_preds)
            slice_sla = evaluate_sla_across_margins(targets_arr, preds_arr)

            seed_results["DLSTM"][sname] = {
                "mean_rmse": float(np.mean(station_rmses)),
                "std_rmse": float(np.std(station_rmses)),
                "mean_mae": float(np.mean(station_maes)),
                "mean_mape": float(np.mean(station_mapes)),
                "sla": slice_sla,
            }
            print(f"    DLSTM {sname}: RMSE = {np.mean(station_rmses):.4f} +/- {np.std(station_rmses):.4f}")

        # 2. Run Centralized LSTM for each slice
        for sname in slices:
            print(f"--> [Seed {seed}] Training Centralized LSTM for slice {sname}...")
            slice_matrix = traffic_data[sname]
            train_loaders = []
            test_loaders = []

            for b in range(num_stations):
                series = slice_matrix[b]
                dl_data = build_client_dataloaders(series, batch_size=32)
                train_loaders.append(dl_data["train_loader"])
                test_loaders.append(dl_data["test_loader"])

            model = TrafficLSTM(input_dim=1, hidden_dim=64, output_dim=1)
            trainer = CentralizedLSTMTrainer(
                model=model,
                train_loaders=train_loaders,
                test_loaders=test_loaders,
                epochs=min(epochs, 25),
            )
            trainer.train(verbose=False)
            eval_metrics = trainer.evaluate()

            seed_results["Centralized"][sname] = {
                "rmse": float(eval_metrics["rmse"]),
                "mse": float(eval_metrics["mse"]),
            }
            print(f"    Centralized {sname}: RMSE = {eval_metrics['rmse']:.4f}")

        all_results[f"seed_{seed}"] = seed_results

    out_path = os.path.join(results_dir, "baselines.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll baseline results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", type=int, default=57)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    run_all_baselines(num_stations=args.stations, duration_days=args.days, epochs=args.epochs)

