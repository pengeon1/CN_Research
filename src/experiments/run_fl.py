"""
Experiment Runner for Federated Learning: FALSTM vs. FPLSTM, mu sweep, and Omega sweep.
Saves comprehensive results to results/fl_experiments.json.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import argparse
import time
import numpy as np
import torch

from src.data.generator import CellularTrafficGenerator
from src.data.dataset import build_client_dataloaders
from src.models.lstm import TrafficLSTM
from src.fl.client import FLClient
from src.fl.server import FLServer
from src.fl.simulator import FLSimulator
from src.metrics.communication import compute_fl_communication_cost


def setup_fl_environment(
    traffic_data: Dict[str, np.ndarray],
    mu: float = 1.0,
    lr: float = 0.001,
    local_epochs: int = 20,
    variant: str = "true_fedprox",
    aggregation: str = "unweighted",
    device: str = "cpu",
) -> Tuple[FLServer, Dict[str, FLClient]]:
    """Initializes global server and 228 local clients."""
    global_model = TrafficLSTM(input_dim=1, hidden_dim=64, output_dim=1).to(device)
    server = FLServer(global_model=global_model, aggregation=aggregation, device=device)

    clients = {}
    for sname, slice_matrix in traffic_data.items():
        num_stations = slice_matrix.shape[0]
        for b in range(num_stations):
            cid = f"{sname}_station_{b}"
            series = slice_matrix[b]
            dl_data = build_client_dataloaders(series, batch_size=16)

            client = FLClient(
                client_id=cid,
                slice_name=sname,
                base_station_id=b,
                model=global_model,
                train_loader=dl_data["train_loader"],
                val_loader=dl_data["val_loader"],
                test_loader=dl_data["test_loader"],
                lr=lr,
                local_epochs=local_epochs,
                mu=mu,
                device=device,
                variant=variant,
            )
            clients[cid] = client

    return server, clients


def run_fl_experiments(
    num_stations: int = 57,
    duration_days: int = 10,
    rounds: int = 20,
    local_epochs: int = 10,
    seeds: List[int] = [42, 1337, 2024],
    results_dir: str = "results",
) -> Dict[str, Any]:
    """Executes full suite of FL reproduction experiments."""
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    for seed in seeds:
        print(f"\n================ Running FL Experiments with Seed {seed} ================")
        torch.manual_seed(seed)
        np.random.seed(seed)

        gen = CellularTrafficGenerator(
            num_base_stations=num_stations,
            duration_days=duration_days,
            seed=seed,
        )
        traffic_data = gen.generate_all_traffic()

        seed_data = {}

        # 1. FALSTM (FedAvg, mu=0)
        print("--> Running FALSTM (FedAvg, mu=0.0)...")
        server_fa, clients_fa = setup_fl_environment(
            traffic_data,
            mu=0.0,
            local_epochs=local_epochs,
            variant="true_fedprox",
            aggregation="unweighted",
        )
        sim_fa = FLSimulator(server=server_fa, clients=clients_fa, rounds=rounds, fraction=1.0, seed=seed)
        fa_res = sim_fa.run(verbose=False)
        seed_data["FALSTM"] = {
            "test_rmse": fa_res["final_test_rmse"],
            "slice_rmse": fa_res["slice_test_rmse"],
            "time_seconds": fa_res["total_time_seconds"],
        }
        print(f"    FALSTM: Mean Test RMSE = {fa_res['final_test_rmse']:.4f}")

        # 2. FPLSTM mu parameter sweep: [0.001, 0.01, 0.1, 0.5, 1.0] (matching Fig 12)
        seed_data["FPLSTM_mu_sweep"] = {}
        for mu_val in [0.001, 0.01, 0.1, 0.5, 1.0]:
            print(f"--> Running FPLSTM (mu = {mu_val})...")
            server_fp, clients_fp = setup_fl_environment(
                traffic_data,
                mu=mu_val,
                local_epochs=local_epochs,
                variant="true_fedprox",
                aggregation="unweighted",
            )
            sim_fp = FLSimulator(server=server_fp, clients=clients_fp, rounds=rounds, fraction=1.0, seed=seed)
            fp_res = sim_fp.run(verbose=False)
            seed_data["FPLSTM_mu_sweep"][f"mu_{mu_val}"] = {
                "test_rmse": fp_res["final_test_rmse"],
                "slice_rmse": fp_res["slice_test_rmse"],
                "time_seconds": fp_res["total_time_seconds"],
            }
            print(f"    FPLSTM (mu={mu_val}): Mean Test RMSE = {fp_res['final_test_rmse']:.4f}")

        # 3. Client participation rate Omega sweep: [0.1, 0.3, 0.5, 0.7, 1.0] (matching Fig 10)
        seed_data["FPLSTM_omega_sweep"] = {}
        for omega in [0.1, 0.3, 0.5, 0.7, 1.0]:
            print(f"--> Running FPLSTM with Omega = {omega}...")
            server_om, clients_om = setup_fl_environment(
                traffic_data,
                mu=1.0,
                local_epochs=local_epochs,
                variant="true_fedprox",
                aggregation="unweighted",
            )
            sim_om = FLSimulator(server=server_om, clients=clients_om, rounds=rounds, fraction=omega, seed=seed)
            om_res = sim_om.run(verbose=False)
            comm_cost = compute_fl_communication_cost(
                rounds=rounds,
                num_clients=len(clients_om),
                selection_fraction=omega,
                model_params_count=server_om.count_parameters(),
            )
            seed_data["FPLSTM_omega_sweep"][f"omega_{omega}"] = {
                "test_rmse": om_res["final_test_rmse"],
                "slice_rmse": om_res["slice_test_rmse"],
                "communication_cost_mb": comm_cost["total_megabytes"],
                "time_seconds": om_res["total_time_seconds"],
            }
            print(f"    Omega={omega}: RMSE = {om_res['final_test_rmse']:.4f}, Comm = {comm_cost['total_megabytes']:.2f} MB")

        # 4. True FedProx vs Paper Alg 2 comparison
        print("--> Running Paper Alg 2 Variant (literal post-SGD argmin)...")
        server_alg2, clients_alg2 = setup_fl_environment(
            traffic_data,
            mu=1.0,
            local_epochs=local_epochs,
            variant="paper_alg2",
            aggregation="unweighted",
        )
        sim_alg2 = FLSimulator(server=server_alg2, clients=clients_alg2, rounds=rounds, fraction=1.0, seed=seed)
        alg2_res = sim_alg2.run(verbose=False)
        seed_data["Paper_Alg2_Variant"] = {
            "test_rmse": alg2_res["final_test_rmse"],
            "slice_rmse": alg2_res["slice_test_rmse"],
            "time_seconds": alg2_res["total_time_seconds"],
        }
        print(f"    Paper Alg 2: Mean Test RMSE = {alg2_res['final_test_rmse']:.4f}")

        all_results[f"seed_{seed}"] = seed_data

    out_path = os.path.join(results_dir, "fl_experiments.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll FL experiment results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", type=int, default=57)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=10)
    args = parser.parse_args()
    run_fl_experiments(
        num_stations=args.stations,
        duration_days=args.days,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
    )
