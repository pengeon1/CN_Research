"""
Experiment Runner for Proposed Improvements:
1. Improvement A: Sample-weighted FedProx (Eq. 1) vs. Unweighted (Alg 1 Line 7)
2. Improvement B: Model-Similarity Clustering (Zero raw telemetry sharing)
3. Improvement C: SLA-aware Asymmetric Pinball Loss (tau = 0.8)
4. Improvement D: Differential Privacy (DP-FedProx gradient clipping + Gaussian noise)
Saves results to results/improvements_experiments.json.
"""

from typing import Dict, Any, List, Optional
import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from src.data.generator import CellularTrafficGenerator
from src.data.dataset import build_client_dataloaders
from src.models.lstm import TrafficLSTM
from src.models.loss_functions import SLAPinballLoss
from src.fl.client import FLClient
from src.fl.server import FLServer
from src.fl.simulator import FLSimulator
from src.metrics.sla import evaluate_sla_across_margins
from src.metrics.regression import compute_rmse


def run_dp_fl_simulation(
    traffic_data: Dict[str, np.ndarray],
    noise_sigma: float = 0.01,
    clip_norm: float = 1.0,
    rounds: int = 20,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Executes Differential Privacy FedProx (DP-FedProx):
    Clips local update norms to clip_norm and injects Gaussian noise N(0, sigma^2).
    """
    global_model = TrafficLSTM(input_dim=1, hidden_dim=64, output_dim=1)
    server = FLServer(global_model=global_model, aggregation="unweighted")

    clients = {}
    for sname, slice_matrix in traffic_data.items():
        for b in range(slice_matrix.shape[0]):
            cid = f"{sname}_b{b}"
            dl = build_client_dataloaders(slice_matrix[b], batch_size=16)
            clients[cid] = FLClient(
                client_id=cid,
                slice_name=sname,
                base_station_id=b,
                model=global_model,
                train_loader=dl["train_loader"],
                test_loader=dl["test_loader"],
                local_epochs=5,
                mu=1.0,
            )

    client_ids = list(clients.keys())
    rng = np.random.RandomState(seed)

    for c in range(rounds):
        current_weights = server.get_parameters()
        updates = []
        for cid in client_ids:
            client = clients[cid]
            u = client.train_round(current_weights)

            # Apply DP clipping and noise to difference (w - w_global)
            noisy_weights = {}
            for k, param in u["weights"].items():
                diff = param - current_weights[k]
                # Gradient/weight norm clipping
                diff_norm = torch.norm(diff)
                if diff_norm > clip_norm:
                    diff = diff * (clip_norm / diff_norm)
                # Add Gaussian perturbation
                noise = torch.randn_like(diff) * noise_sigma
                noisy_weights[k] = current_weights[k] + diff + noise
            u["weights"] = noisy_weights
            updates.append(u)

        server.aggregate(updates)

    # Evaluate
    rmses = []
    for client in clients.values():
        res = client.evaluate(use_test=True)
        rmses.append(res["rmse"])

    return {"mean_test_rmse": float(np.mean(rmses))}


def run_all_improvements(
    num_stations: int = 57,
    duration_days: int = 10,
    rounds: int = 20,
    local_epochs: int = 10,
    seeds: List[int] = [42, 1337, 2024],
    results_dir: str = "results",
) -> Dict[str, Any]:
    """Runs all 4 proposed improvements with multi-seed evaluation."""
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    for seed in seeds:
        print(f"\n================ Running Improvements with Seed {seed} ================")
        torch.manual_seed(seed)
        np.random.seed(seed)

        gen = CellularTrafficGenerator(
            num_base_stations=num_stations,
            duration_days=duration_days,
            seed=seed,
        )
        traffic_data = gen.generate_all_traffic()
        slices = list(traffic_data.keys())

        seed_data = {}

        # -------------------------------------------------------------
        # 1. Improvement A: Sample-Weighted vs. Unweighted Aggregation
        # -------------------------------------------------------------
        print("--> [Improvement A] Comparing Unweighted vs. Sample-Weighted Aggregation...")
        # Create heterogeneous dataset sizes (simulate variable station deployment lengths)
        rng = np.random.RandomState(seed)
        hetero_traffic = {}
        for sname, mat in traffic_data.items():
            het_mat = []
            for b in range(num_stations):
                # Variable length: randomly keep 50% to 100% of data
                keep_len = rng.randint(int(mat.shape[1] * 0.5), mat.shape[1] + 1)
                het_mat.append(mat[b, :keep_len])
            hetero_traffic[sname] = het_mat

        # Train with Unweighted (Paper Alg 1)
        # Train with Weighted (Our Corrected Eq. 1)
        for agg_scheme in ["unweighted", "weighted"]:
            g_model = TrafficLSTM(input_dim=1, hidden_dim=64, output_dim=1)
            server = FLServer(global_model=g_model, aggregation=agg_scheme)
            clients = {}
            for sname, s_list in hetero_traffic.items():
                for b, series in enumerate(s_list):
                    cid = f"{sname}_b{b}"
                    dl = build_client_dataloaders(series, batch_size=16)
                    clients[cid] = FLClient(
                        client_id=cid,
                        slice_name=sname,
                        base_station_id=b,
                        model=g_model,
                        train_loader=dl["train_loader"],
                        test_loader=dl["test_loader"],
                        local_epochs=local_epochs,
                        mu=1.0,
                    )
            sim = FLSimulator(server=server, clients=clients, rounds=rounds, seed=seed)
            res = sim.run(verbose=False)
            seed_data[f"Improvement_A_{agg_scheme}"] = {
                "test_rmse": res["final_test_rmse"],
                "slice_rmse": res["slice_test_rmse"],
            }
            print(f"    Aggregation [{agg_scheme}]: RMSE = {res['final_test_rmse']:.4f}")

        # -------------------------------------------------------------
        # 2. Improvement B: Model-Weight Similarity Clustering
        # -------------------------------------------------------------
        print("--> [Improvement B] Model-Weight Similarity Clustering (Zero Telemetry Shared)...")
        # Step 1: Run 1 warmup round on all clients to gather weight update vectors
        warmup_model = TrafficLSTM(input_dim=1, hidden_dim=64, output_dim=1)
        warmup_clients = {}
        target_slice = "Facebook"
        for b in range(num_stations):
            cid = f"{target_slice}_b{b}"
            dl = build_client_dataloaders(traffic_data[target_slice][b], batch_size=16)
            warmup_clients[cid] = FLClient(
                client_id=cid,
                slice_name=target_slice,
                base_station_id=b,
                model=warmup_model,
                train_loader=dl["train_loader"],
                test_loader=dl["test_loader"],
                local_epochs=2,
            )

        client_vectors = []
        station_order = []
        for b in range(num_stations):
            cid = f"{target_slice}_b{b}"
            c = warmup_clients[cid]
            u = c.train_round(warmup_model.state_dict())
            # Flatten weights
            flat = torch.cat([p.flatten() for p in u["weights"].values()]).cpu().numpy()
            client_vectors.append(flat)
            station_order.append(b)

        # Cluster by weight similarity (Cosine distance)
        weight_mat = np.array(client_vectors)
        kmeans = KMeans(n_clusters=3, random_state=seed, n_init=10)
        weight_clusters_raw = kmeans.fit_predict(weight_mat)
        weight_clusters: Dict[int, List[int]] = {0: [], 1: [], 2: []}
        for b_idx, cl in enumerate(weight_clusters_raw):
            weight_clusters[int(cl)].append(station_order[b_idx])

        # Evaluate clustered FL with model-similarity clusters
        # (imported run_cluster_federations logic)
        sim_scores = []
        for cid, stations in weight_clusters.items():
            if not stations:
                continue
            c_mod = TrafficLSTM(input_dim=1, hidden_dim=64, output_dim=1)
            c_srv = FLServer(global_model=c_mod, aggregation="unweighted")
            c_cls = {}
            for b in stations:
                ckey = f"{target_slice}_b{b}"
                dl = build_client_dataloaders(traffic_data[target_slice][b], batch_size=16)
                c_cls[ckey] = FLClient(
                    client_id=ckey,
                    slice_name=target_slice,
                    base_station_id=b,
                    model=c_mod,
                    train_loader=dl["train_loader"],
                    test_loader=dl["test_loader"],
                    local_epochs=local_epochs,
                )
            sim_c = FLSimulator(server=c_srv, clients=c_cls, rounds=rounds, seed=seed)
            r = sim_c.run(verbose=False)
            sim_scores.append(r["final_test_rmse"])

        model_sim_rmse = float(np.mean(sim_scores))
        seed_data["Improvement_B_Model_Similarity_Clustering"] = {
            "test_rmse": model_sim_rmse,
            "cluster_sizes": {k: len(v) for k, v in weight_clusters.items()},
        }
        print(f"    Model-Similarity Clustering: Mean RMSE = {model_sim_rmse:.4f}")

        # -------------------------------------------------------------
        # 3. Improvement C: SLA-Aware Asymmetric Quantile Loss
        # -------------------------------------------------------------
        print("--> [Improvement C] SLA-Aware Quantile Loss (tau = 0.8) vs. MSE...")
        sla_results = {}
        for loss_type in ["MSE", "SLA_Pinball"]:
            # Train local agents on Facebook slice
            station_targets = []
            station_preds = []
            for b in range(min(15, num_stations)):  # fast subset for ablation
                series = traffic_data["Facebook"][b]
                dl = build_client_dataloaders(series, batch_size=16)
                mod = TrafficLSTM(input_dim=1, hidden_dim=64, output_dim=1)
                opt = torch.optim.SGD(mod.parameters(), lr=0.001)
                crit = nn.MSELoss() if loss_type == "MSE" else SLAPinballLoss(tau=0.8)

                mod.train()
                for _ in range(local_epochs):
                    for xb, yb in dl["train_loader"]:
                        opt.zero_grad()
                        pr = mod(xb)
                        ls = crit(pr, yb)
                        ls.backward()
                        opt.step()

                mod.eval()
                with torch.no_grad():
                    for xb, yb in dl["test_loader"]:
                        pr = mod(xb)
                        station_preds.append(pr.cpu().numpy())
                        station_targets.append(yb.cpu().numpy())

            targets_arr = np.vstack(station_targets)
            preds_arr = np.vstack(station_preds)
            sla_metrics = evaluate_sla_across_margins(targets_arr, preds_arr)
            rmse_val = compute_rmse(targets_arr, preds_arr)
            sla_results[loss_type] = {
                "RMSE": rmse_val,
                "SLA_Violations": sla_metrics,
            }
            print(f"    Loss [{loss_type}]: RMSE = {rmse_val:.4f}, SLA(gamma=0) = {sla_metrics['gamma_0.0']:.2f}%")

        seed_data["Improvement_C_SLA_Loss"] = sla_results

        # -------------------------------------------------------------
        # 4. Improvement D: Differential Privacy (DP-FedProx)
        # -------------------------------------------------------------
        print("--> [Improvement D] Differential Privacy Trade-off Sweep...")
        dp_results = {}
        for sigma in [0.0, 0.005, 0.01, 0.05, 0.1]:
            dp_res = run_dp_fl_simulation(
                traffic_data,
                noise_sigma=sigma,
                clip_norm=1.0,
                rounds=10,
                seed=seed,
            )
            dp_results[f"sigma_{sigma}"] = dp_res["mean_test_rmse"]
            print(f"    DP noise sigma={sigma}: Mean Test RMSE = {dp_res['mean_test_rmse']:.4f}")

        seed_data["Improvement_D_Differential_Privacy"] = dp_results
        all_results[f"seed_{seed}"] = seed_data

    out_path = os.path.join(results_dir, "improvements_experiments.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll improvement results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", type=int, default=57)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=10)
    args = parser.parse_args()
    run_all_improvements(
        num_stations=args.stations,
        duration_days=args.days,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
    )

