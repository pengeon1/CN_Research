"""
Experiment Runner for Clustered FL: IC-FPLSTM (Trend & Lat/Long) and RC-FPLSTM (Buggy vs. Fixed).
Saves comprehensive results to results/clustering_experiments.json.
"""

from typing import Dict, Any, List
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
from src.clustering.ic_fplstm import ic_fplstm_cluster
from src.clustering.rc_fplstm import rc_fplstm_paper_buggy, rc_fplstm_fixed


def run_cluster_federations(
    clusters: Dict[int, List[int]],
    traffic_data: Dict[str, np.ndarray],
    sname: str,
    rounds: int = 20,
    local_epochs: int = 10,
    mu: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs independent FL server and simulation for each cluster in a given slice."""
    slice_matrix = traffic_data[sname]
    cluster_results = {}
    total_time = 0.0
    all_station_rmses = {}

    for cid, station_indices in clusters.items():
        if not station_indices:
            continue

        c_model = TrafficLSTM(input_dim=1, hidden_dim=64, output_dim=1)
        c_server = FLServer(global_model=c_model, aggregation="unweighted")

        c_clients = {}
        for b in station_indices:
            client_key = f"{sname}_b{b}"
            series = slice_matrix[b]
            dl = build_client_dataloaders(series, batch_size=16)
            c_clients[client_key] = FLClient(
                client_id=client_key,
                slice_name=sname,
                base_station_id=b,
                model=c_model,
                train_loader=dl["train_loader"],
                val_loader=dl["val_loader"],
                test_loader=dl["test_loader"],
                local_epochs=local_epochs,
                mu=mu,
                variant="true_fedprox",
            )

        sim = FLSimulator(server=c_server, clients=c_clients, rounds=rounds, seed=seed)
        start_t = time.time()
        res = sim.run(verbose=False)
        elapsed = time.time() - start_t

        total_time += elapsed
        cluster_results[f"cluster_{cid}"] = {
            "num_stations": len(station_indices),
            "test_rmse": res["final_test_rmse"],
            "time_seconds": elapsed,
        }
        all_station_rmses.update(res["per_client_rmse"])

    mean_rmse = float(np.mean(list(all_station_rmses.values()))) if all_station_rmses else 0.0

    return {
        "mean_rmse": mean_rmse,
        "total_time_seconds": total_time,
        "cluster_details": cluster_results,
    }


def run_clustering_experiments(
    num_stations: int = 57,
    duration_days: int = 10,
    k_clusters: int = 3,
    rounds: int = 20,
    local_epochs: int = 10,
    seeds: List[int] = [42, 1337, 2024],
    results_dir: str = "results",
) -> Dict[str, Any]:
    """Executes full suite of clustered FL experiments."""
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    for seed in seeds:
        print(f"\n================ Running Clustering Experiments with Seed {seed} ================")
        torch.manual_seed(seed)
        np.random.seed(seed)

        gen = CellularTrafficGenerator(
            num_base_stations=num_stations,
            duration_days=duration_days,
            seed=seed,
        )
        traffic_data = gen.generate_all_traffic()
        coords = gen.coordinates  # shape (57, 2)
        slices = list(traffic_data.keys())

        seed_data = {
            "IC_FPLSTM_Traffic_Trend": {},
            "IC_FPLSTM_Lat_Long": {},
            "RC_FPLSTM_Fixed": {},
            "RC_FPLSTM_Buggy": {},
        }

        # 1. IC-FPLSTM (Lat/Long clustering - shared across slices)
        print("--> Generating IC-FPLSTM (Lat/Long) clusters...")
        geo_cluster_res = ic_fplstm_cluster(coords, trend=False, num_clusters=k_clusters, random_state=seed)
        geo_clusters = geo_cluster_res["clusters"]

        # 2. RC-FPLSTM (Fixed & Buggy clusters)
        print("--> Generating RC-FPLSTM clusters...")
        station_ids = list(range(num_stations))
        rc_fixed_res = rc_fplstm_fixed(station_ids, k=k_clusters, random_state=seed)
        rc_buggy_res = rc_fplstm_paper_buggy(station_ids, k=k_clusters, random_state=seed)

        for sname in slices:
            print(f"--> [Slice {sname}] Running cluster federations...")
            # Trend-based clustering (per-slice traffic history)
            slice_matrix = traffic_data[sname]
            trend_cluster_res = ic_fplstm_cluster(
                slice_matrix,
                trend=True,
                pca_components=2,
                num_clusters=k_clusters,
                random_state=seed,
            )
            trend_clusters = trend_cluster_res["clusters"]

            # Run Trend
            res_trend = run_cluster_federations(
                trend_clusters, traffic_data, sname, rounds=rounds, local_epochs=local_epochs, seed=seed
            )
            seed_data["IC_FPLSTM_Traffic_Trend"][sname] = res_trend

            # Run Geo
            res_geo = run_cluster_federations(
                geo_clusters, traffic_data, sname, rounds=rounds, local_epochs=local_epochs, seed=seed
            )
            seed_data["IC_FPLSTM_Lat_Long"][sname] = res_geo

            # Run RC Fixed
            res_rc_fixed = run_cluster_federations(
                rc_fixed_res["clusters"], traffic_data, sname, rounds=rounds, local_epochs=local_epochs, seed=seed
            )
            seed_data["RC_FPLSTM_Fixed"][sname] = res_rc_fixed

            # Run RC Buggy
            res_rc_buggy = run_cluster_federations(
                rc_buggy_res["clusters"], traffic_data, sname, rounds=rounds, local_epochs=local_epochs, seed=seed
            )
            seed_data["RC_FPLSTM_Buggy"][sname] = res_rc_buggy

            print(
                f"    {sname} | Trend: {res_trend['mean_rmse']:.4f} | "
                f"Geo: {res_geo['mean_rmse']:.4f} | "
                f"RC-Fixed: {res_rc_fixed['mean_rmse']:.4f} | "
                f"RC-Buggy: {res_rc_buggy['mean_rmse']:.4f}"
            )

        all_results[f"seed_{seed}"] = seed_data

    out_path = os.path.join(results_dir, "clustering_experiments.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll clustering experiment results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", type=int, default=57)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=10)
    args = parser.parse_args()
    run_clustering_experiments(
        num_stations=args.stations,
        duration_days=args.days,
        k_clusters=args.clusters,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
    )

