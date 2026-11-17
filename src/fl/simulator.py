"""
Federated Learning Simulation Orchestrator.
Manages deterministic communication rounds, client sampling, and tracking.
"""

from typing import Dict, List, Any, Optional
import time
import numpy as np
import torch
from src.fl.client import FLClient
from src.fl.server import FLServer


class FLSimulator:
    """
    In-memory simulation engine for multi-agent federated learning in RAN slicing.
    Enforces deterministic seeding and per-round metric aggregation.
    """

    def __init__(
        self,
        server: FLServer,
        clients: Dict[str, FLClient],
        rounds: int = 20,
        fraction: float = 1.0,
        seed: int = 42,
    ):
        self.server = server
        self.clients = clients
        self.rounds = rounds
        self.fraction = fraction
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.history: List[Dict[str, Any]] = []

    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """Executes full federated learning training loop across C rounds."""
        client_ids = list(self.clients.keys())
        start_time = time.time()

        for c in range(self.rounds):
            round_start = time.time()
            selected_ids = self.server.select_clients(
                client_ids=client_ids,
                fraction=self.fraction,
                rng=self.rng,
            )

            current_global_weights = self.server.get_parameters()
            client_updates = []

            # Simulate parallel local training
            for cid in selected_ids:
                client = self.clients[cid]
                update = client.train_round(global_weights=current_global_weights)
                client_updates.append(update)

            # Global aggregation (Algorithm 1 Line 7 or weighted)
            self.server.aggregate(client_updates)

            round_elapsed = time.time() - round_start

            # Evaluate periodically (every 5 rounds or final round)
            if (c + 1) == self.rounds or (c + 1) % 5 == 0:
                eval_metrics = self.evaluate_all_clients(use_test=False)
                mean_val = eval_metrics["mean_rmse"]
                slice_val = eval_metrics["slice_rmse"]
            else:
                mean_val = None
                slice_val = None

            round_info = {
                "round": c + 1,
                "selected_clients": len(selected_ids),
                "round_time_sec": round_elapsed,
                "mean_val_rmse": mean_val,
                "slice_val_rmse": slice_val,
            }
            self.history.append(round_info)

            if verbose and ((c + 1) % 5 == 0 or c == 0 or c == self.rounds - 1):
                val_str = f"{mean_val:.4f}" if mean_val is not None else "N/A"
                print(
                    f"Round {c+1:2d}/{self.rounds} | "
                    f"Selected: {len(selected_ids):3d} | "
                    f"Mean Val RMSE: {val_str} | "
                    f"Time: {round_elapsed:.2f}s"
                )

        total_time = time.time() - start_time
        final_test_metrics = self.evaluate_all_clients(use_test=True)

        return {
            "total_rounds": self.rounds,
            "total_time_seconds": total_time,
            "history": self.history,
            "final_test_rmse": final_test_metrics["mean_rmse"],
            "slice_test_rmse": final_test_metrics["slice_rmse"],
            "per_client_rmse": final_test_metrics["per_client_rmse"],
        }

    def evaluate_all_clients(self, use_test: bool = True) -> Dict[str, Any]:
        """Evaluates every client using its current local model parameters."""
        client_rmses = {}
        slice_rmses: Dict[str, List[float]] = {}

        for cid, client in self.clients.items():
            metrics = client.evaluate(use_test=use_test)
            client_rmses[cid] = metrics["rmse"]
            sname = client.slice_name
            if sname not in slice_rmses:
                slice_rmses[sname] = []
            slice_rmses[sname].append(metrics["rmse"])

        mean_rmse = float(np.mean(list(client_rmses.values()))) if client_rmses else 0.0
        slice_mean_rmse = {s: float(np.mean(vals)) for s, vals in slice_rmses.items()}

        return {
            "mean_rmse": mean_rmse,
            "slice_rmse": slice_mean_rmse,
            "per_client_rmse": client_rmses,
        }

