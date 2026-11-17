"""
Global Agent (GA) implementation for Federated Learning in RAN slicing.
Adheres to Algorithm 1 of Phyu et al. (IEEE TNSM 2023) with configurable aggregation.
"""

from typing import List, Dict, Any, Optional
import copy
import numpy as np
import torch
import torch.nn as nn


class FLServer:
    """
    Represents the Global Agent (GA) hosted by the Infrastructure Provider (InP).
    Coordinates communication rounds, selects participating local agents,
    and aggregates local model parameters.
    """

    def __init__(
        self,
        global_model: nn.Module,
        aggregation: str = "unweighted",  # "unweighted" (Alg 1 Line 7) or "weighted" (Eq. 1)
        device: str = "cpu",
    ):
        self.model = copy.deepcopy(global_model).to(device)
        self.aggregation = aggregation
        self.device = device
        self.round_history: List[Dict[str, Any]] = []

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Returns deep copy of current global parameters."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_parameters(self, weights: Dict[str, torch.Tensor]) -> None:
        """Updates global model with new weights."""
        self.model.load_state_dict(weights)

    def select_clients(
        self,
        client_ids: List[str],
        fraction: float = 1.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[str]:
        """
        Randomly selects a fraction Omega of local agents for round c (Alg 1, Line 3).
        """
        if fraction >= 1.0:
            return list(client_ids)
        num_selected = max(1, int(round(len(client_ids) * fraction)))
        if rng is not None:
            chosen = rng.choice(client_ids, size=num_selected, replace=False).tolist()
        else:
            chosen = np.random.choice(client_ids, size=num_selected, replace=False).tolist()
        return chosen

    def aggregate(
        self,
        client_updates: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates uploaded local model weights into new global model weights.
        - 'unweighted': Paper Alg 1 Line 7: w^(c+1) = (1 / |N^c|) * sum(w_i)
        - 'weighted': Paper Eq. (1) & FedAvg standard: w^(c+1) = sum((a_i / a) * w_i)
        """
        if not client_updates:
            return self.get_parameters()

        first_weights = client_updates[0]["weights"]
        new_weights = {}

        if self.aggregation == "unweighted":
            # Paper Algorithm 1 Line 7
            num_clients = len(client_updates)
            for k in first_weights.keys():
                stacked = torch.stack([u["weights"][k].float() for u in client_updates], dim=0)
                new_weights[k] = torch.mean(stacked, dim=0)

        elif self.aggregation == "weighted":
            # Paper Eq. (1) sample-weighted aggregation
            total_samples = sum(u.get("num_samples", 1) for u in client_updates)
            total_samples = max(1, total_samples)
            for k in first_weights.keys():
                weighted_sum = torch.zeros_like(first_weights[k], dtype=torch.float32)
                for u in client_updates:
                    weight_factor = u.get("num_samples", 1) / total_samples
                    weighted_sum += weight_factor * u["weights"][k].float()
                new_weights[k] = weighted_sum
        else:
            raise ValueError(f"Unknown aggregation scheme: {self.aggregation}")

        self.set_parameters(new_weights)
        return new_weights

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def compute_communication_cost(
        self,
        rounds: int,
        num_clients: int,
        participation_fraction: float,
        precision_bits: int = 32,
    ) -> float:
        """
        Computes federated communication cost per Phyu et al. Section VI-C:
        Cost = 2 * C * N * Omega * psi
        psi = total_trainable_parameters * precision_bits (in bits or converted to bytes/MB)
        Returns cost in Megabytes (MB).
        """
        param_count = self.count_parameters()
        psi_bits = param_count * precision_bits
        psi_bytes = psi_bits / 8.0
        total_bytes = 2.0 * rounds * num_clients * participation_fraction * psi_bytes
        return total_bytes / (1024.0 * 1024.0)  # in MB

