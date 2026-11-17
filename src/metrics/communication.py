"""
Communication cost modeling for Federated Learning and Centralized Baselines.
Implements the model in Phyu et al. Section VI-C (Figure 9).
"""

from typing import Dict


def compute_fl_communication_cost(
    rounds: int,
    num_clients: int,
    selection_fraction: float,
    model_params_count: int,
    precision_bits: int = 32,
) -> Dict[str, float]:
    """
    Computes communication cost per round and total over C rounds:
    Cost = 2 * C * N * Omega * psi (Equation in Section VI-C)
    where psi = model_params_count * precision_bits
    """
    psi_bits = model_params_count * precision_bits
    psi_bytes = psi_bits / 8.0
    psi_kb = psi_bytes / 1024.0

    bytes_per_round = 2.0 * num_clients * selection_fraction * psi_bytes
    total_bytes = rounds * bytes_per_round
    total_mb = total_bytes / (1024.0 * 1024.0)

    return {
        "psi_bytes": psi_bytes,
        "psi_kb": psi_kb,
        "bytes_per_round": bytes_per_round,
        "total_megabytes": total_mb,
    }


def compute_centralized_communication_cost(
    num_clients: int,
    total_samples: int,
    dataset_fraction: float = 1.0,
    precision_bits: int = 32,
) -> Dict[str, float]:
    """
    Computes raw dataset communication cost for centralized baseline:
    Cost = N * (total_samples * dataset_fraction) * precision_bits
    """
    transmitted_samples = int(total_samples * dataset_fraction)
    bytes_per_client = transmitted_samples * (precision_bits / 8.0)
    total_bytes = num_clients * bytes_per_client
    total_mb = total_bytes / (1024.0 * 1024.0)

    return {
        "transmitted_samples_per_client": transmitted_samples,
        "total_megabytes": total_mb,
    }

