"""
Service Level Agreement (SLA) violation metrics for anticipatory RAN slicing.
Matches the formulation in Phyu et al. Section VI-H (Figure 13).
"""

from typing import Dict, List
import numpy as np


def compute_sla_violation_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    margin_gamma: float = 0.0,
) -> float:
    """
    Computes percentage of time intervals where actual traffic exceeds reserved capacity:
    Violation occurred if: y_true > (1 + margin_gamma) * y_pred.
    Returns percentage in [0.0, 100.0].
    """
    allocated = (1.0 + margin_gamma) * y_pred
    violations = y_true > allocated
    return float(np.mean(violations) * 100.0)


def evaluate_sla_across_margins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    margins: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
) -> Dict[str, float]:
    """Evaluates SLA violation rate across a sweep of over-provisioning margins gamma."""
    return {
        f"gamma_{g:.1f}": compute_sla_violation_rate(y_true, y_pred, margin_gamma=g)
        for g in margins
    }

