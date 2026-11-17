"""
Loss functions for RAN traffic forecasting.
Includes standard RMSE/MSE and SLA-aware asymmetric quantile (pinball) loss.
"""

import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(pred, target) + self.eps)


class SLAPinballLoss(nn.Module):
    """
    Asymmetric Pinball / Quantile Loss for SLA-aware capacity allocation.
    Penalizes under-provisioning (y > y_hat) with weight tau,
    and over-provisioning (y_hat > y) with weight (1 - tau).
    When tau > 0.5 (e.g. tau = 0.8), model learns an anticipatory safety margin.
    """

    def __init__(self, tau: float = 0.8):
        super().__init__()
        if not (0.0 < tau < 1.0):
            raise ValueError(f"Quantile parameter tau must be in (0, 1), got {tau}")
        self.tau = tau

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = target - pred
        loss = torch.max(self.tau * diff, (self.tau - 1.0) * diff)
        return torch.mean(loss)

