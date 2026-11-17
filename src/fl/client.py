"""
Local Agent (LA) implementation for Federated Learning in RAN slicing.
Adheres to Algorithm 2 of Phyu et al. (IEEE TNSM 2023) and standard FedProx.
"""

from typing import Dict, Any, Optional
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FLClient:
    """
    Represents a Local Agent (LA) at base station b for slice s.
    Maintains local dataset, private model instance, and local optimization.
    """

    def __init__(
        self,
        client_id: str,
        slice_name: str,
        base_station_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        lr: float = 0.001,
        local_epochs: int = 200,
        mu: float = 1.0,
        device: str = "cpu",
        variant: str = "true_fedprox",
    ):
        self.client_id = client_id
        self.slice_name = slice_name
        self.base_station_id = base_station_id
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = lr
        self.local_epochs = local_epochs
        self.mu = mu
        self.device = device
        self.variant = variant  # "true_fedprox" or "paper_alg2"
        self.num_samples = len(train_loader.dataset) if train_loader and hasattr(train_loader, "dataset") else 0

    def set_parameters(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """Loads latest global model weights into local model."""
        self.model.load_state_dict(global_weights)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Returns current local model weights."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def train_round(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Executes local training for communication round.
        Implements either:
          - 'true_fedprox': in-loop proximal regularization term (mu / 2) * ||w - w_global||^2
          - 'paper_alg2': standard SGD for E epochs, followed by post-hoc proximal adjustment
        """
        self.set_parameters(global_weights)
        self.model.train()
        self.model.to(self.device)

        # Snapshot of global weights for proximal distance
        global_params = {k: v.clone().to(self.device) for k, v in global_weights.items()}

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()  # Paper specifies RMSE/MSE loss

        epoch_losses = []

        if self.variant == "true_fedprox":
            for _ in range(self.local_epochs):
                batch_losses = []
                for x_b, y_b in self.train_loader:
                    x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                    optimizer.zero_grad()
                    pred = self.model(x_b)
                    loss = criterion(pred, y_b)
                    loss.backward()

                    # Exact proximal gradient: mu * (w - w_global) added directly to gradients
                    if self.mu > 0.0:
                        with torch.no_grad():
                            for name, param in self.model.named_parameters():
                                if name in global_params and param.grad is not None:
                                    param.grad.add_(param.data - global_params[name], alpha=self.mu)

                    optimizer.step()
                    batch_losses.append(loss.item())
                if batch_losses:
                    epoch_losses.append(sum(batch_losses) / len(batch_losses))

        elif self.variant == "paper_alg2":
            # Literal interpretation of Algorithm 2:
            # 1. Standard SGD for E epochs without proximal penalty
            for _ in range(self.local_epochs):
                batch_losses = []
                for x_b, y_b in self.train_loader:
                    x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                    optimizer.zero_grad()
                    pred = self.model(x_b)
                    loss = criterion(pred, y_b)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                if batch_losses:
                    epoch_losses.append(sum(batch_losses) / len(batch_losses))

            # 2. Line 9 post-hoc proximal regularization:
            # w^(c+1) = argmin LL(w*) + (mu/2) * ||w* - w^c||^2
            # Modeled as a proximal contraction step toward global model
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in global_params:
                        # Post-hoc proximal update pulling w* toward global weights
                        param.copy_((param + self.mu * global_params[name]) / (1.0 + self.mu))

        else:
            raise ValueError(f"Unknown client variant: {self.variant}")

        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "final_train_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "weights": self.get_parameters(),
        }

    def evaluate(self, use_test: bool = True) -> Dict[str, float]:
        """Evaluates local model on local validation or test set."""
        loader = self.test_loader if use_test else self.val_loader
        if loader is None:
            return {"loss": 0.0, "rmse": 0.0}

        self.model.eval()
        self.model.to(self.device)
        criterion = nn.MSELoss()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                pred = self.model(x_b)
                loss = criterion(pred, y_b)
                total_loss += loss.item() * len(y_b)
                total_samples += len(y_b)

        mse = total_loss / max(1, total_samples)
        rmse = mse ** 0.5
        return {"loss": mse, "rmse": rmse}
