"""
Centralized LSTM Baseline.
Trains a centralized LSTM model across all base stations and slices simultaneously,
representing the non-private cloud centralized benchmark.
"""

from typing import Dict, Any, List, Optional
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from src.models.lstm import TrafficLSTM


class CentralizedLSTMTrainer:
    """
    Centralized model trainer pooling all available client training datasets.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loaders: List[DataLoader],
        val_loaders: Optional[List[DataLoader]] = None,
        test_loaders: Optional[List[DataLoader]] = None,
        lr: float = 0.001,
        epochs: int = 50,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.model = copy.deepcopy(model).to(device)
        self.device = device
        self.lr = lr
        self.epochs = epochs

        # Pool datasets
        train_datasets = [dl.dataset for dl in train_loaders if dl is not None]
        self.combined_train_loader = DataLoader(
            ConcatDataset(train_datasets),
            batch_size=batch_size,
            shuffle=True,
        )

        self.val_loaders = val_loaders or []
        self.test_loaders = test_loaders or []

    def train(self, verbose: bool = False) -> Dict[str, Any]:
        """Trains centralized model over specified epochs."""
        self.model.train()
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        history = []
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = 0
            for x_b, y_b in self.combined_train_loader:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                pred = self.model(x_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            avg_loss = epoch_loss / max(1, batches)
            history.append({"epoch": epoch + 1, "train_loss": avg_loss})
            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"Centralized Epoch {epoch+1:2d}/{self.epochs} | Loss: {avg_loss:.5f}")

        elapsed = time.time() - start_time
        return {"training_time": elapsed, "loss_history": history}

    def evaluate(self, loaders: Optional[List[DataLoader]] = None) -> Dict[str, float]:
        """Evaluates model across specified loaders (validation or test)."""
        target_loaders = loaders if loaders is not None else self.test_loaders
        self.model.eval()
        self.model.to(self.device)
        criterion = nn.MSELoss()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for dl in target_loaders:
                for x_b, y_b in dl:
                    x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                    pred = self.model(x_b)
                    loss = criterion(pred, y_b)
                    total_loss += loss.item() * len(y_b)
                    total_samples += len(y_b)

        mse = total_loss / max(1, total_samples)
        rmse = mse ** 0.5
        return {"mse": mse, "rmse": rmse}

