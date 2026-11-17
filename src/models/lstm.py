"""
LSTM Model Architecture for Mobile Traffic Forecasting.
Implements the architecture specified in Table III of Phyu et al. (IEEE TNSM 2023):
LSTM with 64 hidden units followed by fully-connected readout layers.
"""

from typing import Optional
import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):
    """
    1D LSTM network for cellular traffic prediction.
    Takes historical observation window of shape (batch, H, input_dim)
    and forecasts future window of shape (batch, W).
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 1,
        output_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection head: 64 hidden units -> output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: tensor of shape (batch_size, seq_len, input_dim)
        Returns: tensor of shape (batch_size, output_dim)
        """
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # Take the last time-step hidden representation
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

