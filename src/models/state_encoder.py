"""
State Encoder: combines static patient features with temporal trajectories
using an LSTM to produce a fixed-size latent state representation.
"""

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    Encodes patient state (static features + time-series trajectory)
    into a fixed-size latent vector.

    Architecture:
        static features  → MLP        → 32-dim
        temporal sequence → LSTM       → 64-dim
        concat           → fusion MLP → latent_dim
    """

    def __init__(
        self,
        static_dim: int = 15,
        temporal_dim: int = 20,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # --- Static features encoder ---
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
        )

        # --- Time-series encoder ---
        self.lstm = nn.LSTM(
            input_size=temporal_dim,
            hidden_size=64,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # --- Fusion ---
        self.fusion = nn.Sequential(
            nn.Linear(32 + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, static_features: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        static_features : (batch, static_dim)
        trajectory      : (batch, seq_len, temporal_dim)

        Returns
        -------
        latent : (batch, latent_dim)
        """
        # Encode static
        static_enc = self.static_encoder(static_features)  # (B, 32)

        # Encode trajectory — use last hidden state
        _, (h_n, _) = self.lstm(trajectory)  # h_n: (layers, B, 64)
        temporal_enc = h_n[-1]  # (B, 64)

        # Fuse
        combined = torch.cat([static_enc, temporal_enc], dim=1)  # (B, 96)
        latent = self.fusion(combined)  # (B, latent_dim)
        return latent


class FlatStateEncoder(nn.Module):
    """
    Simpler encoder for flat (non-sequential) state vectors.
    Used when the full trajectory is already collapsed into a single vector.
    """

    def __init__(self, state_dim: int = 50, latent_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
