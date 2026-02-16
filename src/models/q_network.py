"""
Q-Network: estimates Q(s, a) — the expected return from taking action a in state s.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.state_encoder import FlatStateEncoder


class QNetwork(nn.Module):
    """
    Dueling-style Q-network that maps (state, action) → scalar Q-value.

    Can be used standalone or plugged into the CQL agent.
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.action_dim = action_dim

        self.state_encoder = FlatStateEncoder(state_dim=state_dim, latent_dim=64)

        # Q-value head: takes encoded state + one-hot action
        self.q_head = nn.Sequential(
            nn.Linear(64 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state  : (batch, state_dim)
        action : (batch,) — integer action indices

        Returns
        -------
        q_value : (batch, 1)
        """
        state_enc = self.state_encoder(state)
        action_onehot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        sa = torch.cat([state_enc, action_onehot], dim=1)
        return self.q_head(sa)

    def q_values_all_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all possible actions given a batch of states.

        Returns
        -------
        q_values : (batch, action_dim)
        """
        state_enc = self.state_encoder(state)
        batch_size = state_enc.size(0)

        q_vals = []
        for a in range(self.action_dim):
            action_onehot = torch.zeros(batch_size, self.action_dim, device=state.device)
            action_onehot[:, a] = 1.0
            sa = torch.cat([state_enc, action_onehot], dim=1)
            q_vals.append(self.q_head(sa))

        return torch.cat(q_vals, dim=1)  # (batch, action_dim)


class TwinQNetwork(nn.Module):
    """
    Twin (double) Q-networks for reducing overestimation bias.
    Used by CQL with n_critics=2.
    """

    def __init__(self, state_dim: int = 64, action_dim: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.q1(state, action), self.q2(state, action)

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
