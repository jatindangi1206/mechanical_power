"""
Strategy 3: Conservative Q-Learning (CQL) agent for offline RL.

Wraps d3rlpy's CQL implementation and adds clinical-specific helpers
for training, prediction, and confidence estimation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

try:
    import d3rlpy
    from d3rlpy.algos import CQLConfig
    from d3rlpy.dataset import MDPDataset

    D3RLPY_AVAILABLE = True
except ImportError:
    D3RLPY_AVAILABLE = False
    logger.warning("d3rlpy not installed — CQL agent will not be available.")


class MPAdvisorCQL:
    """
    Main RL agent for MP personalisation using Conservative Q-Learning.

    Designed for offline (batch) RL: learns exclusively from historical
    ICU data without online interaction.
    """

    ACTION_NAMES = {
        0: "large_decrease",
        1: "small_decrease",
        2: "maintain",
        3: "small_increase",
        4: "large_increase",
    }
    ACTION_DELTAS = {0: -5, 1: -2, 2: 0, 3: +2, 4: +5}

    def __init__(self, config: dict):
        if not D3RLPY_AVAILABLE:
            raise RuntimeError("d3rlpy is required for Strategy 3. pip install d3rlpy")

        cql_cfg = config["training"]["cql"]

        self.cql_config = CQLConfig(
            actor_learning_rate=cql_cfg["actor_learning_rate"],
            critic_learning_rate=cql_cfg["critic_learning_rate"],
            alpha=cql_cfg["alpha"],
            conservative_weight=cql_cfg["conservative_weight"],
            n_critics=cql_cfg["n_critics"],
            tau=cql_cfg["tau"],
        )
        self.n_epochs = cql_cfg["n_epochs"]
        self.batch_size = cql_cfg["batch_size"]
        self.use_gpu = cql_cfg.get("use_gpu", False)
        self.model = None
        self.config = config

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------
    @staticmethod
    def build_mdp_dataset(
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
    ) -> MDPDataset:
        """
        Wrap flat arrays into a d3rlpy MDPDataset.

        Parameters
        ----------
        observations : (N, state_dim) float32
        actions      : (N,) int64
        rewards      : (N,) float32
        terminals    : (N,) float32 — 1.0 at episode boundaries
        """
        dataset = MDPDataset(
            observations=observations,
            actions=actions.reshape(-1, 1) if actions.ndim == 1 else actions,
            rewards=rewards,
            terminals=terminals,
        )
        return dataset

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        train_dataset: MDPDataset,
        eval_dataset: Optional[MDPDataset] = None,
    ) -> dict:
        """
        Train the CQL agent on an offline MDP dataset.

        Returns a dict of training metrics.
        """
        logger.info(
            f"Training CQL for {self.n_epochs} epochs "
            f"(batch_size={self.batch_size}, gpu={self.use_gpu})"
        )

        self.model = self.cql_config.create(device="cuda:0" if self.use_gpu else "cpu:0")

        self.model.fit(
            train_dataset,
            n_steps_per_epoch=len(train_dataset) // self.batch_size,
            n_steps=self.n_epochs * (len(train_dataset) // self.batch_size),
            evaluators=(
                {"val_td_error": d3rlpy.metrics.TDErrorEvaluator(episodes=eval_dataset.episodes)}
                if eval_dataset
                else None
            ),
            experiment_name="mp_cql",
        )

        logger.info("CQL training complete.")
        return {"status": "trained", "n_epochs": self.n_epochs}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, state: np.ndarray) -> dict:
        """
        Get the recommended action for a single patient state.

        Parameters
        ----------
        state : (state_dim,) or (1, state_dim) float32

        Returns
        -------
        dict with keys: action (int), action_name (str), delta (float),
                        confidence (float), q_values (array)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        state = np.atleast_2d(state).astype(np.float32)
        action = int(self.model.predict(state)[0])

        # Get Q-values for all actions for confidence estimation
        q_values = np.array([
            float(self.model.predict_value(state, np.array([[a]])))
            for a in range(len(self.ACTION_NAMES))
        ])

        confidence = self._estimate_confidence(q_values, action)

        return {
            "action": action,
            "action_name": self.ACTION_NAMES[action],
            "delta": self.ACTION_DELTAS[action],
            "confidence": confidence,
            "q_values": q_values,
        }

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """Predict actions for a batch of states. Returns (N,) int array."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.predict(states.astype(np.float32))

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_confidence(q_values: np.ndarray, chosen_action: int) -> float:
        """
        Estimate confidence in the recommendation.

        Uses softmax over Q-values — the higher the probability mass on the
        chosen action, the more confident the agent is.
        """
        # Softmax with temperature
        temperature = 5.0
        exp_q = np.exp((q_values - q_values.max()) / temperature)
        probs = exp_q / exp_q.sum()
        return float(probs[chosen_action])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save(str(path / "cql_model.d3"))
        logger.info(f"CQL model saved to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        model_file = path / "cql_model.d3"
        if not model_file.exists():
            raise FileNotFoundError(f"No model found at {model_file}")
        self.model = self.cql_config.create()
        self.model.load_model(str(model_file))
        logger.info(f"CQL model loaded from {model_file}")
