"""
Strategy 1 — Static Risk Prediction (Baseline).

XGBoost model: patient demographics + MP at t=0 & t=24 → Mortality risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class Strategy1Static:
    """
    Baseline model that predicts ICU mortality from a single-timepoint
    snapshot: demographics, severity scores, and MP at admission (t=0)
    and 24 hours later.
    """

    def __init__(self, config: dict):
        if not XGB_AVAILABLE:
            raise RuntimeError("xgboost is required. pip install xgboost")

        xgb_cfg = config["training"]["xgboost"]
        self.model = xgb.XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        self.early_stopping = xgb_cfg["early_stopping_rounds"]
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def extract_features(episodes: list[dict], feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        From each episode, extract:
          - static features from the first timestep
          - MP at t=0
          - MP at t=24 (or last available if shorter)

        Returns X (N, n_features+2), y (N,)
        """
        X_list, y_list = [], []

        for ep in episodes:
            transitions = ep["transitions"]
            if len(transitions) == 0:
                continue

            state_0 = transitions[0]["state"]

            # MP at t=0 (assume last feature in state is MP — adjust index as needed)
            mp_0 = state_0[-1] if len(state_0) > 0 else 0.0

            # MP at t=24 or last step
            t24 = min(24, len(transitions) - 1)
            state_24 = transitions[t24]["state"]
            mp_24 = state_24[-1] if len(state_24) > 0 else mp_0

            features = np.concatenate([state_0, [mp_0, mp_24]])
            label = 1 if ep["outcome"] == "died" else 0

            X_list.append(features)
            y_list.append(label)

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        train_episodes: list[dict],
        val_episodes: list[dict],
        feature_cols: list[str],
    ) -> dict:
        X_train, y_train = self.extract_features(train_episodes, feature_cols)
        X_val, y_val = self.extract_features(val_episodes, feature_cols)

        logger.info(f"Strategy 1 — Training on {len(X_train)} patients, validating on {len(X_val)}")

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # Validation metrics
        y_prob = self.model.predict_proba(X_val)[:, 1]
        auroc = roc_auc_score(y_val, y_prob)
        auprc = average_precision_score(y_val, y_prob)

        logger.info(f"Strategy 1 — Val AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
        return {"auroc": auroc, "auprc": auprc}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Strategy 1 model saved to {path}")

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
        logger.info(f"Strategy 1 model loaded from {path}")
