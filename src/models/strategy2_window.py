"""
Strategy 2 — Time-Window Analysis.

XGBoost model: patient demographics + MP snapshots at 0h, 6h, 12h, 18h, 24h
→ Mortality risk.

Captures some temporal dynamics by including multiple MP measurements.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import joblib
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


SNAPSHOT_HOURS = [0, 6, 12, 18, 24]


class Strategy2TimeWindow:
    """
    Improved baseline that includes MP at five time-points (0, 6, 12, 18, 24h)
    plus derived delta features (rate of MP change between windows).
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

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def extract_features(episodes: list[dict], feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        From each episode, extract:
          - static features from t=0
          - MP values at t=0, 6, 12, 18, 24
          - MP deltas between consecutive windows
          - SpO2 at same snapshots
        """
        X_list, y_list = [], []

        for ep in episodes:
            transitions = ep["transitions"]
            if len(transitions) < 2:
                continue

            state_0 = transitions[0]["state"]
            base_features = state_0.copy()

            # Collect MP (last col) and SpO2 at snapshot hours
            mp_snapshots = []
            spo2_snapshots = []
            for h in SNAPSHOT_HOURS:
                idx = min(h, len(transitions) - 1)
                st = transitions[idx]["state"]
                # Assume MP is last feature, SpO2 is feature at a known index
                mp_snapshots.append(st[-1] if len(st) > 0 else 0.0)
                # SpO2 — approximate position (adjust for your actual feature order)
                spo2_idx = min(len(st) - 1, 3)  # placeholder index
                spo2_snapshots.append(st[spo2_idx] if len(st) > spo2_idx else 0.0)

            # MP deltas between consecutive windows
            mp_deltas = [
                mp_snapshots[i + 1] - mp_snapshots[i] for i in range(len(mp_snapshots) - 1)
            ]

            # Summary statistics
            mp_mean = np.mean(mp_snapshots)
            mp_std = np.std(mp_snapshots)
            mp_trend = mp_snapshots[-1] - mp_snapshots[0]

            extra = np.array(
                mp_snapshots + spo2_snapshots + mp_deltas + [mp_mean, mp_std, mp_trend],
                dtype=np.float32,
            )
            features = np.concatenate([base_features, extra])

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

        logger.info(f"Strategy 2 — Training on {len(X_train)} patients, validating on {len(X_val)}")

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        y_prob = self.model.predict_proba(X_val)[:, 1]
        auroc = roc_auc_score(y_val, y_prob)
        auprc = average_precision_score(y_val, y_prob)

        logger.info(f"Strategy 2 — Val AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
        return {"auroc": auroc, "auprc": auprc}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    def get_feature_importance(self, top_n: int = 20) -> dict:
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        return {f"feature_{i}": float(importance[i]) for i in indices}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Strategy 2 model saved to {path}")

    def load(self, path: str | Path) -> None:
        self.model = joblib.load(path)
