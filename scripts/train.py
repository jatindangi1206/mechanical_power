#!/usr/bin/env python3
"""
Train all three strategies (static baseline, time-window, CQL agent).

Usage:
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --config config/config.yaml --strategy 3
"""

import argparse
from pathlib import Path

from loguru import logger

from src.utils.helpers import load_config, set_seed, setup_logging, ensure_dirs
from src.data.preprocessing import preprocess_pipeline
from src.data.dataset import build_episodes, split_episodes, episodes_to_arrays
from src.features.engineering import get_state_feature_cols
from src.models.strategy1_static import Strategy1Static
from src.models.strategy2_window import Strategy2TimeWindow
from src.models.cql_agent import MPAdvisorCQL


def load_raw_tables(config: dict) -> dict:
    """Load parquet files saved by extract_data.py."""
    import pandas as pd

    raw_dir = Path(config["paths"]["raw_data"])
    tables = {}
    for name in ["episodes", "ventilator", "vitals", "labs", "demographics", "outcomes"]:
        path = raw_dir / f"{name}.parquet"
        if path.exists():
            tables[name] = pd.read_parquet(path)
            logger.info(f"Loaded {name}: {len(tables[name])} rows")
        else:
            logger.warning(f"Missing {path}")
    return tables


def main():
    parser = argparse.ArgumentParser(description="Train MP Advisor models")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--strategy", type=int, default=0, help="Train a specific strategy (1/2/3), or 0 for all")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["paths"]["logs"])
    set_seed(config["project"]["random_seed"])
    ensure_dirs(config)

    models_dir = Path(config["paths"]["models"])
    feature_cols = get_state_feature_cols(config)

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    logger.info("Loading raw data...")
    raw_tables = load_raw_tables(config)

    logger.info("Running preprocessing pipeline...")
    processed_df = preprocess_pipeline(raw_tables, config)

    logger.info("Building MDP episodes...")
    episodes = build_episodes(processed_df, config, feature_cols)

    logger.info("Splitting into train / val / test...")
    train_eps, val_eps, test_eps = split_episodes(episodes, config)

    # ------------------------------------------------------------------
    # Strategy 1
    # ------------------------------------------------------------------
    if args.strategy in (0, 1):
        logger.info("=" * 60)
        logger.info("STRATEGY 1 — Static XGBoost Baseline")
        logger.info("=" * 60)
        s1 = Strategy1Static(config)
        s1_metrics = s1.train(train_eps, val_eps, feature_cols)
        s1.save(models_dir / "strategy1_xgboost.pkl")
        logger.info(f"Strategy 1 results: {s1_metrics}")

    # ------------------------------------------------------------------
    # Strategy 2
    # ------------------------------------------------------------------
    if args.strategy in (0, 2):
        logger.info("=" * 60)
        logger.info("STRATEGY 2 — Time-Window XGBoost")
        logger.info("=" * 60)
        s2 = Strategy2TimeWindow(config)
        s2_metrics = s2.train(train_eps, val_eps, feature_cols)
        s2.save(models_dir / "strategy2_xgboost.pkl")
        logger.info(f"Strategy 2 results: {s2_metrics}")

    # ------------------------------------------------------------------
    # Strategy 3
    # ------------------------------------------------------------------
    if args.strategy in (0, 3):
        logger.info("=" * 60)
        logger.info("STRATEGY 3 — Conservative Q-Learning (CQL)")
        logger.info("=" * 60)
        obs_train, act_train, rew_train, _, term_train = episodes_to_arrays(train_eps)
        obs_val, act_val, rew_val, _, term_val = episodes_to_arrays(val_eps)

        agent = MPAdvisorCQL(config)
        train_ds = agent.build_mdp_dataset(obs_train, act_train, rew_train, term_train)
        val_ds = agent.build_mdp_dataset(obs_val, act_val, rew_val, term_val)

        agent.train(train_ds, eval_dataset=val_ds)
        agent.save(models_dir / "strategy3_cql")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
