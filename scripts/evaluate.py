#!/usr/bin/env python3
"""
Evaluate trained models and generate comparison report.

Usage:
    python scripts/evaluate.py --config config/config.yaml
"""

import argparse
import json
from pathlib import Path

from loguru import logger

from src.utils.helpers import load_config, set_seed, setup_logging, ensure_dirs
from src.data.preprocessing import preprocess_pipeline
from src.data.dataset import build_episodes, split_episodes
from src.features.engineering import get_state_feature_cols
from src.models.strategy1_static import Strategy1Static
from src.models.strategy2_window import Strategy2TimeWindow
from src.models.cql_agent import MPAdvisorCQL
from src.evaluation.metrics import comprehensive_evaluation
from src.evaluation.comparison import build_comparison_table, print_comparison, check_success_criteria
from src.evaluation.clinician_validation import generate_validation_cases, save_questionnaire


def main():
    parser = argparse.ArgumentParser(description="Evaluate MP Advisor models")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--generate-questionnaire", action="store_true", help="Generate clinician validation cases")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["paths"]["logs"])
    set_seed(config["project"]["random_seed"])
    ensure_dirs(config)

    models_dir = Path(config["paths"]["models"])
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = get_state_feature_cols(config)

    # ------------------------------------------------------------------
    # Load preprocessed data & rebuild test episodes
    # ------------------------------------------------------------------
    logger.info("Loading and preprocessing data for evaluation...")
    from scripts.train import load_raw_tables

    raw_tables = load_raw_tables(config)
    processed_df = preprocess_pipeline(raw_tables, config)
    episodes = build_episodes(processed_df, config, feature_cols)
    _, _, test_eps = split_episodes(episodes, config)
    logger.info(f"Test set: {len(test_eps)} episodes")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    all_results = {}

    # Strategy 1
    s1_path = models_dir / "strategy1_xgboost.pkl"
    if s1_path.exists():
        logger.info("Evaluating Strategy 1 (Static)...")
        s1 = Strategy1Static(config)
        s1.load(s1_path)
        all_results["Strategy 1 (Static)"] = comprehensive_evaluation(
            model=s1, agent=None, test_episodes=test_eps, feature_cols=feature_cols, config=config
        )

    # Strategy 2
    s2_path = models_dir / "strategy2_xgboost.pkl"
    if s2_path.exists():
        logger.info("Evaluating Strategy 2 (Time-Window)...")
        s2 = Strategy2TimeWindow(config)
        s2.load(s2_path)
        all_results["Strategy 2 (Time-Window)"] = comprehensive_evaluation(
            model=s2, agent=None, test_episodes=test_eps, feature_cols=feature_cols, config=config
        )

    # Strategy 3
    s3_path = models_dir / "strategy3_cql"
    if s3_path.exists():
        logger.info("Evaluating Strategy 3 (CQL)...")
        agent = MPAdvisorCQL(config)
        agent.load(s3_path)
        all_results["Strategy 3 (CQL)"] = comprehensive_evaluation(
            model=None, agent=agent, test_episodes=test_eps, feature_cols=feature_cols, config=config
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    if all_results:
        print_comparison(all_results)

        # Save comparison table
        df = build_comparison_table(all_results)
        df.to_csv(results_dir / "comparison.csv")

        # Check success criteria
        tiers = check_success_criteria(all_results, config)
        logger.info("Success criteria check:")
        for tier, info in tiers.items():
            status = "MET" if info["met"] else "NOT MET"
            logger.info(f"  {tier}: {status}")

        # Save full results
        with open(results_dir / "evaluation_results.json", "w") as f:
            # Convert numpy types for JSON serialisation
            json.dump(all_results, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Clinician questionnaire (optional)
    # ------------------------------------------------------------------
    if args.generate_questionnaire and s3_path.exists():
        logger.info("Generating clinician validation questionnaire...")
        cases = generate_validation_cases(
            test_episodes=test_eps,
            agent=agent,
            feature_names=feature_cols,
            n_cases=config["evaluation"]["clinician_validation"]["n_cases"],
        )
        save_questionnaire(cases, results_dir / "clinician_questionnaire.json")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
