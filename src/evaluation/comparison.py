"""
Cross-strategy comparison: side-by-side performance table and visualisations.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from per-strategy result dicts.

    Parameters
    ----------
    results : e.g. {
        "Strategy 1 (Static)": {"mortality": {...}, ...},
        "Strategy 2 (Windows)": {"mortality": {...}, ...},
        "Strategy 3 (RL)": {"mortality": {...}, "policy": {...}, "safety": {...}},
    }
    """
    rows = []
    for strategy_name, res in results.items():
        row = {"Strategy": strategy_name}

        # Mortality metrics
        mort = res.get("mortality", {})
        row["AUROC"] = mort.get("auroc")
        row["AUPRC"] = mort.get("auprc")
        row["Sensitivity"] = mort.get("sensitivity")
        row["Specificity"] = mort.get("specificity")

        # Policy metrics (RL only)
        pol = res.get("policy", {})
        row["Mean Return"] = pol.get("mean_ai_return")
        row["Return Improvement %"] = pol.get("return_improvement_pct")
        row["Expert Agreement"] = pol.get("expert_agreement_rate")

        # Safety
        safe = res.get("safety", {})
        row["Safety Violation Rate"] = safe.get("excessive_mp_rate")

        rows.append(row)

    df = pd.DataFrame(rows).set_index("Strategy")
    return df


def print_comparison(results: dict[str, dict]) -> None:
    """Pretty-print the comparison table."""
    df = build_comparison_table(results)
    logger.info("Strategy Comparison:\n" + df.to_string())


def check_success_criteria(results: dict, config: dict) -> dict:
    """
    Check which success tier the best strategy achieves.

    Returns a dict with tier name and whether each criterion is met.
    """
    thresholds = config["evaluation"]["thresholds"]
    best_auroc = 0.0
    best_agreement = 0.0

    for res in results.values():
        auroc = res.get("mortality", {}).get("auroc", 0)
        agreement = res.get("policy", {}).get("expert_agreement_rate", 0)
        best_auroc = max(best_auroc, auroc)
        best_agreement = max(best_agreement, agreement)

    tiers = {}
    for tier_name, criteria in thresholds.items():
        met = True
        details = {}
        if "auroc" in criteria:
            passed = best_auroc >= criteria["auroc"]
            details["auroc"] = {"required": criteria["auroc"], "achieved": best_auroc, "passed": passed}
            met = met and passed
        if "expert_agreement" in criteria:
            passed = best_agreement >= criteria["expert_agreement"]
            details["expert_agreement"] = {
                "required": criteria["expert_agreement"],
                "achieved": best_agreement,
                "passed": passed,
            }
            met = met and passed
        tiers[tier_name] = {"met": met, "details": details}

    return tiers
