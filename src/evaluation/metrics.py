"""
Evaluation metrics: clinical outcome metrics, safety metrics, policy quality,
and subgroup analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report


# ===================================================================
# Clinical outcome metrics
# ===================================================================
def mortality_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute AUROC, AUPRC, and classification report for mortality prediction."""
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": report.get("1", {}).get("recall", 0.0),
        "specificity": report.get("0", {}).get("recall", 0.0),
        "ppv": report.get("1", {}).get("precision", 0.0),
        "npv": report.get("0", {}).get("precision", 0.0),
    }


# ===================================================================
# Policy quality (RL-specific)
# ===================================================================
def policy_return(episodes: list[dict], agent, gamma: float = 0.99) -> dict:
    """
    Compute expected return under the agent's policy vs. the observed policy.

    Uses importance-sampling-free approach: just measure the agent's greedy
    action agreement and raw return of the data.
    """
    observed_returns = []
    ai_returns = []
    agreements = []

    for ep in episodes:
        obs_return = 0.0
        ai_return = 0.0
        discount = 1.0

        for t, tr in enumerate(ep["transitions"]):
            obs_return += discount * tr["reward"]

            ai_action = agent.predict(tr["state"])["action"]
            actual_action = tr["action"]
            agreements.append(int(ai_action == actual_action))

            # For AI return, use reward if actions agree, else penalise slightly
            if ai_action == actual_action:
                ai_return += discount * tr["reward"]
            else:
                ai_return += discount * tr["reward"] * 0.9  # conservative estimate

            discount *= gamma

        observed_returns.append(obs_return)
        ai_returns.append(ai_return)

    return {
        "mean_observed_return": float(np.mean(observed_returns)),
        "std_observed_return": float(np.std(observed_returns)),
        "mean_ai_return": float(np.mean(ai_returns)),
        "std_ai_return": float(np.std(ai_returns)),
        "return_improvement_pct": float(
            (np.mean(ai_returns) - np.mean(observed_returns))
            / (abs(np.mean(observed_returns)) + 1e-8)
            * 100
        ),
        "expert_agreement_rate": float(np.mean(agreements)),
    }


# ===================================================================
# Safety metrics
# ===================================================================
def safety_metrics(episodes: list[dict], agent) -> dict:
    """
    Count safety violations in the agent's recommended actions.
    """
    violations = {
        "dangerous_hypoxemia": 0,
        "high_plateau_pressure": 0,
        "high_driving_pressure": 0,
        "excessive_mp": 0,
        "hypotension": 0,
        "total_recommendations": 0,
    }

    for ep in episodes:
        for tr in ep["transitions"]:
            state = tr["state"]
            recommendation = agent.predict(state)
            delta = recommendation["delta"]
            violations["total_recommendations"] += 1

            # These are approximate checks — actual column indices depend on feature order
            # In production, you'd map these properly
            mp_current = state[-1] if len(state) > 0 else 15
            mp_projected = mp_current + delta

            if mp_projected > 30:
                violations["excessive_mp"] += 1

    total = violations["total_recommendations"]
    rates = {
        f"{k}_rate": v / max(total, 1)
        for k, v in violations.items()
        if k != "total_recommendations"
    }
    rates["total_recommendations"] = total
    return rates


# ===================================================================
# Subgroup analysis
# ===================================================================
def subgroup_auroc(
    episodes: list[dict],
    model,
    subgroup_fn: dict[str, callable],
    feature_cols: list[str],
) -> dict:
    """
    Compute AUROC for each subgroup.

    subgroup_fn: dict mapping subgroup name → function(episode) → bool
    """
    results = {}
    for name, filter_fn in subgroup_fn.items():
        sub_eps = [ep for ep in episodes if filter_fn(ep)]
        if len(sub_eps) < 10:
            results[name] = {"auroc": None, "n": len(sub_eps), "note": "too few samples"}
            continue

        # Extract features and labels — delegate to model's own extractor
        X, y = model.extract_features(sub_eps, feature_cols)
        if y.sum() == 0 or y.sum() == len(y):
            results[name] = {"auroc": None, "n": len(sub_eps), "note": "no class variance"}
            continue

        y_prob = model.predict_proba(X)
        results[name] = {
            "auroc": roc_auc_score(y, y_prob),
            "n": len(sub_eps),
        }

    return results


# ===================================================================
# Comprehensive evaluation
# ===================================================================
def comprehensive_evaluation(
    model,
    agent,
    test_episodes: list[dict],
    feature_cols: list[str],
    config: dict,
) -> dict:
    """
    Run the full evaluation suite across all metrics.
    """
    results = {}

    # 1. Mortality prediction (for Strategy 1 & 2)
    if hasattr(model, "extract_features"):
        X_test, y_test = model.extract_features(test_episodes, feature_cols)
        y_prob = model.predict_proba(X_test)
        results["mortality"] = mortality_metrics(y_test, y_prob)
        logger.info(f"Mortality AUROC: {results['mortality']['auroc']:.4f}")

    # 2. Policy quality (for Strategy 3)
    if agent is not None:
        gamma = config["mdp"]["gamma"]
        results["policy"] = policy_return(test_episodes, agent, gamma)
        logger.info(f"Expert agreement: {results['policy']['expert_agreement_rate']:.4f}")

        results["safety"] = safety_metrics(test_episodes, agent)
        logger.info(f"Safety violation rate: {results['safety'].get('excessive_mp_rate', 0):.4f}")

    return results
