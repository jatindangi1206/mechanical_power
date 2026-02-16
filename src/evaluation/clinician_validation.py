"""
Clinician Validation: generate case questionnaires and analyse survey responses.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


def generate_patient_summary(state: np.ndarray, feature_names: list[str]) -> str:
    """Create a human-readable clinical summary from a state vector."""
    lines = []
    for name, val in zip(feature_names, state):
        if np.isnan(val):
            continue
        lines.append(f"  {name}: {val:.1f}")
    return "\n".join(lines)


def action_to_text(action: int) -> str:
    """Convert action index to clinician-friendly description."""
    descriptions = {
        0: "Decrease Mechanical Power by ~5 J/min (large reduction)",
        1: "Decrease Mechanical Power by ~2 J/min (small reduction)",
        2: "Maintain current Mechanical Power settings",
        3: "Increase Mechanical Power by ~2 J/min (small increase)",
        4: "Increase Mechanical Power by ~5 J/min (large increase)",
    }
    return descriptions.get(action, f"Unknown action {action}")


def generate_validation_cases(
    test_episodes: list[dict],
    agent,
    feature_names: list[str],
    n_cases: int = 100,
    seed: int = 42,
) -> list[dict]:
    """
    Sample cases from test episodes and generate questionnaire items.

    Each case includes the patient summary, AI recommendation, and survey questions.
    """
    random.seed(seed)
    sampled = random.sample(test_episodes, min(n_cases, len(test_episodes)))

    cases = []
    for i, ep in enumerate(sampled):
        # Pick a representative time-point (midpoint of the episode)
        mid = len(ep["transitions"]) // 2
        tr = ep["transitions"][mid]
        state = tr["state"]

        recommendation = agent.predict(state)

        case = {
            "case_id": i + 1,
            "stay_id": ep["stay_id"],
            "patient_summary": generate_patient_summary(state, feature_names),
            "ai_recommendation": action_to_text(recommendation["action"]),
            "ai_confidence": round(recommendation["confidence"], 2),
            "actual_clinician_action": action_to_text(tr["action"]),
            "actual_outcome": ep["outcome"],
            "questions": [
                {
                    "id": "q1_follow",
                    "text": "Would you follow this AI recommendation?",
                    "type": "yes_no",
                },
                {
                    "id": "q2_safety",
                    "text": "Rate the safety of this recommendation (1=dangerous, 5=very safe)",
                    "type": "likert_1_5",
                },
                {
                    "id": "q3_reasoning",
                    "text": "Rate the clinical reasoning quality (1=poor, 5=excellent)",
                    "type": "likert_1_5",
                },
                {
                    "id": "q4_alternative",
                    "text": "What would YOU do differently? (free text)",
                    "type": "free_text",
                },
            ],
        }
        cases.append(case)

    return cases


def save_questionnaire(cases: list[dict], output_path: str | Path) -> None:
    """Save the questionnaire as JSON for distribution to clinicians."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cases, f, indent=2)
    logger.info(f"Questionnaire saved: {output_path} ({len(cases)} cases)")


def analyse_responses(responses_path: str | Path) -> dict:
    """
    Analyse collected clinician responses.

    Expects a JSON file where each item has the case_id and filled answers.
    """
    with open(responses_path) as f:
        responses = json.load(f)

    df = pd.DataFrame(responses)

    analysis = {
        "n_responses": len(df),
        "agreement_rate": df["q1_follow"].map({"yes": 1, "no": 0}).mean()
        if "q1_follow" in df.columns
        else None,
        "mean_safety_rating": df["q2_safety"].mean() if "q2_safety" in df.columns else None,
        "mean_reasoning_rating": df["q3_reasoning"].mean() if "q3_reasoning" in df.columns else None,
    }

    logger.info(f"Validation analysis: {analysis}")
    return analysis
