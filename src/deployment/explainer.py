"""
Explanation Generator: produce human-readable explanations for MP recommendations.
"""

from __future__ import annotations

import numpy as np


ACTION_DESCRIPTIONS = {
    0: "a significant reduction in Mechanical Power (~5 J/min decrease)",
    1: "a modest reduction in Mechanical Power (~2 J/min decrease)",
    2: "maintaining current Mechanical Power settings",
    3: "a modest increase in Mechanical Power (~2 J/min increase)",
    4: "a significant increase in Mechanical Power (~5 J/min increase)",
}


class ExplanationGenerator:
    """Generate clinical explanations for AI recommendations."""

    def generate(
        self,
        patient_state: dict,
        action: int,
        confidence: float,
        q_values: np.ndarray | None = None,
        similar_outcomes: dict | None = None,
    ) -> str:
        """
        Produce a multi-part explanation string.

        Parameters
        ----------
        patient_state : current patient features
        action        : recommended action index (0–4)
        confidence    : agent confidence [0, 1]
        q_values      : Q-values for all actions (optional, for comparison)
        similar_outcomes : outcome stats from similar patients (optional)
        """
        parts = []

        # 1. What is being recommended
        desc = ACTION_DESCRIPTIONS.get(action, "an adjustment")
        parts.append(f"Recommendation: {desc}.")

        # 2. Key patient factors driving the recommendation
        factors = self._key_factors(patient_state, action)
        if factors:
            parts.append("Key factors: " + "; ".join(factors) + ".")

        # 3. Confidence
        conf_label = "high" if confidence > 0.7 else ("moderate" if confidence > 0.4 else "low")
        parts.append(f"Confidence: {conf_label} ({confidence:.0%}).")

        # 4. Evidence from similar patients
        if similar_outcomes:
            n = similar_outcomes.get("n_similar", 0)
            surv = similar_outcomes.get("survival_rate_if_followed", 0)
            parts.append(
                f"Evidence: among {n} similar patients where this approach was taken, "
                f"the survival rate was {surv:.0%}."
            )

        # 5. Alternative actions (if Q-values available)
        if q_values is not None:
            alt = self._alternatives(q_values, action)
            if alt:
                parts.append(f"Alternatives considered: {alt}.")

        return " ".join(parts)

    @staticmethod
    def _key_factors(state: dict, action: int) -> list[str]:
        """Identify the clinical factors that most likely influenced the action."""
        factors = []

        mp = state.get("mechanical_power", None)
        spo2 = state.get("spo2", None)
        pp = state.get("plateau_pressure", None)
        dp = state.get("driving_pressure", None)
        pf = state.get("pf_ratio", None)

        if action in (0, 1):  # decrease
            if mp is not None and mp > 17:
                factors.append(f"current MP is elevated ({mp:.1f} J/min)")
            if pp is not None and pp > 25:
                factors.append(f"plateau pressure is high ({pp:.0f} cmH2O)")
            if dp is not None and dp > 13:
                factors.append(f"driving pressure is concerning ({dp:.0f} cmH2O)")
            if spo2 is not None and spo2 > 94:
                factors.append(f"oxygenation is adequate (SpO2 {spo2:.0f}%)")

        elif action in (3, 4):  # increase
            if spo2 is not None and spo2 < 90:
                factors.append(f"oxygenation is low (SpO2 {spo2:.0f}%)")
            if pf is not None and pf < 200:
                factors.append(f"P/F ratio is low ({pf:.0f})")
            if mp is not None and mp < 12:
                factors.append(f"current MP is low ({mp:.1f} J/min)")

        else:  # maintain
            if spo2 is not None:
                factors.append(f"oxygenation is acceptable (SpO2 {spo2:.0f}%)")
            if mp is not None:
                factors.append(f"MP is in a reasonable range ({mp:.1f} J/min)")

        return factors

    @staticmethod
    def _alternatives(q_values: np.ndarray, chosen: int) -> str:
        """Describe the next-best action based on Q-values."""
        ranked = np.argsort(q_values)[::-1]
        if len(ranked) < 2:
            return ""
        second_best = ranked[1] if ranked[0] == chosen else ranked[0]
        desc = ACTION_DESCRIPTIONS.get(int(second_best), "another option")
        gap = q_values[chosen] - q_values[second_best]
        return f"next-best option was {desc} (margin: {gap:.2f})"
