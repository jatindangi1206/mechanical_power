"""
Safety Filter: hard clinical constraints that override model recommendations
when patient safety is at risk.
"""

from __future__ import annotations

from loguru import logger


class SafetyFilter:
    """
    Applies rule-based safety checks to override dangerous RL recommendations.

    These are non-negotiable clinical guard rails that take precedence over
    any model output.
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
        safety_cfg = config["deployment"]["safety"]
        self.min_mp = safety_cfg["min_mp"]
        self.max_mp = safety_cfg["max_mp"]
        self.max_plateau = safety_cfg["max_plateau_pressure"]
        self.max_dp = safety_cfg["max_driving_pressure"]
        self.critical_spo2 = safety_cfg["critical_spo2"]
        self.unstable_map = safety_cfg["unstable_map_threshold"]
        self.unstable_spo2 = safety_cfg["unstable_spo2_threshold"]
        self.unstable_hr = safety_cfg["unstable_hr_threshold"]
        self.unstable_lactate = safety_cfg["unstable_lactate_threshold"]

    def filter(self, patient_state: dict, recommended_action: int) -> tuple[int, list[str]]:
        """
        Apply safety rules. Returns the (possibly modified) action and a list
        of alerts describing any overrides.

        Parameters
        ----------
        patient_state : dict with keys like 'spo2', 'plateau_pressure', etc.
        recommended_action : int (0–4)

        Returns
        -------
        (safe_action, alerts) — safe_action may differ from recommended_action
        """
        alerts = []
        action = recommended_action
        delta = self.ACTION_DELTAS[action]

        spo2 = patient_state.get("spo2", 95)
        pp = patient_state.get("plateau_pressure", 20)
        dp = patient_state.get("driving_pressure", 10)
        mp = patient_state.get("mechanical_power", 15)
        projected_mp = mp + delta

        # Rule 1: Don't reduce support if critically hypoxaemic
        if spo2 < self.critical_spo2 and delta < 0:
            action = 2  # maintain
            alerts.append(
                f"OVERRIDE: SpO2 critically low ({spo2:.0f}%) — "
                "cannot reduce ventilatory support."
            )

        # Rule 2: Don't increase if plateau pressure already dangerous
        if pp > self.max_plateau and delta > 0:
            action = 2
            alerts.append(
                f"OVERRIDE: Plateau pressure high ({pp:.0f} cmH2O) — "
                "cannot increase MP (barotrauma risk)."
            )

        # Rule 3: Reduce if driving pressure too high
        if dp > self.max_dp and delta > 0:
            action = 1  # small decrease
            alerts.append(
                f"OVERRIDE: Driving pressure elevated ({dp:.0f} cmH2O) — "
                "recommending reduction instead."
            )

        # Rule 4: Maximum MP limit
        if projected_mp > self.max_mp:
            action = 2
            alerts.append(
                f"OVERRIDE: Projected MP ({projected_mp:.1f} J/min) would exceed "
                f"maximum of {self.max_mp} J/min."
            )

        # Rule 5: Minimum MP limit
        if projected_mp < self.min_mp:
            action = 2
            alerts.append(
                f"OVERRIDE: Projected MP ({projected_mp:.1f} J/min) would fall below "
                f"minimum of {self.min_mp} J/min."
            )

        # Rule 6: Downgrade large changes if patient is unstable
        if self._is_unstable(patient_state) and abs(delta) > 2:
            if delta > 0:
                action = 3  # small increase
            else:
                action = 1  # small decrease
            alerts.append(
                "OVERRIDE: Patient haemodynamically unstable — "
                "large MP change downgraded to small change."
            )

        if action != recommended_action:
            logger.warning(
                f"Safety override: action {recommended_action} "
                f"({self.ACTION_NAMES[recommended_action]}) → "
                f"{action} ({self.ACTION_NAMES[action]})"
            )

        return action, alerts

    def _is_unstable(self, state: dict) -> bool:
        """Check whether the patient is haemodynamically or respiratory unstable."""
        return (
            state.get("mean_arterial_pressure", 80) < self.unstable_map
            or state.get("spo2", 95) < self.unstable_spo2
            or state.get("heart_rate", 80) > self.unstable_hr
            or state.get("lactate", 1.0) > self.unstable_lactate
        )

    def check_alerts(self, patient_state: dict) -> list[str]:
        """Return informational alerts about the patient's current status (no action override)."""
        alerts = []
        spo2 = patient_state.get("spo2", 95)
        pp = patient_state.get("plateau_pressure", 20)
        mp = patient_state.get("mechanical_power", 15)

        if spo2 < 88:
            alerts.append(f"WARNING: SpO2 low ({spo2:.0f}%)")
        if pp > 28:
            alerts.append(f"WARNING: Plateau pressure elevated ({pp:.0f} cmH2O)")
        if mp > 20:
            alerts.append(f"CAUTION: MP above 20 J/min ({mp:.1f})")
        if self._is_unstable(patient_state):
            alerts.append("CAUTION: Patient haemodynamically unstable")

        return alerts
