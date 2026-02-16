"""Tests for the deployment safety filter."""

import pytest

from src.deployment.safety_filter import SafetyFilter


@pytest.fixture
def config():
    return {
        "deployment": {
            "safety": {
                "min_mp": 8,
                "max_mp": 30,
                "max_plateau_pressure": 30,
                "max_driving_pressure": 15,
                "critical_spo2": 85,
                "unstable_map_threshold": 60,
                "unstable_spo2_threshold": 88,
                "unstable_hr_threshold": 130,
                "unstable_lactate_threshold": 4.0,
            }
        }
    }


@pytest.fixture
def safety(config):
    return SafetyFilter(config)


class TestSafetyFilter:
    def test_no_override_when_safe(self, safety):
        state = {"spo2": 95, "plateau_pressure": 20, "driving_pressure": 10,
                 "mechanical_power": 15, "mean_arterial_pressure": 80,
                 "heart_rate": 80, "lactate": 1.0}
        action, alerts = safety.filter(state, 2)  # maintain
        assert action == 2
        assert len(alerts) == 0

    def test_blocks_decrease_when_critically_hypoxaemic(self, safety):
        state = {"spo2": 80, "plateau_pressure": 20, "driving_pressure": 10,
                 "mechanical_power": 15, "mean_arterial_pressure": 80,
                 "heart_rate": 80, "lactate": 1.0}
        action, alerts = safety.filter(state, 0)  # large_decrease
        assert action == 2  # overridden to maintain
        assert len(alerts) > 0
        assert "SpO2" in alerts[0]

    def test_blocks_increase_when_high_plateau(self, safety):
        state = {"spo2": 95, "plateau_pressure": 32, "driving_pressure": 10,
                 "mechanical_power": 15, "mean_arterial_pressure": 80,
                 "heart_rate": 80, "lactate": 1.0}
        action, alerts = safety.filter(state, 4)  # large_increase
        assert action == 2  # overridden to maintain

    def test_reduces_when_driving_pressure_high(self, safety):
        state = {"spo2": 95, "plateau_pressure": 25, "driving_pressure": 18,
                 "mechanical_power": 15, "mean_arterial_pressure": 80,
                 "heart_rate": 80, "lactate": 1.0}
        action, alerts = safety.filter(state, 3)  # small_increase
        assert action == 1  # overridden to small_decrease

    def test_blocks_exceeding_max_mp(self, safety):
        state = {"spo2": 95, "plateau_pressure": 20, "driving_pressure": 10,
                 "mechanical_power": 28, "mean_arterial_pressure": 80,
                 "heart_rate": 80, "lactate": 1.0}
        action, alerts = safety.filter(state, 4)  # +5 → 33 > max 30
        assert action == 2  # maintain

    def test_blocks_below_min_mp(self, safety):
        state = {"spo2": 95, "plateau_pressure": 20, "driving_pressure": 10,
                 "mechanical_power": 9, "mean_arterial_pressure": 80,
                 "heart_rate": 80, "lactate": 1.0}
        action, alerts = safety.filter(state, 0)  # -5 → 4 < min 8
        assert action == 2

    def test_downgrades_large_change_when_unstable(self, safety):
        state = {"spo2": 95, "plateau_pressure": 20, "driving_pressure": 10,
                 "mechanical_power": 15, "mean_arterial_pressure": 55,  # unstable
                 "heart_rate": 80, "lactate": 1.0}
        action, alerts = safety.filter(state, 4)  # large_increase
        assert action == 3  # downgraded to small_increase

    def test_check_alerts_returns_warnings(self, safety):
        state = {"spo2": 86, "plateau_pressure": 29, "mechanical_power": 22,
                 "mean_arterial_pressure": 55, "heart_rate": 140, "lactate": 5.0}
        alerts = safety.check_alerts(state)
        assert len(alerts) >= 3  # low SpO2, high PP, high MP, unstable
