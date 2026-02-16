"""Tests for data preprocessing functions."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import remove_outliers, resample_to_hourly
from src.features.engineering import calculate_mechanical_power, predicted_body_weight


class TestRemoveOutliers:
    def test_clips_out_of_range(self):
        df = pd.DataFrame({"spo2": [50, 85, 98, 105, -5], "heart_rate": [30, 80, 120, 300, 10]})
        valid_ranges = {"spo2": (50, 100), "heart_rate": (20, 250)}
        result = remove_outliers(df, valid_ranges)
        assert result["spo2"].isna().sum() == 2  # 105 and -5
        assert result["heart_rate"].isna().sum() == 1  # 300

    def test_no_change_when_all_valid(self):
        df = pd.DataFrame({"spo2": [92, 95, 97]})
        result = remove_outliers(df, {"spo2": (50, 100)})
        assert result["spo2"].isna().sum() == 0

    def test_missing_column_ignored(self):
        df = pd.DataFrame({"spo2": [95]})
        result = remove_outliers(df, {"nonexistent": (0, 100)})
        assert len(result) == 1


class TestMechanicalPower:
    def test_basic_calculation(self):
        # MP = 0.098 * RR * VT(L) * (Ppeak - 0.5 * DP)
        mp = calculate_mechanical_power(
            respiratory_rate=20,
            tidal_volume_ml=500,
            peak_pressure=25,
            driving_pressure=10,
        )
        expected = 0.098 * 20 * 0.5 * (25 - 0.5 * 10)
        assert abs(mp - expected) < 0.01

    def test_returns_non_negative(self):
        mp = calculate_mechanical_power(
            respiratory_rate=5, tidal_volume_ml=100, peak_pressure=5, driving_pressure=20
        )
        assert mp >= 0.0


class TestPredictedBodyWeight:
    def test_male(self):
        pbw = predicted_body_weight(170, "M")
        expected = 50 + 0.91 * (170 - 152.4)
        assert abs(pbw - expected) < 0.01

    def test_female(self):
        pbw = predicted_body_weight(160, "F")
        expected = 45.5 + 0.91 * (160 - 152.4)
        assert abs(pbw - expected) < 0.01
