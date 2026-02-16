"""
Feature engineering: Mechanical Power calculation, derived clinical variables,
and feature assembly for model input.
"""

import numpy as np
import pandas as pd


# ===================================================================
# Mechanical Power formula
# ===================================================================
def calculate_mechanical_power(
    respiratory_rate: float,
    tidal_volume_ml: float,
    peak_pressure: float,
    driving_pressure: float,
) -> float:
    """
    Simplified Mechanical Power (Gattinoni equation).

    MP = 0.098 × RR × VT(L) × (Ppeak − 0.5 × ΔP)

    Parameters
    ----------
    respiratory_rate : breaths/min
    tidal_volume_ml  : mL (converted to L internally)
    peak_pressure    : cmH2O
    driving_pressure : cmH2O  (Pplat − PEEP)

    Returns
    -------
    Mechanical power in J/min.
    """
    vt_litres = tidal_volume_ml / 1000.0
    mp = 0.098 * respiratory_rate * vt_litres * (peak_pressure - 0.5 * driving_pressure)
    return max(mp, 0.0)


def calculate_mechanical_power_series(df: pd.DataFrame) -> pd.Series:
    """Vectorised MP calculation for a DataFrame with the required columns."""
    vt_l = df["tidal_volume"] / 1000.0
    dp = df["driving_pressure"]
    return 0.098 * df["respiratory_rate"] * vt_l * (df["peak_pressure"] - 0.5 * dp)


# ===================================================================
# Predicted body weight
# ===================================================================
def predicted_body_weight(height_cm: float, sex: str) -> float:
    """
    ARDSNet predicted (ideal) body weight.

    Male  : PBW = 50 + 0.91 × (height_cm − 152.4)
    Female: PBW = 45.5 + 0.91 × (height_cm − 152.4)
    """
    if sex in ("M", "Male", "male"):
        return 50.0 + 0.91 * (height_cm - 152.4)
    else:
        return 45.5 + 0.91 * (height_cm - 152.4)


# ===================================================================
# Derived features
# ===================================================================
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived clinical variables to the preprocessed DataFrame.

    New columns:
        - driving_pressure   = plateau_pressure − PEEP
        - compliance         = tidal_volume / driving_pressure
        - mechanical_power   = simplified Gattinoni equation
        - tidal_volume_per_kg = tidal_volume / predicted_body_weight
        - pf_ratio           = PaO2 / FiO2
    """
    df = df.copy()

    # Driving pressure
    if "plateau_pressure" in df.columns and "peep" in df.columns:
        df["driving_pressure"] = df["plateau_pressure"] - df["peep"]
        df["driving_pressure"] = df["driving_pressure"].clip(lower=0)

    # Static compliance
    if "tidal_volume" in df.columns and "driving_pressure" in df.columns:
        df["compliance"] = np.where(
            df["driving_pressure"] > 0,
            df["tidal_volume"] / df["driving_pressure"],
            np.nan,
        )

    # Mechanical power
    required_mp = {"respiratory_rate", "tidal_volume", "peak_pressure", "driving_pressure"}
    if required_mp.issubset(df.columns):
        df["mechanical_power"] = calculate_mechanical_power_series(df).clip(lower=0)

    # Tidal volume per kg PBW
    if "tidal_volume" in df.columns and "predicted_body_weight" in df.columns:
        df["tidal_volume_per_kg"] = np.where(
            df["predicted_body_weight"] > 0,
            df["tidal_volume"] / df["predicted_body_weight"],
            np.nan,
        )

    # P/F ratio
    if "pao2" in df.columns and "fio2" in df.columns:
        df["pf_ratio"] = np.where(
            df["fio2"] > 0,
            df["pao2"] / df["fio2"],
            np.nan,
        )

    return df


# ===================================================================
# Feature lists for model input
# ===================================================================
def get_state_feature_cols(config: dict) -> list[str]:
    """Return the ordered list of feature columns that define a state vector."""
    feat = config["features"]
    cols = (
        feat["static"]
        + feat["dynamic_vitals"]
        + feat["dynamic_labs"]
        + feat["ventilator"]
        + feat["interventions"]
    )
    return cols


def get_static_feature_cols(config: dict) -> list[str]:
    return config["features"]["static"]


def get_dynamic_feature_cols(config: dict) -> list[str]:
    feat = config["features"]
    return feat["dynamic_vitals"] + feat["dynamic_labs"] + feat["ventilator"] + feat["interventions"]
