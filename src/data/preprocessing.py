"""
Data preprocessing: cleaning, imputation, normalisation, and temporal sequencing.
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ===================================================================
# Outlier removal
# ===================================================================
def remove_outliers(df: pd.DataFrame, valid_ranges: dict) -> pd.DataFrame:
    """Replace values outside physiological ranges with NaN."""
    df = df.copy()
    for col, (lo, hi) in valid_ranges.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            n_removed = mask.sum()
            if n_removed > 0:
                logger.debug(f"  {col}: {n_removed} outliers clipped to NaN")
            df.loc[mask, col] = np.nan
    return df


# ===================================================================
# Imputation
# ===================================================================
def _forward_fill(series: pd.Series, limit_hours: int, freq_hours: int = 1) -> pd.Series:
    """Forward-fill with a maximum gap limit (expressed in timesteps)."""
    return series.ffill(limit=limit_hours // freq_hours)


def impute_missing(
    df: pd.DataFrame,
    strategies: dict,
    vitals_cols: list[str],
    labs_cols: list[str],
    vent_cols: list[str],
    static_cols: list[str],
) -> pd.DataFrame:
    """
    Apply column-group-specific imputation strategies.

    Parameters
    ----------
    strategies : dict
        Mapping of group name → strategy string, e.g.
        {"vitals": "forward_fill_6h", "labs": "forward_fill_24h", ...}
    """
    df = df.copy()

    group_map = {
        "vitals": vitals_cols,
        "labs": labs_cols,
        "ventilator": vent_cols,
        "static": static_cols,
    }

    for group, cols in group_map.items():
        strategy = strategies.get(group, "median")
        existing = [c for c in cols if c in df.columns]

        if strategy.startswith("forward_fill"):
            hours = int(strategy.split("_")[-1].replace("h", ""))
            for col in existing:
                df[col] = df.groupby("stay_id")[col].transform(
                    lambda s: _forward_fill(s, hours)
                )
        elif strategy == "median":
            for col in existing:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        else:
            logger.warning(f"Unknown imputation strategy '{strategy}' for group '{group}'")

    return df


# ===================================================================
# Normalisation / scaling
# ===================================================================
class FeatureScaler:
    """Fit and transform feature columns using configurable strategies."""

    def __init__(self):
        self.scalers: dict[str, object] = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        continuous_cols: list[str],
        vitals_cols: list[str],
        categorical_cols: list[str],
        method: dict | None = None,
    ) -> pd.DataFrame:
        method = method or {"continuous": "z_score", "vitals": "min_max", "categorical": "one_hot"}
        df = df.copy()

        # Z-score for continuous
        if continuous_cols:
            scaler = StandardScaler()
            existing = [c for c in continuous_cols if c in df.columns]
            df[existing] = scaler.fit_transform(df[existing].values)
            self.scalers["continuous"] = scaler

        # Min-max for vitals
        if vitals_cols:
            scaler = MinMaxScaler()
            existing = [c for c in vitals_cols if c in df.columns]
            df[existing] = scaler.fit_transform(df[existing].values)
            self.scalers["vitals"] = scaler

        # One-hot for categoricals
        if categorical_cols:
            existing = [c for c in categorical_cols if c in df.columns]
            df = pd.get_dummies(df, columns=existing, drop_first=True)

        return df

    def transform(self, df: pd.DataFrame, continuous_cols: list, vitals_cols: list) -> pd.DataFrame:
        df = df.copy()
        if "continuous" in self.scalers:
            existing = [c for c in continuous_cols if c in df.columns]
            df[existing] = self.scalers["continuous"].transform(df[existing].values)
        if "vitals" in self.scalers:
            existing = [c for c in vitals_cols if c in df.columns]
            df[existing] = self.scalers["vitals"].transform(df[existing].values)
        return df


# ===================================================================
# Temporal sequencing
# ===================================================================
def resample_to_hourly(df: pd.DataFrame, time_col: str = "charttime") -> pd.DataFrame:
    """Resample irregular time-series to hourly bins per stay_id."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    resampled = []
    for stay_id, group in df.groupby("stay_id"):
        group = group.set_index(time_col).sort_index()
        # Resample to 1-hour frequency, take last non-null value in each bin
        hourly = group.resample("1h").last()
        hourly["stay_id"] = stay_id
        hourly["hour_index"] = range(len(hourly))
        resampled.append(hourly.reset_index())

    return pd.concat(resampled, ignore_index=True)


def create_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_hours: int = 24,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Create sliding-window sequences for each patient stay.

    Returns
    -------
    X : np.ndarray, shape (N, window_hours, n_features)
    stay_ids : list of stay_id for each sequence
    hour_indices : list of the ending hour index for each sequence
    """
    sequences = []
    ids = []
    hours = []

    for stay_id, group in df.groupby("stay_id"):
        group = group.sort_values("hour_index")
        values = group[feature_cols].values

        if len(values) < window_hours:
            continue

        for end in range(window_hours, len(values) + 1):
            start = end - window_hours
            sequences.append(values[start:end])
            ids.append(stay_id)
            hours.append(end - 1)

    X = np.array(sequences)
    return X, ids, hours


# ===================================================================
# Full pipeline
# ===================================================================
def preprocess_pipeline(
    raw_tables: dict[str, pd.DataFrame],
    config: dict,
) -> pd.DataFrame:
    """
    End-to-end preprocessing: merge tables → clean → impute → derive features → normalise → sequence.
    """
    from src.features.engineering import add_derived_features

    prep_cfg = config["preprocessing"]
    feat_cfg = config["features"]

    # 1. Merge raw tables into a single hourly DataFrame
    logger.info("Merging and resampling to hourly resolution...")
    # -- pivot ventilator wide
    vent = raw_tables["ventilator"]
    vitals = raw_tables["vitals"]
    labs = raw_tables["labs"]
    demo = raw_tables["demographics"]
    outcomes = raw_tables["outcomes"]

    # Combine time-series tables
    ts = pd.concat([vent, vitals, labs], ignore_index=True)
    ts = resample_to_hourly(ts)

    # Merge demographics (static)
    ts = ts.merge(demo, on="stay_id", how="left")
    ts = ts.merge(outcomes, on="stay_id", how="left")

    # 2. Remove outliers
    logger.info("Removing outliers...")
    ts = remove_outliers(ts, prep_cfg["valid_ranges"])

    # 3. Impute
    logger.info("Imputing missing values...")
    ts = impute_missing(
        ts,
        strategies=prep_cfg["imputation"],
        vitals_cols=feat_cfg["dynamic_vitals"],
        labs_cols=feat_cfg["dynamic_labs"],
        vent_cols=feat_cfg["ventilator"],
        static_cols=feat_cfg["static"],
    )

    # 4. Derived features
    logger.info("Computing derived features...")
    ts = add_derived_features(ts)

    # 5. Normalise
    logger.info("Normalising features...")
    scaler = FeatureScaler()
    continuous = feat_cfg["dynamic_labs"] + ["mechanical_power", "driving_pressure", "compliance"]
    ts = scaler.fit_transform(
        ts,
        continuous_cols=continuous,
        vitals_cols=feat_cfg["dynamic_vitals"],
        categorical_cols=["sex", "admission_type"],
    )

    logger.info(f"Preprocessed shape: {ts.shape}")
    return ts
