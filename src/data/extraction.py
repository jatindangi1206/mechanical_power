"""
Data extraction from ICU databases (MIMIC-IV, Amsterdam UMCdb, HiRID).

Extracts ventilator parameters, patient characteristics, labs, vitals,
and outcomes for mechanically ventilated patients.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# MIMIC-IV item IDs for chartevents
# ---------------------------------------------------------------------------
MIMIC_ITEMIDS = {
    # Ventilator parameters
    "tidal_volume_observed": [224685],
    "tidal_volume_set": [224686],
    "respiratory_rate": [224688, 224689, 224690],
    "peep": [220339, 224700],
    "fio2": [223835],
    "plateau_pressure": [224695],
    "peak_pressure": [224696],
    "minute_ventilation": [224687],
    # Vitals
    "heart_rate": [220045],
    "mean_arterial_pressure": [220052, 220181, 225312],
    "spo2": [220277],
    "temperature": [223761, 223762],
    "gcs_total": [228112, 220739],
    # Additional
    "height": [226730],
    "weight": [224639, 226512],
}

MIMIC_LAB_ITEMIDS = {
    "ph": [50820],
    "pao2": [50821],
    "paco2": [50818],
    "lactate": [50813],
    "creatinine": [50912],
    "bilirubin": [50885],
    "platelet_count": [51265],
}


class BaseExtractor(ABC):
    """Abstract base class for ICU data extractors."""

    def __init__(self, config: dict):
        self.config = config
        self.cohort_config = config.get("cohort", {})

    @abstractmethod
    def extract_ventilation_episodes(self) -> pd.DataFrame:
        """Identify patients on invasive mechanical ventilation."""
        ...

    @abstractmethod
    def extract_ventilator_parameters(self, stay_ids: list) -> pd.DataFrame:
        """Extract ventilator settings over time."""
        ...

    @abstractmethod
    def extract_vitals(self, stay_ids: list) -> pd.DataFrame:
        """Extract time-varying vital signs."""
        ...

    @abstractmethod
    def extract_labs(self, stay_ids: list) -> pd.DataFrame:
        """Extract laboratory results."""
        ...

    @abstractmethod
    def extract_demographics(self, stay_ids: list) -> pd.DataFrame:
        """Extract patient demographics and admission info."""
        ...

    @abstractmethod
    def extract_outcomes(self, stay_ids: list) -> pd.DataFrame:
        """Extract mortality / discharge outcomes."""
        ...

    def extract_all(self) -> dict[str, pd.DataFrame]:
        """Run the full extraction pipeline and return all tables."""
        logger.info("Extracting ventilation episodes...")
        episodes = self.extract_ventilation_episodes()
        stay_ids = episodes["stay_id"].unique().tolist()
        logger.info(f"Found {len(stay_ids)} qualifying stays")

        logger.info("Extracting ventilator parameters...")
        vent = self.extract_ventilator_parameters(stay_ids)

        logger.info("Extracting vitals...")
        vitals = self.extract_vitals(stay_ids)

        logger.info("Extracting labs...")
        labs = self.extract_labs(stay_ids)

        logger.info("Extracting demographics...")
        demo = self.extract_demographics(stay_ids)

        logger.info("Extracting outcomes...")
        outcomes = self.extract_outcomes(stay_ids)

        return {
            "episodes": episodes,
            "ventilator": vent,
            "vitals": vitals,
            "labs": labs,
            "demographics": demo,
            "outcomes": outcomes,
        }


# ===================================================================
# MIMIC-IV Extractor (BigQuery)
# ===================================================================
class MIMICExtractor(BaseExtractor):
    """Extract data from MIMIC-IV via Google BigQuery."""

    def __init__(self, config: dict):
        super().__init__(config)
        mimic_cfg = config["data"]["mimic"]
        self.project_id = mimic_cfg["project_id"]
        self.dataset = mimic_cfg["dataset"]
        self.credentials_path = mimic_cfg.get("credentials_path")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google.cloud import bigquery

            if self.credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
            self._client = bigquery.Client(project=self.project_id)
        return self._client

    def _query(self, sql: str) -> pd.DataFrame:
        """Execute a BigQuery SQL query and return a DataFrame."""
        logger.debug(f"Executing query ({len(sql)} chars)")
        return self.client.query(sql).to_dataframe()

    def _table(self, name: str) -> str:
        return f"`{self.project_id}.{self.dataset}.{name}`"

    # ------------------------------------------------------------------
    def extract_ventilation_episodes(self) -> pd.DataFrame:
        min_age = self.cohort_config.get("min_age", 18)
        min_hours = self.cohort_config.get("min_ventilation_hours", 24)

        sql = f"""
        WITH vent_starts AS (
            SELECT
                ie.subject_id,
                ie.hadm_id,
                ie.stay_id,
                pe.starttime AS vent_start,
                pe.endtime   AS vent_end,
                TIMESTAMP_DIFF(pe.endtime, pe.starttime, HOUR) AS vent_hours
            FROM {self._table('procedureevents')} pe
            INNER JOIN {self._table('icustays')} ie
                ON pe.stay_id = ie.stay_id
            WHERE pe.itemid IN (225792)  -- Invasive mechanical ventilation
        ),
        eligible AS (
            SELECT
                vs.*,
                p.anchor_age AS age,
                adm.deathtime
            FROM vent_starts vs
            INNER JOIN {self._table('patients')} p
                ON vs.subject_id = p.subject_id
            INNER JOIN {self._table('admissions')} adm
                ON vs.hadm_id = adm.hadm_id
            WHERE vs.vent_hours >= {min_hours}
              AND p.anchor_age >= {min_age}
        )
        SELECT * FROM eligible
        """
        df = self._query(sql)
        logger.info(f"Ventilation episodes extracted: {len(df)}")
        return df

    def extract_ventilator_parameters(self, stay_ids: list) -> pd.DataFrame:
        ids_str = ",".join(str(s) for s in stay_ids)
        all_items = []
        for items in MIMIC_ITEMIDS.values():
            all_items.extend(items)
        # Only vent-related items
        vent_items = []
        for key in [
            "tidal_volume_observed",
            "tidal_volume_set",
            "respiratory_rate",
            "peep",
            "fio2",
            "plateau_pressure",
            "peak_pressure",
            "minute_ventilation",
        ]:
            vent_items.extend(MIMIC_ITEMIDS[key])

        items_str = ",".join(str(i) for i in vent_items)

        sql = f"""
        SELECT
            stay_id,
            charttime,
            itemid,
            valuenum
        FROM {self._table('chartevents')}
        WHERE stay_id IN ({ids_str})
          AND itemid IN ({items_str})
          AND valuenum IS NOT NULL
        ORDER BY stay_id, charttime
        """
        return self._query(sql)

    def extract_vitals(self, stay_ids: list) -> pd.DataFrame:
        ids_str = ",".join(str(s) for s in stay_ids)
        vital_items = []
        for key in ["heart_rate", "mean_arterial_pressure", "spo2", "temperature", "gcs_total"]:
            vital_items.extend(MIMIC_ITEMIDS[key])
        items_str = ",".join(str(i) for i in vital_items)

        sql = f"""
        SELECT stay_id, charttime, itemid, valuenum
        FROM {self._table('chartevents')}
        WHERE stay_id IN ({ids_str})
          AND itemid IN ({items_str})
          AND valuenum IS NOT NULL
        ORDER BY stay_id, charttime
        """
        return self._query(sql)

    def extract_labs(self, stay_ids: list) -> pd.DataFrame:
        ids_str = ",".join(str(s) for s in stay_ids)
        lab_items = []
        for items in MIMIC_LAB_ITEMIDS.values():
            lab_items.extend(items)
        items_str = ",".join(str(i) for i in lab_items)

        sql = f"""
        SELECT
            ie.stay_id,
            le.charttime,
            le.itemid,
            le.valuenum
        FROM {self._table('labevents')} le
        INNER JOIN {self._table('icustays')} ie
            ON le.hadm_id = ie.hadm_id
        WHERE ie.stay_id IN ({ids_str})
          AND le.itemid IN ({items_str})
          AND le.valuenum IS NOT NULL
        ORDER BY ie.stay_id, le.charttime
        """
        return self._query(sql)

    def extract_demographics(self, stay_ids: list) -> pd.DataFrame:
        ids_str = ",".join(str(s) for s in stay_ids)

        sql = f"""
        SELECT
            ie.stay_id,
            ie.hadm_id,
            p.anchor_age  AS age,
            p.gender      AS sex,
            adm.admission_type,
            adm.ethnicity
        FROM {self._table('icustays')} ie
        INNER JOIN {self._table('patients')} p
            ON ie.subject_id = p.subject_id
        INNER JOIN {self._table('admissions')} adm
            ON ie.hadm_id = adm.hadm_id
        WHERE ie.stay_id IN ({ids_str})
        """
        return self._query(sql)

    def extract_outcomes(self, stay_ids: list) -> pd.DataFrame:
        ids_str = ",".join(str(s) for s in stay_ids)

        sql = f"""
        SELECT
            ie.stay_id,
            ie.hadm_id,
            adm.deathtime,
            adm.hospital_expire_flag,
            ie.los AS icu_los_days
        FROM {self._table('icustays')} ie
        INNER JOIN {self._table('admissions')} adm
            ON ie.hadm_id = adm.hadm_id
        WHERE ie.stay_id IN ({ids_str})
        """
        return self._query(sql)


# ===================================================================
# Amsterdam UMCdb Extractor (PostgreSQL)
# ===================================================================
class AmsterdamExtractor(BaseExtractor):
    """Extract data from Amsterdam UMCdb via PostgreSQL."""

    def __init__(self, config: dict):
        super().__init__(config)
        from sqlalchemy import create_engine

        conn_str = config["data"]["amsterdam"]["connection_string"]
        self.engine = create_engine(conn_str)

    def _query(self, sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, self.engine)

    def extract_ventilation_episodes(self) -> pd.DataFrame:
        # Amsterdam-specific table/column names — adapt as needed
        sql = """
        SELECT
            admissionid AS stay_id,
            start AS vent_start,
            stop  AS vent_end,
            EXTRACT(EPOCH FROM (stop - start)) / 3600 AS vent_hours
        FROM processitems
        WHERE itemid IN (9328, 12290)  -- Mechanical ventilation items
          AND EXTRACT(EPOCH FROM (stop - start)) / 3600 >= 24
        """
        return self._query(sql)

    def extract_ventilator_parameters(self, stay_ids: list) -> pd.DataFrame:
        ids_str = ",".join(str(s) for s in stay_ids)
        sql = f"""
        SELECT admissionid AS stay_id, measuredat AS charttime, itemid, value AS valuenum
        FROM numericitems
        WHERE admissionid IN ({ids_str})
          AND itemid IN (
            12279,  -- Tidal volume
            12283,  -- Respiratory rate
            12275,  -- PEEP
            12271,  -- FiO2
            12277,  -- Peak pressure
            12281   -- Plateau pressure
          )
        ORDER BY admissionid, measuredat
        """
        return self._query(sql)

    def extract_vitals(self, stay_ids: list) -> pd.DataFrame:
        ids_str = ",".join(str(s) for s in stay_ids)
        sql = f"""
        SELECT admissionid AS stay_id, measuredat AS charttime, itemid, value AS valuenum
        FROM numericitems
        WHERE admissionid IN ({ids_str})
          AND itemid IN (6640, 6641, 6642, 6643, 8843)
        ORDER BY admissionid, measuredat
        """
        return self._query(sql)

    def extract_labs(self, stay_ids: list) -> pd.DataFrame:
        ids_str = ",".join(str(s) for s in stay_ids)
        sql = f"""
        SELECT admissionid AS stay_id, measuredat AS charttime, itemid, value AS valuenum
        FROM numericitems
        WHERE admissionid IN ({ids_str})
          AND itemid IN (6848, 9990, 9992, 10053, 9580, 6810, 9553)
        ORDER BY admissionid, measuredat
        """
        return self._query(sql)

    def extract_demographics(self, stay_ids: list) -> pd.DataFrame:
        ids_str = ",".join(str(s) for s in stay_ids)
        sql = f"""
        SELECT
            admissionid AS stay_id,
            agegroup AS age,
            gender AS sex,
            admissionyeargroup
        FROM admissions
        WHERE admissionid IN ({ids_str})
        """
        return self._query(sql)

    def extract_outcomes(self, stay_ids: list) -> pd.DataFrame:
        ids_str = ",".join(str(s) for s in stay_ids)
        sql = f"""
        SELECT
            admissionid AS stay_id,
            dateofdeath,
            destination,
            dischargedat
        FROM admissions
        WHERE admissionid IN ({ids_str})
        """
        return self._query(sql)


# ===================================================================
# Factory
# ===================================================================
def get_extractor(config: dict) -> BaseExtractor:
    """Return the appropriate extractor for the configured data source."""
    source = config["data"]["source"]
    if source == "mimic":
        return MIMICExtractor(config)
    elif source == "amsterdam":
        return AmsterdamExtractor(config)
    else:
        raise ValueError(f"Unsupported data source: {source}. Use 'mimic' or 'amsterdam'.")
