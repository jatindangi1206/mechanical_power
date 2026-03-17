# Mechanical Power Personalisation for ICU Patients

A clinical decision support system that uses machine learning and offline reinforcement learning to recommend personalised **Mechanical Power (MP)** adjustments for mechanically ventilated ICU patients. Three progressively complex strategies are implemented — from static XGBoost mortality prediction to a full Conservative Q-Learning (CQL) policy agent — all with Optuna hyperparameter tuning, SHAP-IQ explainability, and hard clinical safety constraints.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Repository Structure](#2-repository-structure)
3. [How Everything Connects — Data Flow](#3-how-everything-connects--data-flow)
4. [Installation](#4-installation)
5. [Supported Databases](#5-supported-databases)
6. [MIMIC-IV Offline Quickstart (No BigQuery)](#6-mimic-iv-offline-quickstart-no-bigquery)
7. [Running the Notebooks](#7-running-the-notebooks)
8. [Running via CLI Scripts](#8-running-via-cli-scripts)
9. [Configuration Reference](#9-configuration-reference)
10. [What You Can Change and How](#10-what-you-can-change-and-how)
11. [Adding a New Database](#11-adding-a-new-database)
12. [Full Function Reference](#12-full-function-reference)
13. [Reproducibility Checklist](#13-reproducibility-checklist)
14. [Troubleshooting](#14-troubleshooting)
15. [Key References](#15-key-references)

---

## 1. What This Project Does

### The Clinical Problem

Mechanical ventilation is life-sustaining but itself causes lung injury when settings are wrong. **Mechanical Power (MP)** — the energy delivered to the lung per minute — is a key modifiable target. Clinicians currently adjust it based on personal experience with hundreds of patients; this system learns from thousands.

### The Mechanical Power Formula (Gattinoni, 2016)

```
MP (J/min) = 0.098 × RR × VT(L) × (Ppeak − 0.5 × ΔP)
```

Where:
- `RR` = respiratory rate (breaths/min)
- `VT(L)` = tidal volume in litres
- `Ppeak` = peak inspiratory pressure (cmH₂O)
- `ΔP` = driving pressure = plateau_pressure − PEEP

This is computed in `src/features/engineering.py → add_derived_features()`.

### The Three Strategies

| # | Strategy | Algorithm | Input | Output | When to Use |
|---|----------|-----------|-------|--------|-------------|
| 1 | **Static XGBoost** | XGBoost classifier | Demographics + APACHE + MP at t=0 and t=24h | 28-day mortality probability | Rapid risk stratification |
| 2 | **Continuous XGBoost** | XGBoost classifier | 5 MP snapshots (0/6/12/18/24h) + deltas + summary stats | 28-day mortality probability | Trajectory-aware prognosis |
| 3 | **CQL Offline RL** | Conservative Q-Learning | Patient state at each hour (MDP) | Optimal MP action: {−5, −2, 0, +2, +5} J/min | Sequential treatment decisions |

All three strategies use:
- **Optuna** (Bayesian hyperparameter optimisation)
- **SHAP-IQ** (Shapley Interaction Indices — main effects + pairwise feature interactions)
- **Safety Filter** (6 hard clinical constraint rules that override any model output)
- **5-fold stratified cross-validation** (by mortality outcome)

---

## 2. Repository Structure

```
mechanical_power/
│
├── config/
│   └── config.yaml                  # Single source of truth for all settings
│
├── src/
│   ├── data/
│   │   ├── extraction.py            # Data extractors (local CSV, BigQuery, PostgreSQL)
│   │   ├── preprocessing.py         # Outlier removal, imputation, hourly resampling
│   │   └── dataset.py               # MDP construction: (s, a, r, s', done) tuples
│   │
│   ├── features/
│   │   └── engineering.py           # MP formula, driving pressure, P/F ratio, TV/kg
│   │
│   ├── models/
│   │   ├── strategy1_static.py      # Strategy 1: feature extraction helper
│   │   ├── strategy2_window.py      # Strategy 2: time-window feature extraction
│   │   ├── q_network.py             # PyTorch Q-value network (Linear+LayerNorm+ReLU)
│   │   ├── cql_agent.py             # CQL training loop + soft target updates
│   │   └── state_encoder.py         # LSTM encoder (optional, Strategy 3 extension)
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # AUROC, AUPRC, Brier score, calibration, WIS
│   │   ├── comparison.py            # Cross-strategy comparison table
│   │   └── clinician_validation.py  # Clinician survey scoring framework
│   │
│   ├── deployment/
│   │   ├── api.py                   # FastAPI inference endpoint (optional serving)
│   │   ├── safety_filter.py         # 6 hard safety constraint rules
│   │   └── explainer.py             # Natural-language recommendation explanations
│   │
│   └── utils/
│       └── helpers.py               # Shared utilities (logging, config loading)
│
├── notebooks/
│   ├── 00_Data_Pipeline.ipynb       # Extract → preprocess → engineer → save parquet
│   ├── 01_Strategy1_XGBoost_Static.ipynb
│   ├── 02_Strategy2_XGBoost_Continuous.ipynb
│   └── 03_Strategy3_CQL_Offline_RL.ipynb
│
├── data/
│   ├── processed/
│   │   ├── processed_cohort.parquet # Generated by NB00 — hourly patient time-series
│   │   └── cohort_meta.json         # Cohort statistics (n_stays, mortality_rate, etc.)
│   └── raw/                         # (Optional) cached raw extractions
│
├── scripts/
│   ├── extract_data.py              # CLI: run data extraction
│   ├── train.py                     # CLI: train all strategies
│   └── evaluate.py                  # CLI: run full evaluation suite
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_safety_filter.py
│
├── mimic-iv-clinical-database-demo-2.2/   # MIMIC-IV demo dataset (place here)
│   ├── icu/
│   │   ├── chartevents.csv.gz
│   │   ├── icustays.csv.gz
│   │   └── procedureevents.csv.gz
│   └── hosp/
│       ├── patients.csv.gz
│       ├── admissions.csv.gz
│       └── labevents.csv.gz
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 3. How Everything Connects — Data Flow

```
config/config.yaml
        │
        ▼
src/data/extraction.py
  get_extractor(config)
  ├── LocalMIMICExtractor    ← reads .csv.gz files offline (mimic_local)
  ├── MIMICExtractor         ← reads from BigQuery (mimic)
  └── AmsterdamExtractor     ← reads from PostgreSQL (amsterdam)
        │
        │  Raw DataFrames (chartevents, labevents, vitals, demographics)
        ▼
src/data/preprocessing.py
  remove_outliers(df, config)
  impute_missing(df, config)
  resample_to_hourly(df, config)
        │
        │  Clean hourly time-series DataFrame
        ▼
src/features/engineering.py
  add_derived_features(df, config)
  ├── mechanical_power (Gattinoni formula)
  ├── driving_pressure = plateau_pressure − PEEP
  ├── compliance = tidal_volume / driving_pressure
  ├── pf_ratio = PaO2 / FiO2
  └── tidal_volume_per_kg = tidal_volume / predicted_body_weight
        │
        │  data/processed/processed_cohort.parquet  (saved by NB00)
        ▼
┌─────────────────────────────────────────┐
│           Notebooks 01, 02, 03          │
│  (each loads parquet independently)     │
└─────────────────────────────────────────┘
        │
   ┌────┴──────────────────────┐
   │                           │
   ▼                           ▼
NB01/NB02: XGBoost          NB03: CQL Offline RL
  │                           │
  │  src/models/              │  src/data/dataset.py
  │  strategy1_static.py      │  build_episodes()
  │  strategy2_window.py      │  discretise_action()
  │                           │  calculate_reward()
  │  Optuna (40 trials/fold)  │
  │  StratifiedKFold(5)       │  src/models/q_network.py
  │                           │  src/models/cql_agent.py
  │                           │  Optuna (20 trials)
  │                           │
  ├── SHAP-IQ explanations ───┤
  │   (shapiq.TabularExplainer)
  │
  ├── src/evaluation/metrics.py
  │   mortality_metrics() / policy_return() / safety_metrics()
  │
  ├── src/deployment/safety_filter.py
  │   SafetyFilter.filter(patient_state, recommended_action)
  │
  └── src/deployment/explainer.py
      ExplanationGenerator.generate(...)
            │
            ▼
      Doctor Report
      Similar patient evidence
      Cross-strategy comparison (NB03)
```

---

## 4. Installation

### Prerequisites

- Python 3.9 or 3.10
- pip

### Step-by-step

```bash
# Clone the repository
git clone https://github.com/your-org/mechanical_power.git
cd mechanical_power

# Install in editable mode
pip install -e .

# Or install requirements directly
pip install -r requirements.txt
```

### Key dependencies (pinned versions tested)

```
xgboost==2.1.4          # or 3.x — early_stopping_rounds goes in constructor for 3.x
optuna==4.2.1
shapiq==1.2.1
torch>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0          # required for .parquet read/write
loguru>=0.7.0
pyyaml>=6.0
plotly>=5.18.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
```

### XGBoost version note

If you are using **XGBoost 3.x**, `early_stopping_rounds` must be passed in the constructor, not in `fit()`:

```python
# XGBoost 3.x (correct)
model = xgb.XGBClassifier(early_stopping_rounds=20, **other_params)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

# XGBoost 2.x (also works)
model = xgb.XGBClassifier(**other_params)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=20, verbose=False)
```

The notebooks use the constructor style (compatible with both 2.x and 3.x).

---

## 5. Supported Databases

### Comparison

| Feature | `mimic_local` | `mimic` (BigQuery) | `amsterdam` (PostgreSQL) |
|---------|--------------|-------------------|--------------------------|
| Access type | Local CSV files | Google BigQuery | PostgreSQL connection |
| Internet required | **No** | Yes | Yes (or VPN) |
| Setup time | ~5 minutes | ~30 minutes | ~1 hour |
| Demo data available | Yes (66 stays free) | Full 70K+ stays | 20K+ stays (DUA required) |
| Recommended for | Getting started, offline research | Full-scale research | European cohort research |

### Setting the database source

Edit `config/config.yaml`, `data.source` field:

```yaml
data:
  source: "mimic_local"   # options: mimic_local | mimic | amsterdam
```

---

## 6. MIMIC-IV Offline Quickstart (No BigQuery)

This mode uses the free MIMIC-IV demo dataset (100 patients, publicly available without credentialing) or your locally downloaded full MIMIC-IV dataset.

### Step 1: Get the demo data

```bash
# Option A: Already have it in the project folder (default location)
ls mimic-iv-clinical-database-demo-2.2/icu/chartevents.csv.gz   # should exist

# Option B: Download from PhysioNet
wget -r -N -c -np \
  https://physionet.org/files/mimic-iv-demo/2.2/ \
  -P mimic-iv-clinical-database-demo-2.2/
```

For the **full MIMIC-IV** dataset (requires credentialing at physionet.org):
```bash
# After approval, download to the same folder structure:
# mimic-iv-clinical-database-2.2/icu/chartevents.csv.gz  (etc.)
```

### Step 2: Configure the data path

Open `config/config.yaml` and verify:

```yaml
data:
  source: "mimic_local"
  mimic_local:
    data_dir: "mimic-iv-clinical-database-demo-2.2"   # relative to project root
```

If your data is elsewhere (e.g., `/data/mimic-iv/`):
```yaml
data:
  mimic_local:
    data_dir: "/data/mimic-iv"
```

### Step 3: Verify the extractor loads

```bash
cd mechanical_power
python3 -c "
from src.data.extraction import get_extractor
import yaml
config = yaml.safe_load(open('config/config.yaml'))
ext = get_extractor(config)
print(type(ext).__name__)   # Should print: LocalMIMICExtractor
"
```

### Step 4: Run the data pipeline (Notebook 00)

```bash
jupyter notebook notebooks/00_Data_Pipeline.ipynb
# Run all cells (Kernel → Restart & Run All)
# Expected output: data/processed/processed_cohort.parquet (20,000+ rows, 70+ stays)
```

Or via script:
```bash
python scripts/extract_data.py --config config/config.yaml
```

### Expected output with demo data

```
INFO  | LocalMIMICExtractor initialised. Data dir: mimic-iv-clinical-database-demo-2.2
INFO  | Loaded 668,935 chartevents rows
INFO  | Ventilated stays identified: 76
INFO  | After preprocessing: 76 stays, 20,808 hourly rows
INFO  | Stays with full MP parameters: 50
INFO  | Mortality rate: 19.7%
INFO  | Saved to data/processed/processed_cohort.parquet
```

### Using the full MIMIC-IV offline

Simply point `data_dir` at the full dataset download. The extractor reads `.csv.gz` files in exactly the same way regardless of the size:

```yaml
data:
  source: "mimic_local"
  mimic_local:
    data_dir: "/path/to/mimic-iv-clinical-database-2.2"
```

No code changes are needed — `LocalMIMICExtractor` handles both demo and full datasets identically.

---

## 7. Running the Notebooks

Run notebooks **in order** — Notebook 00 must be run first to generate the parquet file that the others load.

### Notebook 00: Data Pipeline

**Purpose:** Extract → preprocess → engineer features → save to parquet.
**Run once.** Re-run only if you change the data source or feature engineering.

```bash
jupyter notebook notebooks/00_Data_Pipeline.ipynb
```

Expected runtime: 2–5 minutes on demo data, 30–60 minutes on full MIMIC-IV.

Outputs:
- `data/processed/processed_cohort.parquet`
- `data/processed/cohort_meta.json`

### Notebook 01: Strategy 1 — Static XGBoost

**Purpose:** Predict 28-day mortality from MP at t=0 and t=24h.

```bash
jupyter notebook notebooks/01_Strategy1_XGBoost_Static.ipynb
```

Key cells:
- Cell 3: Feature extraction (demographics + APACHE + MP at 2 time points)
- Cell 4: Optuna tuning (40 trials × 5 folds = 200 total model fits)
- Cell 6: SHAP-IQ local explanation for one high-risk patient
- Cell 8: Doctor report with k-NN similar patient matching
- Cell 9: Export model to `notebooks/models/strategy1_xgb.json`

Expected runtime: 10–20 minutes.
Expected AUROC (demo data): 0.70–0.80.

### Notebook 02: Strategy 2 — Continuous XGBoost

**Purpose:** Predict mortality using MP trajectory (5 snapshots + deltas + summary stats).

```bash
jupyter notebook notebooks/02_Strategy2_XGBoost_Continuous.ipynb
```

Key additional features over Strategy 1:
- MP at hours 0, 6, 12, 18, 24
- MP deltas between consecutive windows
- MP summary stats: mean, std, max, min, trend (slope), variability
- SpO₂ and driving pressure trajectories

Expected AUROC: equal to or slightly better than Strategy 1.

### Notebook 03: Strategy 3 — CQL Offline RL

**Purpose:** Learn an optimal MP adjustment policy using Conservative Q-Learning.

```bash
jupyter notebook notebooks/03_Strategy3_CQL_Offline_RL.ipynb
```

Key cells:
- Cell 3: MDP construction — episodes, actions, rewards
- Cell 5: CQL training (custom PyTorch — no d3rlpy dependency)
- Cell 7: Safety filter testing (should show 0 hard violations)
- Cell 8: SHAP-IQ on Q-values — what drives action selection
- Cell 10: Doctor report via `ExplanationGenerator`
- Cell 11: **Cross-strategy comparison** — S1 vs S2 vs S3 side-by-side

Expected runtime: 15–30 minutes.

---

## 8. Running via CLI Scripts

All scripts read from `config/config.yaml` by default.

```bash
# Extract and preprocess data
python scripts/extract_data.py \
  --config config/config.yaml \
  --output data/processed/processed_cohort.parquet

# Train all three strategies
python scripts/train.py \
  --config config/config.yaml \
  --data data/processed/processed_cohort.parquet \
  --output notebooks/models/

# Run full evaluation suite
python scripts/evaluate.py \
  --config config/config.yaml \
  --models notebooks/models/ \
  --data data/processed/processed_cohort.parquet
```

---

## 9. Configuration Reference

The single config file at `config/config.yaml` controls every aspect of the project. Key sections:

```yaml
# ─────────────────────────────────────────────────────
# DATA SOURCE
# ─────────────────────────────────────────────────────
data:
  source: "mimic_local"         # mimic_local | mimic | amsterdam
  mimic_local:
    data_dir: "mimic-iv-clinical-database-demo-2.2"
  mimic:
    project: "your-gcp-project"
    dataset: "physionet-data.mimiciv_icu"
  amsterdam:
    host: "localhost"
    port: 5432
    dbname: "amsterdam"

# ─────────────────────────────────────────────────────
# COHORT INCLUSION CRITERIA
# ─────────────────────────────────────────────────────
cohort:
  min_age: 18
  min_ventilation_hours: 24     # Minimum hours on mechanical ventilation
  required_features:            # Drop stays missing any of these
    - tidal_volume
    - respiratory_rate
    - peak_pressure
    - peep

# ─────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────
features:
  mp_formula: "gattinoni"       # gattinoni | simplified
  include_labs: true            # Include blood gas values (PaO2, PaCO2, lactate)
  include_severity: true        # Include SOFA / SAPS-II scores
  hourly_window: 1              # Resampling window in hours

# ─────────────────────────────────────────────────────
# MDP (for Strategy 3 / CQL)
# ─────────────────────────────────────────────────────
mdp:
  actions:
    0: {delta: -5}              # Decrease MP by 5 J/min
    1: {delta: -2}              # Decrease MP by 2 J/min
    2: {delta: 0}               # Maintain current MP
    3: {delta: +2}              # Increase MP by 2 J/min
    4: {delta: +5}              # Increase MP by 5 J/min
  reward:
    survival_bonus: 100         # Awarded at terminal state if survived
    death_penalty: -100         # Awarded at terminal state if died
    spo2_weight: 2.0            # Per % improvement in SpO2
    pf_ratio_weight: 0.1        # Per unit improvement in P/F ratio
    map_improvement_weight: 0.5 # Per mmHg improvement in MAP if MAP was low
    plateau_pressure_30_penalty: -15   # If Pplat > 30 cmH2O
    plateau_pressure_28_penalty: -5    # If Pplat > 28 cmH2O
    driving_pressure_15_penalty: -8    # If ΔP > 15 cmH2O
    mp_over_20_penalty_per_unit: -3    # Per J/min above 20 J/min
    spo2_below_85_penalty: -25         # If SpO2 < 85%
    spo2_below_88_penalty: -10         # If SpO2 < 88%
    map_below_60_penalty: -20          # If MAP < 60 mmHg
    paco2_over_60_penalty: -10         # If PaCO2 > 60 mmHg
    time_penalty_per_hour: -0.1        # Per hour on ventilator (encourages weaning)

# ─────────────────────────────────────────────────────
# DEPLOYMENT / SAFETY
# ─────────────────────────────────────────────────────
deployment:
  safety:
    min_mp: 5.0                 # Hard lower bound on projected MP (J/min)
    max_mp: 30.0                # Hard upper bound on projected MP (J/min)
    max_plateau_pressure: 30.0  # cmH2O — block any increase if exceeded
    max_driving_pressure: 15.0  # cmH2O — force decrease if exceeded
    critical_spo2: 85.0         # % — block any decrease if below this
    unstable_map_threshold: 60  # mmHg — triggers instability flag
    unstable_spo2_threshold: 88 # % — triggers instability flag
    unstable_hr_threshold: 130  # bpm — triggers instability flag
    unstable_lactate_threshold: 4.0  # mmol/L — triggers instability flag

# ─────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────
training:
  split:
    train: 0.70
    val: 0.15
    test: 0.15
  random_seed: 42
  n_cv_folds: 5
  optuna_trials: 40             # Per inner fold for XGBoost strategies
  cql_optuna_trials: 20         # For CQL architecture search
```

---

## 10. What You Can Change and How

### Change the patient cohort

Edit `config.yaml → cohort`:

```yaml
cohort:
  min_ventilation_hours: 48     # Increase to 48h minimum
  min_age: 16                   # Include adolescents
```

To add exclusion criteria (e.g., exclude ECMO patients), edit `src/data/extraction.py → LocalMIMICExtractor.extract_ventilation_episodes()`. Add a filter on the relevant ITEMID:

```python
# Example: exclude stays with ECMO (ITEMID 225441)
ecmo_stays = chartevents[chartevents["itemid"] == 225441]["stay_id"].unique()
vent = vent[~vent["stay_id"].isin(ecmo_stays)]
```

### Change the features used

Add a new derived feature in `src/features/engineering.py → add_derived_features()`:

```python
def add_derived_features(df, config):
    # ... existing features ...

    # Add your new feature:
    df["my_new_ratio"] = df["some_col"] / df["other_col"]
    df["my_new_ratio"] = df["my_new_ratio"].clip(lower=0, upper=200)

    return df
```

The new column will automatically appear in `processed_cohort.parquet` and be available to all notebooks.

### Change the MP formula

The formula is in `src/features/engineering.py`. To switch from Gattinoni to a simplified version:

```python
# Simplified MP formula (without the 0.5 × ΔP correction):
df["mechanical_power"] = 0.098 * df["respiratory_rate"] * \
                         (df["tidal_volume"] / 1000) * df["peak_pressure"]
```

Or set `config.yaml → features.mp_formula: "simplified"` if you add the branch logic.

### Change the reward function

All reward weights are in `config.yaml → mdp.reward`. Change them without touching any code:

```yaml
reward:
  survival_bonus: 200           # Make survival more important
  mp_over_20_penalty_per_unit: -5   # Stronger penalty for high MP
  time_penalty_per_hour: -0.2       # Stronger pressure to wean
```

Reward logic is in `src/data/dataset.py → calculate_reward()`. To add a new clinical component (e.g., penalise high FiO₂):

```python
fio2 = state_t1.get("fio2", 0.5)
if fio2 > 0.6:
    reward += reward_cfg.get("fio2_over_60_penalty", -5)
```

Add the corresponding weight to `config.yaml`.

### Change the safety thresholds

Edit `config.yaml → deployment.safety`. No code changes needed:

```yaml
deployment:
  safety:
    max_plateau_pressure: 28.0   # Tighter limit (from 30 → 28)
    critical_spo2: 88.0          # Higher threshold for blocking reductions
```

To add a new safety rule, edit `src/deployment/safety_filter.py → SafetyFilter.filter()`:

```python
# Rule 7: Don't increase if PaCO2 already very high (risk of permissive hypercapnia)
paco2 = patient_state.get("paco2", 40)
if paco2 > 55 and delta > 0:
    action = 2
    alerts.append(f"OVERRIDE: PaCO2 elevated ({paco2:.0f} mmHg) — cannot increase MP.")
```

### Change the action space

Edit `config.yaml → mdp.actions`. For finer granularity:

```yaml
mdp:
  actions:
    0: {delta: -5}
    1: {delta: -3}
    2: {delta: -1}
    3: {delta: 0}
    4: {delta: +1}
    5: {delta: +3}
    6: {delta: +5}
```

The Q-network output size is set automatically from `len(config["mdp"]["actions"])`.

### Change Optuna search bounds

In the notebooks, the `objective()` function defines search ranges. For example, to search deeper trees:

```python
params = {
    'max_depth': trial.suggest_int('max_depth', 3, 12),    # was 2–8
    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),  # was 50–500
}
```

---

## 11. Adding a New Database

To add a new data source (e.g., eICU, HiRID, your hospital's EHR):

### Step 1: Create a new extractor class

In `src/data/extraction.py`, subclass `BaseExtractor`:

```python
class MyHospitalExtractor(BaseExtractor):
    """Extract from MyHospital EHR system."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.connection_string = config["data"]["my_hospital"]["connection_string"]

    def extract_chartevents(self) -> pd.DataFrame:
        """Return DataFrame with columns: stay_id, itemid, charttime, valuenum"""
        # Your implementation here
        ...

    def extract_ventilation_episodes(self) -> pd.DataFrame:
        """Return DataFrame with: stay_id, subject_id, hadm_id,
           intime, outtime, anchor_age, gender, hospital_expire_flag"""
        ...

    def extract_labevents(self) -> pd.DataFrame:
        """Return DataFrame with: stay_id, itemid, charttime, valuenum"""
        ...

    # Implement remaining abstract methods:
    # extract_demographics(), extract_admissions(), extract_procedures()
```

### Step 2: Register in the factory function

```python
def get_extractor(config: dict) -> BaseExtractor:
    source = config["data"]["source"]
    if source == "mimic_local":
        return LocalMIMICExtractor(config)
    elif source == "mimic":
        return MIMICExtractor(config)
    elif source == "amsterdam":
        return AmsterdamExtractor(config)
    elif source == "my_hospital":              # ← add this
        return MyHospitalExtractor(config)
    else:
        raise ValueError(f"Unknown data source: {source}")
```

### Step 3: Add config block

In `config/config.yaml`:

```yaml
data:
  source: "my_hospital"
  my_hospital:
    connection_string: "postgresql://user:pass@host:5432/mydb"
```

No other changes needed — the preprocessing, feature engineering, and all notebooks load whatever the extractor returns.

---

## 12. Full Function Reference

### `src/data/extraction.py`

| Function / Method | What It Does | Called By |
|---|---|---|
| `get_extractor(config)` | Factory — returns the right extractor based on `config["data"]["source"]` | `00_Data_Pipeline.ipynb`, `extract_data.py` |
| `LocalMIMICExtractor.__init__(config)` | Loads all 6 CSV.GZ files into memory | `get_extractor()` |
| `LocalMIMICExtractor.extract_chartevents()` | Pivots long chartevents → wide (one row per stay_id, hour) using `MIMIC_ITEMIDS` dict | `build_cohort()` |
| `LocalMIMICExtractor.extract_ventilation_episodes()` | Filters procedureevents for ITEMID 225792 (InvasiveMechVent), merges with patients/admissions | `build_cohort()` |
| `LocalMIMICExtractor.extract_labevents()` | Pivots labevents using `MIMIC_LAB_ITEMIDS` dict | `build_cohort()` |
| `MIMIC_ITEMIDS` (dict) | Maps column name → list of chartevents ITEMIDs (e.g., `"heart_rate": [211, 220045]`) | All extractors |
| `MIMIC_LAB_ITEMIDS` (dict) | Maps column name → labevents ITEMIDs (e.g., `"pao2": [50821]`) | Lab extraction |

### `src/data/preprocessing.py`

| Function | What It Does | Called By |
|---|---|---|
| `remove_outliers(df, config)` | Clips physiological values to valid clinical ranges (e.g., HR 20–300, SpO₂ 50–100) | `00_Data_Pipeline.ipynb` |
| `impute_missing(df, config)` | Forward-fills within stays, then fills with column medians | `00_Data_Pipeline.ipynb` |
| `resample_to_hourly(df, config)` | Groups by (stay_id, hour_index), takes mean per hour | `00_Data_Pipeline.ipynb` |

### `src/features/engineering.py`

| Function | What It Does | Called By |
|---|---|---|
| `add_derived_features(df, config)` | Computes MP (Gattinoni), driving_pressure, compliance, pf_ratio, tidal_volume_per_kg, predicted_body_weight | `00_Data_Pipeline.ipynb` |
| `compute_mechanical_power(row)` | Single-row MP computation: `0.098 × RR × VT(L) × (Ppeak − 0.5 × ΔP)` | `add_derived_features()` |

### `src/data/dataset.py`

| Function | What It Does | Called By |
|---|---|---|
| `discretise_action(mp_delta, action_bins)` | Maps observed MP change to nearest discrete action (0–4) | `build_episodes()` |
| `calculate_reward(state_t, state_t1, is_terminal, outcome, reward_cfg)` | Multi-objective reward: survival/death bonus + SpO₂ weight + safety penalties | `build_episodes()` |
| `build_episodes(df, config, feature_cols)` | Converts hourly time-series → list of episode dicts with `(s, a, r, s', done)` tuples | `03_Strategy3_CQL_Offline_RL.ipynb` |
| `split_episodes(episodes, config)` | Patient-level stratified split (train/val/test) by mortality outcome | `03_Strategy3_CQL_Offline_RL.ipynb` |
| `episodes_to_arrays(episodes)` | Flattens episode dicts → numpy arrays for training | `03_Strategy3_CQL_Offline_RL.ipynb` |

### `src/models/`

| Class / Function | What It Does | Called By |
|---|---|---|
| `strategy1_static.py → Strategy1Static.extract_features(df)` | Pivots parquet → one row per patient with t=0 and t=24h features | `01_Strategy1_XGBoost_Static.ipynb` |
| `strategy2_window.py → Strategy2TimeWindow.extract_features(df)` | Pivots parquet → per-patient with 5-snapshot MP trajectory features | `02_Strategy2_XGBoost_Continuous.ipynb` |
| `q_network.py → QNetwork` | PyTorch `nn.Module`: Linear → LayerNorm → ReLU → Dropout, outputs Q-values for each action | `cql_agent.py` |
| `cql_agent.py → CQLAgent` | CQL training loop: TD loss + conservative penalty (`logsumexp − Q_taken`) + soft target updates | `03_Strategy3_CQL_Offline_RL.ipynb` |

### `src/evaluation/metrics.py`

| Function | What It Does | Called By |
|---|---|---|
| `mortality_metrics(y_true, y_prob)` | Returns dict: AUROC, AUPRC, Brier score, ECE (calibration error) | Notebooks 01, 02 |
| `policy_return(episodes, agent)` | Mean cumulative reward of learned policy on test episodes | Notebook 03 |
| `safety_metrics(episodes, agent, safety_filter)` | Counts hard constraint violations, override rate | Notebook 03 |

### `src/evaluation/comparison.py`

| Function | What It Does | Called By |
|---|---|---|
| `build_comparison_table(s1_metrics, s2_metrics, s3_metrics)` | Returns a pandas DataFrame with side-by-side metrics | `03_Strategy3_CQL_Offline_RL.ipynb` (final section) |

### `src/deployment/safety_filter.py`

| Method | What It Does |
|---|---|
| `SafetyFilter.__init__(config)` | Loads all 8 thresholds from `config["deployment"]["safety"]` |
| `SafetyFilter.filter(patient_state, recommended_action)` | Applies 6 safety rules, returns `(safe_action, alerts)` |
| `SafetyFilter.check_alerts(patient_state)` | Returns informational warnings without overriding action |
| `SafetyFilter._is_unstable(state)` | Returns True if MAP < 60 or SpO₂ < 88 or HR > 130 or lactate > 4 |

**The 6 safety rules:**
1. Don't reduce support if SpO₂ < 85% (critically hypoxaemic)
2. Don't increase if plateau pressure > 30 cmH₂O (barotrauma risk)
3. Force decrease if driving pressure > 15 cmH₂O
4. Block if projected MP would exceed 30 J/min
5. Block if projected MP would fall below 5 J/min
6. Downgrade large changes to small changes if patient is haemodynamically unstable

### `src/deployment/explainer.py`

| Method | What It Does |
|---|---|
| `ExplanationGenerator.generate(patient_state, action, confidence, q_values, similar_outcomes)` | Returns a multi-sentence natural-language explanation for doctors |
| `ExplanationGenerator._key_factors(state, action)` | Identifies clinical factors driving the recommendation (MP level, SpO₂, pressures) |
| `ExplanationGenerator._alternatives(q_values, chosen)` | Describes the next-best action and Q-value margin |

---

## 13. Reproducibility Checklist

For researchers who want to reproduce published results:

### Seeds

All random seeds are set to 42 throughout:

```python
# In notebooks — set at the top of every notebook:
import numpy as np, random, torch
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Optuna:
sampler = optuna.samplers.TPESampler(seed=SEED)

# XGBoost:
params = {'random_state': SEED, ...}

# Scikit-learn:
StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
```

These are set in `config.yaml → training.random_seed: 42` and respected throughout.

### Package versions

Pin all versions by exporting your environment:

```bash
pip freeze > requirements_exact.txt
```

Or use conda:
```bash
conda env export > environment.yml
conda env create -f environment.yml
```

### Data versioning

MIMIC-IV demo v2.2 is the reference dataset. The version is in the folder name: `mimic-iv-clinical-database-demo-2.2/`. The full MIMIC-IV version should be noted in publications (currently v3.1).

### Expected results with demo data (n=76 stays, 50 with MP)

| Strategy | AUROC | AUPRC | Notes |
|----------|-------|-------|-------|
| S1 Static XGBoost | 0.70–0.80 | 0.45–0.60 | Small sample → wide CI |
| S2 Continuous XGBoost | 0.72–0.82 | 0.48–0.65 | Slight improvement from trajectory |
| S3 CQL Agent | N/A (policy) | N/A | Safety override rate < 5% |

Note: with n=50 patients having MP data, results will vary between runs even with fixed seeds due to stratified splits. The demo dataset is for development only — full MIMIC-IV (70K+ stays) produces stable results.

### Citation

If you use this codebase in a publication, please cite:

```bibtex
@software{mechanical_power_icu,
  author  = {Jatin Dangi},
  title   = {Mechanical Power Personalisation for ICU Patients},
  year    = {2024},
  url     = {https://github.com/your-org/mechanical_power}
}
```

---

## 14. Troubleshooting

### `ModuleNotFoundError: No module named 'loguru'`

```bash
pip install loguru pyyaml optuna shapiq pyarrow --break-system-packages -q
```

### `ModuleNotFoundError: No module named 'pyarrow'`

Required for `.parquet` read/write. Install with:
```bash
pip install pyarrow --break-system-packages
```

### `KeyError: 'mechanical_power'`

The parquet file was generated before MP was added to feature engineering, or your cohort has no stays with all required MP components (RR, VT, Ppeak, PEEP). Check:

```python
df = pd.read_parquet("data/processed/processed_cohort.parquet")
print("mp_stays:", df.groupby("stay_id")["mechanical_power"].apply(lambda x: x.notna().any()).sum())
```

Re-run Notebook 00 to regenerate the parquet.

### `TypeError: XGBClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'`

You have XGBoost 3.x. Move `early_stopping_rounds` to the constructor:

```python
# Wrong (XGBoost 3.x):
model.fit(X, y, early_stopping_rounds=20)

# Correct (works in both 2.x and 3.x):
model = xgb.XGBClassifier(early_stopping_rounds=20, **other_params)
model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
```

### SHAP-IQ is slow

With `sample_size=64` and `max_order=2`, SHAP-IQ takes ~5 seconds per patient. For faster development:

```python
# Speed up SHAP-IQ:
explainer = shapiq.TabularExplainer(
    model=predict_fn,
    data=X_background,
    index='SII',
    max_order=1,          # Only main effects (no interactions)
    sample_size=32,       # Fewer samples
)
```

For the doctor report, `max_order=1` is sufficient and 5-10× faster.

### `FileNotFoundError: No such file or directory: 'data/processed/processed_cohort.parquet'`

Run Notebook 00 first. The parquet is generated by that notebook and is not included in the repository (it's 10–500 MB depending on the dataset).

### Notebook 03 CQL training is very slow

Reduce the number of training epochs in the Optuna search bounds:

```python
n_epochs = trial.suggest_int('n_epochs', 5, 20)   # was 5–50
```

Or reduce `cql_optuna_trials` in `config.yaml` from 20 to 5 for quick testing.

### `OperationalError` or `google.auth` errors when using BigQuery

You're likely using `source: "mimic"` but don't have Google Cloud credentials set up. Switch to offline mode:

```yaml
data:
  source: "mimic_local"
  mimic_local:
    data_dir: "mimic-iv-clinical-database-demo-2.2"
```

---

## 15. Key References

**Mechanical Power:**
- Gattinoni L, et al. (2016). *Ventilator-related causes of lung injury: the mechanical power.* Intensive Care Med. [doi:10.1007/s00134-016-4505-2](https://doi.org/10.1007/s00134-016-4505-2)
- Amato M, et al. (2015). *Driving pressure and survival in ARDS.* NEJM. [doi:10.1056/NEJMsa1410639](https://doi.org/10.1056/NEJMsa1410639)

**Offline Reinforcement Learning:**
- Komorowski M, et al. (2018). *The AI Clinician learns optimal treatment strategies for sepsis in intensive care.* Nature Medicine. [doi:10.1038/s41591-018-0213-5](https://doi.org/10.1038/s41591-018-0213-5)
- Kumar A, et al. (2020). *Conservative Q-Learning for Offline Reinforcement Learning.* NeurIPS. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)

**Explainability:**
- Fumagalli N, et al. (2023). *SHAP-IQ: Unified Approximation of Any-order Shapley Interactions.* NeurIPS. [arXiv:2303.01179](https://arxiv.org/abs/2303.01179)

**MIMIC-IV:**
- Johnson A, et al. (2023). *MIMIC-IV, a freely accessible electronic health record dataset.* Scientific Data. [doi:10.1038/s41597-022-01899-x](https://doi.org/10.1038/s41597-022-01899-x)

**Causal Inference:**
- Hernán M, Robins J. (2020). *Causal Inference: What If.* Chapman & Hall. [Online](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)

---

## Contact

**Jatin Dangi**
Ashoka University
[jatin@ashoka.edu.in](mailto:jatin@ashoka.edu.in)

---

*This project is for research purposes only. The system is not approved for clinical use and must not be used to make treatment decisions without physician oversight.*
