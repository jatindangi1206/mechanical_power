# Mechanical Power Personalisation for ICU Patients

A Clinical Decision Support System that uses Offline Reinforcement Learning to recommend personalised Mechanical Power (MP) adjustments for mechanically ventilated ICU patients.

## Problem

Doctors adjust mechanical ventilation settings based on limited personal experience (~500–600 patients), but optimal settings vary by patient characteristics. This project builds an AI system that learns from thousands of ICU patient records to recommend personalised MP adjustments that minimise mortality while maintaining safe oxygenation.

## Approach

**Offline Reinforcement Learning** (Conservative Q-Learning) combined with **Causal Inference** to learn optimal treatment policies from historical data.

Three progressively complex strategies are implemented:

| Strategy | Method | Description |
|----------|--------|-------------|
| 1 — Static | XGBoost | MP at t=0 & t=24 → Mortality risk |
| 2 — Time-Window | XGBoost | MP snapshots at 0h, 6h, 12h, 18h, 24h → Mortality risk |
| 3 — Sequential RL | CQL Agent | Learn optimal MP adjustment policy over time |

## Project Structure

```
mechanical_power/
├── config/
│   └── config.yaml                 # All project configuration
├── src/
│   ├── data/
│   │   ├── extraction.py           # MIMIC-IV / AmsterdamUMC data extraction
│   │   ├── preprocessing.py        # Cleaning, imputation, normalisation
│   │   └── dataset.py              # MDP dataset creation for RL
│   ├── features/
│   │   └── engineering.py          # MP calculation, derived variables
│   ├── models/
│   │   ├── state_encoder.py        # LSTM-based patient state encoder
│   │   ├── q_network.py            # Q-value network
│   │   ├── cql_agent.py            # Conservative Q-Learning agent
│   │   ├── strategy1_static.py     # Baseline static XGBoost
│   │   └── strategy2_window.py     # Time-window XGBoost
│   ├── evaluation/
│   │   ├── metrics.py              # AUROC, AUPRC, safety metrics
│   │   ├── comparison.py           # Cross-strategy comparison
│   │   └── clinician_validation.py # Clinician survey framework
│   ├── deployment/
│   │   ├── api.py                  # FastAPI inference endpoint
│   │   ├── safety_filter.py        # Hard safety constraints
│   │   └── explainer.py            # Recommendation explanations
│   └── utils/
│       └── helpers.py              # Shared utilities
├── scripts/
│   ├── extract_data.py             # CLI: run data extraction
│   ├── train.py                    # CLI: train all strategies
│   └── evaluate.py                 # CLI: run evaluation suite
├── tests/
│   ├── test_preprocessing.py
│   └── test_safety_filter.py
├── notebooks/                      # Exploration & visualisation
├── requirements.txt
└── setup.py
```

## Data Sources

| Database | Size | Access |
|----------|------|--------|
| **MIMIC-IV** (primary) | 70,000+ ICU admissions | [PhysioNet](https://physionet.org/content/mimiciv/) — free after CITI training |
| **Amsterdam UMCdb** | 20,000+ ICU admissions | Data use agreement required |
| **HiRID** | 33,000+ ICU admissions | [PhysioNet](https://physionet.org/content/hirid/) |

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-org/mechanical_power.git
cd mechanical_power
pip install -e .

# 2. Configure database credentials
cp config/config.yaml config/config.local.yaml
# Edit config.local.yaml with your MIMIC-IV credentials

# 3. Extract and preprocess data
python scripts/extract_data.py --source mimic --config config/config.local.yaml

# 4. Train all strategies
python scripts/train.py --config config/config.local.yaml

# 5. Evaluate
python scripts/evaluate.py --config config/config.local.yaml
```

## Inclusion Criteria

- Invasive mechanical ventilation ≥ 24 hours
- Age ≥ 18 years
- Complete outcome data
- **Excluded:** ECMO, comfort care within 24h, traumatic brain injury

## Success Criteria

| Level | AUROC | Expert Agreement | Notes |
|-------|-------|------------------|-------|
| MVP | > 0.80 | > 60% | No safety violations |
| Strong | > 0.85 | > 70% | Improvement over observed policy |
| Exceptional | > 0.88 | > 75% | Ready for prospective validation |

## Key References

- Komorowski et al. (2018). *The AI Clinician learns optimal treatment strategies for sepsis.* Nature Medicine.
- Amato et al. (2015). *Driving pressure and survival in ARDS.* NEJM.
- Hernán & Robins (2020). *Causal Inference: What If.*
- Sendak et al. (2020). *Real-world integration of a sepsis deep learning technology.* NPJ Digital Medicine.

## License

This project is for research purposes. See LICENSE for details.

## Contact

jatin@ashoka.edu.in
