<div align="center">

# F1 Pit Stop Effectiveness Predictor

**An end-to-end ML pipeline that predicts whether an F1 pit stop will be strategically effective — trained on 3,100+ real race events across 60+ Grand Prix.**

ROC-AUC 87.4% · Precision 80% · Recall 81% · SHAP-explained predictions

[![Live App](https://img.shields.io/badge/Live_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://f1-pitstop-predictor-ig.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32?style=flat-square)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=flat-square)](https://shap.readthedocs.io/)
[![FastF1](https://img.shields.io/badge/FastF1-Telemetry-e4002b?style=flat-square)](https://theoehrly.github.io/Fast-F1/)

</div>

---

## What This Is

A classification model that answers one question: given the race state at the moment a team considers pitting, will that pit stop improve the driver's relative position? This is a non-trivial prediction — it depends on tyre degradation rate, the undercut/overcut window, traffic in the pit lane, stint history, and circuit-specific pace characteristics.

The project is end-to-end: automated telemetry ingestion via FastF1, feature engineering, cross-validated LightGBM training, and a production Streamlit app with SHAP force plot integration for per-prediction interpretability.

**Live app:** https://f1-pitstop-predictor-ig.streamlit.app/

---

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | **87.4%** |
| Precision | 80% |
| Recall | 81% |
| F1 Score | 80.6% |
| Accuracy | 79.6% |
| Holdout size | 3,183 pit stop events |

ROC-AUC of 87.4% on F1 pit stop data is meaningful — the baseline (always predict the majority class) achieves around 50% AUC. The model is correctly ordering effective vs. ineffective stops with genuine discriminative power.

---

## Pipeline Architecture

```
FastF1 API
    │  Race telemetry: lap times, tyre compounds,
    │  stint lengths, positional data — 60+ races (2022–2024)
    ▼
Data Ingestion Layer
    │  Automated per-race download
    │  Lap-wise sequence construction
    │  Race-level joins (weather, circuit metadata)
    ▼
Feature Engineering (19 features)
    │  Tyre degradation rate        — pace loss per lap on current compound
    │  Undercut delta               — predicted time gain from pitting now vs. staying out
    │  Stint length                 — laps completed on current set
    │  Compound age relative to typical stint length for that compound/circuit
    │  Gap to car ahead / behind    — traffic context
    │  Current track position       — whether undercut is viable
    │  Circuit-specific pace model  — Monaco vs Bahrain have fundamentally different strategies
    │  ... + 12 more domain features
    ▼
Class Balancing
    │  Effective stops are not evenly distributed.
    │  Applied class weighting in LightGBM to prevent
    │  the model collapsing to always predicting "ineffective".
    ▼
LightGBM Classifier
    │  Cross-validated hyperparameter tuning
    │  GridSearchCV over: n_estimators, max_depth,
    │  learning_rate, min_child_samples, subsample
    ▼
Evaluation on holdout (3,183 events)
    │  ROC-AUC 87.4%
    │  Precision 80%, Recall 81%
    ▼
SHAP Integration
    │  TreeExplainer on LightGBM model
    │  Force plots for per-prediction feature attribution
    │  Summary plots for global feature importance
    ▼
Streamlit App
    └── Real-time predictions + SHAP force plots
        + driver/circuit stratified comparison views
```

---

## Why LightGBM

LightGBM was chosen over alternatives for several concrete reasons:

**vs. XGBoost:** LightGBM uses leaf-wise tree growth (grows the leaf with the highest loss reduction) rather than level-wise. On tabular F1 telemetry data with mixed feature types (continuous tyre degradation rates + categorical compound names + integer stint lengths), this consistently produces better accuracy with faster training.

**vs. a neural network:** With 3,100 samples and 19 features, a neural network would overfit heavily. Gradient boosted trees handle mixed feature types natively, require no normalisation, and are substantially easier to interpret via SHAP.

**vs. Random Forest:** LightGBM's boosting sequentially corrects errors from prior trees. For a prediction problem where the signal is subtle (a 0.8s undercut window vs a 1.2s window), boosting better captures the non-linear thresholds that separate effective from ineffective stops.

---

## SHAP Explainability

The app integrates SHAP TreeExplainer, which produces exact Shapley values for LightGBM (not approximate). For every prediction:

- A **force plot** shows which features pushed the prediction toward "effective" (red) or "ineffective" (blue) and by how much
- The **base value** is the model's expected output across the training set
- Each feature's SHAP value represents its marginal contribution to the deviation from that base value

This is not decorative. For a pit strategy recommendation to be trusted by a team, the reasoning has to be legible. SHAP provides that — "this stop was flagged as high-risk primarily because tyre degradation rate was low (you're not in a degradation window) and the gap to the car behind is 8 seconds (no threat)."

---

## Data

- **Source:** FastF1 API — official F1 telemetry
- **Races:** 60+ Grand Prix across 2022, 2023, 2024 seasons
- **Samples:** 3,183 pit stop events in holdout set
- **Target variable:** Binary — did this pit stop result in net position gain within the next 5 laps?

The ingestion pipeline is designed to be reusable across seasons with minimal reconfiguration — adding a new season requires updating the race round range, not restructuring the pipeline.

---

## Project Structure

```
f1-pitstop-predictor/
├── data/
│   ├── raw/                    ← FastF1 cache (gitignored, ~500MB)
│   └── processed/
│       └── pitstops_features.csv
├── notebooks/
│   ├── 01_ingestion.ipynb      ← FastF1 download + lap construction
│   ├── 02_features.ipynb       ← Feature engineering
│   ├── 03_training.ipynb       ← LightGBM + cross-validation
│   └── 04_evaluation.ipynb     ← Metrics + SHAP analysis
├── model/
│   └── lgbm_pitstop.pkl        ← Trained model
├── app.py                      ← Streamlit inference app
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/Ishaan2510/f1-pitstop-predictor
cd f1-pitstop-predictor

pip install -r requirements.txt

streamlit run app.py
```

To retrain from scratch, run the notebooks in order (01 → 04). FastF1 downloads race data on first run and caches locally. Expect ~500MB for 3 seasons.

---

## Limitations

**Data freshness.** The model is trained on 2022–2024 data. The 2026 regulations introduced significant car design changes — tyre behaviour and undercut windows may differ enough that retraining on 2025–2026 data would be warranted before deploying as a real strategy tool.

**No real-time inference.** The current app is a prediction tool given manually entered race state, not a live feed that ingests telemetry during a race. Building that would require a streaming data source (the FastF1 live timing client) and a low-latency serving layer.

**Target variable approximation.** "Effective" is defined as net position gain within 5 laps. This is a reasonable proxy but doesn't capture longer-term strategic outcomes — a stop that costs position in lap 28 but wins the race by lap 57 would be labelled "ineffective" in this dataset.

---

*Built by [Ishaan Goswami](https://github.com/Ishaan2510) — CS undergrad, PDEU + IIT Madras*
