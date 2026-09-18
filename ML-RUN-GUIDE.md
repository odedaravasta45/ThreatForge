# ThreatLens ML Upgrade — Run Guide

## Backend

```bash
cd src/backend
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

On first start, the included broad synthetic corpus trains the XGBoost and creates:
- `src/backend/ml/threat_model.joblib`
- `src/backend/ml/model_metrics.json`

The pre-trained artifact is included in this ZIP, so normal startup loads it. Delete those two files if you want to retrain.

## Frontend

```bash
cd src/frontend
npm install
npm run dev
```

## ML verification

Open:

`http://localhost:8000/api/v1/ml/model-info`

This returns the actual cross-validation, validation and untouched-test metrics, plus the train-to-test F1 gap.

## What changed

- Core threat assessment is XGBoost ML, not the old hardcoded threat-score formula.
- Priority is predicted by a second XGBoost classifier.
- Training corpus contains 44 malicious/benign behaviour scenarios and is varied with noise.
- The original demo's 17 examples are not the model's threat universe.
- Correlation guardrails remain deterministic because correlation is a separate task from threat classification.
- MITRE mapping remains explainable keyword mapping in this version; it is not falsely labelled as ML.
