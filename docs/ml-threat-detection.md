# ML Threat Detection

ThreatLens now uses a XGBoost classifier for the core threat assessment instead of the old hardcoded score/classification formula.

## What the model predicts
- Threat probability (0–100)
- `GENUINE_THREAT`, `INVESTIGATING`, or `LIKELY_FALSE_POSITIVE`
- `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW` priority

## Training design
The training corpus contains 44 behaviour scenarios, including malicious behaviours and benign operational activity. Each scenario is varied with source, severity, entity, numeric and contextual noise. It is intentionally broader than the original 17 demo examples.

The training process uses:
1. Stratified held-out test set (15%).
2. Train/validation split from the remaining data.
3. 5-fold stratified cross-validation on the training portion.
4. Small hyperparameter search over tree depth, leaf size, feature sampling and estimator count.
5. Final evaluation on the untouched test set.

The generated `ml/model_metrics.json` records the actual validation/test metrics and train-to-test F1 gap. The application never uses the held-out test data for model selection.

## Runtime architecture

`raw feeds -> normalization -> correlation evidence -> ML model -> threat probability + classification + priority -> incident -> MITRE/BLUF`

The existing deterministic correlation guardrail remains responsible for deciding which alerts belong to an incident. The threat assessment itself is ML-driven.

## Important demo limitation
The included corpus is synthetic and broad enough to demonstrate the ML pipeline. Its metrics must not be presented as real-world SOC accuracy. For production deployment, retrain on analyst-labelled historical SIEM/EDR/network/intelligence data and monitor drift/calibration.

## Model API
`GET /api/v1/ml/model-info` returns the model name, feature list, cross-validation metrics, validation metrics, held-out test metrics and generalization gap.
