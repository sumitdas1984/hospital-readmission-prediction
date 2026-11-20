# Phase 3 — Step 3: Model Development

## Models trained

1. Logistic Regression baseline (`C=1.0`, `max_iter=1000`).
2. Random Forest, using three limited configurations.

All models use the same 20 features and the Step 2 preprocessing design. The models were fitted on the 2021–2022 training partition and evaluated on the Jan–Jun 2023 validation partition.

## Validation results

| Model | Precision@5% | Recall@5% | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (baseline) | 0.552 | 0.175 | 0.399 | 0.752 | 0.116 |\n| Random Forest candidate 1 | 0.516 | 0.163 | 0.376 | 0.740 | 0.119 |\n| Random Forest candidate 2 | 0.514 | 0.163 | 0.372 | 0.739 | 0.118 |\n| Random Forest candidate 3 | 0.530 | 0.168 | 0.375 | 0.742 | 0.118 |\n
## Random Forest selection

Selected parameters: `{'n_estimators': 200, 'max_depth': 15, 'min_samples_leaf': 10, 'max_features': 'sqrt'}`

Selection rule: highest validation Precision@5%; ties are broken by Recall@5%, then PR-AUC. This aligns model selection with the team's 5% outreach capacity. The three RF candidates are a small, predeclared search rather than an extensive tuning exercise.

## Interpretation and caveats

- The validation set contains 9,910 admissions; the top-5% call list contains 496 admissions.
- Precision@5% is the fraction of the selected top 5% who were readmitted.
- Recall@5% is the fraction of all readmitted admissions captured in that top 5%.
- PR-AUC and ROC-AUC provide broader ranking summaries.
- Brier score is reported as an initial probability-quality diagnostic; formal calibration analysis belongs in Phase 4.
- These are synthetic data results, not evidence of clinical effectiveness.
- No class weighting was used.
- The internal test partition and Jan–Mar 2024 holdout were not used for this model selection run.

## Reproduction

From repository root:

```bash
uv run python code/train_models.py
```

Outputs:
- `models/logistic_regression_baseline.joblib`
- `models/random_forest_selected.joblib`
- `results/validation_model_comparison.csv`
- `results/model_selection.json`
- `results/validation_predictions.csv`

## Decision

Retain Logistic Regression as the baseline and the selected Random Forest as the nonlinear candidate. Phase 4 should evaluate both on the untouched Jul–Dec 2023 internal-test partition, examine calibration, and document the final model recommendation before any holdout scoring.
