# Phase 3 — Step 1: Temporal Data Split

## Split definitions and results

| Partition | Admission date range | Rows | Unique patients | Readmission prevalence |
|---|---|---:|---:|---:|
| train | 2021-01-01 to 2022-12-31 | 43,323 | 26,090 | 15.26% |\n| validation | 2023-01-01 to 2023-06-30 | 9,910 | 7,737 | 15.82% |\n| internal_test | 2023-07-01 to 2023-12-31 | 6,767 | 5,986 | 14.63% |\n
## Patient overlap

| Earlier → Later | Shared patients | Share of later-period patients |
|---|---:|---:|
| train->validation | 2,188 | 28.28% |\n| train->internal_test | 539 | 9.00% |\n| validation->internal_test | 1,125 | 18.79% |\n
Patient overlap is intentional in this temporal evaluation: returning patients can appear in earlier and later periods, as they would in deployment. `patient_id` will not be used directly as a predictor. Historical utilization features must be calculated only from events preceding the current admission.

## Holdout protection

The Jan–Mar 2024 holdout was not read or used by the split script. It remains reserved for final scoring after model selection.

## Outputs

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/internal_test.csv`
- `data/processed/temporal_split_summary.csv`

Recreate from the repository root:

```bash
uv run python code/create_temporal_splits.py
```

## Decision

Use the chronological partitions for Phase 3. Fit preprocessing and models on the training partition, use validation for limited hyperparameter selection, and reserve internal test for the final comparison between Logistic Regression and Random Forest.
