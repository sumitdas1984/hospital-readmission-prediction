# Synthetic Data Generation Specification

## Purpose
Create reproducible synthetic hospital-admission data for the Lakeshore Health Network 30-day readmission assignment. No real patient data, external datasets, lookup tables, or pretrained models are used.

## Dataset sizes and time windows
- Training: 60,000 admissions, admission dates from 2021-01-01 through 2023-12-31, with all 26 schema columns including `readmitted_30d`.
- Holdout: 8,000 admissions, admission dates from 2024-01-01 through 2024-03-31, with 25 columns and no `readmitted_30d`.
- Random seed: 42.
- Admission IDs are unique across the combined datasets.
- Approximately 30% of the synthetic patient population is assigned multiple admissions.

## Prediction moment and leakage controls
The intended prediction moment is hospital discharge.
- `admission_id`: identifier only; not a model feature.
- `patient_id`: identifier/history key; exclude as a direct model feature.
- `readmitted_30d`: training target only; never a predictor.
- `post_discharge_call_logged`: retained in the schema, but excluded from model features because the call may be logged after discharge.
- `followup_visit_scheduled`: generated as known at or before discharge.
- Prior admissions and ED visits refer only to events before the current admission.
- `weekend_discharge` is derived from `discharge_date`.

## Population and categorical assumptions
These are synthetic design assumptions, not statistics supplied by the assignment.

| Field | Assumption |
|---|---|
| Age groups | 18–35: 15%; 36–50: 20%; 51–65: 25%; 66–80: 25%; >80: 15% |
| Sex | F: 52%; M: 48% |
| Hospital site | Site_1: 25%; Site_2: 20%; Site_3: 18%; Site_4: 15%; Site_5: 12%; Site_6: 10% |
| Insurance | Medicare: 45%; Commercial: 30%; Medicaid: 20%; Self-pay: 5% |
| Admission source | Emergency: 60%; Referral: 25%; Transfer: 15% |
| Diagnosis | Cardiovascular 15%; Respiratory 12%; Endocrine/Diabetes 12%; Renal 8%; Gastrointestinal 9%; Neurological 7%; Musculoskeletal 7%; Infectious Diseases 10%; Oncology 6%; Mental Health 5%; Hematology 3%; Other 6% |
| Disposition | Home 55%; Home with services 20%; Skilled nursing 15%; Rehab 10% |
| Follow-up scheduled | Approximately 65% overall; probability varies modestly with age and comorbidity |
| Post-discharge call logged | Approximately 30%; excluded from predictors |

## Numeric feature assumptions
All values are synthetic and are generated with plausible ranges and dependencies rather than claimed clinical reference distributions.

| Field | Generation approach |
|---|---|
| `comorbidity_count` | Integer, non-negative, tends to increase with age |
| `los_days` | Integer, minimum 1; generated from a skewed distribution with complexity effects |
| `n_prior_admissions_12m` | Derived from earlier generated admissions for the same patient within 365 days |
| `n_ed_visits_6m` | Non-negative count, correlated with prior admissions and patient utilization propensity |
| `num_medications` | Non-negative count, tends to increase with age and comorbidity |
| `bmi` | Continuous mixture across the agreed BMI bands; approximately 14–55 |
| `lab_creatinine_last` | Approximately 0.4–5.0 mg/dL |
| `lab_sodium_last` | Approximately 125–150 mmol/L |
| `lab_hemoglobin_last` | Approximately 6–18 g/dL |
| `systolic_bp_last` | Approximately 75–210 mmHg |
| `heart_rate_last` | Approximately 40–150 beats/minute |
| Missing clinical values | Approximately 4% missing independently in each of the five lab/vital fields |

## Target generation
`readmitted_30d` is generated from a multi-factor logistic probability using age, comorbidity, prior utilization, length of stay, medication count, diagnosis, selected clinical measurements, admission source, discharge disposition, BMI, and follow-up scheduling, with random noise. The intercept is calibrated so expected overall prevalence is approximately 15%; the realized rate varies through Bernoulli sampling. The same generation mechanism is used for both periods. Holdout labels are generated internally for synthetic consistency diagnostics but are not written to `holdout.csv` and must not be used for training, model selection, or tuning.

## Important interpretation
This is a synthetic dataset designed to exercise data auditing, leakage prevention, temporal validation, ranking under a 5% outreach constraint, and probability calibration. It is not a clinically validated simulation and must not be presented as representing real hospital populations or clinical effect sizes.
