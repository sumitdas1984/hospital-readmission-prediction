# Synthetic Data Generation Specification

## Purpose and source constraints

Generate a reproducible, entirely synthetic dataset for the Lakeshore Health Network 30-day readmission assignment. The assignment requires approximately 60,000 labelled training admissions (January 2021–December 2023), approximately 8,000 holdout admissions (January–March 2024), 26 training columns, and 25 holdout columns. The holdout file must not contain `readmitted_30d`.

All distributions and clinical relationships below are design assumptions agreed for this synthetic exercise; they are not claimed to represent real hospital statistics.

## Confirmed design choices

- Random seed: `42`.
- Target prevalence: approximately 15% across generated admissions, achieved by calibrating the intercept of a multi-factor probabilistic outcome model and then sampling binary outcomes. Period-specific prevalence may vary naturally.
- Approximately 30% of synthetic patients have multiple admissions. Patient IDs are stable across periods where a patient has admissions in both periods; age and sex are patient-level attributes. Age advances with elapsed time. The holdout includes both previously seen patients and new patients.
- Site mix: Site_1 25%, Site_2 20%, Site_3 18%, Site_4 15%, Site_5 12%, Site_6 10%.
- Age bands: 18–35 15%, 36–50 20%, 51–65 25%, 66–80 25%, >80 15%; minimum age 18.
- Sex: F 52%, M 48%.
- Insurance: Medicare 45%, Commercial 30%, Medicaid 20%, Self-pay 5%.
- Admission source: Emergency 60%, Referral 25%, Transfer 15%.
- Primary diagnosis mix: Cardiovascular 15%, Respiratory 12%, Endocrine / Diabetes 12%, Renal 8%, Gastrointestinal 9%, Neurological 7%, Musculoskeletal 7%, Infectious Diseases 10%, Oncology 6%, Mental Health 5%, Hematology 3%, Other 6%.
- Comorbidity count: 0 (15%), 1–2 (35%), 3–4 (30%), 5–6 (15%), 7+ (5%). Older patients tend to have more comorbidities.
- Length of stay: 1–2 days 25%, 3–4 days 35%, 5–7 days 25%, 8–14 days 12%, >14 days 3%; minimum 1 day.
- Prior admissions in preceding 12 months: intended approximate mix 0 (60%), 1 (25%), 2 (10%), 3+ (5%). Calculated strictly from earlier generated admissions. The realized distribution is constrained by the separate requirement that 30% of patients have multiple admissions and by the per-patient admission-count design; observed values are reported in the Phase 2 audit rather than forced independently.
- ED visits in preceding 6 months: 0 (55%), 1 (25%), 2 (12%), 3+ (8%). Correlated with prior admissions and based only on history before the current admission.
- Discharge medications: 0–4 (15%), 5–9 (35%), 10–14 (30%), 15–19 (15%), 20+ (5%). Older age and more comorbidities tend to increase counts.
- BMI bands: <18.5 (5%), 18.5–24.9 (25%), 25–29.9 (30%), 30–34.9 (20%), 35–39.9 (12%), >=40 (8%); generated range approximately 14–55.
- Creatinine: approximately 0.4–5.0 mg/dL; sodium 125–150 mmol/L; hemoglobin 6–18 g/dL; systolic BP 75–210 mmHg; heart rate 40–150 bpm. Each of these five fields has approximately 4% missing values.
- Disposition target mix: Home 55%, Home with services 20%, Skilled nursing 15%, Rehab 10%; the generator uses these as baseline probabilities with small severity-related shifts, so observed proportions are approximate.
- `weekend_discharge` is derived from `discharge_date` (Saturday/Sunday = 1).
- `followup_visit_scheduled`: approximately 65%, generated as information known by discharge.
- `post_discharge_call_logged`: approximately 30% positive. Retained in both raw datasets but excluded from model features because it may occur after the discharge-time prediction point.
- Prediction moment: discharge. `admission_id` is an identifier, `patient_id` is retained for traceability but not a model predictor, and `readmitted_30d` is the training target only.

## Temporal and history rules

Admission dates are generated in the required windows. Discharge dates are admission date plus length of stay and may extend beyond the admission-date window. Admissions are processed chronologically. A patient’s next admission is scheduled no earlier than 56 days after the prior admission date, allowing up to 21 days of stay plus a 35-day separation. Prior-admission and ED-visit counts never use future events. Holdout allocation includes 2,000 repeat patients also represented in training and 6,000 single-admission patients represented only in holdout. Training labels represent the 30 days after discharge, even when that observation window extends into 2024. Holdout outcomes are generated internally to preserve a realistic synthetic process, then omitted from the holdout CSV; they are not available to model development or evaluation.

## Reproducibility and output

Run `python synthetic_data_generator/generate.py` from the repository root. It writes:

- `data/raw/train.csv` — 60,000 rows and all 26 schema columns.
- `data/raw/holdout.csv` — 8,000 rows and the 25 non-target columns.

The script prints row counts, target prevalence for training (and an internal holdout prevalence diagnostic), missingness rates, and repeated-patient share. No external data or pretrained models are used.


## Date-boundary clarification

Training admission dates are generated across the complete inclusive window 2021-01-01 through 2023-12-31. Holdout admission dates are generated across 2024-01-01 through 2024-03-31. Discharge dates are calculated as admission date plus LOS, so a small number of training admissions may discharge in January 2024; the specified periods apply to admission dates.
