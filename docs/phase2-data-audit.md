# Phase 2 — Data Audit (after date-boundary correction)

Generated with `python synthetic_data_generator/generate.py` using seed 42.

## Structural checks

| Check | Training | Holdout | Result |
|---|---:|---:|---|
| Rows | 60,000 | 8,000 | PASS |
| Columns | 26 | 25 | PASS |
| Target column | Present | Absent | PASS |
| Duplicate admission IDs | 0 | 0 | PASS |
| Admission ID overlap | — | 0 | PASS |
| Admission date window | 2021-01-01 to 2023-12-31 | 2024-01-01 to 2024-03-31 | PASS |

The training admission-date window now covers the full assignment-specified period. Some training admissions discharge in January 2024 because discharge date is admission date plus LOS; the assignment's date windows are admission-date windows.

## Corrected distributions (training)

| Measure | Observed | Design target | Assessment |
|---|---:|---:|---|
| LOS 1–2 days | 24.70% | 25% | Close |
| LOS 3–4 days | 34.68% | 35% | Close |
| LOS 5–7 days | 25.18% | 25% | Close |
| LOS 8–14 days | 12.28% | 12% | Close |
| LOS 15+ days | 3.16% | 3% | Close |
| Home | 54.34% | 55% | Close |
| Home with services | 20.23% | 20% | Close |
| Skilled nursing | 15.22% | 15% | Close |
| Rehab | 10.22% | 10% | Close |

## History and chronology

- Exactly 30.0% of patients across the combined generated population have multiple admissions.
- 2,000 patient IDs occur in both training and holdout; 6,000 holdout patients are new relative to training.
- No overlapping stays were found. The minimum gap between a previous discharge and the next admission is 35 days.
- `n_prior_admissions_12m` matched a recomputation from earlier admissions for every training row.
- The observed training distribution of `n_prior_admissions_12m` is 0: 60.83%, 1: 19.02%, 2: 7.08%, 3+: 13.06%.

**Important distribution caveat:** the agreed prior-admission marginal target (60%, 25%, 10%, 5%) cannot be independently imposed without conflicting with the 30% multi-admission patient assumption and the chosen patient-level admission-count mix. The feature is derived from actual synthetic history, not independently sampled to force its marginal distribution. Keep the observed distribution transparent in the decision log.

## Other checks

- Training readmission prevalence: 15.28% (target approximately 15%).
- Holdout internal prevalence diagnostic: 12.54%; labels are not exported and must not be used for model selection or tuning.
- Missingness in the five specified lab/vital fields is approximately 4%: creatinine 3.99%, sodium 4.02%, hemoglobin 3.97%, systolic BP 3.84%, heart rate 4.14% in training.
- No duplicate admission IDs, invalid category values, admission/discharge duration mismatches, or weekend-flag mismatches were found.
- All training admission dates are within 2021-01-01 and 2023-12-31; all holdout admission dates are within 2024-01-01 and 2024-03-31.

## Conclusion

The training date boundary has been corrected to cover the full assignment-specified period. Row counts, schemas, date logic, patient histories, LOS and disposition distributions, and the required missingness pattern pass the audited checks. The prior-admission marginal distribution remains different from its initial target for the documented structural reason above. Phase 2 data audit is complete; retain the caveats and feature-availability assumptions in the decision log before model development.
