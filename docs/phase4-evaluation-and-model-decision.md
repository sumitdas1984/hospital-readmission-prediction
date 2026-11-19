# Phase 4 — Internal-test evaluation and probability assessment

## Scope and protocol
The models selected in Phase 3 were evaluated on the untouched July–December 2023 internal-test partition (6,767 admissions). The 2024 holdout was not read, scored, or used. The test partition contains 990 readmissions (14.63%). At 5% capacity, the call list contains `ceil(0.05 × 6,767) = 339` admissions.

## Internal-test results

| Model | Precision@5% | Recall@5% | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 46.31% | 15.86% | 0.3412 | 0.7432 | 0.1119 |
| Random Forest | 45.72% | 15.66% | 0.3280 | 0.7302 | 0.1133 |

Logistic Regression leads the selected Random Forest on every reported internal-test metric. At the 339-slot capacity, approximately 157 of the selected admissions were readmitted within 30 days (precision 46.31%). The model captures 15.86% of all readmissions in the test partition. This is ranking performance, not evidence that an individual score is a perfectly calibrated probability.

## Comparison with validation
Both models' metrics declined from the January–June 2023 validation period to the July–December 2023 internal test. Logistic Regression Precision@5% moved from 55.24% to 46.31%, PR-AUC from 0.3989 to 0.3412, and ROC-AUC from 0.7522 to 0.7432. The direction is consistent with some temporal performance degradation; it does not by itself identify the cause. Logistic Regression remains ahead of Random Forest on both periods.

## Probability interpretation
On the internal test, Logistic Regression has Brier score 0.1119 versus 0.1133 for Random Forest (lower is better). In Logistic Regression's probability deciles, the mean predicted probability and observed event rate are reasonably ordered, but they do not match perfectly. For example, the lowest decile averages 3.19% predicted versus 2.22% observed; the highest decile averages 42.88% predicted versus 40.47% observed. The middle deciles show some underprediction/overprediction. Therefore, treat the output as a risk estimate that supports ranking and prioritization, not as a guaranteed individual probability. The decile table and calibration plot are included in `results/`.

No post-hoc calibration method was fitted using the internal-test labels. This preserves the test set as an evaluation set. If calibrated probabilities are required before operational use, fit a calibration method using a separate calibration/validation dataset and re-evaluate it on a later untouched temporal dataset.

## Model decision
Select **Logistic Regression** as the Phase 4 candidate. It is simpler and outperformed the selected Random Forest across validation and internal-test metrics, including the primary capacity-constrained metrics and Brier score. Do not claim the Random Forest is stronger. Keep the 2024 holdout untouched for the final required prediction file.

## Limitations and interpretation
- Data are synthetic; these results do not establish clinical effectiveness or generalization to real hospitals.
- The synthetic label-generation mechanism may make patterns easier or different from real-world readmission risk.
- Temporal performance declined between validation and internal test, so monitoring and prospective evaluation would be necessary.
- Precision@5% depends on the 5% capacity assumption and the test-period prevalence.
- Scores should support, not replace, clinical judgment; do not interpret associations or model coefficients as causal effects.
