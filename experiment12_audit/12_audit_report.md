# Experiment 12 — Integrity audit

Automated checks are limited: PASS does not certify absence of leakage.

## Findings

- **PASS — Source audit:** Python source parses. 01_dataset_inspection.py
- **PASS — Source audit:** Python source parses. 02_eeg_exploration.py
- **PASS — Source audit:** Python source parses. 03_frequency_analysis.py
- **PASS — Source audit:** Python source parses. 04_motor_imagery_epochs.py
- **PASS — Source audit:** Python source parses. 05_within_subject_baseline.py
- **PASS — Source audit:** Python source parses. 06_cross_subject_baseline.py
- **PASS — Source audit:** Python source parses. 07_full_cross_subject_evaluation.py
- **PASS — Source audit:** Python source parses. 08_methodological_validation.py
- **PASS — Source audit:** Python source parses. 09_class_conditional_covariance.py
- **PASS — Source audit:** Python source parses. 10_riemannian_cross_subject.py
- **PASS — Source audit:** Python source parses. 11_neural_representation_analysis.py
- **PASS — Source audit:** Python source parses. 12_pipeline_integrity_audit.py
- **REVIEW — Leakage certification:** Source keyword/AST inspection cannot prove fitting order or absence of leakage. Inspect preprocessing, event extraction, train/test masks, and all fitted transforms manually.. See source_review_locations.csv
- **PASS — Session results:** CSV read successfully. 10_riemannian_session_results.csv: 36 rows
- **PASS — Session results:** All ROC-AUC scores in [0,1]. 
- **PASS — Session results:** Unique participant/session/model rows. 0
- **PASS — Session results:** Expected complete 9 × 2 × 2 result grid. missing=[], extra=[]
- **PASS — Trial counts:** Reported counts consistent with 8 × 288 training and 144 session-test trials. Reported counts are NOT independent verification of actual indices
- **PASS — Cross-file agreement:** CSV read successfully. 10_riemannian_subject_comparison.csv: 9 rows
- **PASS — Cross-file agreement:** CSP_LDA agrees with session-level recomputation. max_abs_diff=1.1102230246251565e-16
- **PASS — Cross-file agreement:** Riemannian_MDM agrees with session-level recomputation. max_abs_diff=2.220446049250313e-16
- **PASS — Cross-file agreement:** MDM_minus_CSP agrees with session-level recomputation. max_abs_diff=9.71445146547012e-17
- **PASS — Statistical analysis:** Participant-level bootstrap and paired Wilcoxon computed. mean_delta=+0.001490; CI=[-0.055609,+0.057818]; p=0.910156
- **REVIEW — Dataset protocol:** Confirm MOABB paradigm epoch boundaries, event labels, filtering, referencing, and subject/session IDs against raw dataset and library documentation. Not independently checked by this script.. 
- **REVIEW — Data independence:** CSV n_train/n_test columns cannot verify unique sample IDs or preprocessing order. Inspect or instrument actual folds before claiming leakage-free evaluation.. 
- **REVIEW — Generalization:** All 9 participants are from one benchmark dataset; external-dataset generalization untested.. 

## Paired model comparison
- CSP mean AUC: 0.7383
- MDM mean AUC: 0.7398
- MDM − CSP: +0.0015
- Subject bootstrap 95% CI: [-0.05560913923182439, 0.057817858367626804]
- Exact Wilcoxon p: 0.9102

## Mandatory follow-up
1. Review the actual source lines listed in 12_source_review_locations.csv.
2. Verify fit/transform operations happen inside each training fold only.
3. Confirm subject masks are disjoint and trial/session IDs match source data.
4. Freeze preprocessing and model selection before independent dataset evaluation.
5. Do not treat the 9 participants as 18 independent observations.
