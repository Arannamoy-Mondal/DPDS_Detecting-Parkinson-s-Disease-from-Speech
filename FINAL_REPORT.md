# FINAL REPORT - Parkinson's disease detection from speech (FINAL_V1)

**Report status: READY (all mandatory runs completed and audited).** Generated 2026-10-20T13:06:52Z only from verified FINAL_V1 artifacts (registry, metrics, prediction files, manifests). Historical results of earlier work are not used anywhere in this report.

## 1. Title and objective
Classify Parkinson's disease (positive class) versus elderly healthy controls from speech, estimating generalization to **unseen speakers**, with acoustic features, frozen WavLM-large, wav2vec2-large-lv60 and XLS-R 300M embeddings and their hybrids, under one frozen protocol.

## 2. Abstract
On 47 speakers (25 PD, 22 elderly controls; 785 recordings, 16 tasks pooled) we ran 56 of 56 planned conditions (280 of 280 outer-fold runs) under stratified speaker-disjoint five-fold cross-validation with nested speaker-disjoint tuning. Pooled out-of-fold speaker-level AUROC ranged from 0.867 to 1.000 (median 0.995) and AUPRC from 0.887 to 1.000 across completed conditions; 40 of 56 conditions made at most one speaker error at the fixed threshold 0.5. The highest-scoring condition (AC Logistic regression: AUPRC 1.000, AUROC 1.000) is the best of 56 and is therefore an optimistic estimate. Acoustic features alone (40 hand-crafted values) already reach a median AUROC of 0.996, and every recording date in the corpus holds a single class, so these scores must not be read as evidence of disease detection. Results are internal; an independent corpus is required.

## 3. Research questions and pre-specified comparisons
1. How well does each representation separate PD from controls for unseen speakers (acoustic, WavLM, wav2vec2, XLS-R)?
2. Do hybrids (encoder + acoustic, encoder combinations) improve on their constituent baselines on identical folds?
3. Do linear SVM, RBF SVM and XGBoost differ from logistic regression?
4. Do forward or backward selection of 10 acoustic features change the acoustic result?
5. FWL: blocked (section 12).
All comparisons were fixed in `FINAL_V1_PROTOCOL.md` before any model was trained on the project data.

## 4. Dataset, labels and task composition
- 785 recordings from 47 speakers; 436 PD recordings / 25 speakers; 349 control recordings / 22 speakers; 4.98 audio hours; class prevalence 53.2% PD (speakers).
- Evaluation unit: speaker (mean of recording probabilities); sample-level metrics are supporting.
- Tasks pooled (recordings per task code): {'B1': 49, 'B2': 44, 'D1': 50, 'D2': 50, 'FB1': 47, 'PR1': 50, 'VA1': 50, 'VA2': 49, 'VE1': 50, 'VE2': 49, 'VI1': 50, 'VI2': 49, 'VO1': 50, 'VO2': 49, 'VU1': 50, 'VU2': 49}.

## 5. Inclusion and exclusion; data-quality findings
- Demographics from the corpus spreadsheets: patients 66.8 +/- 9.1 years (40 to 80), 16 men / 9 women; controls 67.1 +/- 5.2 years (60 to 77), 10 men / 12 women. The two-digit number in the file names is the year of birth (erratum E-1), not the age; age is never a model input.
- Four controls are flagged with an asterisk in the control spreadsheet (poor-quality recordings, reading difficulty; one has no passage or phrase recordings): HC_001, HC_007, HC_008, HC_020. They are included.

- Excluded: 45 young-control recordings (age confound); 1 byte-identical duplicate recording (Giulia P B2); both decisions are in `DECISION_LOG.md`.
- Three people recorded in two sessions were merged into one speaker each (D-003); speakers with fewer than 16 recordings were kept without imputation.
- Findings (`DATA_AUDIT.md`): every recording date holds one class; all 160 recordings at 44.1 kHz are PD (all audio resampled to 16 kHz); no non-finite audio; no duplicated decoded audio after exclusion.

## 6. Speaker-disjoint evaluation and leakage prevention
Stratified speaker-disjoint 5-fold outer CV (seed 42; folds of 10, 10, 9, 9, 9 speakers); nested stratified speaker-disjoint 3-fold inner CV for tuning and for feature selection; all learned transformations fit on the applicable training partition only; threshold fixed at 0.5. The audit (`OOF_LEAKAGE_AUDIT.md`): **all checks passed** (20 checks; 280 completed runs, 0 failed). The primary estimate is for unseen speakers, not unseen sessions of known speakers.

## 7. Preprocessing and representation extraction
- Audio: mono, polyphase resampling to 16 kHz, no peak normalization, no silence removal.
- Acoustic (40 values): 10 frame features (log energy, zero-crossing rate, spectral centroid, rolloff 85%%, flatness, five band energies) x mean, SD, p10, p90 over 25 ms / 10 ms frames.
- Encoders (frozen, CPU): W2V = `facebook/wav2vec2-large-lv60` revision `0cde644b64da`; WLM = `microsoft/wavlm-large` revision `c1423ed94bb0`; XLSR = `facebook/wav2vec2-xls-r-300m` revision `1a640f32ac3e`. Recipe: 10-second windows (last window end-aligned), checkpoint feature extractor normalization, mean over time of last_hidden_state (1,024 values), mean over windows; every cache entry stores the source-audio hash and the checkpoint revision (verified in the audit).

## 8. Protocol and hyperparameter tuning
Models: logistic regression (20 configurations), linear SVM (10), RBF SVM (24), XGBoost (24 of a 256-configuration grid, fixed subset). Inner objective: pooled speaker-level AUPRC over three speaker-disjoint inner folds; selected configuration refitted on the whole outer-training partition and applied once to the untouched outer-test fold. Search spaces were not changed after any result.

## 9. Results (all completed conditions)
Pooled out-of-fold, speaker level (47 speakers). Sens. = sensitivity, Spec. = specificity; CI = 95%% percentile bootstrap over speakers. Threshold-dependent metrics at the fixed 0.5.

| Condition | AUPRC [CI] | AUROC [CI] | F1 | Sens. | Spec. | Bal.acc. | FN/FP | Fold AUROC mean +/- SD |
|---|---|---|---|---|---|---|---|---|
| FV1-AC-ALL-LR | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-AC-ALL-LSVM | 0.998 [0.991, 1.000] | 0.998 [0.989, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-AC-ALL-RSVM | 0.991 [0.971, 1.000] | 0.989 [0.964, 1.000] | 0.923 | 0.960 | 0.864 | 0.912 | 1/3 | 1.000 +/- 0.000 |
| FV1-AC-ALL-XGB | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-AC-FORWARD-LR | 0.997 [0.985, 1.000] | 0.996 [0.983, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-AC-FORWARD-LSVM | 0.997 [0.985, 1.000] | 0.996 [0.983, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-AC-FORWARD-RSVM | 0.997 [0.987, 1.000] | 0.996 [0.984, 1.000] | 0.960 | 0.960 | 0.955 | 0.957 | 1/1 | 1.000 +/- 0.000 |
| FV1-AC-FORWARD-XGB | 0.997 [0.985, 1.000] | 0.996 [0.983, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-AC-BACKWARD-LR | 0.990 [0.966, 1.000] | 0.987 [0.957, 1.000] | 0.941 | 0.960 | 0.909 | 0.935 | 1/2 | 0.980 +/- 0.045 |
| FV1-AC-BACKWARD-LSVM | 0.994 [0.977, 1.000] | 0.993 [0.971, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 0.990 +/- 0.022 |
| FV1-AC-BACKWARD-RSVM | 0.990 [0.963, 1.000] | 0.985 [0.947, 1.000] | 0.923 | 0.960 | 0.864 | 0.912 | 1/3 | 1.000 +/- 0.000 |
| FV1-AC-BACKWARD-XGB | 0.996 [0.982, 1.000] | 0.995 [0.978, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-WLM-ALL-LR | 0.996 [0.981, 1.000] | 0.995 [0.977, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 0.990 +/- 0.022 |
| FV1-WLM-ALL-LSVM | 0.996 [0.982, 1.000] | 0.995 [0.977, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-WLM-ALL-RSVM | 0.887 [0.765, 0.972] | 0.873 [0.752, 0.966] | 0.889 | 0.960 | 0.773 | 0.866 | 1/5 | 1.000 +/- 0.000 |
| FV1-WLM-ALL-XGB | 0.998 [0.991, 1.000] | 0.998 [0.989, 1.000] | 0.960 | 0.960 | 0.955 | 0.957 | 1/1 | 1.000 +/- 0.000 |
| FV1-W2V-ALL-LR | 0.992 [0.971, 1.000] | 0.989 [0.960, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 0.960 +/- 0.089 |
| FV1-W2V-ALL-LSVM | 0.986 [0.948, 1.000] | 0.975 [0.913, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 0.960 +/- 0.089 |
| FV1-W2V-ALL-RSVM | 0.997 [0.987, 1.000] | 0.996 [0.984, 1.000] | 0.962 | 1.000 | 0.909 | 0.955 | 0/2 | 1.000 +/- 0.000 |
| FV1-W2V-ALL-XGB | 0.989 [0.957, 1.000] | 0.982 [0.935, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 0.990 +/- 0.022 |
| FV1-XLSR-ALL-LR | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-XLSR-ALL-LSVM | 0.997 [0.987, 1.000] | 0.996 [0.984, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-XLSR-ALL-RSVM | 0.986 [0.950, 1.000] | 0.976 [0.919, 1.000] | 0.941 | 0.960 | 0.909 | 0.935 | 1/2 | 1.000 +/- 0.000 |
| FV1-XLSR-ALL-XGB | 0.997 [0.986, 1.000] | 0.996 [0.983, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H1-ALL-LR | 0.998 [0.992, 1.000] | 0.998 [0.989, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H1-ALL-LSVM | 0.991 [0.967, 1.000] | 0.987 [0.955, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H1-ALL-RSVM | 0.901 [0.796, 0.969] | 0.867 [0.749, 0.954] | 0.828 | 0.960 | 0.591 | 0.775 | 1/9 | 1.000 +/- 0.000 |
| FV1-H1-ALL-XGB | 0.998 [0.991, 1.000] | 0.998 [0.989, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H2-ALL-LR | 0.997 [0.986, 1.000] | 0.996 [0.984, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 0.980 +/- 0.045 |
| FV1-H2-ALL-LSVM | 0.989 [0.958, 1.000] | 0.982 [0.935, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 0.980 +/- 0.045 |
| FV1-H2-ALL-RSVM | 0.956 [0.866, 1.000] | 0.962 [0.891, 1.000] | 0.893 | 1.000 | 0.727 | 0.864 | 0/6 | 0.992 +/- 0.018 |
| FV1-H2-ALL-XGB | 0.998 [0.991, 1.000] | 0.998 [0.989, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H3-ALL-LR | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H3-ALL-LSVM | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H3-ALL-RSVM | 0.985 [0.946, 1.000] | 0.973 [0.905, 1.000] | 0.941 | 0.960 | 0.909 | 0.935 | 1/2 | 0.990 +/- 0.022 |
| FV1-H3-ALL-XGB | 0.998 [0.991, 1.000] | 0.998 [0.989, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H4-ALL-LR | 0.996 [0.981, 1.000] | 0.995 [0.977, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 0.990 +/- 0.022 |
| FV1-H4-ALL-LSVM | 0.992 [0.970, 1.000] | 0.989 [0.960, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H4-ALL-RSVM | 0.986 [0.947, 1.000] | 0.975 [0.912, 1.000] | 0.960 | 0.960 | 0.955 | 0.957 | 1/1 | 1.000 +/- 0.000 |
| FV1-H4-ALL-XGB | 0.994 [0.978, 1.000] | 0.993 [0.971, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H5-ALL-LR | 0.997 [0.985, 1.000] | 0.996 [0.983, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 0.990 +/- 0.022 |
| FV1-H5-ALL-LSVM | 0.998 [0.991, 1.000] | 0.998 [0.989, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H5-ALL-RSVM | 0.986 [0.947, 1.000] | 0.975 [0.912, 1.000] | 0.941 | 0.960 | 0.909 | 0.935 | 1/2 | 1.000 +/- 0.000 |
| FV1-H5-ALL-XGB | 0.993 [0.974, 1.000] | 0.991 [0.964, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 0.990 +/- 0.022 |
| FV1-H6-ALL-LR | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H6-ALL-LSVM | 0.992 [0.969, 1.000] | 0.989 [0.961, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H6-ALL-RSVM | 0.986 [0.947, 1.000] | 0.975 [0.912, 1.000] | 0.960 | 0.960 | 0.955 | 0.957 | 1/1 | 1.000 +/- 0.000 |
| FV1-H6-ALL-XGB | 0.998 [0.991, 1.000] | 0.998 [0.989, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H7-ALL-LR | 0.997 [0.985, 1.000] | 0.996 [0.983, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 0.990 +/- 0.022 |
| FV1-H7-ALL-LSVM | 0.997 [0.987, 1.000] | 0.996 [0.984, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H7-ALL-RSVM | 0.984 [0.943, 1.000] | 0.971 [0.901, 1.000] | 0.941 | 0.960 | 0.909 | 0.935 | 1/2 | 0.980 +/- 0.045 |
| FV1-H7-ALL-XGB | 0.993 [0.974, 1.000] | 0.991 [0.964, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 0.990 +/- 0.022 |
| FV1-H8-ALL-LR | 0.997 [0.987, 1.000] | 0.996 [0.984, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H8-ALL-LSVM | 0.998 [0.991, 1.000] | 0.998 [0.989, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |
| FV1-H8-ALL-RSVM | 0.986 [0.947, 1.000] | 0.975 [0.912, 1.000] | 0.941 | 0.960 | 0.909 | 0.935 | 1/2 | 1.000 +/- 0.000 |
| FV1-H8-ALL-XGB | 0.998 [0.991, 1.000] | 0.998 [0.989, 1.000] | 0.980 | 0.960 | 1.000 | 0.980 | 1/0 | 1.000 +/- 0.000 |

### 9.1 Acoustic only, encoders only, hybrids (median AUROC over the four models)
| Representation | Median AUROC | Min | Max |
|---|---|---|---|
| AC | 0.999 | 0.989 | 1.000 |
| H3 | 0.999 | 0.973 | 1.000 |
| H8 | 0.997 | 0.975 | 0.998 |
| AC/FORWARD | 0.996 | 0.996 | 0.996 |
| XLSR | 0.996 | 0.976 | 1.000 |
| WLM | 0.995 | 0.873 | 0.998 |
| H6 | 0.994 | 0.975 | 1.000 |
| H7 | 0.994 | 0.971 | 0.996 |
| H5 | 0.994 | 0.975 | 0.998 |
| H1 | 0.993 | 0.867 | 0.998 |
| H4 | 0.991 | 0.975 | 0.995 |
| AC/BACKWARD | 0.990 | 0.985 | 0.995 |
| H2 | 0.989 | 0.962 | 0.998 |
| W2V | 0.985 | 0.975 | 0.996 |

### 9.2 Model variants (median AUROC over all representations)
| Model | Median AUROC | Min | Max |
|---|---|---|---|
| Logistic regression | 0.996 | 0.987 | 1.000 |
| Linear SVM | 0.995 | 0.975 | 1.000 |
| RBF SVM | 0.975 | 0.867 | 0.996 |
| XGBoost | 0.997 | 0.982 | 1.000 |

### 9.3 Hybrids against constituent baselines
76 paired comparisons (same model, same folds); median AUROC difference 0.000 (range -0.122 to 0.102); the 95% bootstrap interval excludes zero in 5 (positive 4, negative 1). Unadjusted and descriptive; no significance claim.

### 9.4 Models against logistic regression
- Linear SVM: 14 comparisons, median AUROC difference -0.001 (range -0.015 to 0.005), interval excludes zero in 0.
- RBF SVM: 14 comparisons, median AUROC difference -0.023 (range -0.131 to 0.007), interval excludes zero in 2.
- XGBoost: 14 comparisons, median AUROC difference -0.001 (range -0.007 to 0.007), interval excludes zero in 0.

### 9.5 Acoustic feature selection
Selected minus all 40 features (same model): median AUROC difference -0.004 (range -0.013 to 0.007); interval excludes zero in 0 of 8.

## 10. Metrics and aggregation
Primary: pooled out-of-fold speaker-level AUPRC and AUROC. Supporting: F1, sensitivity, specificity, balanced accuracy, precision, accuracy at threshold 0.5; fold-wise values with mean and sample SD over five folds; sample-level AUROC/AUPRC (in `metrics/FINAL_V1/condition_metrics.csv`). The positive class is PD (prevalence 53.2%% of speakers, 55.5%% of recordings), which matters for AUPRC (chance level is the prevalence).

## 11. Fold variability and uncertainty
Median fold SD of AUROC over completed conditions: 0.000 (many folds are perfectly separated); only 9 to 10 speakers per fold, so a single speaker changes a fold metric by 10 to 20 points. Median width of the 95% bootstrap AUROC interval over conditions: 0.020 (range 0.000 to 0.214); differences between most conditions are within these intervals. When the pooled AUROC is 1.000 every bootstrap resample is also perfectly ranked, so the percentile interval collapses to [1.000, 1.000]; it does not mean there is no uncertainty (exact Clopper-Pearson intervals for sensitivity and specificity are given in the whitepaper). Errors are concentrated: PD_005 misclassified in 54 of 56 conditions; HC_001 misclassified in 14 of 56 conditions; HC_020 misclassified in 11 of 56 conditions; HC_006 misclassified in 5 of 56 conditions. Of the false-positive events over all conditions, 29 of 44 come from the four flagged controls.

## 12. Failed, blocked, skipped and missing experiments
- Fold runs by status: {'COMPLETED': 280}.
- FWL: **BLOCKED** (`PROTOCOL_DECISION_REQUIRED`); no FWL result exists.
- Sequential feature selection for embeddings and hybrids: `SKIPPED_BY_PROTOCOL` (not authorized for high-dimensional inputs).
- Failed runs: 0; quarantined prediction files: see `OOF_LEAKAGE_AUDIT.md`.
- Incomplete conditions: none.

## 13. Runtime and compute
CPU only (32 logical cores, 61 GB RAM, no GPU). Embedding extraction: the three encoders ran in parallel (3 processes x 3 threads each) on 785 recordings (4.98 audio hours) each, from 09:39 to 10:40 UTC (WLM 61 min; W2V 53 min; XLSR 54 min); i.e. about one hour of wall time. Model runs: 280 outer-fold runs executed in parallel between 09:42 and 11:48 UTC; the sum of per-run wall-clock times is 15.5 run-hours (XGBoost accounts for 83% of it).

## 14. Limitations
- 47 speakers (25 PD, 22 controls): one speaker moves speaker-level metrics by about two points; fold SDs are large.
- **Recording conditions are confounded with class** (every recording date holds one class; all 44.1 kHz audio is PD). Acoustic energy and band features alone separate the classes almost perfectly, and speaker-disjoint folds do not remove a shared recording-condition cue. The scores therefore cannot support the claim that speech pathology is being detected.
- One corpus, one language; no independent test corpus; internal cross-validation only.
- The best-of-56 condition is selection-biased.
- SVM scores are uncalibrated pseudo-probabilities; threshold 0.5 equals the margin.
- No disease severity, duration or medication data.
- Research prototype; no clinical readiness is implied.

## 14b. Reproducibility note
Re-executing all 70 logistic-regression runs with single-threaded BLAS reproduced the official selected configurations, probabilities and metrics bit-identically; refitting in a process with unrestricted multi-threaded BLAS changes some L2-logistic-regression probabilities by up to 4e-4 (the liblinear trust-region solver uses BLAS dot products; tolerance 1e-4) without changing any decision or pooled metric. See `audit_opus/OPUS_AUDIT.md`.

## 15. Reproduction instructions
See `REPRODUCIBILITY.md` and `ENVIRONMENT.md`. Order: `stage_a_inventory.py`, `stage_a_build.py`, `splits.py`, `download_models.py`, `freeze_protocol.py`, `acoustic.py`, `extract_embeddings.py` (three encoders, then `collect`), `runner.py run`, `aggregate.py`, `audit_oof.py`, `group_pdfs.py`, `final_report.py`. Seeds are fixed in `protocol_defs.py`.

## 16. Artifact inventory and protocol hash
- Protocol JSON SHA-256: `5535d1a440a57005fa05d474f5e2d1016df36712b5696f2d1f667569582287c0`
- Protocol document SHA-256: `b84b9ed94443a1646527e7a4e39db70c5e2843f8e4b39a89ca5bf96006737099`
- Split manifest SHA-256: `31c40ed902165aff483dd8e3ab246e028c985bed029074f4588804c7c8def837`
- Freeze time: 2026-10-20T09:37:38Z
- Artifact list with hashes: `ARTIFACT_MANIFEST.csv`; registry: `EXPERIMENT_REGISTRY.csv`; workspace manifest: `WORKSPACE_MANIFEST.csv`.

## 17. Completed versus planned
Completed: 56 of 56 conditions (280 of 280 outer-fold runs). Planned and not run: FWL (blocked). Nothing else is outstanding.
