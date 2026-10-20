# FINAL_V1 PROTOCOL (drafted by the execution agent; no supplied protocol existed)

Status: **FROZEN on the date recorded in `FREEZE_RECORD.json`** (the project owner instructed on 2026-10-20: no protocol file exists, the execution prompt `Final_Prompt.md` rules replace `AGENTS.md`, use all speech tasks, run the complete experiment A to F and deliver a short English PDF per stage and per experiment group).
Controlling machine-readable copy: `configs/FINAL_V1/protocol.json` (its SHA-256 is recorded in `FREEZE_RECORD.json`). If the two disagree, the JSON controls.

## 1. Question and unit
- Task: Parkinson's disease (positive class, label 1) versus elderly healthy control (label 0), from speech.
- Primary evaluation unit: **speaker**. A speaker score is the mean of the probabilities of that speaker's recordings. Recording-level metrics are supporting only and are labelled "sample-level".
- The primary evaluation estimates generalization to **unseen speakers**, not to unseen sessions of known speakers.

## 2. Data
- Source: Italian Parkinson's Voice and Speech corpus (Dimauro and Girardi, IEEE DataPort, doi:10.21227/aw6b-tg17), copied read-only into `data_input/`.
- Included: PD folder and Elderly Healthy Control folder, **all 16 task codes** (passages B1/B2, syllables D1/D2, phrase FB1, word list PR1, vowels VA..VU x2). Tasks are pooled; the task code is **not** a model input.
- Excluded (decision D-001): the 15 young healthy controls (age 19 to 29, three tasks only, age would confound the label). 45 files.
- Excluded (decision D-002): one byte-identical duplicate recording (Giulia P B2 equals B1); the first copy is kept.
- Speaker identity (decision D-003): folder name plus sex plus age plus letter-anagram of the name code across recording sessions. Three people recorded in two sessions are merged (same folder name, sex, age, same letters in a different order or the same code). One file with a different age token (typo) is kept under its person. Metadata (age, sex, native sample rate, date) are never model inputs.
- Resample every recording to 16 kHz with polyphase filtering. No peak normalization, no silence removal.

## 3. Splits
- Outer: stratified 5-fold, speaker-disjoint, seed 42 (class-wise shuffle, round-robin). Saved in `SPLIT_MANIFEST.csv`.
- Inner (tuning and feature selection): stratified 3-fold, speaker-disjoint, inside each outer-training partition, seed 142 + outer fold; feature-selection internal folds use seed 242 + outer fold.
- Session folds, record-level folds and any other split are **not** part of FINAL_V1.

## 4. Representations (frozen)
| Key | Definition | Dimension |
|---|---|---|
| AC | 10 frame features x 4 statistics (see `src/final_v1/acoustic.py`): log energy, zero-crossing rate, spectral centroid, rolloff 85 %, flatness, five band energies; statistics mean, SD, p10, p90 | 40 |
| WLM | microsoft/wavlm-large, last_hidden_state, mean over time then over 10-s windows | 1024 |
| W2V | facebook/wav2vec2-large-lv60 (English, pretrained only), same recipe | 1024 |
| XLSR | facebook/wav2vec2-xls-r-300m (multilingual), same recipe | 1024 |
| H1..H8 | concatenations: WLM+AC, W2V+AC, XLSR+AC, WLM+W2V, WLM+XLSR, W2V+XLSR, WLM+W2V+XLSR, WLM+W2V+XLSR+AC | 1064, 1064, 1064, 2048, 2048, 2048, 3072, 3112 |
Checkpoint revisions and file hashes: `features/manifests/encoder_checkpoints.json`. Joins use stable `sample_id` only.

## 5. Models and search spaces (frozen)
All models are trained on recordings; class weighting is part of each search space; the decision threshold is fixed at 0.5 (support vector machines use the logistic function of the decision value as an uncalibrated pseudo-probability, so 0.5 equals the margin 0).
- Logistic regression (liblinear): C in {0.01,0.1,1,10,100} x penalty in {l1,l2} x class_weight in {none,balanced} (20 configurations).
- Linear SVM (sklearn SVC, kernel linear): C in {0.01,0.1,1,10,100} x class_weight (10).
- RBF SVM: C in {0.1,1,10,100} x gamma = m / n_features with m in {0.1,1,10} (features standardized) x class_weight (24).
- XGBoost (hist): full grid learning_rate {0.03,0.1} x max_depth {2,4} x n_estimators {100,300} x min_child_weight {1,5} x subsample {0.7,1.0} x colsample_bytree {0.3,1.0} x reg_lambda {1,10} x scale_pos_weight {1, n_neg/n_pos} = 256; a fixed random subset of 24 configurations (RandomState 42, stored in the JSON) is searched.
- Standardization (mean and SD fitted on the training rows) for all models except XGBoost.
- Inner-validation objective: **speaker-level AUPRC** pooled over the three inner validation folds; ties go to the earliest configuration in the listed order. The selected configuration is refitted on the whole outer-training partition.

## 6. Feature selection
- Only for AC. Conditions: ALL (40), FORWARD (to exactly 10 features), BACKWARD (from 40 down to exactly 10 features). k = 10 is fixed here and is not tuned.
- The selector is logistic regression (l2, C = 1, balanced) scored by the speaker-level AUPRC of a 3-fold speaker-disjoint cross-validation **inside the training partition**; the selection is repeated inside every inner-training partition used for tuning, and once on the outer-training partition for the final model. Selected feature names are saved per fold.
- Embeddings and hybrids are not subject to sequential selection (`SKIPPED_BY_PROTOCOL`).

## 7. Experiment matrix (56 conditions x 5 folds = 280 runs)
- AC: {ALL, FORWARD, BACKWARD} x {LR, LSVM, RSVM, XGB} = 12 conditions.
- WLM, W2V, XLSR, H1..H8 (11 representations): ALL x 4 models = 44 conditions.
- **FWL: BLOCKED** (`PROTOCOL_DECISION_REQUIRED`): no scientific purpose, response, nuisance covariates, residualization procedure, statistic or stopping rule was supplied; the agent does not improvise it. It is recorded as one blocked registry row.
- Run ID format: `FV1-<rep>-<fs>-<model>-F<fold>`; condition ID `FV1-<rep>-<fs>-<model>`.

## 8. Metrics
- Primary (speaker level, pooled over the five outer test folds): AUPRC (average precision, positive class PD) and AUROC.
- Supporting (speaker level, threshold 0.5): F1, sensitivity, specificity, precision, accuracy, balanced accuracy; fold-wise values with mean and sample SD over five folds; sample-level AUROC and AUPRC.
- Uncertainty: percentile bootstrap over speakers (2,000 resamples, seed 42) for pooled AUPRC and AUROC. Paired speaker-bootstrap differences (same resamples) for the pre-specified comparisons: each hybrid against each of its constituent baselines with the same model, and each model against logistic regression in the same representation. These intervals are descriptive and unadjusted; no significance claim is made.
- Class prevalence is reported: 25 PD and 22 control speakers (53.2 % PD).
- Selection caveat: choosing the best of 56 conditions on pooled OOF scores gives an optimistic estimate; the report ranks conditions for description only and does not present the best one as an unbiased estimate.

## 9. Integrity gates (every run)
Each run is quarantined and marked FAILED if any gate fails: zero speaker overlap train/test; every test sample predicted exactly once; probabilities finite in [0,1]; labels equal the audited manifest; folds equal the frozen split manifest; protocol and split hashes equal the frozen values. See `OOF_LEAKAGE_AUDIT.md`.

## 10. Reporting
Every table is generated from the verified registry and prediction files. Historical results of earlier work are not used. A short English PDF summary is produced for each stage (A to F) and each experiment group (acoustic only, single encoders, hybrids, models, feature selection).
