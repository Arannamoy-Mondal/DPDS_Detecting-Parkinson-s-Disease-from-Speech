# Pre-declared settings: WavLM + Random Forest (exploratory amendment A2)

Declared before the run, on 21 October 2026, by the owner's request to complete the WavLM + Random Forest
comparison. It is an exploratory supplementary analysis under the same rules as amendment A1 (D-010): it is
written under `supplementary/`, it is labelled exploratory, and it does not change any frozen FINAL_V1 number.

- Representation: WLM (frozen WavLM-large, 1,024 values), the same vectors used by FV1-WLM-ALL-LR and FV1-WLM-ALL-XGB.
- Outer folds: `SPLIT_MANIFEST.csv` (5 stratified, speaker-disjoint folds, seed 42).
- Inner folds: three speaker-disjoint folds inside each outer-training partition, seed 142 + outer fold (same function as the frozen pipeline).
- Model: `sklearn.ensemble.RandomForestClassifier`, `n_estimators=300`, `random_state=42`, no feature selection, no standardisation.
- Search space (8 configurations, listed in this order):
  `max_depth` in {4, None} x `min_samples_leaf` in {1, 2} x `class_weight` in {None, "balanced"}.
- Selection: highest mean inner speaker-level AUPRC over the three inner validation folds; ties go to the earliest configuration.
  The selected configuration is refitted on the whole outer-training partition.
- Threshold: 0.5 on the speaker mean of recording probabilities (fixed, as in FINAL_V1).
- Metrics: pooled speaker-level metrics over the five outer test folds; fold-wise mean and sample SD; paired bootstrap comparison
  with logistic regression on the same resamples (2,000 resamples, seed 42), descriptive only.
- Reference rows for comparison: FV1-WLM-ALL-LR, FV1-WLM-ALL-XGB (already saved in `predictions/FINAL_V1/`).
