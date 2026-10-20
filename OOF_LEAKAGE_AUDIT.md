# OOF LEAKAGE AUDIT

Generated 2026-10-20T13:08:26Z by `src/final_v1/audit_oof.py` (machine-readable: `metrics/FINAL_V1/oof_audit.json`).

Overall: **ALL CHECKS PASSED** (280 completed runs, 0 failed, 56 of 56 conditions complete).

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Every speaker belongs to exactly one outer fold (all records of a speaker share the fold) | PASS | 47 speakers, max folds per speaker 1 |
| 2 | Speaker overlap between training and test in every completed outer-fold run is zero (recomputed from the frozen split manifest) | PASS | 280 runs checked, 0 with overlap |
| 3 | Each test sample has exactly one prediction in its run; ids equal the split manifest | PASS | 280 runs checked, 0 violations |
| 4 | True labels in prediction files equal the audited sample manifest | PASS | 0 violations |
| 5 | Probabilities are finite and in [0,1] | PASS | 0 violations |
| 6 | Folds in prediction files equal the frozen split manifest | PASS | 0 violations |
| 7 | Protocol hash and split-manifest hash in every run equal the frozen values | PASS | 0 violations; protocol and split files unchanged since freeze |
| 8 | Threshold policy: predicted class equals probability >= 0.5 fixed in advance; no calibration or threshold search exists in the code | PASS | 0 violations |
| 9 | Run configurations on disk equal the configuration hashes stored in predictions and metrics | PASS | 0 violations |
| 10 | No byte-identical source audio appears in two outer folds; sample ids unique | PASS | 785 unique audio hashes, 785 samples |
| 11 | No duplicated decoded audio among included recordings | PASS | 785 unique decoded hashes |
| 12 | Features and embeddings are joined by stable sample_id; shapes, finiteness, checkpoint revision and source-audio hash verified | PASS | acoustic ok; WLM ok (cache entries with wrong source/revision: 0); W2V ok (cache entries with wrong source/revision: 0); XLSR ok (cache entries with wrong source/revision: 0) |
| 13 | Modelling code is unchanged since the freeze (hashes); only execution harness files were added after the freeze | PASS | changed modelling files: []; files added after freeze (harness/audit only): ['runner.py', 'aggregate.py', 'audit_oof.py'] |
| 14 | Preprocessing, tuning and feature-selection fit boundaries verified by the instrumented Stage C tests (T06, T07, T07b-d) and the null/positive controls (T09, T09b) | PASS | 25 of 25 tests passed |
| 15 | Inner tuning logs exist for every run, train+test speakers equal 47, and selected hyperparameters belong to the frozen grid | PASS | 280 runs |
| 16 | Feature-selection runs record exactly 10 selected acoustic features per fold | PASS | 40 runs |
| 17 | Every completed condition has exactly one OOF prediction for each of the 785 recordings (47 speakers) | PASS | 56 completed conditions, 0 with coverage problems |
| 18 | Pooled speaker-level AUROC and AUPRC of every completed condition were recomputed with independent code and agree to 1e-9 | PASS | 56 conditions, 0 disagreements |
| 19 | Failed or quarantined runs | PASS | FAILED runs: 0, quarantined prediction files: 0 (excluded from all metrics) |
| 20 | Registry statuses are consistent with the metric files | PASS | registry fold-run statuses: {'COMPLETED': 280}; metric files completed: 280 |

Scope note: the audit shows the pipeline and the saved predictions obey the speaker-disjoint rules. It cannot show that the score reflects disease: recording date and class are fully confounded in this corpus (`DATA_AUDIT.md`), and speaker-disjoint folds do not remove that.
