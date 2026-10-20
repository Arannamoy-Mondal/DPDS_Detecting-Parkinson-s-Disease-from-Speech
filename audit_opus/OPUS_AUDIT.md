# Independent audit of FINAL_V1 (Opus 5.5, max effort)

Date: 2026-10-20T13:09:03Z. Scope: data pipeline, features, modelling code, execution, aggregation, reporting truthfulness and protocol compliance. Method: code reading plus independent re-implementation and re-execution; evidence in `audit_opus/evidence/`, scripts in `audit_opus/scripts/`.

**Verdict: no critical bug and no change to any reported prediction or metric.** Two major documentation errors and several minor glitches were found and fixed; frozen files were left byte-identical and corrections are recorded as errata.

## Findings

| ID | Severity | Finding | Evidence | Fix | Changes a result? |
|---|---|---|---|---|---|
| AUD-01 | MAJOR (documentation) | The two-digit number in the file names was described as age (protocol D-003 text, DATA_AUDIT, DECISION_LOG, stage_a_build docstring, error analysis of the guideline report). | It is the year of birth: file-number means/SDs equal spreadsheet ages after subtraction from the recording year (controls 2017-1949.9=67.1, SD 5.16 both; patients 66.8, SD 9.13 both). The identity rule never used it. | Errata E-1 (PROTOCOL_ERRATA.md); documents and reports corrected; sheet demographics added. | No |
| AUD-02 | MAJOR (data documentation) | Spreadsheet notes on four starred controls (poor-quality recordings; no passage/no phrases; reading difficulty) were not reported. | Tab 3.xlsx footnotes. The four controls (HC_001, HC_007, HC_008, HC_020) account for 29 of 44 false-positive events over all 56 conditions. | Errata E-2; DATA_AUDIT, FINAL_REPORT and guideline report updated. | No |
| AUD-03 | MINOR (reporting) | Embedding extraction stated as 'roughly 1.5 hours'; summed run times called 'core-hours'. | Cache file times: 09:39-10:41 UTC (about 1 h). Run times are wall-clock per run (15.5 run-hours in total). | FINAL_REPORT.md/.pdf and the Stage D PDF corrected. | No |
| AUD-04 | MINOR (reporting) | Guideline report wording: 'different equipment' (not documented), 'which is why extensions cannot improve' (causal), encoders '315-317M' parameters, 'about 630' training recordings per fold, '41 to 3,113 parameters' including XGBoost. | Measured: 315,453,120 (WavLM) and 315,438,720 (wav2vec2, XLS-R) parameters; 611-644 training recordings per fold. | Wording corrected; reports regenerated. | No |
| AUD-05 | MINOR (execution harness) | Runner gate 'protocol_and_split_hashes_match_freeze' compared the freeze record with itself (always true). | Code inspection of runner.py. | Gate now hashes the on-disk files; the OOF audit had verified the on-disk hashes independently (check 7). Stage C tests re-run: 25/25 pass. | No |
| AUD-06 | MINOR (reproducibility scope) | 'Bit-identical replay' was shown for 15 runs only. | Refitting L2 logistic regression with unrestricted multi-threaded BLAS moves probabilities by up to 4e-4 (liblinear TRON uses BLAS); all 70 LR runs re-executed with single-threaded BLAS are bit-identical (configurations, probabilities, metrics). | Documented in FINAL_REPORT (14b), ENVIRONMENT.md, guideline report. | No |
| AUD-07 | MINOR (code hygiene) | pipeline.py silences all warnings. | scikit-learn 1.9.1 deprecates 'penalty' (L1 still applied: verified sparse fit); 0 ConvergenceWarnings in the 70 final LR fits. | Not changed (frozen modelling code); documented. | No |
| AUD-08 | MINOR (statistics presentation) | Percentile bootstrap intervals collapse to [1.000, 1.000] when the pooled AUROC is 1. | Property of the percentile bootstrap at the boundary. | Noted in FINAL_REPORT; exact Clopper-Pearson intervals for sensitivity/specificity added in the whitepaper. | No |
| AUD-09 | MINOR (compliance) | FAILURE_LOG.csv was empty although development failures and the FWL block occurred. | Session record. | 7 rows added (all RESOLVED except FWL BLOCKED). | No |
| AUD-10 | MINOR (presentation) | Feature-selection stability table repeats identical rows for the four models. | Selection uses a fixed LR selector; selected sets identical across models in 10/10 folds. | Explanatory note added to the PDF. | No |
| AUD-11 | INFO | Two runner gates (labels, folds) are consistency checks by construction. | Label provenance verified independently from the raw folder tree (A02). | None needed. | No |
| AUD-12 | INFO | ENVIRONMENT.md / REPRODUCIBILITY.md incomplete (torch build, packages added later, full command list). | uv pip freeze. | Both updated. | No |

## Verified without findings

| Area | Evidence |
|---|---|
| Data identity and labels | Independent re-parse of the raw tree: 785/785 files matched, 0 label mismatches, 47 speakers, identical partition: True. |
| Duplicates | No cross-speaker recording pair with WavLM cosine >= 0.999 (max 0.9948; within-speaker max 0.9976). One byte-identical duplicate already excluded. |
| Encoders | Load reports list only unused pre-training heads (no missing weights). Fresh-process re-extraction of 5 recordings per encoder: max |difference| 3.1e-06, cosine 1.0. |
| Acoustic features | Independent implementation (25 recordings x 40 features): max relative difference 9.6e-06. |
| Nested CV | Independent re-implementation for WavLM-LR: max |dp| 1.3e-14, same configuration in 5/5 folds; XLS-R-LR: 3.9e-04 (BLAS threading), 5/5 folds. |
| Metrics | All 56 conditions x 6 metrics recomputed from prediction files and compared with condition_metrics.csv and the registry: 0 disagreements. |
| Bootstrap | Percentile intervals and paired differences recomputed with independent code for two comparisons: identical to 1e-12. |
| Thread sensitivity | 70 LR runs re-executed with single-threaded BLAS: max |dp| 4.4e-16; same configurations in all folds; identical pooled metrics. |
| OOF audit | 20 checks, all passed after the fixes (280 runs, 0 failed). |
| Isolation | Original project unchanged since the start snapshot; 831 original audio files re-hashed, 0 changed. |

## Scope limits
- The audit verifies that the pipeline obeys the speaker-disjoint rules and that the numbers are computed correctly. It cannot show that the classifiers detect disease: recording date and class are confounded in this corpus (see the whitepaper's confounding diagnostics).
