"""Write audit_opus/OPUS_AUDIT.md and the audit PDF from the evidence files."""
import json, sys
import pandas as pd
W = "/root/Projects/Project4a_FINAL_V1_ISOLATED"
sys.path.insert(0, W + "/src/final_v1")
from common import *  # noqa
import make_pdf
E = W + "/audit_opus/evidence/"
a1 = json.load(open(E + "a01_reextract.json")); a2 = json.load(open(E + "a02_data_identity.json")); a3 = json.load(open(E + "a03_independent_recompute.json"))
a4 = pd.read_csv(E + "a04_thread_sensitivity.csv"); a5 = json.load(open(E + "a05_demographics_and_flags.json"))
oof = json.load(open(W + "/metrics/FINAL_V1/oof_audit.json")); iso = json.load(open(W + "/manifests/isolation_check_audit.json"))
reex = max(x["max_abs_diff"] for v in a1.values() for x in v)
findings = [
 ["AUD-01", "MAJOR (documentation)", "The two-digit number in the file names was described as age (protocol D-003 text, DATA_AUDIT, DECISION_LOG, stage_a_build docstring, error analysis of the guideline report).",
  "It is the year of birth: file-number means/SDs equal spreadsheet ages after subtraction from the recording year (controls 2017-1949.9=67.1, SD 5.16 both; patients 66.8, SD 9.13 both). The identity rule never used it.",
  "Errata E-1 (PROTOCOL_ERRATA.md); documents and reports corrected; sheet demographics added.", "No"],
 ["AUD-02", "MAJOR (data documentation)", "Spreadsheet notes on four starred controls (poor-quality recordings; no passage/no phrases; reading difficulty) were not reported.",
  "Tab 3.xlsx footnotes. The four controls (HC_001, HC_007, HC_008, HC_020) account for %d of %d false-positive events over all 56 conditions." % (a5["false_positive_events_from_flagged_controls"], a5["false_positive_events_total"]),
  "Errata E-2; DATA_AUDIT, FINAL_REPORT and guideline report updated.", "No"],
 ["AUD-03", "MINOR (reporting)", "Embedding extraction stated as 'roughly 1.5 hours'; summed run times called 'core-hours'.", "Cache file times: 09:39-10:41 UTC (about 1 h). Run times are wall-clock per run (15.5 run-hours in total).",
  "FINAL_REPORT.md/.pdf and the Stage D PDF corrected.", "No"],
 ["AUD-04", "MINOR (reporting)", "Guideline report wording: 'different equipment' (not documented), 'which is why extensions cannot improve' (causal), encoders '315-317M' parameters, 'about 630' training recordings per fold, '41 to 3,113 parameters' including XGBoost.",
  "Measured: 315,453,120 (WavLM) and 315,438,720 (wav2vec2, XLS-R) parameters; 611-644 training recordings per fold.", "Wording corrected; reports regenerated.", "No"],
 ["AUD-05", "MINOR (execution harness)", "Runner gate 'protocol_and_split_hashes_match_freeze' compared the freeze record with itself (always true).", "Code inspection of runner.py.",
  "Gate now hashes the on-disk files; the OOF audit had verified the on-disk hashes independently (check 7). Stage C tests re-run: 25/25 pass.", "No"],
 ["AUD-06", "MINOR (reproducibility scope)", "'Bit-identical replay' was shown for 15 runs only.", "Refitting L2 logistic regression with unrestricted multi-threaded BLAS moves probabilities by up to 4e-4 (liblinear TRON uses BLAS); all 70 LR runs re-executed with single-threaded BLAS are bit-identical (configurations, probabilities, metrics).",
  "Documented in FINAL_REPORT (14b), ENVIRONMENT.md, guideline report.", "No"],
 ["AUD-07", "MINOR (code hygiene)", "pipeline.py silences all warnings.", "scikit-learn 1.9.1 deprecates 'penalty' (L1 still applied: verified sparse fit); 0 ConvergenceWarnings in the 70 final LR fits.",
  "Not changed (frozen modelling code); documented.", "No"],
 ["AUD-08", "MINOR (statistics presentation)", "Percentile bootstrap intervals collapse to [1.000, 1.000] when the pooled AUROC is 1.", "Property of the percentile bootstrap at the boundary.",
  "Noted in FINAL_REPORT; exact Clopper-Pearson intervals for sensitivity/specificity added in the whitepaper.", "No"],
 ["AUD-09", "MINOR (compliance)", "FAILURE_LOG.csv was empty although development failures and the FWL block occurred.", "Session record.", "7 rows added (all RESOLVED except FWL BLOCKED).", "No"],
 ["AUD-10", "MINOR (presentation)", "Feature-selection stability table repeats identical rows for the four models.", "Selection uses a fixed LR selector; selected sets identical across models in 10/10 folds.", "Explanatory note added to the PDF.", "No"],
 ["AUD-11", "INFO", "Two runner gates (labels, folds) are consistency checks by construction.", "Label provenance verified independently from the raw folder tree (A02).", "None needed.", "No"],
 ["AUD-12", "INFO", "ENVIRONMENT.md / REPRODUCIBILITY.md incomplete (torch build, packages added later, full command list).", "uv pip freeze.", "Both updated.", "No"],
]
verified = [
 ["Data identity and labels", "Independent re-parse of the raw tree: %d/%d files matched, %d label mismatches, %d speakers, identical partition: %s." % (a2["merge_status"]["both"], a2["manifest_files"], a2["label_mismatches"], a2["manifest_speakers"], a2["speaker_partition_identical"])],
 ["Duplicates", "No cross-speaker recording pair with WavLM cosine >= 0.999 (max %.4f; within-speaker max %.4f). One byte-identical duplicate already excluded." % (a2["max_cosine_cross_speaker_WLM"], a2["max_cosine_within_speaker_WLM"])],
 ["Encoders", "Load reports list only unused pre-training heads (no missing weights). Fresh-process re-extraction of 5 recordings per encoder: max |difference| %.1e, cosine 1.0." % reex],
 ["Acoustic features", "Independent implementation (25 recordings x 40 features): max relative difference %.1e." % a3["acoustic_independent_max_relative_difference"]],
 ["Nested CV", "Independent re-implementation for WavLM-LR: max |dp| %.1e, same configuration in 5/5 folds; XLS-R-LR: %.1e (BLAS threading), 5/5 folds." % (a3["independent_nested_cv_WLM_LR"]["max_abs_probability_difference"], a3["independent_nested_cv_XLSR_LR"]["max_abs_probability_difference"])],
 ["Metrics", "All 56 conditions x 6 metrics recomputed from prediction files and compared with condition_metrics.csv and the registry: %d disagreements." % len(a3["threshold_metric_disagreements"])],
 ["Bootstrap", "Percentile intervals and paired differences recomputed with independent code for two comparisons: identical to 1e-12."],
 ["Thread sensitivity", "70 LR runs re-executed with single-threaded BLAS: max |dp| %.1e; same configurations in all folds; identical pooled metrics." % a4.max_abs_prob_diff.max()],
 ["OOF audit", "%d checks, all passed after the fixes (%d runs, %d failed)." % (len(oof["checks"]), oof["completed_runs"], oof["failed_runs"])],
 ["Isolation", "Original project unchanged since the start snapshot; %d original audio files re-hashed, %d changed." % (iso["original_audio_files_checked"], len(iso["original_audio_files_changed"]))],
]
md = ["# Independent audit of FINAL_V1 (Opus 5.5, max effort)", "", "Date: %s. Scope: data pipeline, features, modelling code, execution, aggregation, reporting truthfulness and protocol compliance. Method: code reading plus independent re-implementation and re-execution; evidence in `audit_opus/evidence/`, scripts in `audit_opus/scripts/`." % now_utc(), "",
      "**Verdict: no critical bug and no change to any reported prediction or metric.** Two major documentation errors and several minor glitches were found and fixed; frozen files were left byte-identical and corrections are recorded as errata.", "",
      "## Findings", "", "| ID | Severity | Finding | Evidence | Fix | Changes a result? |", "|---|---|---|---|---|---|"]
md += ["| " + " | ".join(r) + " |" for r in findings]
md += ["", "## Verified without findings", "", "| Area | Evidence |", "|---|---|"] + ["| %s | %s |" % tuple(v) for v in verified]
md += ["", "## Scope limits", "- The audit verifies that the pipeline obeys the speaker-disjoint rules and that the numbers are computed correctly. It cannot show that the classifiers detect disease: recording date and class are confounded in this corpus (see the whitepaper's confounding diagnostics)."]
write_text(W + "/audit_opus/OPUS_AUDIT.md", "\n".join(md) + "\n")
pdf = make_pdf.summary_pdf("AUDIT_independent_opus_audit", "Independent audit of the FINAL_V1 experiment", "Audit by Opus 5.5 (max effort): bugs, glitches, fixes and verification evidence",
    [("Verdict", ["No critical bug; no reported prediction or metric changed.", "2 major documentation errors and 8 minor glitches found and fixed; 2 informational notes. Frozen protocol and modelling code left byte-identical; corrections recorded in PROTOCOL_ERRATA.md and DECISION_LOG.md (D-009).",
                  "All 20 OOF-audit checks and all 25 pipeline tests pass after the fixes."]),
     ("Verified without findings", ["%s: %s" % tuple(v) for v in verified])],
    [("Findings and fixes", ["ID", "Severity", "Finding", "Fix", "Result changed?"], [[r[0], r[1], r[2], r[4], r[5]] for r in findings], "lp{2.2cm}p{6.6cm}p{5.2cm}l")])
print("wrote", pdf)
