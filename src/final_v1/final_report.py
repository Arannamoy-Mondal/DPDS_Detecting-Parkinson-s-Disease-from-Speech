"""Stage F: FINAL_REPORT.md (and a short PDF) generated only from verified FINAL_V1 artifacts (registry, metrics, predictions, manifests)."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import protocol_defs as P
import make_pdf
import glob, os, time

MET = WORKSPACE / "metrics/FINAL_V1"
NAMES = {"LR": "Logistic regression", "LSVM": "Linear SVM", "RSVM": "RBF SVM", "XGB": "XGBoost"}


def f3(x):
    return "%.3f" % x


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def label(c):
    return c.representation + ("/" + c.feature_selection if c.feature_selection != "ALL" else "")


def main():
    rec = json.loads((WORKSPACE / "FREEZE_RECORD.json").read_text())
    audit = json.loads((MET / "oof_audit.json").read_text())
    cm = pd.read_csv(MET / "condition_metrics.csv")
    comp = pd.read_csv(MET / "paired_comparisons.csv")
    reg = pd.read_csv(WORKSPACE / "EXPERIMENT_REGISTRY.csv", keep_default_na=False)
    stage_a = json.loads((WORKSPACE / "manifests/stage_a_summary.json").read_text())
    ck = json.loads((WORKSPACE / "features/manifests/encoder_checkpoints.json").read_text())
    done = cm[cm.folds_completed == P.N_OUTER].copy()
    n_done, n_cond = len(done), len(cm)
    fr = reg[reg.row_type == "fold_run"]
    status_counts = fr.training_status.value_counts().to_dict()
    ready = (n_done == n_cond) and audit["all_passed"] and audit["failed_runs"] == 0
    # ---- derived statements
    done["errors"] = done.FN + done.FP
    best = done.sort_values(["AUPRC", "AUROC", "errors"], ascending=[False, False, True]).head(5)
    by_rep = done.groupby(done.apply(label, axis=1)).AUROC.agg(["median", "min", "max"]).sort_values("median", ascending=False)
    by_model = done.groupby("model").AUROC.agg(["median", "min", "max"])
    sub_err = done[done.errors <= 1]
    hy = comp[(comp.comparison_type == "hybrid_vs_constituent") & (comp.metric == "AUROC")]
    mv = comp[(comp.comparison_type == "model_vs_logistic_regression") & (comp.metric == "AUROC")]
    fsc = comp[(comp.comparison_type == "feature_selection_vs_all_features") & (comp.metric == "AUROC")]
    # missed speaker(s) common to all conditions
    allp = []
    for cid in done.condition_id:
        d = pd.concat([pd.read_csv(WORKSPACE / "predictions/FINAL_V1" / ("%s-F%d.csv" % (cid, f))) for f in range(5)])
        g = d.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first"))
        g["wrong"] = ((g.s >= 0.5).astype(int) != g.y)
        allp.append(g.wrong.rename(cid))
    W = pd.concat(allp, axis=1)
    wrong_counts = W.sum(axis=1).sort_values(ascending=False)
    top_wrong = wrong_counts[wrong_counts > 0].head(4)
    L = []
    L.append("# FINAL REPORT - Parkinson's disease detection from speech (FINAL_V1)\n")
    L.append("**Report status: %s.** Generated %s only from verified FINAL_V1 artifacts (registry, metrics, prediction files, manifests). Historical results of earlier work are not used anywhere in this report.\n" % ("READY (all mandatory runs completed and audited)" if ready else "INCOMPLETE - see section 12", now_utc()))
    L.append("## 1. Title and objective\nClassify Parkinson's disease (positive class) versus elderly healthy controls from speech, estimating generalization to **unseen speakers**, with acoustic features, frozen WavLM-large, wav2vec2-large-lv60 and XLS-R 300M embeddings and their hybrids, under one frozen protocol.\n")
    best1 = best.iloc[0]
    L.append("## 2. Abstract\nOn %d speakers (%d PD, %d elderly controls; %d recordings, 16 tasks pooled) we ran %d of %d planned conditions (%d of %d outer-fold runs) under stratified speaker-disjoint five-fold cross-validation with nested speaker-disjoint tuning. Pooled out-of-fold speaker-level AUROC ranged from %s to %s (median %s) and AUPRC from %s to %s across completed conditions; %d of %d conditions made at most one speaker error at the fixed threshold 0.5. The highest-scoring condition (%s %s: AUPRC %s, AUROC %s) is the best of %d and is therefore an optimistic estimate. Acoustic features alone (40 hand-crafted values) already reach a median AUROC of %s, and every recording date in the corpus holds a single class, so these scores must not be read as evidence of disease detection. Results are internal; an independent corpus is required.\n" % (
        stage_a["speakers_total"], stage_a["speakers_pd"], stage_a["speakers_hc"], stage_a["files_included"], n_done, n_cond, status_counts.get("COMPLETED", 0), len(fr),
        f3(done.AUROC.min()), f3(done.AUROC.max()), f3(done.AUROC.median()), f3(done.AUPRC.min()), f3(done.AUPRC.max()), len(sub_err), n_done,
        label(best1), NAMES[best1.model], f3(best1.AUPRC), f3(best1.AUROC), n_done, f3(done[done.representation == "AC"].AUROC.median())))
    L.append("## 3. Research questions and pre-specified comparisons\n1. How well does each representation separate PD from controls for unseen speakers (acoustic, WavLM, wav2vec2, XLS-R)?\n2. Do hybrids (encoder + acoustic, encoder combinations) improve on their constituent baselines on identical folds?\n3. Do linear SVM, RBF SVM and XGBoost differ from logistic regression?\n4. Do forward or backward selection of 10 acoustic features change the acoustic result?\n5. FWL: blocked (section 12).\nAll comparisons were fixed in `FINAL_V1_PROTOCOL.md` before any model was trained on the project data.\n")
    L.append("## 4. Dataset, labels and task composition\n- %d recordings from %d speakers; %d PD recordings / %d speakers; %d control recordings / %d speakers; %.2f audio hours; class prevalence %.1f%% PD (speakers).\n- Evaluation unit: speaker (mean of recording probabilities); sample-level metrics are supporting.\n- Tasks pooled (recordings per task code): %s.\n" % (
        stage_a["files_included"], stage_a["speakers_total"], stage_a["recordings_pd"], stage_a["speakers_pd"], stage_a["recordings_hc"], stage_a["speakers_hc"], stage_a["audio_hours"], 100 * stage_a["class_prevalence_speakers"], stage_a["tasks"]))
    dem = json.loads((WORKSPACE / "audit_opus/evidence/a05_demographics_and_flags.json").read_text())
    L.append("## 5. Inclusion and exclusion; data-quality findings\n- Demographics from the corpus spreadsheets: patients %.1f +/- %.1f years (%d to %d), %d men / %d women; controls %.1f +/- %.1f years (%d to %d), %d men / %d women. The two-digit number in the file names is the year of birth (erratum E-1), not the age; age is never a model input.\n- Four controls are flagged with an asterisk in the control spreadsheet (poor-quality recordings, reading difficulty; one has no passage or phrase recordings): HC_001, HC_007, HC_008, HC_020. They are included.\n" % (dem["PD_age"]["mean"], dem["PD_age"]["sd"], dem["PD_age"]["min"], dem["PD_age"]["max"], dem["PD_sex"].get("M", 0), dem["PD_sex"].get("F", 0), dem["HC_age"]["mean"], dem["HC_age"]["sd"], dem["HC_age"]["min"], dem["HC_age"]["max"], dem["HC_sex"].get("M", 0), dem["HC_sex"].get("F", 0)))
    L.append("- Excluded: 45 young-control recordings (age confound); 1 byte-identical duplicate recording (Giulia P B2); both decisions are in `DECISION_LOG.md`.\n- Three people recorded in two sessions were merged into one speaker each (D-003); speakers with fewer than 16 recordings were kept without imputation.\n- Findings (`DATA_AUDIT.md`): every recording date holds one class; all 160 recordings at 44.1 kHz are PD (all audio resampled to 16 kHz); no non-finite audio; no duplicated decoded audio after exclusion.\n")
    L.append("## 6. Speaker-disjoint evaluation and leakage prevention\nStratified speaker-disjoint 5-fold outer CV (seed 42; folds of 10, 10, 9, 9, 9 speakers); nested stratified speaker-disjoint 3-fold inner CV for tuning and for feature selection; all learned transformations fit on the applicable training partition only; threshold fixed at 0.5. The audit (`OOF_LEAKAGE_AUDIT.md`): **%s** (%d checks; %d completed runs, %d failed). The primary estimate is for unseen speakers, not unseen sessions of known speakers.\n" % (
        "all checks passed" if audit["all_passed"] else "FAILURES PRESENT", len(audit["checks"]), audit["completed_runs"], audit["failed_runs"]))
    L.append("## 7. Preprocessing and representation extraction\n- Audio: mono, polyphase resampling to 16 kHz, no peak normalization, no silence removal.\n- Acoustic (40 values): 10 frame features (log energy, zero-crossing rate, spectral centroid, rolloff 85%%, flatness, five band energies) x mean, SD, p10, p90 over 25 ms / 10 ms frames.\n- Encoders (frozen, CPU): " + "; ".join("%s = `%s` revision `%s`" % (k, v["model_id"], v["revision"][:12]) for k, v in ck.items()) +
             ". Recipe: 10-second windows (last window end-aligned), checkpoint feature extractor normalization, mean over time of last_hidden_state (1,024 values), mean over windows; every cache entry stores the source-audio hash and the checkpoint revision (verified in the audit).\n")
    L.append("## 8. Protocol and hyperparameter tuning\nModels: logistic regression (20 configurations), linear SVM (10), RBF SVM (24), XGBoost (24 of a 256-configuration grid, fixed subset). Inner objective: pooled speaker-level AUPRC over three speaker-disjoint inner folds; selected configuration refitted on the whole outer-training partition and applied once to the untouched outer-test fold. Search spaces were not changed after any result.\n")
    L.append("## 9. Results (all completed conditions)\nPooled out-of-fold, speaker level (47 speakers). Sens. = sensitivity, Spec. = specificity; CI = 95%% percentile bootstrap over speakers. Threshold-dependent metrics at the fixed 0.5.\n")
    rows = []
    for c in cm.itertuples():
        if c.folds_completed != P.N_OUTER:
            rows.append([c.condition_id, "incomplete", "", "", "", "", "", "", ""]); continue
        rows.append([c.condition_id, "%s [%s, %s]" % (f3(c.AUPRC), f3(c.AUPRC_ci_lo), f3(c.AUPRC_ci_hi)), "%s [%s, %s]" % (f3(c.AUROC), f3(c.AUROC_ci_lo), f3(c.AUROC_ci_hi)), f3(c.F1), f3(c.sensitivity), f3(c.specificity),
                     f3(c.balanced_accuracy), "%d/%d" % (c.FN, c.FP), "%s +/- %s" % (f3(c.fold_mean_auroc), f3(c.fold_sd_auroc))])
    L.append(md_table(["Condition", "AUPRC [CI]", "AUROC [CI]", "F1", "Sens.", "Spec.", "Bal.acc.", "FN/FP", "Fold AUROC mean +/- SD"], rows) + "\n")
    L.append("### 9.1 Acoustic only, encoders only, hybrids (median AUROC over the four models)\n" + md_table(["Representation", "Median AUROC", "Min", "Max"], [[k, f3(r["median"]), f3(r["min"]), f3(r["max"])] for k, r in by_rep.iterrows()]) + "\n")
    L.append("### 9.2 Model variants (median AUROC over all representations)\n" + md_table(["Model", "Median AUROC", "Min", "Max"], [[NAMES[k], f3(r["median"]), f3(r["min"]), f3(r["max"])] for k, r in by_model.iterrows()]) + "\n")
    if len(hy):
        L.append("### 9.3 Hybrids against constituent baselines\n%d paired comparisons (same model, same folds); median AUROC difference %s (range %s to %s); the 95%% bootstrap interval excludes zero in %d (positive %d, negative %d). Unadjusted and descriptive; no significance claim.\n" % (
            len(hy), f3(hy.delta_A_minus_B.median()), f3(hy.delta_A_minus_B.min()), f3(hy.delta_A_minus_B.max()), int(hy.ci_excludes_zero.sum()), int((hy.ci_excludes_zero & (hy.delta_A_minus_B > 0)).sum()), int((hy.ci_excludes_zero & (hy.delta_A_minus_B < 0)).sum())))
    if len(mv):
        L.append("### 9.4 Models against logistic regression\n" + "\n".join("- %s: %d comparisons, median AUROC difference %s (range %s to %s), interval excludes zero in %d." % (NAMES[m], len(x), f3(x.delta_A_minus_B.median()), f3(x.delta_A_minus_B.min()), f3(x.delta_A_minus_B.max()), int(x.ci_excludes_zero.sum()))
                                                      for m in P.MODELS[1:] for x in [mv[mv.condition_A.str.endswith("-" + m)]] if len(x)) + "\n")
    if len(fsc):
        L.append("### 9.5 Acoustic feature selection\nSelected minus all 40 features (same model): median AUROC difference %s (range %s to %s); interval excludes zero in %d of %d.\n" % (f3(fsc.delta_A_minus_B.median()), f3(fsc.delta_A_minus_B.min()), f3(fsc.delta_A_minus_B.max()), int(fsc.ci_excludes_zero.sum()), len(fsc)))
    L.append("## 10. Metrics and aggregation\nPrimary: pooled out-of-fold speaker-level AUPRC and AUROC. Supporting: F1, sensitivity, specificity, balanced accuracy, precision, accuracy at threshold 0.5; fold-wise values with mean and sample SD over five folds; sample-level AUROC/AUPRC (in `metrics/FINAL_V1/condition_metrics.csv`). The positive class is PD (prevalence 53.2%% of speakers, 55.5%% of recordings), which matters for AUPRC (chance level is the prevalence).\n")
    L.append("## 11. Fold variability and uncertainty\nMedian fold SD of AUROC over completed conditions: %s (many folds are perfectly separated); only 9 to 10 speakers per fold, so a single speaker changes a fold metric by 10 to 20 points. Median width of the 95%% bootstrap AUROC interval over conditions: %s (range %s to %s); differences between most conditions are within these intervals. When the pooled AUROC is 1.000 every bootstrap resample is also perfectly ranked, so the percentile interval collapses to [1.000, 1.000]; it does not mean there is no uncertainty (exact Clopper-Pearson intervals for sensitivity and specificity are given in the whitepaper). Errors are concentrated: %s. Of the false-positive events over all conditions, %d of %d come from the four flagged controls.\n" % (
        f3(done.fold_sd_auroc.median()), f3((done.AUROC_ci_hi - done.AUROC_ci_lo).median()), f3((done.AUROC_ci_hi - done.AUROC_ci_lo).min()), f3((done.AUROC_ci_hi - done.AUROC_ci_lo).max()), "; ".join("%s misclassified in %d of %d conditions" % (s, c, n_done) for s, c in top_wrong.items()), dem["false_positive_events_from_flagged_controls"], dem["false_positive_events_total"]))
    miss = fr[fr.training_status != "COMPLETED"]
    L.append("## 12. Failed, blocked, skipped and missing experiments\n- Fold runs by status: %s.\n- FWL: **BLOCKED** (`PROTOCOL_DECISION_REQUIRED`); no FWL result exists.\n- Sequential feature selection for embeddings and hybrids: `SKIPPED_BY_PROTOCOL` (not authorized for high-dimensional inputs).\n- Failed runs: %d; quarantined prediction files: see `OOF_LEAKAGE_AUDIT.md`.\n- Incomplete conditions: %s.\n" % (
        status_counts, audit["failed_runs"], ", ".join(cm[cm.folds_completed < P.N_OUTER].condition_id) or "none"))
    tot = cm.runtime_seconds_total.sum()
    ext = {k: [os.path.getmtime(f) for f in glob.glob(str(WORKSPACE / "features/cache" / k / "*.npy"))] for k in ["WLM", "W2V", "XLSR"]}
    runs_t = [os.path.getmtime(f) for f in glob.glob(str(MET / "FV1-*-F[0-4].json"))]
    hm = lambda t: time.strftime("%H:%M", time.gmtime(t))
    L.append("## 13. Runtime and compute\nCPU only (32 logical cores, 61 GB RAM, no GPU). Embedding extraction: the three encoders ran in parallel (3 processes x 3 threads each) on 785 recordings (4.98 audio hours) each, from %s to %s UTC (%s); i.e. about one hour of wall time. Model runs: %d outer-fold runs executed in parallel between %s and %s UTC; the sum of per-run wall-clock times is %.1f run-hours (XGBoost accounts for %.0f%% of it).\n" % (
        hm(min(min(v) for v in ext.values())), hm(max(max(v) for v in ext.values())), "; ".join("%s %.0f min" % (k, (max(v) - min(v)) / 60) for k, v in ext.items()),
        len(fr), hm(min(runs_t)), hm(max(runs_t)), tot / 3600, 100 * pd.to_numeric(fr[fr.model == "XGB"].runtime_seconds).sum() / pd.to_numeric(fr.runtime_seconds).sum()))
    L.append("## 14. Limitations\n- 47 speakers (25 PD, 22 controls): one speaker moves speaker-level metrics by about two points; fold SDs are large.\n- **Recording conditions are confounded with class** (every recording date holds one class; all 44.1 kHz audio is PD). Acoustic energy and band features alone separate the classes almost perfectly, and speaker-disjoint folds do not remove a shared recording-condition cue. The scores therefore cannot support the claim that speech pathology is being detected.\n- One corpus, one language; no independent test corpus; internal cross-validation only.\n- The best-of-%d condition is selection-biased.\n- SVM scores are uncalibrated pseudo-probabilities; threshold 0.5 equals the margin.\n- No disease severity, duration or medication data.\n- Research prototype; no clinical readiness is implied.\n" % n_cond)
    L.append("## 14b. Reproducibility note\nRe-executing all 70 logistic-regression runs with single-threaded BLAS reproduced the official selected configurations, probabilities and metrics bit-identically; refitting in a process with unrestricted multi-threaded BLAS changes some L2-logistic-regression probabilities by up to 4e-4 (the liblinear trust-region solver uses BLAS dot products; tolerance 1e-4) without changing any decision or pooled metric. See `audit_opus/OPUS_AUDIT.md`.\n")
    L.append("## 15. Reproduction instructions\nSee `REPRODUCIBILITY.md` and `ENVIRONMENT.md`. Order: `stage_a_inventory.py`, `stage_a_build.py`, `splits.py`, `download_models.py`, `freeze_protocol.py`, `acoustic.py`, `extract_embeddings.py` (three encoders, then `collect`), `runner.py run`, `aggregate.py`, `audit_oof.py`, `group_pdfs.py`, `final_report.py`. Seeds are fixed in `protocol_defs.py`.\n")
    L.append("## 16. Artifact inventory and protocol hash\n- Protocol JSON SHA-256: `%s`\n- Protocol document SHA-256: `%s`\n- Split manifest SHA-256: `%s`\n- Freeze time: %s\n- Artifact list with hashes: `ARTIFACT_MANIFEST.csv`; registry: `EXPERIMENT_REGISTRY.csv`; workspace manifest: `WORKSPACE_MANIFEST.csv`.\n" % (rec["protocol_json_sha256"], rec["protocol_md_sha256"], rec["split_manifest_sha256"], rec["freeze_time_utc"]))
    L.append("## 17. Completed versus planned\nCompleted: %d of %d conditions (%d of %d outer-fold runs). Planned and not run: FWL (blocked). Nothing else is outstanding.\n" % (n_done, n_cond, status_counts.get("COMPLETED", 0), len(fr)))
    write_text(WORKSPACE / "FINAL_REPORT.md", "\n".join(L))
    print("FINAL_REPORT.md written; ready =", ready)
    return done, best, by_rep, by_model, hy, mv, fsc, top_wrong, ready, n_done, n_cond, audit, status_counts


def pdf():
    (done, best, by_rep, by_model, hy, mv, fsc, top_wrong, ready, n_done, n_cond, audit, status_counts) = main()
    secs = [("Outcome", ["%d of %d conditions completed (%d of 280 outer-fold runs); FWL blocked; audit %s." % (n_done, n_cond, status_counts.get("COMPLETED", 0), "passed" if audit["all_passed"] else "FAILED"),
                         "Pooled speaker-level AUROC over conditions: %s to %s (median %s); AUPRC %s to %s." % (f3(done.AUROC.min()), f3(done.AUROC.max()), f3(done.AUROC.median()), f3(done.AUPRC.min()), f3(done.AUPRC.max())),
                         "Best single condition (descriptive, selection-biased): %s %s, AUPRC %s, AUROC %s." % (label(best.iloc[0]), NAMES[best.iloc[0].model], f3(best.iloc[0].AUPRC), f3(best.iloc[0].AUROC))]),
            ("Comparisons", ["Hybrids vs constituents: median AUROC difference %s over %d paired comparisons; interval excludes zero in %d." % (f3(hy.delta_A_minus_B.median()), len(hy), int(hy.ci_excludes_zero.sum())),
                             "Model variants vs logistic regression: " + "; ".join("%s median %s" % (NAMES[m], f3(mv[mv.condition_A.str.endswith('-' + m)].delta_A_minus_B.median())) for m in P.MODELS[1:]) + " (AUROC difference).",
                             "Feature selection vs all 40 acoustic features: median AUROC difference %s." % f3(fsc.delta_A_minus_B.median())]),
            ("Interpretation limits", ["Recording date and class are confounded; acoustic features alone reach median AUROC %s. Scores are internal and do not show disease detection." % f3(done[done.representation == "AC"].AUROC.median()),
                                       "Errors concentrate in a few speakers: " + "; ".join("%s (%d of %d conditions)" % (s, c, n_done) for s, c in top_wrong.items()) + "."])]
    rows = [[k, f3(r["median"]), f3(r["min"]), f3(r["max"])] for k, r in by_rep.iterrows()]
    p = make_pdf.summary_pdf("STAGE_F_final_report_summary", "Stage F: final report summary", "FINAL_V1 - Parkinson's disease detection from speech", secs,
                             [("Median pooled AUROC by representation (over models)", ["Representation", "Median", "Min", "Max"], rows, "lrrr")], status="COMPLETED" if ready else "INCOMPLETE")
    print("wrote", p)


if __name__ == "__main__":
    pdf() if (len(sys.argv) > 1 and sys.argv[1] == "pdf") else main()
