"""Short English PDF per experiment group, generated from condition_metrics.csv / paired_comparisons.csv only. Usage: python group_pdfs.py GROUP"""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import protocol_defs as P
import make_pdf

MET = WORKSPACE / "metrics/FINAL_V1"
NAMES = {"LR": "Logistic regression", "LSVM": "Linear SVM", "RSVM": "RBF SVM", "XGB": "XGBoost"}
GROUPS = {
    "acoustic_all": ("EXP_acoustic_only", "Acoustic features only (all 40 features)", lambda c: c.representation == "AC" and c.feature_selection == "ALL"),
    "acoustic_fs": ("EXP_acoustic_feature_selection", "Acoustic features with forward and backward selection (k = 10)", lambda c: c.representation == "AC"),
    "encoders": ("EXP_single_encoders", "Frozen encoders alone: WavLM-large, wav2vec2-large-lv60, XLS-R 300M", lambda c: c.representation in P.ENCODERS),
    "hybrids": ("EXP_hybrid_representations", "Hybrid representations (encoders with acoustic features and encoder combinations)", lambda c: c.representation in P.HYBRIDS),
}


def fmt(x, nd=3):
    return "%.*f" % (nd, x)


def rows_for(cm, sel):
    out = []
    for c in cm[cm.apply(sel, axis=1)].itertuples():
        if c.folds_completed != P.N_OUTER:
            out.append([c.representation + ("/" + c.feature_selection if c.feature_selection != "ALL" else ""), NAMES[c.model], "incomplete (%d/5 folds)" % c.folds_completed, "", "", "", "", ""])
            continue
        out.append([c.representation + ("/" + c.feature_selection if c.feature_selection != "ALL" else ""), NAMES[c.model],
                    "%s [%s, %s]" % (fmt(c.AUPRC), fmt(c.AUPRC_ci_lo), fmt(c.AUPRC_ci_hi)), "%s [%s, %s]" % (fmt(c.AUROC), fmt(c.AUROC_ci_lo), fmt(c.AUROC_ci_hi)),
                    fmt(c.F1), fmt(c.sensitivity), fmt(c.specificity), fmt(c.balanced_accuracy)])
    return out


HEAD = ["Representation", "Model", "AUPRC [95% CI]", "AUROC [95% CI]", "F1", "Sens.", "Spec.", "Bal.acc."]
SPEC = "llllllll"


def common_caveats():
    return ["Primary metrics are pooled speaker-level out-of-fold AUPRC and AUROC (47 speakers: 25 PD, 22 controls; prevalence 53.2% PD); F1, sensitivity, specificity and balanced accuracy use the fixed threshold 0.5.",
            "Intervals are percentile bootstrap intervals over speakers (2,000 resamples); with 47 speakers one person moves a metric by roughly 2 points. Comparisons are descriptive; no significance is claimed.",
            "Recording date and class are confounded in this corpus (every date holds one class; all 44.1 kHz audio is PD), so a high score does not show disease detection."]


def build(group):
    cm = pd.read_csv(MET / "condition_metrics.csv")
    comp = pd.read_csv(MET / "paired_comparisons.csv")
    name, title, sel = GROUPS[group]
    sub = cm[cm.apply(sel, axis=1)]
    done = sub[sub.folds_completed == P.N_OUTER]
    bullets = []
    if len(done):
        b = done.sort_values(["AUPRC", "AUROC"], ascending=False).iloc[0]
        w = done.sort_values(["AUPRC", "AUROC"]).iloc[0]
        bullets.append("%d of %d conditions in this group are complete." % (len(done), len(sub)))
        bullets.append("Highest pooled AUPRC (descriptive, selection-biased among %d conditions): %s %s, AUPRC %s, AUROC %s." % (len(done), b.representation + ("/" + b.feature_selection if b.feature_selection != "ALL" else ""), NAMES[b.model], fmt(b.AUPRC), fmt(b.AUROC)))
        bullets.append("Lowest: %s %s, AUPRC %s, AUROC %s." % (w.representation + ("/" + w.feature_selection if w.feature_selection != "ALL" else ""), NAMES[w.model], fmt(w.AUPRC), fmt(w.AUROC)))
        bullets.append("Range of AUROC over the group: %s to %s; median %s." % (fmt(done.AUROC.min()), fmt(done.AUROC.max()), fmt(done.AUROC.median())))
        nerr = (done.FN + done.FP)
        bullets.append("Speaker errors at threshold 0.5 (false negatives + false positives) per condition: minimum %d, median %d, maximum %d (out of 47 speakers)." % (nerr.min(), nerr.median(), nerr.max()))
        fm = done[["fold_mean_auroc", "fold_sd_auroc"]].median()
        bullets.append("Fold variability: median over conditions of the mean fold AUROC is %s with a median SD of %s over the five outer folds (9 to 10 speakers per fold)." % (fmt(fm.fold_mean_auroc), fmt(fm.fold_sd_auroc)))
    sections = [("Results", bullets)]
    tables = [("Pooled out-of-fold speaker-level results", HEAD, rows_for(cm, sel), SPEC)]
    ids = set(sub.condition_id)
    if group == "hybrids":
        c = comp[(comp.comparison_type == "hybrid_vs_constituent") & (comp.condition_A.isin(ids)) & (comp.metric == "AUROC")]
        if len(c):
            ex = c[c.ci_excludes_zero]
            sections.append(("Hybrid versus its constituent baselines (same model, same folds)",
                             ["%d paired comparisons; the 95%% bootstrap interval of the AUROC difference excludes zero in %d (positive in %d, negative in %d); unadjusted and descriptive." % (len(c), len(ex), int((ex.delta_A_minus_B > 0).sum()), int((ex.delta_A_minus_B < 0).sum())),
                              "Median AUROC difference (hybrid minus constituent): %s; range %s to %s." % (fmt(c.delta_A_minus_B.median()), fmt(c.delta_A_minus_B.min()), fmt(c.delta_A_minus_B.max()))]))
            tab = c.groupby("note").first().reset_index()
    if group == "acoustic_fs":
        sel_rows = []
        for fs in ["FORWARD", "BACKWARD"]:
            for m in P.MODELS:
                cnt = Counter()
                for f in range(P.N_OUTER):
                    r = json.loads((MET / ("FV1-AC-%s-%s-F%d.json" % (fs, m, f))).read_text())
                    cnt.update(r["selected_features"])
                top = ", ".join("%s (%d/5)" % (k.replace("ac_", ""), v) for k, v in cnt.most_common(4))
                stable = sum(1 for v in cnt.values() if v == 5)
                sel_rows.append([fs, NAMES[m], str(stable), str(len(cnt)), top])
        sections.append(("Note on the selection table", ["The selector is a fixed logistic regression (C = 1), so the selected features in a fold do not depend on the final model; the four rows of each method are therefore identical (verified in the audit)."]))
        tables.append(("Selection stability across the five outer folds (features chosen in all 5 folds / distinct features ever chosen; most frequent)", ["Method", "Model", "In 5/5", "Distinct", "Most frequent features"], sel_rows, "llrrl"))
        c = comp[(comp.comparison_type == "feature_selection_vs_all_features") & (comp.metric == "AUROC")]
        if len(c):
            sections.append(("Selection versus all 40 features (same model)", ["AUROC difference (selected minus all), median %s, range %s to %s; interval excludes zero in %d of %d comparisons (descriptive)." % (fmt(c.delta_A_minus_B.median()), fmt(c.delta_A_minus_B.min()), fmt(c.delta_A_minus_B.max()), int(c.ci_excludes_zero.sum()), len(c))]))
    sections.append(("Reading the table", common_caveats()))
    pdf = make_pdf.summary_pdf(name, title, "FINAL_V1 experiment group summary", sections, tables,
                               status="COMPLETED" if len(done) == len(sub) else "INCOMPLETE")
    print("wrote", pdf)


def build_models():
    cm = pd.read_csv(MET / "condition_metrics.csv")
    comp = pd.read_csv(MET / "paired_comparisons.csv")
    done = cm[cm.folds_completed == P.N_OUTER]
    reps = P.BASE_REPS + list(P.HYBRIDS)
    rows = []
    for rep in reps:
        for fs in (["ALL", "FORWARD", "BACKWARD"] if rep == "AC" else ["ALL"]):
            r = [rep + ("/" + fs if fs != "ALL" else "")]
            for m in P.MODELS:
                c = done[(done.representation == rep) & (done.feature_selection == fs) & (done.model == m)]
                r.append("%s / %s" % (fmt(c.AUPRC.iloc[0]), fmt(c.AUROC.iloc[0])) if len(c) else "pending")
            rows.append(r)
    sections = []
    bl = []
    for m in P.MODELS:
        d = done[done.model == m]
        if len(d):
            bl.append("%s: %d conditions, median AUROC %s, median AUPRC %s, median speaker errors %d." % (NAMES[m], len(d), fmt(d.AUROC.median()), fmt(d.AUPRC.median()), int((d.FN + d.FP).median())))
    sections.append(("Model variants over all completed representations", bl))
    c = comp[(comp.comparison_type == "model_vs_logistic_regression") & (comp.metric == "AUROC")]
    cb = []
    for m in P.MODELS[1:]:
        x = c[c.condition_A.str.endswith("-" + m)]
        if len(x):
            cb.append("%s minus logistic regression (same representation): %d comparisons, median AUROC difference %s (range %s to %s); interval excludes zero in %d (positive %d, negative %d). Unadjusted, descriptive." % (
                NAMES[m], len(x), fmt(x.delta_A_minus_B.median()), fmt(x.delta_A_minus_B.min()), fmt(x.delta_A_minus_B.max()), int(x.ci_excludes_zero.sum()),
                int((x.ci_excludes_zero & (x.delta_A_minus_B > 0)).sum()), int((x.ci_excludes_zero & (x.delta_A_minus_B < 0)).sum())))
    sections.append(("Paired comparison with logistic regression", cb))
    sections.append(("Reading the table", common_caveats() + ["The SVM pseudo-probability is the logistic function of the decision value, not a calibrated probability; the 0.5 threshold equals the SVM margin."]))
    pdf = make_pdf.summary_pdf("EXP_model_variants", "Model variants: logistic regression, linear SVM, RBF SVM, XGBoost", "FINAL_V1 experiment group summary", sections,
                               [("AUPRC / AUROC (pooled, speaker level) by representation and model", ["Representation"] + [NAMES[m] for m in P.MODELS], rows, "lllll")],
                               status="COMPLETED" if len(done) == len(cm) else "INCOMPLETE")
    print("wrote", pdf)


def build_fwl():
    pdf = make_pdf.summary_pdf("EXP_FWL_blocked", "Frisch-Waugh-Lovell (FWL) experiment: BLOCKED", "FINAL_V1 experiment group summary",
        [("Status", ["BLOCKED: PROTOCOL_DECISION_REQUIRED. The execution prompt requires the scientific purpose, response, nuisance covariates, residualization procedure, statistic and stopping rule to be frozen before use.",
                     "No FINAL_V1 protocol was supplied and none of these items was defined by the project owner; the agent did not improvise an FWL algorithm.",
                     "No FWL result exists in this study and none is implied. The registry carries one row (FV1-FWL) with status BLOCKED."]),
         ("What is needed to unblock", ["A written definition of: purpose (for example removing the linear effect of recording-condition variables from the features), the response variable, the nuisance covariates and how they are measured, whether residualization is fitted inside each training partition only, the reported statistic, and a stopping rule.",
                                        "Because this changes the frozen protocol, it must be approved and recorded as an amendment; amended runs must not be described as following the original specification."])],
        status="BLOCKED")
    print("wrote", pdf)


if __name__ == "__main__":
    args = sys.argv[1:] or list(GROUPS) + ["models", "fwl"]
    for g in args:
        if g == "models": build_models()
        elif g == "fwl": build_fwl()
        else: build(g)
