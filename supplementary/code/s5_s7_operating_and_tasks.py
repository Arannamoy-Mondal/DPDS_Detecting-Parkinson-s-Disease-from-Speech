"""S5: operating characteristics (exact Clopper-Pearson CIs at threshold 0.5, ROC points, predictive values at screening prevalences,
validation sample-size planning). S7: task-family analysis from the official out-of-fold predictions (no retraining)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
from scipy.stats import beta, norm
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1); hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k); return float(lo), float(hi)

cm = pd.read_csv(ROOT / "metrics/FINAL_V1/condition_metrics.csv")
rows = []
for c in cm.itertuples():
    sl, sh = cp(int(c.TP), int(c.TP + c.FN)); pl_, ph = cp(int(c.TN), int(c.TN + c.FP))
    rows.append({"condition_id": c.condition_id, "sensitivity": c.sensitivity, "sens_cp_lo": sl, "sens_cp_hi": sh, "specificity": c.specificity, "spec_cp_lo": pl_, "spec_cp_hi": ph,
                 "TP": int(c.TP), "FN": int(c.FN), "TN": int(c.TN), "FP": int(c.FP)})
cpdf = pd.DataFrame(rows); cpdf.to_csv(RES / "S5_exact_intervals.csv", index=False)

# predictive values at screening prevalences (point estimate and pessimistic: lower CP bounds)
def ppv(se, sp, pr): return se * pr / (se * pr + (1 - sp) * (1 - pr))
def npv(se, sp, pr): return sp * (1 - pr) / (sp * (1 - pr) + (1 - se) * pr)
prev = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
pv = []
for cid in ["FV1-WLM-ALL-LR", "FV1-AC-ALL-LR", "FV1-XLSR-ALL-LR", "FV1-H8-ALL-XGB"]:
    r = cpdf[cpdf.condition_id == cid].iloc[0]
    for p in prev:
        pv.append({"condition_id": cid, "prevalence": p, "PPV_point": ppv(r.sensitivity, min(r.specificity, 0.999999), p), "NPV_point": npv(r.sensitivity, r.specificity, p),
                   "PPV_pessimistic": ppv(r.sens_cp_lo, r.spec_cp_lo, p), "NPV_pessimistic": npv(r.sens_cp_lo, r.spec_cp_lo, p)})
pd.DataFrame(pv).to_csv(RES / "S5_predictive_values.csv", index=False)

# sample-size planning for a prospective validation: n per class so that the exact 95% CI half-width / lower bound meets a target
plan = []
for true_val in [0.85, 0.90, 0.95]:
    for lb in [0.75, 0.80, 0.85, 0.90]:
        if lb >= true_val: continue
        n = 10
        while n < 5000:
            k = int(round(true_val * n)); lo, _ = cp(k, n)
            if lo >= lb: break
            n += 1
        plan.append({"assumed_true_sensitivity_or_specificity": true_val, "required_lower_95_bound": lb, "people_needed_in_that_class": n})
pd.DataFrame(plan).to_csv(RES / "S5_sample_size_plan.csv", index=False)
# AUROC precision (Hanley-McNeil) for equal groups
hm = []
for A in [0.80, 0.85, 0.90]:
    for n in [50, 100, 200, 300, 500]:
        q1, q2 = A / (2 - A), 2 * A * A / (1 + A); se = np.sqrt((A * (1 - A) + (n - 1) * (q1 - A * A) + (n - 1) * (q2 - A * A)) / (n * n))
        hm.append({"true_auroc": A, "people_per_class": n, "se": se, "ci_halfwidth": 1.96 * se})
pd.DataFrame(hm).to_csv(RES / "S5_auroc_precision.csv", index=False)

# ROC points for figures
sm = pl.load_manifest()
roc_rows = []
for cid in ["FV1-AC-ALL-LR", "FV1-WLM-ALL-LR", "FV1-W2V-ALL-LR", "FV1-XLSR-ALL-LR", "FV1-H8-ALL-XGB", "FV1-WLM-ALL-RSVM"]:
    d = pd.concat([pd.read_csv(ROOT / "predictions/FINAL_V1" / ("%s-F%d.csv" % (cid, f))) for f in range(5)])
    g = d.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first"))
    fpr, tpr, thr = roc_curve(g.y, g.s)
    for a, b, t in zip(fpr, tpr, thr): roc_rows.append({"condition_id": cid, "fpr": a, "tpr": b, "threshold": t})
pd.DataFrame(roc_rows).to_csv(RES / "S5_roc_points.csv", index=False)

# S7: task families (no retraining): speaker score from one task family only
fam = lambda t: "vowels" if t.startswith("V") else ("syllables" if t.startswith("D") else ("passages" if t.startswith("B") else ("word list" if t == "PR1" else "phrase")))
trows = []
for cid in ["FV1-AC-ALL-LR", "FV1-WLM-ALL-LR", "FV1-W2V-ALL-LR", "FV1-XLSR-ALL-LR", "FV1-H3-ALL-LR", "FV1-H8-ALL-XGB"]:
    d = pd.concat([pd.read_csv(ROOT / "predictions/FINAL_V1" / ("%s-F%d.csv" % (cid, f))) for f in range(5)]).merge(sm[["sample_id", "task_code", "seconds"]], on="sample_id")
    d["family"] = d.task_code.map(fam)
    for fm, g in list(d.groupby("family")) + [("all tasks", d)]:
        s = g.groupby("speaker_id").agg(p=("predicted_probability", "mean"), y=("true_label", "first"))
        m = pl.binary_metrics(s.p.values, s.y.values.astype(int))
        trows.append({"condition_id": cid, "task_family": fm, "speakers": len(s), "recordings": len(g), "median_seconds_per_recording": float(g.seconds.median()),
                      "speaker_auroc": m["auroc"], "speaker_auprc": m["auprc"], "speaker_errors_at_0.5": m["fn"] + m["fp"], "recording_auroc": roc_auc_score(g.true_label, g.predicted_probability)})
pd.DataFrame(trows).to_csv(RES / "S7_task_families.csv", index=False)
print(cpdf.head(3).round(3).to_string()); print(pd.DataFrame(trows).round(3).to_string()); print(pd.DataFrame(plan).to_string())
