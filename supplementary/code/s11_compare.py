"""S11 comparison: recompute saved WavLM LR and XGB predictions, then compare with WavLM RF on the same speakers.
Paired speaker bootstrap (2,000 resamples, seed 42) for differences to logistic regression. Descriptive only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *

def saved(model):
    frames = [pd.read_csv(ROOT / ("predictions/FINAL_V1/FV1-WLM-ALL-%s-F%d.csv" % (model, f))) for f in range(5)]
    d = pd.concat(frames, ignore_index=True)
    g = d.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first")).sort_index()
    return g

def rf_scores():
    d = pd.read_csv(RES / "S11_rf_wlm_sample_predictions.csv")
    return d.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first")).sort_index()

lr, xg, rf = saved("LR"), saved("XGB"), rf_scores()
assert (lr.index == rf.index).all() and (xg.index == rf.index).all() and (lr.y.values == rf.y.values).all()
y = lr.y.values.astype(int)
cond = pd.read_csv(ROOT / "metrics/FINAL_V1/condition_metrics.csv").set_index("condition_id")
rows = []
for name, g in [("WLM logistic regression (FV1-WLM-ALL-LR)", lr), ("WLM XGBoost (FV1-WLM-ALL-XGB)", xg), ("WLM Random Forest (S11, exploratory)", rf)]:
    m = pl.binary_metrics(g.s.values, y)
    rows.append({"model": name, "people": len(y), "tp": m["tp"], "fn": m["fn"], "tn": m["tn"], "fp": m["fp"], "f1": m["f1"], "sensitivity": m["sensitivity"],
                 "specificity": m["specificity"], "precision": m["precision"], "accuracy": m["accuracy"], "auroc": m["auroc"], "auprc": m["auprc"]})
out = pd.DataFrame(rows)
# consistency: the recomputed LR and XGB pooled values must equal the saved condition_metrics row
for cid, row in [("FV1-WLM-ALL-LR", out.iloc[0]), ("FV1-WLM-ALL-XGB", out.iloc[1])]:
    ok = abs(row.f1 - cond.loc[cid, "F1"]) < 1e-9 and abs(row.auprc - cond.loc[cid, "AUPRC"]) < 1e-9 and abs(row.auroc - cond.loc[cid, "AUROC"]) < 1e-9
    print("consistency", cid, "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)
rng = np.random.RandomState(42)
n = len(y)
boot = {}
for b in range(2000):
    idx = rng.randint(0, n, n)
    if len(set(y[idx])) < 2:
        continue
    m_lr = pl.binary_metrics(lr.s.values[idx], y[idx]); m_rf = pl.binary_metrics(rf.s.values[idx], y[idx]); m_xg = pl.binary_metrics(xg.s.values[idx], y[idx])
    for name, m_other in [("rf", m_rf), ("xgb", m_xg)]:
        for metric in ["auprc", "auroc", "f1"]:
            boot.setdefault("%s_%s_minus_lr" % (name, metric), []).append(m_other[metric] - m_lr[metric])
ci = []
for k, v in boot.items():
    v = np.asarray(v, dtype=float); v = v[~np.isnan(v)]
    ci.append({"difference": k, "mean": float(v.mean()), "ci_2.5": float(np.percentile(v, 2.5)), "ci_97.5": float(np.percentile(v, 97.5)), "resamples": len(v)})
out.to_csv(RES / "S11_comparison_metrics.csv", index=False)
pd.DataFrame(ci).to_csv(RES / "S11_paired_bootstrap.csv", index=False)
print(out.round(4).to_string(index=False))
print(pd.DataFrame(ci).round(4).to_string(index=False))
