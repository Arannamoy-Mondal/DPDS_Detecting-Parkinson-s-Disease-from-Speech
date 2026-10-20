"""S1: acoustic feature-family and statistic ablations with the same nested protocol (LR and XGBoost, frozen grids, frozen folds)."""
import sys, time
from joblib import Parallel, delayed
sys.path.insert(0, str(Path := __import__("pathlib").Path(__file__).resolve().parent))
from supp_common import *

sm, X, names = load_ac()
idx = {n: i for i, n in enumerate(names)}
fam_cols = {f: [idx["ac_%s_%s" % (f, s)] for s in STATS] for f in FAMILIES}
stat_cols = {s: [idx["ac_%s_%s" % (f, s)] for f in FAMILIES] for s in STATS}
subsets = {}
for f in FAMILIES:
    subsets["drop_" + f] = [i for i in range(40) if i not in fam_cols[f]]
    subsets["only_" + f] = fam_cols[f]
for s in STATS:
    subsets["only_stat_" + s] = stat_cols[s]
    subsets["drop_stat_" + s] = [i for i in range(40) if i not in stat_cols[s]]
subsets["only_energy"] = fam_cols["log_energy_db"]
subsets["only_spectral_shape"] = sum([fam_cols[f] for f in ["centroid_hz", "rolloff85_hz", "flatness"]], [])
subsets["only_band_energies"] = sum([fam_cols[f] for f in FAMILIES if f.startswith("band")], [])
subsets["only_zcr_and_shape"] = sum([fam_cols[f] for f in ["zcr", "centroid_hz", "rolloff85_hz", "flatness"]], [])
subsets["no_level_features"] = sum([fam_cols[f] for f in ["zcr", "centroid_hz", "rolloff85_hz", "flatness"]], [])  # same as above, kept for naming clarity in the table
of = outer_idx(sm)


def one(name, model, fold):
    out = RES / "S1_preds" / ("%s__%s__F%d.csv" % (name, model, fold))
    if out.exists():
        return name, model, fold, "cached"
    cols = subsets[name]
    r = pl.run_fold({"model": model, "feature_selection": "ALL"}, fold, X[:, cols], [names[c] for c in cols], sm)
    p = r["pred"]; p["subset"] = name; p["model"] = model
    out.parent.mkdir(parents=True, exist_ok=True); p.to_csv(out, index=False)
    json.dump({"best_params": r["best_params"], "runtime": r["runtime_seconds"]}, open(str(out) + ".json", "w"))
    return name, model, fold, "ok"


if __name__ == "__main__":
    jobs = [(n, m, f) for n in subsets for m in ["LR", "XGB"] for f in range(5)]
    t = time.time()
    res = Parallel(n_jobs=14, verbose=0)(delayed(one)(*j) for j in jobs)
    print("done", len(res), "jobs in", round(time.time() - t), "s")
    rows = []
    for n in subsets:
        for m in ["LR", "XGB"]:
            frames = [pd.read_csv(RES / "S1_preds" / ("%s__%s__F%d.csv" % (n, m, f))) for f in range(5)]
            met, d, g = pooled_metrics_from_pred(frames)
            ok = (len(d) == len(sm)) and not d.sample_id.duplicated().any() and set(d.sample_id) == set(sm.sample_id)
            rows.append({"subset": n, "model": m, "n_features": len(subsets[n]), "AUROC": met["auroc"], "AUPRC": met["auprc"], "F1": met["f1"], "sensitivity": met["sensitivity"],
                         "specificity": met["specificity"], "balanced_accuracy": met["balanced_accuracy"], "FN": met["fn"], "FP": met["fp"], "coverage_ok": ok})
    pd.DataFrame(rows).to_csv(RES / "S1_ablation.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
