"""S12 (amendment A3, exploratory): hyperparameter study of LR, RF and XGB on the frozen WavLM vectors.

Settings are fixed in supplementary/PREDECLARED_A3.md. Three outputs:
  1. nested result with wide grids (each outer fold chooses by the FINAL_V1 inner rule),
  2. landscape: every configuration scored on every outer test fold (descriptive only, never used to choose),
  3. training-partition scores of the selected configurations (official LR/XGB, S11 RF, and the wide-grid choices).
Frozen FINAL_V1 code is imported read-only.
"""
import sys, time, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from splits import assign_folds

GRIDS = {
    "LR": [{"C": c, "penalty": p, "class_weight": w}
           for c, p, w in itertools.product([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], ["l1", "l2"], [None, "balanced"])],
    "RF": [{"n_estimators": n, "max_depth": d, "min_samples_leaf": l, "max_features": f, "class_weight": w}
           for n, d, l, f, w in itertools.product([100, 300, 1000], [2, 4, 8, None], [1, 5], ["sqrt", 0.1], [None, "balanced"])],
    "XGB": [{"max_depth": d, "learning_rate": lr, "n_estimators": n, "subsample": s, "colsample_bytree": c,
             "min_child_weight": 1, "reg_lambda": 1, "scale_pos_weight": 1}
            for d, lr, n, s, c in itertools.product([1, 2, 4, 6], [0.01, 0.03, 0.1, 0.3], [100, 300], [0.7, 1.0], [0.3, 1.0])],
}


def fit_predict(model, params, Xtr, ytr, Xte):
    if model == "RF":
        clf = RandomForestClassifier(random_state=P.SEED, n_jobs=1, **params)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    return pl.fit_predict(model, params, Xtr, ytr, Xte)


def load():
    sm = pl.load_manifest()
    X, _ = pl.load_representation("WLM", sm)
    return sm, X, sm.label.values.astype(int), sm.speaker_id.values, outer_idx(sm)


def inner_assignment(spk, y, tr, fold):
    labels = {s: int(y[spk == s][0]) for s in np.unique(spk[tr])}
    m = assign_folds(list(labels), labels, P.N_INNER, 142 + fold)
    return np.array([m[s] for s in spk[tr]])


def job(kind, model, ci, fold, k):
    sm, X, y, spk, of = load()
    tr, te = np.where(of != fold)[0], np.where(of == fold)[0]
    params = GRIDS[model][ci]
    if kind == "outer":
        return kind, model, ci, fold, k, te, fit_predict(model, params, X[tr], y[tr], X[te])
    inner_f = inner_assignment(spk, y, tr, fold)
    a, b = tr[inner_f != k], tr[inner_f == k]
    return kind, model, ci, fold, k, b, fit_predict(model, params, X[a], y[a], X[b])


def speaker_metrics(prob, idx, y, spk):
    s, t, _ = pl.speaker_scores(prob, spk[idx], y[idx])
    return pl.binary_metrics(s, t.astype(int))


def train_job(name, model, params, fold):
    """Refit on the outer-training rows and score the same rows (training-partition diagnostic)."""
    sm, X, y, spk, of = load()
    tr, te = np.where(of != fold)[0], np.where(of == fold)[0]
    p_tr = fit_predict(model, params, X[tr], y[tr], X[tr])
    return name, fold, speaker_metrics(p_tr, tr, y, spk)


def main():
    t0 = time.time()
    sm, X, y, spk, of = load()
    jobs = []
    for model, grid in GRIDS.items():
        for ci in range(len(grid)):
            for fold in range(5):
                jobs.append(("outer", model, ci, fold, -1))
                for k in range(P.N_INNER):
                    jobs.append(("inner", model, ci, fold, k))
    print("jobs", len(jobs), flush=True)
    out = Parallel(n_jobs=24, verbose=5)(delayed(job)(*j) for j in jobs)
    outer = {}
    inner = {}
    for kind, model, ci, fold, k, idx, p in out:
        if kind == "outer":
            outer[(model, ci, fold)] = (idx, p)
        else:
            inner.setdefault((model, ci, fold), []).append((idx, p))
    # inner scores and nested selection
    inner_rows, selection, nested_rows = [], {}, []
    for model, grid in GRIDS.items():
        for fold in range(5):
            scores = []
            for ci in range(len(grid)):
                idx = np.concatenate([a for a, _ in inner[(model, ci, fold)]])
                prob = np.concatenate([b for _, b in inner[(model, ci, fold)]])
                sc = pl.speaker_auprc(prob, spk[idx], y[idx])
                scores.append(sc)
                inner_rows.append({"model": model, "config_index": ci, "fold": fold, "inner_speaker_auprc": sc})
            best = max(range(len(grid)), key=lambda c: (round(scores[c], 12), -c))
            selection[(model, fold)] = best
            idx, p = outer[(model, best, fold)]
            nested_rows.append(pd.DataFrame({"model": model, "sample_id": sm.sample_id.values[idx], "speaker_id": spk[idx], "outer_fold": fold,
                                             "true_label": y[idx], "predicted_probability": p, "config_index": best}))
    pd.DataFrame(inner_rows).to_csv(RES / "S12_inner_scores.csv", index=False)
    nested = pd.concat(nested_rows, ignore_index=True)
    nested.to_csv(RES / "S12_nested_predictions.csv", index=False)
    # landscape (descriptive)
    land = []
    for model, grid in GRIDS.items():
        for ci, params in enumerate(grid):
            prob = np.full(len(y), np.nan)
            for fold in range(5):
                idx, p = outer[(model, ci, fold)]
                prob[idx] = p
            m = speaker_metrics(prob, np.arange(len(y)), y, spk)
            land.append({"model": model, "config_index": ci, "params": json.dumps(params, sort_keys=True), **{k: params[k] for k in params},
                         "auroc": m["auroc"], "auprc": m["auprc"], "f1": m["f1"], "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                         "errors": m["fn"] + m["fp"], "selected_in_folds": sum(1 for f in range(5) if selection[(model, f)] == ci)})
    pd.DataFrame(land).to_csv(RES / "S12_landscape.csv", index=False)
    # training-partition scores
    tjobs = []
    for f in range(5):
        for m in ["LR", "XGB"]:
            r = json.loads((ROOT / ("metrics/FINAL_V1/FV1-WLM-ALL-%s-F%d.json" % (m, f))).read_text())
            tjobs.append(("%s_official" % m, m, r["selected_hyperparameters"], f))
        s11 = json.loads((RES / "S11_rf_wlm_summary.json").read_text())["selection"][str(f)]["selected"]
        tjobs.append(("RF_S11", "RF", dict(s11, n_estimators=300), f))
        for m in GRIDS:
            tjobs.append(("%s_wide" % m, m, GRIDS[m][selection[(m, f)]], f))
    tout = Parallel(n_jobs=16)(delayed(train_job)(*j) for j in tjobs)
    trows = [{"variant": n, "fold": f, **{k: v for k, v in mt.items()}} for n, f, mt in tout]
    pd.DataFrame(trows).to_csv(RES / "S12_train_scores.csv", index=False)
    summ = {}
    for model in GRIDS:
        d = nested[nested.model == model]
        g = d.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first"))
        summ[model] = {"pooled": pl.binary_metrics(g.s.values, g.y.values.astype(int)),
                       "selected": {str(f): GRIDS[model][selection[(model, f)]] for f in range(5)}}
    json.dump({"nested_wide": summ, "grid_sizes": {m: len(g) for m, g in GRIDS.items()}, "runtime_seconds": time.time() - t0},
              open(RES / "S12_summary.json", "w"), indent=2, default=float)
    for model in GRIDS:
        print(model, {k: round(summ[model]["pooled"][k], 4) for k in ["f1", "sensitivity", "specificity", "auroc", "auprc"]})
    print("runtime %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
