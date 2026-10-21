"""S11 (amendment A2, exploratory): WavLM + Random Forest, same folds and nested tuning as FINAL_V1.

Settings are fixed in supplementary/PREDECLARED_RF_GRID.md before this file was run. Frozen FINAL_V1 code is imported
read-only and is never edited. Output: supplementary/results/S11_*.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from splits import assign_folds

GRID = [{"max_depth": d, "min_samples_leaf": l, "class_weight": w}
        for d in [4, None] for l in [1, 2] for w in [None, "balanced"]]


def rf_fit_predict(params, Xtr, ytr, Xte):
    clf = RandomForestClassifier(n_estimators=300, random_state=P.SEED, n_jobs=8, **params)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def main():
    sm = pl.load_manifest()
    X, _ = pl.load_representation("WLM", sm)
    y = sm.label.values.astype(int)
    spk = sm.speaker_id.values
    of = outer_idx(sm)
    fold_rows, pred_rows, selection = [], [], {}
    for fold in range(5):
        tr, te = np.where(of != fold)[0], np.where(of == fold)[0]
        labels = {s: int(y[spk == s][0]) for s in np.unique(spk[tr])}
        inner_map = assign_folds(list(labels), labels, P.N_INNER, 142 + fold)
        inner_f = np.array([inner_map[s] for s in spk[tr]])
        table = []
        for ci, params in enumerate(GRID):
            probs = np.full(len(tr), np.nan)
            pos = {i: n for n, i in enumerate(tr)}
            for k in range(P.N_INNER):
                a, b = tr[inner_f != k], tr[inner_f == k]
                for i, v in zip(b, rf_fit_predict(params, X[a], y[a], X[b])):
                    probs[pos[i]] = v
            table.append({"config_index": ci, "params": params, "inner_speaker_auprc": pl.speaker_auprc(probs, spk[tr], y[tr])})
        best = max(table, key=lambda r: (round(r["inner_speaker_auprc"], 12), -r["config_index"]))
        p = rf_fit_predict(best["params"], X[tr], y[tr], X[te])
        selection[fold] = {"selected": best["params"], "inner_table": table}
        pred_rows.append(pd.DataFrame({"sample_id": sm.sample_id.values[te], "speaker_id": spk[te], "outer_fold": fold,
                                       "true_label": y[te], "predicted_probability": p}))
        print("fold", fold, "selected", best["params"], "inner AUPRC %.4f" % best["inner_speaker_auprc"], flush=True)
    pred = pd.concat(pred_rows, ignore_index=True)
    pred.to_csv(RES / "S11_rf_wlm_sample_predictions.csv", index=False)
    rows = []
    for fold in range(5):
        part = pred[pred.outer_fold == fold]
        g = part.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first"))
        rows.append(dict(fold=fold, people=len(g), **pl.binary_metrics(g.s.values, g.y.values.astype(int))))
    fold_df = pd.DataFrame(rows); fold_df.to_csv(RES / "S11_rf_wlm_fold_metrics.csv", index=False)
    m, _, g = pooled_metrics_from_pred([pred])
    g.to_csv(RES / "S11_rf_wlm_speaker_scores.csv")
    json.dump({"pooled": m, "selection": {str(k): v for k, v in selection.items()}, "grid": GRID},
              open(RES / "S11_rf_wlm_summary.json", "w"), indent=2, default=float)
    print("pooled RF:", {k: round(m[k], 4) for k in ["f1", "sensitivity", "specificity", "auroc", "auprc"]})


if __name__ == "__main__":
    main()
