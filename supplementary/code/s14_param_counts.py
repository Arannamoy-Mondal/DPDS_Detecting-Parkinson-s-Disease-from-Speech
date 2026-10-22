"""S14 (amendment A3, exploratory): exact sizes of the fitted classifiers (for the model-description table).

For each outer fold, the configuration chosen by the official runs (LR, XGB) or by S11 (RF) is refitted on the
outer-training rows exactly as in the pipeline, and its size is counted. No prediction is made or reported here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def main():
    sm = pl.load_manifest()
    X, _ = pl.load_representation("WLM", sm)
    y = sm.label.values.astype(int)
    of = outer_idx(sm)
    rows = []
    s11 = json.loads((RES / "S11_rf_wlm_summary.json").read_text())["selection"]
    for f in range(5):
        tr = np.where(of != f)[0]
        A, _ = pl._standardize(X[tr], X[tr])
        p = json.loads((ROOT / ("metrics/FINAL_V1/FV1-WLM-ALL-LR-F%d.json" % f)).read_text())["selected_hyperparameters"]
        lr = LogisticRegression(solver="liblinear", C=p["C"], penalty=p["penalty"], class_weight=p["class_weight"], max_iter=3000, random_state=P.SEED).fit(A, y[tr])
        rows.append({"model": "LR", "fold": f, "parameters": int(lr.coef_.size + 1), "nonzero_weights": int((lr.coef_ != 0).sum()),
                     "trees": 0, "nodes": 0, "leaves": 0, "settings": json.dumps(p, sort_keys=True)})
        p = s11[str(f)]["selected"]
        rf = RandomForestClassifier(n_estimators=300, random_state=P.SEED, n_jobs=8, **p).fit(X[tr], y[tr])
        nodes = sum(t.tree_.node_count for t in rf.estimators_)
        leaves = sum(t.tree_.n_leaves for t in rf.estimators_)
        rows.append({"model": "RF", "fold": f, "parameters": int(nodes), "nonzero_weights": 0, "trees": 300, "nodes": int(nodes), "leaves": int(leaves),
                     "settings": json.dumps(p, sort_keys=True)})
        p = json.loads((ROOT / ("metrics/FINAL_V1/FV1-WLM-ALL-XGB-F%d.json" % f)).read_text())["selected_hyperparameters"]
        spw = float((y[tr] == 0).sum()) / max(1.0, float((y[tr] == 1).sum())) if p["scale_pos_weight"] == "balanced" else p["scale_pos_weight"]
        xg = XGBClassifier(tree_method="hist", n_jobs=8, random_state=P.SEED, eval_metric="logloss", verbosity=0, learning_rate=p["learning_rate"],
                           max_depth=p["max_depth"], n_estimators=p["n_estimators"], min_child_weight=p["min_child_weight"], subsample=p["subsample"],
                           colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"], scale_pos_weight=spw).fit(X[tr], y[tr])
        df = xg.get_booster().trees_to_dataframe()
        rows.append({"model": "XGB", "fold": f, "parameters": int(len(df)), "nonzero_weights": 0, "trees": int(df.Tree.nunique()), "nodes": int(len(df)),
                     "leaves": int((df.Feature == "Leaf").sum()), "settings": json.dumps(p, sort_keys=True)})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "S14_param_counts.csv", index=False)
    print(out[["model", "fold", "parameters", "nonzero_weights", "trees", "leaves"]].to_string(index=False))


if __name__ == "__main__":
    main()
