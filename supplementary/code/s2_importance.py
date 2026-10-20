"""S2: feature importance. (a) acoustic features: LR standardized coefficients, XGBoost total gain, permutation importance (single feature,
family, statistic) measured on the outer-test folds of the official runs with the officially selected hyperparameters; (b) block importance in
hybrids (permute a whole representation block). Post-hoc explanation only; no model selection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

sm = pl.load_manifest(); of = outer_idx(sm); y = sm.label.values.astype(int); spk = np.asarray(sm.speaker_id.astype(str).tolist())
R = 20


def hp_of(cid, f):
    return json.load(open(ROOT / "metrics/FINAL_V1" / ("%s-F%d.json" % (cid, f))))["selected_hyperparameters"]


def fit_model(model, params, Xa, ya):
    """Return a predict function (same preprocessing as pipeline.fit_predict)."""
    if model == "XGB":
        from xgboost import XGBClassifier
        spw = params["scale_pos_weight"]; spw = float((ya == 0).sum()) / max(1.0, float((ya == 1).sum())) if spw == "balanced" else spw
        clf = XGBClassifier(tree_method="hist", n_jobs=2, random_state=P.SEED, eval_metric="logloss", verbosity=0, learning_rate=params["learning_rate"], max_depth=params["max_depth"],
                            n_estimators=params["n_estimators"], min_child_weight=params["min_child_weight"], subsample=params["subsample"], colsample_bytree=params["colsample_bytree"],
                            reg_lambda=params["reg_lambda"], scale_pos_weight=spw).fit(Xa, ya)
        return (lambda Xb: clf.predict_proba(Xb)[:, 1]), clf
    mu, sd = Xa.mean(0), Xa.std(0); sd[sd < 1e-12] = 1
    clf = LogisticRegression(solver="liblinear", C=params["C"], penalty=params["penalty"], class_weight=params["class_weight"], max_iter=3000, random_state=P.SEED).fit((Xa - mu) / sd, ya)
    return (lambda Xb: clf.predict_proba((Xb - mu) / sd)[:, 1]), clf


def pooled_auc_ll(p):
    d = pd.DataFrame({"p": p, "s": spk, "y": y}).groupby("s").agg(p=("p", "mean"), y=("y", "first"))
    return roc_auc_score(d.y, d.p), log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))


def permutation_importance(cid, model, X, groups, seed=0):
    """groups: dict name -> list of columns. Returns baseline and per-group mean drop of pooled speaker AUROC and increase of recording log-loss."""
    fits = []
    for f in range(5):
        tr, te = np.where(of != f)[0], np.where(of == f)[0]
        pred, clf = fit_model(model, hp_of(cid, f), X[tr], y[tr]); fits.append((te, pred, clf))
    base = np.zeros(len(y))
    for te, pred, _ in fits: base[te] = pred(X[te])
    b_auc, b_ll = pooled_auc_ll(base)
    rng = np.random.RandomState(seed); out = []
    for name, cols in groups.items():
        aucs, lls = [], []
        for r in range(R):
            p = np.zeros(len(y))
            for te, pred, _ in fits:
                Xp = X[te].copy(); perm = rng.permutation(len(te)); Xp[:, cols] = Xp[perm][:, cols]; p[te] = pred(Xp)
            a, l = pooled_auc_ll(p); aucs.append(a); lls.append(l)
        out.append({"group": name, "n_columns": len(cols), "auroc_drop": b_auc - np.mean(aucs), "auroc_drop_sd": np.std(aucs), "logloss_increase": np.mean(lls) - b_ll, "logloss_increase_sd": np.std(lls)})
    return b_auc, b_ll, pd.DataFrame(out), fits


def run_acoustic():
    X, names = pl.load_representation("AC", sm)
    singles = {n: [i] for i, n in enumerate(names)}
    fams = {f: [names.index("ac_%s_%s" % (f, s)) for s in STATS] for f in FAMILIES}
    stats_ = {"stat_" + s: [names.index("ac_%s_%s" % (f, s)) for f in FAMILIES] for s in STATS}
    res = {}
    for cid, model in [("FV1-AC-ALL-LR", "LR"), ("FV1-AC-ALL-XGB", "XGB")]:
        b_auc, b_ll, df1, fits = permutation_importance(cid, model, X, {**singles, **fams, **stats_})
        df1["condition"] = cid; df1["baseline_auroc"] = b_auc; df1["baseline_logloss"] = b_ll
        if model == "LR":
            coefs = np.array([f[2].coef_[0] for f in fits]); df1 = df1.merge(pd.DataFrame({"group": names, "coef_mean": coefs.mean(0), "coef_sd": coefs.std(0),
                                                                                              "coef_sign_consistent": (np.sign(coefs) == np.sign(coefs.mean(0))).all(0)}), on="group", how="left")
        else:
            gains = []
            for f in fits:
                g = f[2].get_booster().get_score(importance_type="total_gain"); v = np.array([g.get("f%d" % i, 0.0) for i in range(len(names))]); gains.append(v / v.sum() if v.sum() else v)
            gains = np.array(gains); df1 = df1.merge(pd.DataFrame({"group": names, "gain_share_mean": gains.mean(0), "gain_share_sd": gains.std(0)}), on="group", how="left")
        res[cid] = df1
    pd.concat(res.values()).to_csv(RES / "S2_acoustic_importance.csv", index=False)
    print("acoustic importance done")


def run_blocks():
    rows = []
    for rep in ["H1", "H2", "H3", "H7", "H8"]:
        comps = P.components(rep)
        X, names = pl.load_representation(rep, sm)
        groups, start = {}, 0
        for c in comps:
            d = P.DIMS[c]; groups[c] = list(range(start, start + d)); start += d
        for model in ["LR", "XGB"]:
            cid = "FV1-%s-ALL-%s" % (rep, model)
            b_auc, b_ll, df1, fits = permutation_importance(cid, model, X, groups, seed=1)
            df1["condition"] = cid; df1["baseline_auroc"] = b_auc; df1["baseline_logloss"] = b_ll
            if model == "LR":
                coefs = np.abs(np.array([f[2].coef_[0] for f in fits]))
                share = {c: float(coefs[:, cols].sum(1).mean() / coefs.sum(1).mean()) for c, cols in groups.items()}
                df1["coef_mass_share"] = df1.group.map(share)
            else:
                shares = []
                for f in fits:
                    g = f[2].get_booster().get_score(importance_type="total_gain"); v = np.array([g.get("f%d" % i, 0.0) for i in range(X.shape[1])])
                    shares.append({c: v[cols].sum() / v.sum() for c, cols in groups.items()})
                df1["gain_share"] = df1.group.map({c: float(np.mean([s[c] for s in shares])) for c in groups})
            rows.append(df1)
            print("block", cid, "done", flush=True)
    pd.concat(rows).to_csv(RES / "S2_block_importance.csv", index=False)


if __name__ == "__main__":
    which = sys.argv[1]
    {"acoustic": run_acoustic, "blocks": run_blocks}[which]()
