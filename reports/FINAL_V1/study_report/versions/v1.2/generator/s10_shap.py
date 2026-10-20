"""S10: out-of-fold SHAP values for the acoustic models (XGBoost: exact TreeSHAP via pred_contribs; LR: linear SHAP on standardized inputs),
using each outer fold's refit model with the officially selected hyperparameters. Post-hoc explanation only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_importance import *  # fit_model, hp_of, sm, of, y
import xgboost as xgb
X, names = pl.load_representation("AC", sm)
out = {}
for cid, model in [("FV1-AC-ALL-LR", "LR"), ("FV1-AC-ALL-XGB", "XGB")]:
    S = np.zeros_like(X)
    for f in range(5):
        tr, te = np.where(of != f)[0], np.where(of == f)[0]
        pred, clf = fit_model(model, hp_of(cid, f), X[tr], y[tr])
        if model == "XGB":
            S[te] = clf.get_booster().predict(xgb.DMatrix(X[te]), pred_contribs=True)[:, :-1]
        else:
            mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd < 1e-12] = 1
            Z, Zt = (X[te] - mu) / sd, (X[tr] - mu) / sd
            S[te] = clf.coef_[0] * (Z - Zt.mean(0))
    np.save(RES / ("S10_shap_%s.npy" % model), S)
    out[model] = S
rows = [{"feature": n, "LR_mean_abs_shap": float(np.abs(out["LR"][:, i]).mean()), "XGB_mean_abs_shap": float(np.abs(out["XGB"][:, i]).mean()),
         "LR_corr_value_shap": float(np.corrcoef(X[:, i], out["LR"][:, i])[0, 1]) if out["LR"][:, i].std() > 0 else 0.0,
         "XGB_corr_value_shap": float(np.corrcoef(X[:, i], out["XGB"][:, i])[0, 1]) if out["XGB"][:, i].std() > 0 else 0.0} for i, n in enumerate(names)]
pd.DataFrame(rows).to_csv(RES / "S10_shap_summary.csv", index=False)
print(pd.DataFrame(rows).sort_values("XGB_mean_abs_shap", ascending=False).head(8).round(3).to_string())
