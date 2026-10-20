"""Fixed-layout importance figures (no overlapping labels) and SHAP figures, for report v1.2. New file names; older figures untouched."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, PUR = "#2B5FAF", "#6B4FA3"
FAMN = {"log_energy_db": "log energy", "zcr": "zero-crossing rate", "centroid_hz": "spectral centroid", "rolloff85_hz": "roll-off 85%", "flatness": "flatness",
        "band0_log": "band 0-500 Hz", "band1_log": "band 0.5-1 kHz", "band2_log": "band 1-2 kHz", "band3_log": "band 2-4 kHz", "band4_log": "band 4-8 kHz"}
STN = {"stat_mean": "mean", "stat_std": "std", "stat_p10": "p10 (low end)", "stat_p90": "p90 (high end)"}
d = pd.read_csv(RES / "S2_acoustic_importance.csv"); lr = d[d.condition == "FV1-AC-ALL-LR"].set_index("group"); xg = d[d.condition == "FV1-AC-ALL-XGB"].set_index("group")

def hbars(ax, keys, labels, title):
    y = np.arange(len(keys)); h = .38
    ax.barh(y - h / 2, [lr.loc[k, "logloss_increase"] for k in keys], h, color=BLUE, label="Logistic regression")
    ax.barh(y + h / 2, [xg.loc[k, "logloss_increase"] for k in keys], h, color=PUR, label="XGBoost")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlabel("increase in recording log-loss when shuffled"); ax.set_title(title, fontsize=10)

fams = sorted(FAMN, key=lambda k: -lr.loc[k, "logloss_increase"])
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.55}, constrained_layout=False)
hbars(axes[0], fams, [FAMN[k] for k in fams], "Permutation importance by feature family")
hbars(axes[1], list(STN), list(STN.values()), "Permutation importance by statistic")
axes[0].legend(fontsize=8, loc="lower right")
fig.savefig(FIGS / "importance_acoustic_v2.pdf", bbox_inches="tight"); plt.close(fig)

g = xg[xg.index.str.startswith("ac_")].sort_values("gain_share_mean", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.barh(np.arange(len(g)), g.gain_share_mean, xerr=g.gain_share_sd, color=PUR, ecolor="#999", capsize=2)
ax.set_yticks(np.arange(len(g))); ax.set_yticklabels([n.replace("ac_", "") for n in g.index]); ax.invert_yaxis()
ax.set_xlabel("share of total gain (mean over 5 folds, bar = SD)"); ax.set_title("XGBoost: the 10 features with the largest gain share", fontsize=10)
fig.savefig(FIGS / "xgb_gain_top10_v2.pdf", bbox_inches="tight"); plt.close(fig)

sm, X, names = load_ac()
for model, title in [("XGB", "XGBoost (TreeSHAP)"), ("LR", "Logistic regression (linear SHAP)")]:
    S = np.load(RES / ("S10_shap_%s.npy" % model)); order = np.argsort(-np.abs(S).mean(0))[:12]
    fig, ax = plt.subplots(figsize=(7.4, 4.8)); rng = np.random.RandomState(0)
    for r, i in enumerate(order):
        v = X[:, i]; c = (v - np.percentile(v, 5)) / (np.percentile(v, 95) - np.percentile(v, 5) + 1e-12)
        sc = ax.scatter(S[:, i], r + rng.uniform(-.3, .3, len(v)), c=np.clip(c, 0, 1), cmap="coolwarm", s=5, alpha=.7)
    ax.axvline(0, c="#888", lw=.8); ax.set_yticks(range(len(order))); ax.set_yticklabels([names[i].replace("ac_", "") for i in order]); ax.invert_yaxis()
    ax.set_xlabel("SHAP value (effect on log-odds of PD; > 0 pushes towards PD)"); ax.set_title("%s: out-of-fold SHAP values of every recording" % title, fontsize=10)
    cb = fig.colorbar(sc, ax=ax, pad=.02); cb.set_label("feature value (blue low, red high)")
    fig.savefig(FIGS / ("shap_beeswarm_%s.pdf" % model), bbox_inches="tight"); plt.close(fig)
print("figures ok")
