"""Figures for the whitepaper (all from verified FINAL_V1 files and supplementary results)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GRAY, RED, AMB, GRN, PUR = "#2B5FAF", "#8A9099", "#C0392B", "#D98A00", "#2E7D4F", "#6B4FA3"
NAMES = {"LR": "Logistic regression", "LSVM": "Linear SVM", "RSVM": "RBF SVM", "XGB": "XGBoost"}
MC = {"LR": BLUE, "LSVM": GRN, "RSVM": AMB, "XGB": PUR}
cm = pd.read_csv(ROOT / "metrics/FINAL_V1/condition_metrics.csv"); cm["label"] = cm.representation + cm.feature_selection.map(lambda s: "" if s == "ALL" else "/" + s[:3])
sm = pd.read_csv(ROOT / "manifests/SAMPLE_MANIFEST.csv")
REPS = ["AC", "AC/FOR", "AC/BAC", "WLM", "W2V", "XLSR", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]


def save(fig, name):
    fig.savefig(FIGS / (name + ".pdf"), bbox_inches="tight"); plt.close(fig)


def f_confound():
    d = sm.groupby(["date", "label", "sample_rate"]).speaker_id.nunique().reset_index().sort_values("date")
    fig, ax = plt.subplots(figsize=(7.2, 2.4)); x = np.arange(len(d))
    ax.bar(x, d.speaker_id, color=[BLUE if l == 1 else GRAY for l in d.label], edgecolor=["black" if r == 44100 else "none" for r in d.sample_rate], hatch=None)
    for i, r in enumerate(d.itertuples()):
        ax.text(i, r.speaker_id + .25, "%d" % r.speaker_id, ha="center", fontsize=7)
        if r.sample_rate == 44100: ax.bar(i, r.speaker_id, color="none", edgecolor="black", hatch="///")
    ax.set_xticks(x); ax.set_xticklabels([pd.Timestamp(t).strftime("%d %b %Y") for t in d.date], rotation=30, fontsize=7)
    ax.axvline(3.5, ls=":", c="#999"); ax.text(1.5, d.speaker_id.max() + 1.2, "Parkinson's disease", ha="center", color=BLUE, fontsize=8); ax.text(6.5, d.speaker_id.max() + 1.2, "controls", ha="center", color="#555", fontsize=8)
    ax.set_ylim(0, d.speaker_id.max() + 2.5); ax.set_ylabel("speakers recorded"); ax.set_title("Recording sessions: every date holds one class (hatched = recorded at 44.1 kHz)", fontsize=8.5)
    save(fig, "wp_confound_timeline")


def f_forest():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, m in zip(axes, ["LR", "XGB"]):
        x = cm[cm.model == m].set_index("label").loc[REPS]
        yy = np.arange(len(REPS))[::-1]
        ax.errorbar(x.AUROC, yy, xerr=[x.AUROC - x.AUROC_ci_lo, x.AUROC_ci_hi - x.AUROC], fmt="o", color=MC[m], ecolor="#9aa", capsize=2, ms=4)
        ax.set_yticks(yy); ax.set_yticklabels(REPS); ax.set_xlim(0.8, 1.005); ax.axvline(1, c="#ccc", lw=.6); ax.set_title(NAMES[m] + ": pooled AUROC and 95% bootstrap CI", fontsize=8.5); ax.set_xlabel("speaker-level AUROC")
    save(fig, "wp_forest_auroc")


def f_univariate():
    u = pd.read_csv(RES / "S0_univariate_features.csv").sort_values("AUROC_abs", ascending=True)
    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    y = np.arange(len(u))
    ax.barh(y, u.AUROC_abs, color=[BLUE if d == "PD higher" else AMB for d in u.direction], alpha=.85, label="all 47 speakers")
    ax.scatter([max(a, 1 - a) for a in u.AUROC_class_native16k_only], y, s=12, c="black", marker="|", label="16-kHz recordings only (40 speakers)")
    ax.set_yticks(y); ax.set_yticklabels([n.replace("ac_", "") for n in u.feature], fontsize=6.5); ax.set_xlim(.5, 1); ax.axvline(.8, ls=":", c="#999")
    ax.set_xlabel("speaker-level AUROC of the single feature (direction-free: max(AUC, 1-AUC))"); ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Univariate separation of each acoustic feature (blue: higher in PD, amber: lower in PD)", fontsize=8.5)
    save(fig, "wp_univariate_features")


def f_ablation():
    a = pd.read_csv(RES / "S1_ablation.csv")
    a = a[~a.subset.isin(["only_energy", "no_level_features"])]
    fams = ["log_energy_db", "zcr", "centroid_hz", "rolloff85_hz", "flatness", "band0_log", "band1_log", "band2_log", "band3_log", "band4_log"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=False)
    for ax, (pref, title, keys) in zip(axes, [("only_", "Single feature family (4 values)", fams), ("drop_", "All features except one family (36 values)", fams),
                                              ("only_stat_", "One statistic of all families (10 values)", ["mean", "std", "p10", "p90"])]):
        x = np.arange(len(keys)); w = .38
        for j, m in enumerate(["LR", "XGB"]):
            v = [a[(a.subset == pref + k) & (a.model == m)].AUROC.iloc[0] for k in keys]
            ax.bar(x + (j - .5) * w, v, w, color=MC[m], label=NAMES[m])
        ax.set_xticks(x); ax.set_xticklabels([k.replace("_log", "").replace("_hz", "").replace("log_energy_db", "energy") for k in keys], rotation=60, fontsize=7)
        ax.set_ylim(.5, 1.01); ax.set_title(title, fontsize=8.5); ax.axhline(1, c="#ddd", lw=.6)
    axes[0].set_ylabel("pooled speaker-level AUROC"); axes[0].legend(fontsize=7, loc="lower right")
    save(fig, "wp_ablation")


def f_importance():
    d = pd.read_csv(RES / "S2_acoustic_importance.csv")
    fams = ["log_energy_db", "zcr", "centroid_hz", "rolloff85_hz", "flatness", "band0_log", "band1_log", "band2_log", "band3_log", "band4_log"]
    stats = ["stat_mean", "stat_std", "stat_p10", "stat_p90"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    for ax, keys, title in [(axes[0], fams, "Permutation importance by family"), (axes[1], stats, "Permutation importance by statistic")]:
        x = np.arange(len(keys)); w = .38
        for j, cid in enumerate(["FV1-AC-ALL-LR", "FV1-AC-ALL-XGB"]):
            v = [d[(d.condition == cid) & (d.group == k)].logloss_increase.iloc[0] for k in keys]
            ax.bar(x + (j - .5) * w, v, w, color=MC[cid.split("-")[-1]], label=NAMES[cid.split("-")[-1]])
        ax.set_xticks(x); ax.set_xticklabels([k.replace("stat_", "").replace("_log", "").replace("_hz", "").replace("log_energy_db", "energy") for k in keys], rotation=60, fontsize=7)
        ax.set_title(title, fontsize=8.5); ax.set_ylabel("increase in recording log-loss")
    axes[0].legend(fontsize=7)
    g = d[(d.condition == "FV1-AC-ALL-XGB") & d.group.str.startswith("ac_")].sort_values("gain_share_mean", ascending=False).head(10)[::-1]
    axes[2].barh(np.arange(len(g)), g.gain_share_mean, color=PUR, xerr=g.gain_share_sd, ecolor="#bbb")
    axes[2].set_yticks(np.arange(len(g))); axes[2].set_yticklabels([n.replace("ac_", "") for n in g.group], fontsize=7); axes[2].set_title("XGBoost: share of total gain (top 10)", fontsize=8.5)
    save(fig, "wp_importance_acoustic")


def f_blocks():
    b = pd.read_csv(RES / "S2_block_importance.csv")
    conds = ["FV1-H1-ALL-LR", "FV1-H2-ALL-LR", "FV1-H3-ALL-LR", "FV1-H8-ALL-LR", "FV1-H1-ALL-XGB", "FV1-H2-ALL-XGB", "FV1-H3-ALL-XGB", "FV1-H8-ALL-XGB"]
    col = {"WLM": BLUE, "W2V": GRN, "XLSR": AMB, "AC": RED}
    fig, ax = plt.subplots(figsize=(7.4, 2.9)); x = np.arange(len(conds))
    for i, c in enumerate(conds):
        x_ = b[b.condition == c]; share = x_.coef_mass_share if c.endswith("LR") else x_.gain_share; bottom = 0
        for g, s in zip(x_.group, share):
            ax.bar(i, s, bottom=bottom, color=col[g], label=g if i == 0 or (c == "FV1-H8-ALL-LR" and g == "W2V") else None); bottom += s
    ax.set_xticks(x); ax.set_xticklabels([c.replace("FV1-", "").replace("-ALL-", " ") for c in conds], fontsize=7)
    ax.set_ylabel("share of model weight"); ax.set_title("Which block carries the model? LR: share of |standardized coefficients|; XGBoost: share of total gain", fontsize=8.5)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=col[k]) for k in ["WLM", "W2V", "XLSR", "AC"]], ["WavLM-large", "wav2vec2-large", "XLS-R 300M", "40 acoustic features"], fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.15))
    save(fig, "wp_block_importance")


def f_tuning():
    s = pd.read_csv(RES / "S3_selected_and_ties.csv"); L = pd.read_csv(RES / "S3_landscape.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for m in P.MODELS:
        v = s[s.model == m].n_tied_at_best
        axes[0].hist(v, bins=np.arange(0.5, 25.5, 1), alpha=.55, color=MC[m], label="%s (grid %d)" % (NAMES[m], len(P.grid(m))))
    axes[0].set_xlabel("configurations tied at the best inner AUPRC (per run)"); axes[0].set_ylabel("runs"); axes[0].legend(fontsize=7); axes[0].set_title("Inner-validation saturation", fontsize=8.5)
    reps = ["AC", "WLM", "W2V", "XLSR", "H3"]; i = 0
    for r in reps:
        for m in P.MODELS:
            x = L[(L.representation == r) & (L.model == m)]
            sel = cm[(cm.representation == r) & (cm.model == m) & (cm.feature_selection == "ALL")].AUROC.iloc[0]
            axes[1].plot([i, i], [x.pooled_auroc.min(), x.pooled_auroc.max()], c="#bbb", lw=3)
            axes[1].scatter(i, x.pooled_auroc.median(), c="black", s=10, zorder=3); axes[1].scatter(i, sel, c=MC[m], marker="D", s=22, zorder=4)
            i += 1
    axes[1].set_xticks(np.arange(0, 20, 4) + 1.5); axes[1].set_xticklabels(reps); axes[1].set_ylim(.5, 1.01)
    axes[1].set_title("Grid landscape on outer folds: range (grey), median (black), nested choice (colour)", fontsize=8.5); axes[1].set_ylabel("pooled AUROC")
    save(fig, "wp_tuning")


def f_roc():
    r = pd.read_csv(RES / "S5_roc_points.csv")
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    lab = {"FV1-AC-ALL-LR": ("Acoustic, LR", RED), "FV1-WLM-ALL-LR": ("WavLM, LR", BLUE), "FV1-W2V-ALL-LR": ("wav2vec2, LR", GRN), "FV1-XLSR-ALL-LR": ("XLS-R, LR", AMB),
           "FV1-H8-ALL-XGB": ("All four, XGBoost", PUR), "FV1-WLM-ALL-RSVM": ("WavLM, RBF SVM", GRAY)}
    for cid, (n, c) in lab.items():
        x = r[r.condition_id == cid]; ax.step(x.fpr, x.tpr, where="post", c=c, lw=1.2, label=n)
    ax.plot([0, 1], [0, 1], ls=":", c="#aaa"); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.legend(fontsize=6.5, loc="lower right"); ax.set_title("Speaker-level ROC (pooled out-of-fold)", fontsize=8.5)
    save(fig, "wp_roc")


def f_tasks():
    t = pd.read_csv(RES / "S7_task_families.csv")
    fams = ["vowels", "syllables", "passages", "word list", "phrase", "all tasks"]
    conds = ["FV1-AC-ALL-LR", "FV1-WLM-ALL-LR", "FV1-W2V-ALL-LR", "FV1-XLSR-ALL-LR", "FV1-H8-ALL-XGB"]
    cols = [RED, BLUE, GRN, AMB, PUR]
    fig, ax = plt.subplots(figsize=(7.6, 2.8)); w = .16; x = np.arange(len(fams))
    for j, (c, col) in enumerate(zip(conds, cols)):
        v = [t[(t.condition_id == c) & (t.task_family == f)].speaker_auroc.iloc[0] for f in fams]
        ax.bar(x + (j - 2) * w, v, w, color=col, label=c.replace("FV1-", "").replace("-ALL-", " "))
    ax.set_xticks(x); ax.set_xticklabels(fams); ax.set_ylim(.9, 1.005); ax.set_ylabel("speaker AUROC"); ax.legend(fontsize=6.5, ncol=5, loc="upper center", bbox_to_anchor=(.5, -.12))
    ax.set_title("Speaker-level AUROC when the speaker score uses only one task family (same out-of-fold models)", fontsize=8.5)
    save(fig, "wp_tasks")


def f_session():
    s = pd.read_csv(RES / "S6_session_predictability.csv"); s = s[s.model == "LR"]
    tasks = list(dict.fromkeys(s.task)); reps = ["AC", "WLM", "W2V", "XLSR"]; cols = [RED, BLUE, GRN, AMB]
    fig, ax = plt.subplots(figsize=(7.4, 2.8)); w = .2; x = np.arange(len(tasks))
    for j, (r, c) in enumerate(zip(reps, cols)):
        ax.bar(x + (j - 1.5) * w, [s[(s.task == t) & (s.representation == r)].unit_auroc.iloc[0] for t in tasks], w, color=c, label=r)
    ax.axhline(.5, ls=":", c="#888"); ax.set_xticks(x)
    ax.set_xticklabels(["T1: PD 24 Jan vs 10 Feb 2017\n(same class, same 16 kHz)", "T2: controls early vs late\nMarch 2017 (same class)", "T3: PD 2016 (44.1 kHz)\nvs PD 2017 (16 kHz)"], fontsize=7)
    ax.set_ylabel("AUROC for unseen speakers"); ax.set_ylim(.4, 1.02); ax.legend(fontsize=7, ncol=4); ax.set_title("Can the representations recognise the recording SESSION inside one class? (LR, nested, speaker-disjoint)", fontsize=8.5)
    save(fig, "wp_session")


def f_compute():
    c = pd.read_csv(RES / "S8_compute.csv"); c = c[c.encoder.isin(["WLM", "W2V", "XLSR"])]
    fig, ax = plt.subplots(figsize=(5.2, 2.6)); w = .25
    for j, th in enumerate([1, 4, 8]):
        v = [c[(c.encoder == e) & (c.threads == th)].seconds_per_10s_window_median.iloc[0] for e in ["WLM", "W2V", "XLSR"]]
        ax.bar(np.arange(3) + (j - 1) * w, v, w, label="%d thread%s" % (th, "s" if th > 1 else ""), color=[GRAY, BLUE, PUR][j])
    ax.set_xticks(range(3)); ax.set_xticklabels(["WavLM-large", "wav2vec2-large", "XLS-R 300M"]); ax.set_ylabel("seconds per 10-s window (CPU)"); ax.legend(fontsize=7); ax.set_title("Encoder inference cost on a desktop CPU", fontsize=8.5)
    save(fig, "wp_compute")


def f_ppv():
    fig, ax = plt.subplots(figsize=(5.4, 3.0)); prev = np.linspace(.002, .3, 300); se = 0.96
    for sp, c in zip([0.80, 0.90, 0.95, 0.99], [RED, AMB, BLUE, GRN]):
        ax.plot(prev * 100, se * prev / (se * prev + (1 - sp) * (1 - prev)) * 100, c=c, label="specificity %.2f" % sp)
    ax.set_xscale("log"); ax.set_xlabel("prevalence in the tested population (%)"); ax.set_ylabel("positive predictive value (%)"); ax.legend(fontsize=7)
    ax.set_title("What a positive result means at sensitivity 0.96", fontsize=8.5); ax.grid(alpha=.3)
    save(fig, "wp_ppv")


if __name__ == "__main__":
    for f in [f_confound, f_forest, f_univariate, f_ablation, f_importance, f_blocks, f_tuning, f_roc, f_tasks, f_session, f_compute, f_ppv]:
        f(); print("figure", f.__name__)
