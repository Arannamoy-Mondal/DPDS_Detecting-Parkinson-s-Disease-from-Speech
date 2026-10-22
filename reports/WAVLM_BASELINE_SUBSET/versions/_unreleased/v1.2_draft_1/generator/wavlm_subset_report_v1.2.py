"""WavLM baseline subset report (LaTeX -> PDF), document version 1.2.

Scope: the WavLM part of the FINAL_V1 comparison, written to the Final Report column of the CSE 431 guideline.
  Baseline     frozen WavLM-large vectors + logistic regression (official run FV1-WLM-ALL-LR)
  Extensions   Random Forest (S11, amendment A2), XGBoost (official FV1-WLM-ALL-XGB),
               wide hyperparameter study for LR/RF/XGB (S12, amendment A3),
               partial fine-tuning of WavLM-large (S13, amendment A3)
Every number is read from saved predictions and recomputed here; the build stops and writes nothing if a check fails.
Existing versions are never overwritten.

    .venv/bin/python src/final_v1/wavlm_subset_report.py build
"""
import hashlib, json, re, shutil, statistics, subprocess, sys, zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wavlm_subset_refs import REFS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/WAVLM_BASELINE_SUBSET"
FIG = OUT / "figures_v1.2"
RES = ROOT / "supplementary/results"
PRED = ROOT / "predictions/FINAL_V1"
METRICS = ROOT / "metrics/FINAL_V1"
STUDYFIG = ROOT / "reports/FINAL_V1/study_report/figures"
REPORT_VERSION = "1.2"
REPORT_DATE = "22 October 2026"
VOUT = OUT / "versions" / ("v" + REPORT_VERSION)
STEM = "WAVLM_BASELINE_SUBSET_REPORT_v" + REPORT_VERSION
FOLDS = range(5)
THRESHOLD = 0.5
NBOOT = 2000
MAIN = [("LR", "Logistic regression (baseline)"), ("RF", "Random Forest"), ("XGB", "XGBoost"), ("FT", "Fine-tuned WavLM")]
WIDE = [("LR_wide", "LR, wide grid"), ("RF_wide", "RF, wide grid"), ("XGB_wide", "XGB, wide grid")]
SHORT = {"LR": "LR", "RF": "RF", "XGB": "XGB", "FT": "FT", "LR_wide": "LR wide", "RF_wide": "RF wide", "XGB_wide": "XGB wide"}
COLORS = {"LR": "#2B5FAF", "RF": "#3C8D5A", "XGB": "#B5651D", "FT": "#7A3E9D", "LR_wide": "#7FA3D9", "RF_wide": "#8CC49E", "XGB_wide": "#E0A872"}
FAMILY = {"B1": "passages", "B2": "passages", "D1": "syllables", "D2": "syllables", "FB1": "phrase", "PR1": "word list"}
problems, checks_log, B = [], [], []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, ok, detail=""):
    checks_log.append((name, "PASS" if ok else "FAIL", detail))
    if not ok:
        problems.append(name)
        print("FAIL  " + name + "  " + detail)


def cite(keyword):
    hits = [i + 1 for i, r in enumerate(REFS) if keyword in r]
    if len(hits) != 1:
        print("citation keyword not unique:", keyword, hits); sys.exit(7)
    return hits[0]


# ------------------------------------------------------------------ data
def metrics_from(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= THRESHOLD).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (tp + fp) and (prec + sens) > 0 else 0.0
    two = len(set(labels.tolist())) == 2
    return {"people": len(labels), "pd": int(labels.sum()), "controls": int(len(labels) - labels.sum()), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "errors": fn + fp, "recall": sens, "specificity": spec, "precision": prec, "f1": f1, "accuracy": (tp + tn) / len(labels),
            "auroc": float(roc_auc_score(labels, scores)) if two else float("nan"),
            "auprc": float(average_precision_score(labels, scores)) if two else float("nan")}


def sample_predictions(name):
    if name in ("LR", "XGB"):
        return pd.concat([pd.read_csv(PRED / ("FV1-WLM-ALL-%s-F%d.csv" % (name, f))) for f in FOLDS], ignore_index=True)
    if name == "RF":
        return pd.read_csv(RES / "S11_rf_wlm_sample_predictions.csv")
    if name == "FT":
        return pd.read_csv(RES / "S13_ft_sample_predictions.csv")
    d = pd.read_csv(RES / "S12_nested_predictions.csv")
    return d[d.model == name.split("_")[0]].drop(columns=["model", "config_index"]).reset_index(drop=True)


def speaker_table(d):
    return d.groupby("speaker_id").agg(score=("predicted_probability", "mean"), label=("true_label", "first"),
                                       fold=("outer_fold", "first")).sort_index()


def load_all():
    global SM, REC, SPK, POOLED, FOLDM
    SM = pd.read_csv(ROOT / "manifests/SAMPLE_MANIFEST.csv").set_index("sample_id")
    REC = {n: sample_predictions(n) for n, _ in MAIN + WIDE}
    SPK = {n: speaker_table(REC[n]) for n in REC}
    POOLED = {n: metrics_from(SPK[n].label, SPK[n].score) for n in SPK}
    rows = []
    for n in SPK:
        for f in FOLDS:
            part = SPK[n][SPK[n].fold == f]
            rows.append(dict(model=n, fold=f, **metrics_from(part.label, part.score)))
    FOLDM = pd.DataFrame(rows)


def boot(labels, a, b=None, seed=42):
    """Percentile bootstrap over people. With b: paired difference a - b on the same resamples."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    out = {"auroc": [], "auprc": []}
    for _ in range(NBOOT):
        idx = rng.randint(0, n, n)
        if len(set(labels[idx])) < 2:
            continue
        for k, fn in [("auroc", roc_auc_score), ("auprc", average_precision_score)]:
            v = fn(labels[idx], a[idx])
            if b is not None:
                v -= fn(labels[idx], b[idx])
            out[k].append(v)
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


# ------------------------------------------------------------------ compute and check
def compute():
    global CI, DELTA, VERDICT, TVT, SEL, LAND, S12SUM, FTSUM, FTCURVE, COUNTS, FAMTAB, S6, S9, cond
    load_all()
    cond = pd.read_csv(METRICS / "condition_metrics.csv").set_index("condition_id")
    for n, cid in [("LR", "FV1-WLM-ALL-LR"), ("XGB", "FV1-WLM-ALL-XGB")]:
        r = cond.loc[cid]
        ok = (abs(POOLED[n]["f1"] - r["F1"]) < 1e-9 and abs(POOLED[n]["auroc"] - r["AUROC"]) < 1e-9 and abs(POOLED[n]["auprc"] - r["AUPRC"]) < 1e-9
              and POOLED[n]["fn"] == int(r["FN"]) and POOLED[n]["fp"] == int(r["FP"]))
        check("official WavLM %s reproduced from its saved predictions" % n, ok)
    base_ids = set(SPK["LR"].index)
    for n in SPK:
        check("%s scores the same 47 people with the same labels" % SHORT[n],
              set(SPK[n].index) == base_ids and (SPK[n].label.values == SPK["LR"].label.values).all() and len(base_ids) == 47)
        d = REC[n]
        check("%s: every person in exactly one test fold; every recording predicted once" % SHORT[n],
              int(d.groupby("speaker_id").outer_fold.nunique().max()) == 1 and d.sample_id.is_unique and set(d.sample_id) == set(REC["LR"].sample_id))
        p = d.predicted_probability.values
        check("%s: probabilities finite and in [0, 1]" % SHORT[n], bool(np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1))
    split = pd.read_csv(ROOT / "SPLIT_MANIFEST.csv").drop_duplicates("speaker_id").set_index("speaker_id").outer_fold
    check("folds of all models equal SPLIT_MANIFEST", all((SPK[n].fold == split.reindex(SPK[n].index)).all() for n in SPK))
    gates = all(json.loads((METRICS / ("FV1-WLM-ALL-%s-F%d.json" % (m, f))).read_text())["status"] == "COMPLETED"
                and all(v in (True, "PASS", "passed") for v in json.loads((METRICS / ("FV1-WLM-ALL-%s-F%d.json" % (m, f))).read_text())["gates"].values())
                for m in ["LR", "XGB"] for f in FOLDS)
    check("saved integrity gates PASS for the official WavLM LR and XGB runs", gates)
    cache = json.loads((RES / "S13_cache_summary.json").read_text())
    check("fine-tuning cache reproduces the saved FINAL_V1 WavLM vectors (max |diff| %.1e, cosine %.8f)"
          % (cache["max_abs_cache_vs_saved"], cache["min_cosine_cache_vs_saved"]), bool(cache["check_passed"]))
    a3 = (ROOT / "supplementary/PREDECLARED_A3.md").read_text()
    check("amendment A3 pre-declaration includes the decision rule", "Decision rule" in a3)
    # bootstrap intervals and paired differences against the baseline
    lab = SPK["LR"].label.values.astype(int)
    CI = {n: boot(lab, SPK[n].score.values) for n in SPK}
    DELTA = {n: boot(lab, SPK[n].score.values, SPK["LR"].score.values) for n in SPK if n != "LR"}
    VERDICT = {}
    for n in DELTA:
        fewer = POOLED["LR"]["errors"] - POOLED[n]["errors"]
        VERDICT[n] = {"fewer_errors": fewer, "auroc_ci_excludes_zero": DELTA[n]["auroc"][1] > 0,
                      "improvement": fewer >= 2 and DELTA[n]["auroc"][1] > 0}
    # selected settings, validation and training scores
    s11 = json.loads((RES / "S11_rf_wlm_summary.json").read_text())["selection"]
    FTSUM = json.loads((RES / "S13_ft_summary.json").read_text())
    S12SUM = json.loads((RES / "S12_summary.json").read_text())
    tr = pd.read_csv(RES / "S12_train_scores.csv")
    SEL, TVT = {}, []
    for f in FOLDS:
        for m in ["LR", "XGB"]:
            r = json.loads((METRICS / ("FV1-WLM-ALL-%s-F%d.json" % (m, f))).read_text())
            SEL[(m, f)] = (r["selected_hyperparameters"], max(x["inner_speaker_auprc"] for x in r["inner_table"]))
        SEL[("RF", f)] = (s11[str(f)]["selected"], max(x["inner_speaker_auprc"] for x in s11[str(f)]["inner_table"]))
        fi = FTSUM["folds"][f]
        SEL[("FT", f)] = ({"k": fi["selected"]["k"], "lr": fi["selected"]["lr"], "epochs": fi["epochs"]}, fi["inner_auprc_selected"])
        ft_train = pd.read_csv(RES / ("S13_folds/fold%d_train.csv" % f))
        g = ft_train.groupby("speaker_id").agg(s=("p", "mean"), y=("label", "first"))
        ftm = metrics_from(g.y, g.s)
        for m, variant in [("LR", "LR_official"), ("RF", "RF_S11"), ("XGB", "XGB_official")]:
            t = tr[(tr.variant == variant) & (tr.fold == f)].iloc[0]
            TVT.append({"model": m, "fold": f, "train_auprc": t.auprc, "train_auroc": t.auroc, "train_f1": t.f1, "val_auprc": SEL[(m, f)][1]})
        TVT.append({"model": "FT", "fold": f, "train_auprc": ftm["auprc"], "train_auroc": ftm["auroc"], "train_f1": ftm["f1"], "val_auprc": SEL[("FT", f)][1]})
    TVT = pd.DataFrame(TVT)
    for col in ["auprc", "auroc", "f1"]:
        TVT["test_" + col] = [float(FOLDM[(FOLDM.model == r.model) & (FOLDM.fold == r.fold)][col].iloc[0]) for r in TVT.itertuples()]
    check("training-partition scores exist for all four models and five folds", len(TVT) == 20)
    LAND = pd.read_csv(RES / "S12_landscape.csv")
    check("S12 landscape covers 32 + 96 + 128 configurations", LAND.groupby("model").size().to_dict() == {"LR": 32, "RF": 96, "XGB": 128})
    FTCURVE = pd.concat([pd.read_csv(RES / ("S13_folds/fold%d_curves.csv" % f)) for f in FOLDS], ignore_index=True)
    check("fine-tuning learning curves for 5 folds x 4 settings x 3 inner folds x 6 epochs", len(FTCURVE) == 5 * 4 * 3 * 6, "rows %d" % len(FTCURVE))
    COUNTS = pd.read_csv(RES / "S14_param_counts.csv")
    # sensitivity to input changes: one speech-task family at a time
    rows = []
    for n, _ in MAIN:
        d = REC[n].copy()
        d["family"] = [FAMILY.get(t, "vowels") for t in SM.loc[d.sample_id, "task_code"]]
        for fam, g in d.groupby("family"):
            s = g.groupby("speaker_id").agg(score=("predicted_probability", "mean"), label=("true_label", "first"))
            m = metrics_from(s.label, s.score)
            rows.append({"model": n, "family": fam, "people": m["people"], "recordings": len(g), "auroc": m["auroc"], "errors": m["errors"]})
    FAMTAB = pd.DataFrame(rows)
    S6 = pd.read_csv(RES / "S6_session_predictability.csv")
    S9 = json.loads((RES / "S9_digital_silence_summary.json").read_text())
    if problems:
        print("\nSTOP: %d check(s) failed. Nothing was written." % len(problems)); sys.exit(1)


# ------------------------------------------------------------------ figures
def style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9.5})


def box_at(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#6B7280", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#6B7280", lw=0.9))


def fig_overview():
    fig, ax = plt.subplots(figsize=(10.2, 4.0)); ax.axis("off"); ax.set_xlim(0, 10.2); ax.set_ylim(0, 4.0)
    box_at(ax, 0.05, 1.55, 1.35, 0.9, "Recording\n16 kHz, cut into\n10-s windows\n(160,000 samples)", "#F3F4F6")
    box_at(ax, 1.7, 1.55, 1.5, 0.9, "WavLM-large\nCNN + layers 0-19\nfrozen\n499 x 1,024 / window", "#E8F3EC")
    ax.text(5.6, 3.75, "Baseline and classifier extensions (frozen encoder)", fontsize=8.5, weight="bold", ha="center")
    box_at(ax, 3.55, 2.65, 1.5, 0.9, "Layers 20-23 +\nlayer norm, frozen\n499 x 1,024", "#E8F3EC")
    box_at(ax, 5.35, 2.65, 1.45, 0.9, "Mean over time,\nthen over windows\n1 x 1,024 / recording", "#FFF7E6")
    box_at(ax, 7.1, 2.65, 1.45, 0.9, "LR (baseline)\nRF  |  XGB\nnested settings", "#E8EFFA")
    ax.text(5.6, 0.12, "Extension: partial fine-tuning (top k layers trained, k = 2 or 4)", fontsize=8.5, weight="bold", ha="center")
    box_at(ax, 3.55, 0.45, 1.5, 0.9, "Layers 20..23-k frozen\nlayers 24-k..23\nTRAINED", "#F6E8F6")
    box_at(ax, 5.35, 0.45, 1.45, 0.9, "Layer norm (trained)\nmean over time\n1 x 1,024 / window", "#F6E8F6")
    box_at(ax, 7.1, 0.45, 1.45, 0.9, "Dropout 0.1 +\nlinear 1,024 -> 1\nsigmoid / window", "#F6E8F6")
    box_at(ax, 8.85, 1.55, 1.3, 0.9, "Mean per recording,\nmean per person\nPD if >= 0.5", "#FDECEC")
    arrow(ax, 1.4, 2.0, 1.7, 2.0); arrow(ax, 3.2, 2.15, 3.55, 3.0); arrow(ax, 3.2, 1.85, 3.55, 0.95)
    arrow(ax, 5.05, 3.1, 5.35, 3.1); arrow(ax, 6.8, 3.1, 7.1, 3.1); arrow(ax, 5.05, 0.9, 5.35, 0.9); arrow(ax, 6.8, 0.9, 7.1, 0.9)
    arrow(ax, 8.55, 3.1, 8.85, 2.2); arrow(ax, 8.55, 0.9, 8.85, 1.8)
    fig.savefig(FIG / "method_overview.pdf", bbox_inches="tight"); plt.close(fig)


def fig_roc():
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    for n, name in MAIN:
        fpr, tpr, _ = roc_curve(SPK[n].label.values.astype(int), SPK[n].score.values)
        ax.plot(fpr, tpr, color=COLORS[n], lw=1.7, label="%s (AUROC %.3f)" % (name, POOLED[n]["auroc"]))
    ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=0.8, ls="--")
    ax.set_xlabel("false positive rate (controls called PD)"); ax.set_ylabel("true positive rate (patients found)")
    ax.legend(fontsize=7, loc="lower right", frameon=False); ax.set_title("Pooled person-level ROC, held-out folds")
    fig.savefig(FIG / "roc.pdf", bbox_inches="tight"); plt.close(fig)


def fig_folds():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for ax, key, lab in [(axes[0], "f1", "F1 (%)"), (axes[1], "auroc", "AUROC (%)")]:
        for i, (n, name) in enumerate(MAIN):
            v = 100 * FOLDM[FOLDM.model == n][key].values
            ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), v, color=COLORS[n], s=26, zorder=3)
            ax.hlines(v.mean(), i - 0.28, i + 0.28, color=COLORS[n], lw=2)
        ax.set_xticks(range(len(MAIN))); ax.set_xticklabels([SHORT[n] for n, _ in MAIN]); ax.set_ylabel("held-out " + lab)
        ax.set_title("Per-fold held-out %s (line = mean)" % key.upper() if key == "auroc" else "Per-fold held-out F1 (line = mean)")
    axes[0].set_ylim(60, 102); axes[1].set_ylim(60, 102)
    fig.tight_layout(); fig.savefig(FIG / "folds.pdf", bbox_inches="tight"); plt.close(fig)


def fig_tvt():
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for i, (n, name) in enumerate(MAIN):
        t = TVT[TVT.model == n]
        for j, (col, lab, mk) in enumerate([("train_auprc", "training", "o"), ("val_auprc", "inner validation", "s"), ("test_auprc", "held-out test", "D")]):
            v = 100 * t[col].values
            x = i + (j - 1) * 0.22
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt=mk, color=COLORS[n], ms=6, capsize=3, label=lab if i == 0 else None,
                        mfc=COLORS[n] if j == 0 else ("white" if j == 1 else COLORS[n]), alpha=1.0 if j != 2 else 0.85)
    ax.set_xticks(range(len(MAIN))); ax.set_xticklabels([name for _, name in MAIN], fontsize=8)
    ax.set_ylabel("person-level AUPRC (%), mean +/- SD over folds"); ax.set_ylim(85, 101)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left"); ax.set_title("Training, validation and held-out test, per model")
    fig.savefig(FIG / "train_val_test_models.pdf", bbox_inches="tight"); plt.close(fig)


def fig_hparams():
    L = LAND
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    ax = axes[0]
    lr = L[L.model == "LR"].copy(); lr["cw"] = lr.class_weight.fillna("none")
    for (pen, cw), g in lr.groupby(["penalty", "cw"]):
        g = g.sort_values("C")
        ax.plot(g.C, 100 * g.auroc, marker="o", ms=3, lw=1.2, label="%s, weight %s" % (pen.upper(), cw), ls="-" if pen == "l1" else "--")
    ax.set_xscale("log"); ax.set_xlabel("C (inverse penalty strength)"); ax.set_ylabel("pooled held-out AUROC (%)")
    ax.set_title("Logistic regression: effect of C"); ax.legend(fontsize=6.5, frameon=False); ax.set_ylim(min(90, 100 * lr.auroc.min() - 1), 100.5)
    ax = axes[1]
    rf = L[L.model == "RF"].copy(); rf["depth"] = rf.max_depth.fillna(0).astype(int).map(lambda d: "none" if d == 0 else str(d))
    piv = rf.pivot_table(index="depth", columns="n_estimators", values="auroc", aggfunc="mean").reindex(["2", "4", "8", "none"])
    im = ax.imshow(100 * piv.values, cmap="Greens", aspect="auto", vmin=100 * piv.values.min() - 0.5, vmax=100 * piv.values.max())
    for (a, b), v in np.ndenumerate(100 * piv.values):
        ax.text(b, a, "%.2f" % v, ha="center", va="center", fontsize=7)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns); ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    ax.set_xlabel("number of trees"); ax.set_ylabel("maximum depth"); ax.set_title("Random Forest: mean AUROC (%) over other settings")
    ax = axes[2]
    xg = L[L.model == "XGB"]
    piv = xg.pivot_table(index="max_depth", columns="learning_rate", values="auroc", aggfunc="mean")
    ax.imshow(100 * piv.values, cmap="Oranges", aspect="auto", vmin=100 * piv.values.min() - 0.5, vmax=100 * piv.values.max())
    for (a, b), v in np.ndenumerate(100 * piv.values):
        ax.text(b, a, "%.2f" % v, ha="center", va="center", fontsize=7)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns); ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    ax.set_xlabel("learning rate"); ax.set_ylabel("maximum depth"); ax.set_title("XGBoost: mean AUROC (%) over other settings")
    fig.tight_layout(); fig.savefig(FIG / "hparam_effects.pdf", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    rng = np.random.RandomState(3)
    for i, m in enumerate(["LR", "RF", "XGB"]):
        g = L[L.model == m]
        axes[0].scatter(i + rng.uniform(-0.25, 0.25, len(g)), 100 * g.auroc, s=9, color=COLORS[m], alpha=0.6)
        axes[0].scatter([i + 0.38], [100 * POOLED[m + "_wide"]["auroc"]], marker="*", s=110, color="black", zorder=4)
        axes[0].scatter([i + 0.38], [100 * POOLED[m]["auroc"]], marker="D", s=30, color=COLORS[m], edgecolor="black", zorder=4)
        cnt = g.errors.value_counts().sort_index()
        axes[1].bar(np.array(cnt.index) + (i - 1) * 0.27, 100 * cnt.values / len(g), width=0.27, color=COLORS[m], label="%s (%d settings)" % (m, len(g)))
    axes[0].set_xticks([0, 1, 2]); axes[0].set_xticklabels(["LR (32)", "RF (96)", "XGB (128)"]); axes[0].set_ylabel("pooled held-out AUROC (%)")
    axes[0].set_title("Every setting (dots); star = nested wide grid;\ndiamond = original nested grid")
    axes[1].set_xlabel("person errors at threshold 0.5 (FN + FP)"); axes[1].set_ylabel("share of settings (%)")
    axes[1].set_title("How many people each setting gets wrong"); axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "hparam_landscape.pdf", bbox_inches="tight"); plt.close(fig)


def fig_ft_curves():
    C = FTCURVE.copy()
    C["cfg"] = ["k=%d, lr=%.0e" % (k, lr) for k, lr in zip(C.k, C.lr)]
    pal = ["#C9A3E0", "#9B5FC0", "#6E8FD0", "#2B4FA0"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    for c, (cfg, g) in zip(pal, C.groupby("cfg", sort=False)):
        e = g.groupby("epoch").agg(tl=("train_loss", "mean"), vl=("val_loss", "mean"))
        axes[0].plot(e.index, e.tl, marker="o", ms=3, color=c, label=cfg)
        axes[1].plot(e.index, e.vl, marker="o", ms=3, color=c, label=cfg)
    pooled = pd.DataFrame([dict(fold=f, **r) for f in FOLDS for r in FTSUM["folds"][f]["inner_auprc_table"]])
    for c, (ci, g) in zip(pal, pooled.groupby("config_index")):
        e = g.groupby("epoch").pooled_inner_auprc.mean()
        axes[2].plot(e.index, 100 * e.values, marker="o", ms=3, color=c)
    axes[0].set_title("Training loss (BCE), mean over folds"); axes[1].set_title("Inner-validation loss (BCE)")
    axes[2].set_title("Inner-validation person AUPRC (%)")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "ft_curves.pdf", bbox_inches="tight"); plt.close(fig)


def fig_scores():
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.3), sharey=True)
    rng = np.random.RandomState(7)
    for ax, (n, name) in zip(axes, MAIN):
        t = SPK[n]
        for cls in (0, 1):
            part = t[t.label == cls]
            xs = cls + rng.uniform(-0.18, 0.18, len(part))
            ax.scatter(xs, part.score, s=14, color=("#8A8A8A" if cls == 0 else COLORS[n]), zorder=3)
            for x, (sid, r) in zip(xs, part.iterrows()):
                if (r.score >= THRESHOLD) != bool(r.label):
                    ax.annotate(sid, (x, r.score), xytext=(5, 0), textcoords="offset points", fontsize=6, va="center")
        ax.axhline(THRESHOLD, color="black", lw=0.8, ls="--")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "PD"]); ax.set_xlim(-0.5, 1.9); ax.set_title(SHORT[n] if n != "LR" else "LR (baseline)")
    axes[0].set_ylabel("person score"); axes[0].set_ylim(-0.04, 1.04)
    fig.tight_layout(); fig.savefig(FIG / "person_scores.pdf", bbox_inches="tight"); plt.close(fig)


def fig_tasks(people):
    order = ["B1", "B2", "D1", "D2", "FB1", "PR1", "VA1", "VA2", "VE1", "VE2", "VI1", "VI2", "VO1", "VO2", "VU1", "VU2"]
    fig, axes = plt.subplots(1, len(people), figsize=(4.4 * len(people), 4.0), squeeze=False)
    for ax, sid in zip(axes[0], people):
        mat = np.full((len(order), len(MAIN)), np.nan)
        for j, (n, _) in enumerate(MAIN):
            d = REC[n][REC[n].speaker_id == sid]
            tasks = SM.loc[d.sample_id, "task_code"].values
            for t, p in zip(tasks, d.predicted_probability.values):
                if t in order:
                    mat[order.index(t), j] = p
        im = ax.imshow(mat, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
        for (a, b), v in np.ndenumerate(mat):
            if np.isfinite(v):
                ax.text(b, a, "%.2f" % v, ha="center", va="center", fontsize=6.3, color="white" if abs(v - 0.5) > 0.3 else "black")
        ax.set_xticks(range(len(MAIN))); ax.set_xticklabels([SHORT[n] for n, _ in MAIN]); ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
        lab = "patient" if int(SPK["LR"].loc[sid, "label"]) == 1 else "control"
        ax.set_title("%s (%s): P(PD) per recording" % (sid, lab))
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="probability of PD (red = PD)")
    fig.savefig(FIG / "task_breakdown.pdf", bbox_inches="tight"); plt.close(fig)


def fig_forest():
    names = [n for n in ["RF", "XGB", "FT", "LR_wide", "RF_wide", "XGB_wide"] if n in DELTA]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
    for ax, key in zip(axes, ["auroc", "auprc"]):
        for i, n in enumerate(names):
            m, lo, hi = DELTA[n][key]
            ax.errorbar(100 * m, i, xerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="o", color=COLORS[n], capsize=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(names))); ax.set_yticklabels([SHORT[n] for n in names]); ax.invert_yaxis()
        ax.set_xlabel("%s minus baseline (percentage points)" % key.upper()); ax.set_title("Paired bootstrap difference vs LR baseline (95%)")
    fig.tight_layout(); fig.savefig(FIG / "paired_differences.pdf", bbox_inches="tight"); plt.close(fig)


def fig_families():
    fams = ["passages", "phrase", "word list", "syllables", "vowels"]
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    for i, (n, name) in enumerate(MAIN):
        v = [100 * FAMTAB[(FAMTAB.model == n) & (FAMTAB.family == f)].auroc.iloc[0] for f in fams]
        ax.bar(np.arange(len(fams)) + (i - 1.5) * 0.2, v, width=0.2, color=COLORS[n], label=SHORT[n])
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams); ax.set_ylabel("person AUROC (%) from one task family")
    ax.set_ylim(min(80, 100 * FAMTAB.auroc.min() - 2), 100.5); ax.legend(fontsize=7, frameon=False, ncol=4, loc="lower right")
    ax.set_title("Sensitivity to the input: scores from one kind of speech task at a time")
    fig.savefig(FIG / "task_families.pdf", bbox_inches="tight"); plt.close(fig)


def make_figures(err_people):
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    fig_overview(); fig_roc(); fig_folds(); fig_tvt(); fig_hparams(); fig_ft_curves(); fig_scores(); fig_tasks(err_people); fig_forest(); fig_families()
    for name in ["dataset_examples", "nested_cv"]:
        shutil.copy2(STUDYFIG / (name + ".pdf"), FIG / (name + ".pdf"))


# ------------------------------------------------------------------ latex helpers
def esc(text):
    return (str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#").replace("{", r"\{").replace("}", r"\}"))


def inline(s):
    maths = []
    def keep(m):
        maths.append(m.group(1)); return "MATHTOKEN%dZ" % (len(maths) - 1)
    s = re.sub(r"\$(.+?)\$", keep, s)
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r"`(.+?)`", lambda m: "\\texttt{" + m.group(1).replace("\\_", "\\_\\allowbreak{}") + "}", s)
    for i, m in enumerate(maths):
        s = s.replace("MATHTOKEN%dZ" % i, "$" + m + "$")
    return s


def A(x):
    B.append(x)


def P(text):
    A(inline(text) + "\n")


def H(level, title, key=None):
    if level == 0:
        A("\\section*{%s}\\addcontentsline{toc}{section}{%s}" % (inline(title), inline(title)))
        return
    A("\\%s{%s}%s" % ({1: "section", 2: "subsection", 3: "subsubsection"}[level], inline(title), ("\\label{sec:%s}" % key) if key else ""))


def box(kind, text):
    style = {"plain": ("In plain words", "green!40!black", "green!3"), "read": ("What this tells us", "blue!50!black", "blue!3"),
             "key": ("Key finding", "orange!70!black", "orange!5"), "rule": ("Decision rule (fixed before the A3 runs)", "red!50!black", "red!3")}[kind]
    A("\\begin{tcolorbox}[colback=%s,colframe=%s,boxrule=0.5pt,arc=2pt,left=5pt,right=5pt,top=3pt,bottom=3pt,title={\\small\\bfseries %s}]\\small %s\\end{tcolorbox}"
      % (style[2], style[1], style[0], inline(text)))


def bullets(items, enum=False):
    env = "enumerate" if enum else "itemize"
    A("\\begin{%s}" % env + "".join("\\item " + inline(i) + "\n" for i in items) + "\\end{%s}" % env)


def cell(x):
    x = str(x)
    return x if (x.startswith("\\") or "$" in x) else inline(x)


def table(caption, header, rows, spec, size="small"):
    lines = ["\\begin{table}[H]\\centering\\%s\\renewcommand{\\arraystretch}{1.15}" % size, "\\caption{%s}" % inline(caption),
             "\\begin{tabular}{%s}" % spec, "\\toprule", " & ".join("\\textbf{%s}" % inline(h) for h in header) + "\\\\", "\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r == "--" else " & ".join(cell(c) for c in r) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}\\end{table}"]
    A("\n".join(lines))


def fig(name, caption, width="0.95"):
    A("\\begin{figure}[H]\\centering\\includegraphics[width=%s\\linewidth]{%s.pdf}\\caption{%s}\\label{fig:%s}\\end{figure}"
      % (width, name, inline(caption), name))


def math(s):
    A(s)


def pct(x):
    return "%.2f" % (100 * x)


def ci_txt(t):
    return "[%.2f, %.2f]" % (100 * t[1], 100 * t[2])


def pm(vals):
    vals = [float(v) for v in vals]
    return "%.2f $\\pm$ %.2f" % (100 * statistics.mean(vals), 100 * statistics.stdev(vals))


PRE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.1cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype}
\usepackage{booktabs,float,graphicx,amsmath,amssymb,enumitem,xcolor,tcolorbox}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black,citecolor=blue!50!black]{hyperref}
\setlist{itemsep=2pt,topsep=3pt}\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}\linespread{1.08}
\usepackage{titlesec}\titleformat{\section}{\Large\bfseries}{\thesection.}{.5em}{}\titleformat{\subsection}{\large\bfseries}{\thesubsection}{.5em}{}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\fancyfoot[C]{\small\thepage}\fancyhead[L]{\small Frozen and fine-tuned WavLM for Parkinson's speech}\fancyhead[R]{\small FINAL\_V1, baseline subset, v1.2}\renewcommand{\headrulewidth}{.3pt}
\begin{document}
"""


def title_page():
    A("\\begin{center}{\\LARGE\\bfseries Does the Classifier or Fine-Tuning Matter?\\\\[3pt]\\Large Frozen and Fine-Tuned WavLM for Detecting Parkinson's Disease from Speech\\par}\\vspace{8pt}"
      "{\\large Baseline-and-extensions study on the FINAL\\_V1 folds (CSE 431, topic 4a)\\par}\\vspace{4pt}"
      "{\\large Version %s \\textperiodcentered{} %s\\par}\\vspace{6pt}"
      "{\\small Evidence: frozen protocol FINAL\\_V1 (47 people, 785 recordings, five speaker-disjoint folds); amendments A2 and A3, declared before their runs.\\par}\\end{center}\n"
      "\\vspace{4pt}\\hrule\\vspace{6pt}\n\\tableofcontents\n\\newpage\n" % (REPORT_VERSION, REPORT_DATE))


# ------------------------------------------------------------------ body
def write_body():
    p = POOLED
    cit = {k: cite(k) for k in ["End-to-end deep learning approach", "machine learning method to process voice", "Egbo", "Mandarin language dataset",
                                 "Klempir", "WavLM: large-scale", "wav2vec 2.0: a framework", "XLS-R: self-supervised", "Crawford", "Vaiciukynas",
                                 "Italian Parkinson's Voice and Speech", "Assessment of speech intelligibility", "NeuroVoz", "LIBLINEAR", "XGBoost: a scalable",
                                 "Scikit-learn", "Trail"]}
    c = lambda k: "[%d]" % cit[k]
    errs = {n: SPK[n][(SPK[n].score >= THRESHOLD) != SPK[n].label.astype(bool)] for n, _ in MAIN}
    missed_all = sorted(set.intersection(*[set(errs[n][errs[n].label == 1].index) for n, _ in MAIN]))
    any_err = sorted(set().union(*[set(errs[n].index) for n, _ in MAIN]))
    improved = [n for n in VERDICT if VERDICT[n]["improvement"]]
    ft = p["FT"]; lr = p["LR"]
    k_counts = pd.Series([SEL[("FT", f)][0]["k"] for f in FOLDS]).value_counts().to_dict()
    ft_params = {fi["selected"]["k"]: fi["trainable_parameters"] for fi in FTSUM["folds"]}
    ft_runtime = sum(fi["runtime_seconds"] for fi in FTSUM["folds"]) / 3600

    H(0, "Key numbers at a glance")
    table("Held-out person-level results on the 47 FINAL_V1 people (five folds pooled, threshold 0.5). Percent; errors = missed patients + false alarms.",
          ["Model", "Errors", "F1", "Recall", "Specificity", "AUROC", "AUPRC"],
          [[name, str(p[n]["errors"]), pct(p[n]["f1"]), pct(p[n]["recall"]), pct(p[n]["specificity"]), pct(p[n]["auroc"]), pct(p[n]["auprc"])]
           for n, name in MAIN + WIDE], "lrrrrrr", size="footnotesize")
    box("key", ("No extension met the pre-declared improvement rule. " if not improved else "Extensions meeting the pre-declared rule: %s. " % ", ".join(SHORT[n] for n in improved))
        + "The frozen baseline makes %d error(s); the best extension makes %d. Every model misses the same patient (%s), and the spread across folds is larger than the differences between models."
        % (lr["errors"], min(p[n]["errors"] for n in VERDICT), ", ".join(missed_all) if missed_all else "none"))

    H(0, "Revision history")
    table("Versions of this report.", ["Version", "Date", "Changes"],
          [["1.0", "21 October 2026", "Internal draft built from the separate PatchTST proposal run (v4.0). Not delivered. Superseded."],
           ["1.1", "21 October 2026", "Rebuilt on FINAL_V1: LR and XGB official runs, Random Forest added (amendment A2)."],
           ["1.2", REPORT_DATE, "Rewritten to the guideline's Final Report column. New experiments (amendment A3): wide hyperparameter study for LR, RF and XGB; nested partial fine-tuning of WavLM-large. New sections on training versus validation versus test, parameter counts, failure analysis by task, sensitivity to input changes, and detailed future directions. v4.0 cross-check removed."]],
          "llp{11.4cm}", size="footnotesize")

    # 1 ---------------------------------------------------------------
    H(1, "Abstract", "abstract")
    P("Parkinson's disease changes how people speak: the voice becomes quieter and flatter, and consonants become less clear. A short voice recording could therefore help to flag the disease early, cheaply and without a clinic visit. "
      "We study one concrete question on a public Italian corpus of 785 recordings from 47 people (25 with Parkinson's disease, 22 healthy older adults): "
      "when a large pretrained speech model, WavLM-large, turns each recording into numbers, does it matter which classifier reads those numbers, how its settings are chosen, or whether the speech model itself is fine-tuned on our data? "
      "Every person is kept entirely on one side of every train/test split. The baseline is the frozen speech model with logistic regression; it misses %d of 25 patients and raises %d false alarms (F1 %s%%, AUROC %s%%). "
      "Random Forest, XGBoost, 256 alternative classifier settings and partial fine-tuning of the top layers of WavLM reach between %d and %d errors and AUROC between %s%% and %s%%. "
      "None meets the improvement rule we fixed before running the new experiments. All models miss the same patient, and the evidence suggests that much of the separation may come from recording conditions rather than from the disease. "
      "We therefore treat the high scores as an upper bound for this corpus, not as clinical accuracy, and we give a concrete plan to test this."
      % (lr["fn"], lr["fp"], pct(lr["f1"]), pct(lr["auroc"]), min(p[n]["errors"] for n in VERDICT), max(p[n]["errors"] for n in VERDICT),
         pct(min(p[n]["auroc"] for n in VERDICT)), pct(max(p[n]["auroc"] for n in VERDICT))))

    # 2 ---------------------------------------------------------------
    H(1, "Motivation and objectives", "motivation")
    P("About one person in a hundred over the age of 60 has Parkinson's disease, and speech changes are among its early signs %s. A test that needs only a microphone would be cheap, painless and repeatable, so it could help screening where neurologists are scarce. "
      "Patients would gain an earlier referral; clinics would gain a tool for follow-up. The obstacle is trust: many published voice studies report near-perfect accuracy, but often because recordings of the same person appear in both training and testing %s." % (c("Trail"), c("Vaiciukynas")))
    P("**Qualitative objective.** Find out whether a frozen self-supervised speech model is enough, or whether a stronger classifier, better-tuned settings or fine-tuning of the speech model adds anything, under a protocol in which no person is ever seen in both training and testing.")
    P("**Quantitative objectives.**")
    bullets(["O1. Report held-out person-level recall, specificity, precision, F1, accuracy, AUROC and AUPRC, with 95% bootstrap intervals, for the baseline and every extension.",
             "O2. Decide improvement with the rule fixed before the new runs: at least two fewer person errors than the baseline and a paired AUROC interval that excludes zero.",
             "O3. Report training, validation and test results separately for every model, and the spread across the five folds.",
             "O4. Identify at least one failure mode, and measure how results change with hyperparameters and with the type of speech task."])

    # 3 ---------------------------------------------------------------
    H(1, "Related work", "related")
    P("We group prior work into four research lines and state how each one shapes this study.")
    P("**End-to-end deep networks.** Quan et al. (2022) %s train convolutional networks directly on speech spectrograms. Iyer et al. (2023) %s compare a convolutional network with Random Forest and logistic regression on telephone voice recordings. "
      "Such models learn every parameter from small clinical datasets. *Connection:* our fine-tuning extension sits between these models and a frozen encoder: it trains only the top layers of a large pretrained model." % (c("End-to-end deep learning approach"), c("machine learning method to process voice")))
    P("**Hand-made acoustic features with classical classifiers.** Wang et al. (2022) %s select features from a Mandarin corpus and compare four classifiers; Egbo et al. (2025) %s use XGBoost with SHAP explanations on vocal biomarkers. "
      "*Connection:* they motivate our tree-based extensions (Random Forest, XGBoost) and their tuning." % (c("Mandarin language dataset"), c("Egbo")))
    P("**Self-supervised speech representations.** wav2vec 2.0 %s, XLS-R %s and WavLM %s learn speech representations from tens of thousands of hours of unlabelled audio. "
      "Klempir and Krupicka (2024) %s classify frozen wav2vec embeddings with logistic regression, Random Forest and XGBoost across several Parkinson's databases. "
      "*Connection:* this is the closest design to ours; we use the stronger WavLM-large encoder, add fine-tuning, and evaluate strictly by person." % (c("wav2vec 2.0: a framework"), c("XLS-R: self-supervised"), c("WavLM: large-scale"), c("Klempir")))
    P("**Evaluation methodology and data.** Vaiciukynas et al. (2017) %s show that scores are inflated unless all recordings of a person stay together, and that results change with the microphone. Crawford (2025) %s uses what people say rather than how they sound. "
      "The Italian corpus we use is described by Dimauro and Girardi %s; Dimauro et al. %s study intelligibility in Parkinson's disease, and NeuroVoz %s is a Spanish corpus of the same kind. "
      "*Connection:* they justify person-level evaluation and our concern about recording conditions." % (c("Vaiciukynas"), c("Crawford"), c("Italian Parkinson's Voice and Speech"), c("Assessment of speech intelligibility"), c("NeuroVoz")))

    # 4 ---------------------------------------------------------------
    H(1, "Background", "background")
    H(2, "Why these metrics")
    P("The unit is the person, because a diagnosis is about a person, not a file. Following Vaiciukynas et al. %s, all of a person's recordings are averaged to one score. "
      "Recall (sensitivity) and specificity are the clinical pair used throughout the Parkinson's voice literature %s: recall counts patients found, specificity counts healthy people left alone. "
      "F1 is the comparison number of several of the cited studies %s, but it ignores true negatives, so we never report it alone. AUROC and AUPRC measure ranking quality without a threshold." % (c("Vaiciukynas"), c("machine learning method to process voice"), c("Egbo")))
    H(2, "Why these baselines and extensions")
    P("Logistic regression on frozen self-supervised embeddings is the reference design of Klempir and Krupicka %s and the simplest model with readable weights %s. Random Forest and XGBoost %s are the two tree ensembles those authors and Egbo et al. %s compare against it. "
      "Fine-tuning is the natural next step for a self-supervised encoder %s: WavLM was trained for general speech, not for disease, so adapting its top layers could make it more sensitive to Parkinson's speech." % (c("Klempir"), c("LIBLINEAR"), c("XGBoost: a scalable"), c("Egbo"), c("WavLM: large-scale")))
    H(2, "Frozen versus fine-tuned, in plain words")
    box("plain", "A frozen model is a fixed listener: it turns each recording into 1,024 numbers and never changes. Only the small classifier on top learns from our data. A fine-tuned model lets the top part of the listener change too, so it can learn to pay attention to the cues that matter for Parkinson's disease, but with only 47 people it can also learn things that do not generalise.")

    # 5 ---------------------------------------------------------------
    H(1, "Methodology", "method")
    H(2, "Model description", "model")
    fig("method_overview", "The two model families. Top: the baseline and classifier extensions read a frozen 1,024-number vector per recording. Bottom: the fine-tuning extension trains the top k WavLM layers, the final layer norm and a linear head on 10-second windows.", "1.0")
    table("Inputs, outputs and shapes at each stage (one 10-second window, batch size B).",
          ["Stage", "Input shape", "Output shape", "Notes"],
          [["Audio window", "B x 160,000 samples", "B x 160,000", "16 kHz, zero mean and unit variance per window"],
           ["Convolutional feature encoder", "B x 160,000", "B x 499 x 512", "7 conv layers, 20 ms frame step (frozen)"],
           ["Projection + positional conv", "B x 499 x 512", "B x 499 x 1,024", "frozen"],
           ["Transformer layers 0-19", "B x 499 x 1,024", "B x 499 x 1,024", "frozen; cached once for fine-tuning"],
           ["Transformer layers 20-23 + layer norm", "B x 499 x 1,024", "B x 499 x 1,024", "frozen (baseline) or top k trained (fine-tuning)"],
           ["Mean over frames", "B x 499 x 1,024", "B x 1,024", "baseline also averages windows of a recording"],
           ["Classifier / head", "1,024", "1 probability", "LR, RF, XGB, or dropout + linear head"],
           ["Person score", "probabilities of one person's recordings", "1 score", "mean; PD if score >= 0.5"]],
          "p{4.3cm}p{3.6cm}p{3.0cm}p{5.0cm}", size="footnotesize")
    rows = []
    for m, name in [("LR", "Logistic regression"), ("RF", "Random Forest"), ("XGB", "XGBoost")]:
        g = COUNTS[COUNTS.model == m]
        if m == "LR":
            rows.append([name, "1,025 (1,024 weights + bias)", "non-zero weights per fold: " + ", ".join(str(v) for v in g.nonzero_weights)])
        else:
            rows.append([name, "%d trees per fold (%s)" % (g.trees.iloc[0], ", ".join(str(v) for v in g.trees)) if m == "XGB" else "300 trees per fold",
                         "tree nodes per fold: " + ", ".join("{:,}".format(v) for v in g.nodes)])
    rows.append(["Fine-tuned WavLM", ", ".join("k = %d: {:,} trainable".format(v) % k for k, v in sorted(ft_params.items())), "frozen part: about 315 million weights (whole WavLM-large)"])
    table("Parameter counts of the fitted models (chosen settings, refitted on each fold's training people).", ["Model", "Size", "Detail"], rows, "p{3.2cm}p{5.6cm}p{7.2cm}", size="footnotesize")
    P("The L1 penalty chosen by every baseline fold keeps only a handful of the 1,024 WavLM dimensions (between %d and %d non-zero weights). In other words, the baseline separates the groups with very few directions of the representation, a first hint that the signal is strong and simple."
      % (COUNTS[COUNTS.model == "LR"].nonzero_weights.min(), COUNTS[COUNTS.model == "LR"].nonzero_weights.max()))

    H(2, "Dataset", "dataset")
    P("We use the Italian Parkinson's Voice and Speech corpus %s. Each person performs 16 short tasks: two read passages (B1, B2), syllable repetition /pa/ and /ta/ (D1, D2), a phrase (FB1), a word list (PR1) and five vowels held twice (VA to VU). "
      "We keep the folders of people with Parkinson's disease and of elderly healthy controls. The 15 young controls are excluded because they recorded only three tasks and are 40 years younger, so age would be confused with disease. "
      "One byte-identical duplicate is removed. The result is 785 recordings from 47 people (25 with Parkinson's disease, 22 controls); three people recorded in two sessions are merged into one identity each. "
      "Audio is resampled to 16 kHz with polyphase filtering, without peak normalisation or silence removal (loudness is itself a symptom)." % c("Italian Parkinson's Voice and Speech"))
    fig("dataset_examples", "Representative recordings: a sustained vowel and a phrase from a person with Parkinson's disease and from a control (waveform and spectrogram).", "0.9")
    H(2, "Training and evaluation steps, including batch sampling", "steps")
    fig("nested_cv", "Nested, speaker-disjoint cross-validation. The outer folds give the held-out results; the inner folds, built only from training people, choose all settings.", "0.95")
    bullets(["**Outer split.** The 47 people are dealt into five folds, stratified by class (seed 42). Each person is tested exactly once.",
             "**Inner split.** Inside each outer training set, people are dealt into three inner folds (seed 142 + fold).",
             "**Selection.** Every setting is trained on two inner folds and scored on the third, three times; the pooled person AUPRC chooses the setting (ties: the earliest setting).",
             "**Refit and test.** The chosen setting is refitted on all training people of the fold and predicts the held-out people. Threshold 0.5 is fixed.",
             "**Batch sampling, classical models.** LR, RF and XGB are fitted on all training recordings at once; no mini-batches are used.",
             "**Batch sampling, fine-tuning.** Each mini-batch holds 8 ten-second windows: 4 from people with Parkinson's disease and 4 from controls. Within a class a person is drawn uniformly, then one of their windows uniformly, so every person and both classes have equal expected weight regardless of how long they spoke. One epoch is the number of training windows divided by 8."], enum=True)

    H(2, "Evaluation metrics", "metrics")
    P("Let $\\mathcal{P}$ be the set of test people, $y_i \\in \\{0, 1\\}$ the true label of person $i$ (1 = Parkinson's disease) and $s_i$ the person score, the mean of the predicted probabilities of person $i$'s recordings. "
      "The decision is $\\hat{y}_i = 1$ if $s_i \\geq 0.5$. Counting over people:")
    math("\\begin{align*}TP&=\\textstyle\\sum_i[\\hat y_i=1,\\,y_i=1]\\ \\text{(patients found)}, & FN&=\\textstyle\\sum_i[\\hat y_i=0,\\,y_i=1]\\ \\text{(patients missed)},\\\\ "
         "TN&=\\textstyle\\sum_i[\\hat y_i=0,\\,y_i=0]\\ \\text{(controls cleared)}, & FP&=\\textstyle\\sum_i[\\hat y_i=1,\\,y_i=0]\\ \\text{(false alarms)},\\\\ "
         "\\text{Recall}&=\\frac{TP}{TP+FN}, \\quad \\text{Specificity}=\\frac{TN}{TN+FP}, & \\text{Precision}&=\\frac{TP}{TP+FP}, \\quad F_1=\\frac{2\\,\\text{Precision}\\cdot\\text{Recall}}{\\text{Precision}+\\text{Recall}},\\\\ "
         "\\text{AUROC}&=\\frac{1}{N_1N_0}\\sum_{i:y_i=1}\\sum_{j:y_j=0}\\Big([s_i>s_j]+\\tfrac12[s_i=s_j]\\Big), & \\text{AUPRC}&=\\sum_{n}(R_n-R_{n-1})\\,P_n .\\end{align*}")
    P("$N_1$ and $N_0$ are the numbers of patients and controls; $P_n$ and $R_n$ are precision and recall at the n-th threshold (average precision). "
      "AUROC is the chance that a random patient scores above a random control. Uncertainty comes from 2,000 bootstrap resamples of people (seed 42); differences between models use the same resamples (paired).")

    H(2, "Loss functions and optimisation", "loss")
    P("With $x_r \\in \\mathbb{R}^{1024}$ a recording vector (standardised with training statistics) and $y_r$ its label, the models minimise the following objectives on the training data of each fold.")
    math("\\begin{align*}\\textbf{LR:}\\quad &\\min_{w,b}\\ \\lVert w\\rVert_q^{q'}\\!/\\,\\!c + C\\sum_r \\log\\big(1+e^{-\\tilde y_r(w^\\top x_r+b)}\\big),\\quad \\tilde y_r\\in\\{-1,1\\},\\ (q,q',c)=(1,1,1)\\ \\text{or}\\ (2,2,2),\\\\ "
         "\\textbf{RF:}\\quad &\\text{each split maximises } G(\\text{node})-\\tfrac{n_L}{n}G(L)-\\tfrac{n_R}{n}G(R),\\quad G=1-\\textstyle\\sum_{k\\in\\{0,1\\}}p_k^2,\\\\ "
         "\\textbf{XGB:}\\quad &\\min\\ \\textstyle\\sum_r \\ell\\big(y_r,\\hat y_r\\big)+\\sum_{t}\\Omega(f_t),\\quad \\Omega(f)=\\gamma T+\\tfrac12\\lambda\\lVert w\\rVert^2,\\quad \\ell=\\text{log-loss},\\\\ "
         "\\textbf{Fine-tuning:}\\quad &\\min_{\\theta}\\ -\\frac{1}{|\\mathcal{B}|}\\sum_{u\\in\\mathcal{B}}\\Big[y_u\\log\\sigma(z_u)+(1-y_u)\\log\\big(1-\\sigma(z_u)\\big)\\Big],\\quad z_u=v^\\top\\,\\overline{h}_\\theta(u)+b .\\end{align*}")
    P("For LR, liblinear solves the problem %s; class weights, when chosen, rescale each class's terms. For XGBoost, $T$ is the number of leaves of a tree and $w$ its leaf values; trees are added by second-order boosting %s. "
      "For fine-tuning, $u$ is a window in mini-batch $\\mathcal{B}$, $\\overline{h}_\\theta(u)$ the mean over frames of the last layer norm, and $\\theta$ the weights of the top k layers. "
      "Optimisation uses AdamW (weight decay 0.01; learning rate 1e-5 or 3e-5 for WavLM layers, 1e-3 for the head), 10%% linear warm-up, gradient clipping at norm 1 and bfloat16 arithmetic." % (c("LIBLINEAR"), c("XGBoost: a scalable")))

    H(2, "Experimental depth and iteration", "depth")
    n_fits_classic = 5 * 4 * (20 + 8 + 24) + 5 * 4 * (32 + 96 + 128) + 5 * (32 + 96 + 128)
    table("Every configuration tried. Each was trained and scored inside every outer fold.",
          ["Stage", "Model", "Settings searched", "Number"],
          [["1. Baseline (frozen protocol)", "LR", "C in {0.01, 0.1, 1, 10, 100} x penalty {L1, L2} x class weight {none, balanced}", "20"],
           ["2. Classifier extensions", "XGB (frozen protocol)", "24 random draws from 256 combinations of rate, depth, trees, child weight, subsample, columns, lambda, class weight", "24"],
           ["", "RF (amendment A2)", "depth {4, none} x min leaf {1, 2} x class weight {none, balanced}; 300 trees", "8"],
           ["3. Hyperparameter study (A3)", "LR", "C in {1e-4 ... 1e3} (8 values) x penalty x class weight", "32"],
           ["", "RF", "trees {100, 300, 1000} x depth {2, 4, 8, none} x min leaf {1, 5} x features {sqrt, 10%} x class weight", "96"],
           ["", "XGB", "depth {1, 2, 4, 6} x rate {0.01, 0.03, 0.1, 0.3} x trees {100, 300} x subsample {0.7, 1} x columns {0.3, 1}", "128"],
           ["4. Fine-tuning (A3)", "WavLM top k + head", "k in {2, 4} x learning rate {1e-5, 3e-5} x epochs 1-6 (chosen on inner folds)", "4 x 6"]],
          "p{3.4cm}p{2.9cm}p{8.0cm}r", size="footnotesize")
    P("In total about %s classical model fits and 65 fine-tuning runs (5 folds x 4 settings x 3 inner folds, plus 5 refits) were made, %.1f GPU hours for fine-tuning. The development path was: baseline, then other classifiers on the same vectors, then a much wider search over their settings, then adapting the speech model itself."
      % ("{:,}".format(n_fits_classic), ft_runtime))

    # 6 ---------------------------------------------------------------
    H(1, "Baseline and extensions", "baseline")
    H(2, "Baseline choice and justification", "baselinechoice")
    P("The baseline is frozen WavLM-large with logistic regression. We chose it for four reasons. (1) It is the reference design for frozen self-supervised embeddings in Parkinson's speech %s. (2) It has the fewest learned parameters, which matters with 47 people. "
      "(3) Its weights are readable, and with an L1 penalty it shows how many representation dimensions are needed. (4) WavLM was chosen as the encoder because it is the strongest of the three encoders in the FINAL_V1 comparison on general speech benchmarks %s and it was the predeclared baseline of the protocol." % (c("Klempir"), c("WavLM: large-scale")))
    H(2, "Systematic evaluation of the baseline", "baselineeval")
    rows = []
    for f in FOLDS:
        q = FOLDM[(FOLDM.model == "LR") & (FOLDM.fold == f)].iloc[0]
        s, inner = SEL[("LR", f)]
        nz = int(COUNTS[(COUNTS.model == "LR") & (COUNTS.fold == f)].nonzero_weights.iloc[0])
        rows.append([str(f + 1), "%d (%d PD)" % (q.people, q.pd), "C=%s, %s, weight %s" % (s["C"], s["penalty"].upper(), s["class_weight"] or "none"), str(nz), pct(inner),
                     "%d/%d/%d/%d" % (q.tp, q.fn, q.tn, q.fp), pct(q.recall), pct(q.specificity), pct(q.auroc)])
    rows.append(["Pooled", "47 (25 PD)", "", "", "", "%d/%d/%d/%d" % (lr["tp"], lr["fn"], lr["tn"], lr["fp"]), pct(lr["recall"]), pct(lr["specificity"]), pct(lr["auroc"])])
    table("Baseline, fold by fold. Settings and inner AUPRC come from the training people only; the last four columns are held-out.",
          ["Fold", "Test people", "Chosen setting", "Non-zero weights", "Inner AUPRC", "TP/FN/TN/FP", "Recall", "Spec.", "AUROC"], rows, "llp{3.1cm}rrlrrr", size="scriptsize")
    H(2, "Baseline outputs confirming reproduction", "reproduction")
    rows = []
    for f in FOLDS:
        r = json.loads((METRICS / ("FV1-WLM-ALL-LR-F%d.json" % f)).read_text())
        rows.append([r["run_id"], r["status"], "all %d PASS" % len(r["gates"]), r["protocol_hash"][:12], r["split_manifest_hash"][:12], "%.0f s" % r["runtime_seconds"]])
    table("Run log of the five official baseline runs (from the saved run records).", ["Run", "Status", "Integrity gates", "Protocol hash", "Split hash", "Runtime"], rows, "llllll", size="footnotesize")
    P("Recomputing the pooled baseline from its saved predictions gives exactly the saved values (TP %d, FN %d, TN %d, FP %d; AUROC %s%%). The same holds for XGBoost. All checks are listed in Appendix A." % (lr["tp"], lr["fn"], lr["tn"], lr["fp"], pct(lr["auroc"])))

    H(2, "Extension 1: other classifiers on the same vectors", "ext1")
    P("**Idea.** If the WavLM vectors contain non-linear patterns, tree ensembles could use them where a linear model cannot. **Design.** Random Forest and XGBoost read exactly the same vectors, folds, selection rule and threshold as the baseline (Table 1 settings, rows 2). "
      "**Result.** Random Forest makes %d error(s) and XGBoost %d, against %d for the baseline. Random Forest and the baseline give the same decision for every person. XGBoost ranks people slightly better (AUROC %s%% against %s%%) but adds a false alarm."
      % (p["RF"]["errors"], p["XGB"]["errors"], lr["errors"], pct(p["XGB"]["auroc"]), pct(lr["auroc"])))
    H(2, "Extension 2: changing the hyperparameters", "ext2")
    land = LAND
    rows = []
    for m in ["LR", "RF", "XGB"]:
        g = land[land.model == m]
        rows.append([m, str(len(g)), pct(g.auroc.min()), pct(g.auroc.median()), pct(g.auroc.max()), "%d to %d" % (g.errors.min(), g.errors.max()),
                     "%.0f%%" % (100 * (g.errors <= lr["errors"]).mean()), str(p[m + "_wide"]["errors"]), pct(p[m + "_wide"]["auroc"])])
    table("Wide hyperparameter study. Columns 3-7 describe every setting scored on the held-out folds (descriptive only); the last two columns are the honest nested result, where each fold chose its own setting on inner folds.",
          ["Model", "Settings", "Min AUROC", "Median AUROC", "Max AUROC", "Errors range", "Settings with <= baseline errors", "Nested errors", "Nested AUROC"],
          rows, "lrrrrrrrr", size="scriptsize")
    P("**Idea.** Perhaps the original grids were too narrow. **Design.** We widened every grid (256 settings in total, Section 5.6) and repeated nested selection. **Result.** The best single settings look slightly better than the baseline, but choosing them would use the test people. "
      "When each fold chooses on its own training people, the wide grids give %s errors (LR, RF, XGB) against %d for the baseline." % (", ".join(str(p[m + "_wide"]["errors"]) for m in ["LR", "RF", "XGB"]), lr["errors"]))
    H(2, "Extension 3: fine-tuning WavLM", "ext3")
    rows = []
    for f in FOLDS:
        s, inner = SEL[("FT", f)]
        q = FOLDM[(FOLDM.model == "FT") & (FOLDM.fold == f)].iloc[0]
        fi = FTSUM["folds"][f]
        rows.append([str(f + 1), "k = %d, lr = %.0e" % (s["k"], s["lr"]), str(s["epochs"]), "{:,}".format(fi["trainable_parameters"]), pct(inner),
                     "%d/%d/%d/%d" % (q.tp, q.fn, q.tn, q.fp), pct(q.auroc), "%.0f min" % (fi["runtime_seconds"] / 60)])
    rows.append(["Pooled", "", "", "", "", "%d/%d/%d/%d" % (ft["tp"], ft["fn"], ft["tn"], ft["fp"]), pct(ft["auroc"]), "%.1f h" % ft_runtime])
    table("Fine-tuning, fold by fold. Setting, epochs and inner AUPRC are chosen on training people only.",
          ["Fold", "Chosen setting", "Epochs", "Trainable weights", "Inner AUPRC", "TP/FN/TN/FP", "AUROC", "Runtime"], rows, "llrrrlrr", size="footnotesize")
    P("**Idea.** Let the top of the speech model adapt to Parkinson's speech. **Design.** The top 2 or 4 transformer layers, the final layer norm and a linear head are trained on 10-second windows with balanced mini-batches; layers below are frozen and their output is cached once "
      "(the cache reproduces the frozen vectors to within %.1e). Settings and the number of epochs are chosen on inner folds only. **Result.** Fine-tuning makes %d error(s) (recall %s%%, specificity %s%%) with AUROC %s%%. "
      "The inner folds chose k = 4 in %d of 5 folds."
      % (json.loads((RES / "S13_cache_summary.json").read_text())["max_abs_cache_vs_saved"], ft["errors"], pct(ft["recall"]), pct(ft["specificity"]), pct(ft["auroc"]), k_counts.get(4, 0)))
    H(2, "Did any extension improve on the baseline?", "verdict")
    box("rule", "An extension counts as an improvement only if it makes at least two fewer person errors than the baseline AND the 95% paired bootstrap interval of its AUROC minus the baseline AUROC excludes zero. Fixed in supplementary/PREDECLARED_A3.md before any A3 result existed.")
    rows = []
    for n in ["RF", "XGB", "FT", "LR_wide", "RF_wide", "XGB_wide"]:
        v = VERDICT[n]
        rows.append([dict(MAIN + WIDE)[n], str(p[n]["errors"]), "%+d" % (-v["fewer_errors"]), "%+.2f %s" % (100 * DELTA[n]["auroc"][0], ci_txt(DELTA[n]["auroc"])),
                     "%+.2f %s" % (100 * DELTA[n]["auprc"][0], ci_txt(DELTA[n]["auprc"])), "yes" if v["improvement"] else "no"])
    table("Each extension against the baseline (%d errors). Differences in percentage points with 95%% paired bootstrap intervals." % lr["errors"],
          ["Extension", "Errors", "Change in errors", "AUROC difference", "AUPRC difference", "Improvement?"], rows, "lrrllc", size="footnotesize")
    fig("paired_differences", "Paired bootstrap differences from the baseline. An interval that crosses zero means the data cannot tell the two models apart.", "0.95")
    P("**Justified analysis of the failure to improve.** Three facts explain why no extension can show an improvement here. First, there is almost no room: the baseline already ranks %s%% of patient-control pairs correctly, so the most an extension could gain is about %.1f AUROC points. "
      "Second, the remaining errors are the same people for every model (Section 7.6), so they are not caused by the classifier. Third, with 47 people one person is worth 4 points of recall, and the bootstrap intervals are wider than any difference between the models. "
      "A larger, harder or external test set is needed to separate them (Section 9)." % (pct(lr["auroc"]), 100 * (1 - lr["auroc"])))

    # 7 ---------------------------------------------------------------
    H(1, "Results and analysis", "results")
    H(2, "Training, validation and test, kept separate", "tvt")
    rows = []
    for n, name in MAIN:
        t = TVT[TVT.model == n]
        rows.append([name, pm(t.train_auprc), pm(t.val_auprc), pm(t.test_auprc), pm(t.train_auroc), pm(t.test_auroc), pm(t.train_f1), pm(t.test_f1)])
    table("Mean $\\pm$ SD over the five folds (percent). Training = the refitted model scored on its own training people; validation = the inner-fold score that chose the setting; test = held-out people.",
          ["Model", "Train AUPRC", "Val. AUPRC", "Test AUPRC", "Train AUROC", "Test AUROC", "Train F1", "Test F1"], rows, "lrrrrrrr", size="scriptsize")
    fig("train_val_test_models", "Training, inner-validation and held-out AUPRC for every model (mean and SD over folds).", "0.8")
    P("Every model fits its training people %s; the tree models and the fine-tuned model reach a perfect or near-perfect training score. Validation and test scores stay high and close to each other, which means the selection step did not overfit the inner folds in a way that hurt the test folds. "
      "The gap between training and test is the cost of meeting new people; it is small here, which is itself suspicious for a medical task with 47 people (Section 8)." % ("almost perfectly" if TVT.train_auroc.min() > 0.99 else "well"))

    H(2, "Held-out comparison of all models", "heldout")
    rows = []
    for n, name in MAIN + WIDE:
        q = p[n]
        rows.append([name, "%d/%d/%d/%d" % (q["tp"], q["fn"], q["tn"], q["fp"]), pct(q["f1"]), pct(q["recall"]), pct(q["specificity"]), pct(q["precision"]),
                     pct(q["accuracy"]), "%s %s" % (pct(q["auroc"]), ci_txt(CI[n]["auroc"])), "%s %s" % (pct(q["auprc"]), ci_txt(CI[n]["auprc"]))])
    table("Pooled held-out person-level results with 95% bootstrap intervals (percent).",
          ["Model", "TP/FN/TN/FP", "F1", "Recall", "Spec.", "Prec.", "Acc.", "AUROC [95% CI]", "AUPRC [95% CI]"], rows, "lrrrrrrll", size="scriptsize")
    fig("roc", "Pooled person-level ROC curves on the held-out folds.", "0.62")

    H(2, "Variation across folds", "folds")
    rows = []
    for n, name in MAIN:
        g = FOLDM[FOLDM.model == n]
        rows.append([name, pm(g.f1), pm(g.recall), pm(g.specificity), pm(g.auroc)])
    table("Held-out results per fold: mean $\\pm$ sample SD over five folds (percent).", ["Model", "F1", "Recall", "Specificity", "AUROC"], rows, "lrrrr", size="footnotesize")
    fig("folds", "Every dot is one held-out fold of 9 or 10 people. One missed patient lowers a fold's recall by about 20 points.", "0.95")

    H(2, "How the hyperparameters change the result", "hparams")
    fig("hparam_effects", "Effect of the main hyperparameters on pooled held-out AUROC (descriptive; never used to choose). Left: logistic regression against C. Middle and right: tree models, averaged over the other settings.", "1.0")
    fig("hparam_landscape", "Left: held-out AUROC of every setting; the star is the honest nested result with the wide grid, the diamond the original grid. Right: how many people each setting misclassifies.", "0.98")
    lrl = land[land.model == "LR"]
    P("Three patterns stand out. **Logistic regression** is stable over most of the range of C; only the strongest L1 penalties remove every weight, so the model predicts the same score for everyone and the AUROC falls to %s%%. "
      "**Random Forest** gains little from more trees or deeper trees once each tree can make a few splits. **XGBoost** is the most sensitive model: very shallow, slow-learning settings and very aggressive ones differ most. "
      "Across all 256 settings, %d%% make at most as many errors as the baseline. The choice of setting therefore matters less than which people are hard."
      % (pct(lrl.auroc.min()), round(100 * (land.errors <= lr["errors"]).mean())))

    H(2, "Fine-tuning dynamics", "ftdyn")
    fig("ft_curves", "Fine-tuning learning curves, averaged over outer and inner folds. Training loss falls steadily; inner-validation loss and AUPRC show how quickly the model starts to fit the training people.", "1.0")
    vc = FTCURVE.groupby("epoch").val_loss.mean()
    P("Training loss falls with every epoch. The inner-validation loss is lowest at epoch %d and rises afterwards, the usual sign of overfitting; the inner folds chose between %d and %d epochs. "
      "Training four layers instead of two, or using the larger learning rate, makes the training loss fall faster but does not improve validation, which matches the held-out result: the extra capacity is spent on the training people."
      % (int(vc.idxmin()), min(SEL[("FT", f)][0]["epochs"] for f in FOLDS), max(SEL[("FT", f)][0]["epochs"] for f in FOLDS)))

    H(2, "Error and failure analysis", "errors")
    rows = []
    for n, name in MAIN:
        for sid, r in errs[n].iterrows():
            rows.append([name, sid, "missed patient" if r.label == 1 else "false alarm", "%.3f" % r.score, "fold %d" % (r.fold + 1)])
    table("Every misclassified person in the held-out folds.", ["Model", "Person", "Error", "Score", "Fold"], rows, "llllr", size="footnotesize")
    fig("person_scores", "Person scores of the four main models on the held-out folds. The dashed line is the threshold; labelled points are errors.", "1.0")
    fig("task_breakdown", "Failure analysis by speech task: probability of Parkinson's disease for each recording of the misclassified people, for every model.", "0.98")
    if missed_all:
        sid = missed_all[0]
        d = REC["LR"][REC["LR"].speaker_id == sid]
        tasks = SM.loc[d.sample_id, "task_code"].values
        above = int((d.predicted_probability.values >= THRESHOLD).sum())
        P("**Failure mode 1: one patient sounds like a control to every model.** Person %s is missed by all four models. Only %d of their %d recordings are scored above 0.5 by the baseline, and the pattern is the same for the other models (figure above). "
          "Because the classifiers and even the fine-tuned encoder agree, the error is a property of this person's recordings, not of a model: either their speech is mildly affected, or their recordings share the acoustic conditions of the control sessions. "
          "This is the kind of case a screening tool would miss." % (sid, above, len(tasks)))
    fa = [s for s in any_err if int(SPK["LR"].loc[s, "label"]) == 0]
    if fa:
        P("**Failure mode 2: borderline false alarms.** Control %s is called Parkinson's by %s only, with a score just above the threshold. Its task-by-task scores are mixed, so a small change in the training data moves it across 0.5. "
          "Such borderline cases are where a calibrated threshold or a second test would help." % (", ".join(fa), ", ".join(SHORT[n] for n, _ in MAIN if any(s in errs[n].index for s in fa))))

    H(2, "Sensitivity to the input: which speech tasks carry the signal", "families")
    fig("task_families", "Person AUROC when each model's score is computed from one family of speech tasks only.", "0.85")
    rows = []
    for fam in ["passages", "phrase", "word list", "syllables", "vowels"]:
        g = FAMTAB[FAMTAB.family == fam]
        rows.append([fam, str(int(g.people.iloc[0])), str(int(g.recordings.iloc[0]))] + ["%s (%d)" % (pct(g[g.model == n].auroc.iloc[0]), int(g[g.model == n].errors.iloc[0])) for n, _ in MAIN])
    table("Person AUROC (errors in brackets) from one task family at a time (percent).", ["Task family", "People", "Recordings", "LR", "RF", "XGB", "FT"], rows, "lrrllll", size="footnotesize")
    P("All task families separate the groups well, including sustained vowels, which contain no words at all. This is consistent with a disease signal in the voice, but it is also exactly what one would expect if the models recognised recording conditions, which are shared by every task of a session (Section 8.2).")

    H(2, "Key findings", "findings")
    bullets(["A frozen WavLM-large with a sparse logistic regression already reaches %s%% AUROC with %d errors in 47 people." % (pct(lr["auroc"]), lr["errors"]),
             "Neither other classifiers, nor 256 alternative settings, nor fine-tuning of the top layers improve on it under the pre-declared rule.",
             "All models fail on the same patient, so the remaining errors belong to the data, not to the model.",
             "Fine-tuning fits the training people quickly and then overfits; with 47 people the extra capacity does not help.",
             "Every task family, even vowels, separates the groups, which is consistent with both a voice signal and a recording-condition shortcut."])

    # 8 ---------------------------------------------------------------
    H(1, "Discussion", "discussion")
    H(2, "Significance of the results", "significance")
    P("For a course project the main lesson is methodological: when a strong frozen representation already separates the classes, adding model capacity is not where the gains are. The models agree on almost every person, and their differences are smaller than the uncertainty from having 47 people. "
      "This matches the observation that scores on small Parkinson's speech corpora are often near the ceiling %s and should be read with care." % c("Vaiciukynas"))
    H(2, "Risks: recording conditions", "risks")
    s6 = S6[(S6.representation == "WLM") & (S6.model == "LR")].set_index("task")
    P("The largest risk is that the models detect *how the recordings were made* rather than the disease. Three facts from FINAL_V1 support this risk. "
      "(1) Digital silence: %d of %d control recordings start with a short block of exact zeros, which no patient recording has, and %d of %d patient recordings end with at least half a second of exact zeros, which no control recording has. "
      "(2) Sessions are recognisable: within the controls, the same WavLM vectors tell two recording periods apart with AUROC %s%%, and within the patients they separate 2016 from 2017 recordings with AUROC %s%%. "
      "(3) Each recording date belongs to one group only. A model can therefore score well by recognising the session. Our results cannot rule this out."
      % (S9["recordings_with_leading_zero_block_ge_5ms"]["0"], S9["recordings_per_label"]["0"], S9["recordings_with_trailing_zeros_ge_0.5s"]["1"], S9["recordings_per_label"]["1"],
         pct(s6.loc["T2_controls_17-21Mar_vs_23Mar-2Apr", "unit_auroc"]), pct(s6.loc["T3_PD_2016_44k_vs_2017_16k", "unit_auroc"])))
    H(2, "Sensitivity of the results to input changes", "inputsens")
    P("Changing the input from all tasks to one task family changes AUROC by at most %.1f points for any model (Section 7.7), and changing the hyperparameters changes it by at most %.1f points (Section 7.4). Fine-tuning changes the representation itself and still leaves the same patient misclassified. "
      "The result is therefore robust to the design choices we varied, but this robustness says nothing about robustness to a new microphone, room or language."
      % (100 * (FAMTAB.groupby("model").auroc.max() - FAMTAB.groupby("model").auroc.min()).max(), 100 * (land.auroc.max() - land.auroc.min())))
    H(2, "Limitations", "limitations")
    bullets(["One corpus, one language, one recording setup; no external test set.",
             "47 people: one person moves recall by 4 points and specificity by 4.5 points.",
             "One outer split (seed 42). Fine-tuning on a GPU is not bit-exactly repeatable; we fixed seeds but did not repeat runs with other seeds.",
             "Random Forest, the wide grids and fine-tuning are exploratory amendments added after the frozen comparison; they were declared before their runs but are reported beside, not instead of, the frozen results.",
             "Only the top 2 or 4 of 24 layers were fine-tuned, with at most 6 epochs; deeper or longer fine-tuning was not tested."])

    # 9 ---------------------------------------------------------------
    H(1, "Future directions", "future")
    P("Our results point to the data, not the model, as the limit. The next steps are ordered by how much they would change what we can claim.")
    table("Planned next steps, each tied to a result of this report.", ["#", "Next step", "Why (from our results)", "How we would measure success"],
          [["1", "Remove the recording-condition shortcut: trim leading and trailing digital silence, then retrain; train a model on silent frames only", "Digital-silence blocks and session predictability (Section 8.2)", "If the silence-only model is near chance and the trimmed model keeps its AUROC, the signal is in the voice"],
           ["2", "Test across corpora: train on the Italian corpus, test on NeuroVoz (Spanish) and vice versa", "No external test set; all errors are shared by all models", "Person AUROC and recall on the other corpus, with intervals"],
           ["3", "Session-aware evaluation: hold out whole recording dates, not only people", "Each date belongs to one group", "Drop in AUROC when sessions are held out"],
           ["4", "Better adaptation with little data: low-rank adapters, layer-wise learning-rate decay, more epochs with stronger regularisation, several seeds", "Fine-tuning overfits after few epochs (Section 7.5)", "Inner and held-out AUPRC averaged over seeds"],
           ["5", "Choose a screening threshold on training people for high recall, and report predictive values at a realistic prevalence", "Threshold 0.5 is arbitrary; the clinical goal is to miss few patients", "Recall at 95% specificity; positive predictive value at 1% prevalence"],
           ["6", "Collect more people, with matched recording sessions for patients and controls", "One person moves recall by 4 points", "Narrower intervals; a power plan from FINAL_V1 suggests about 33 to 70 people per class to bound sensitivity above 75%"]],
          "rp{4.2cm}p{4.6cm}p{5.4cm}", size="footnotesize")
    P("Steps 1 and 3 are cheap and decisive: they reuse the saved models and data and directly test the main risk. Step 2 is the real test of usefulness. Step 4 only makes sense after steps 1 to 3, because a better-adapted model on confounded data would only learn the shortcut better.")

    # 10 --------------------------------------------------------------
    H(1, "Conclusion", "conclusion")
    rows = [["O1", "Report all metrics with intervals", "Done for 7 models (Sections 7.1-7.3)"],
            ["O2", "Decide improvement by the pre-declared rule", "No extension improved; analysed in Section 6.6"],
            ["O3", "Training, validation and test separately", "Done (Section 7.1)"],
            ["O4", "Failure modes and sensitivity", "Shared missed patient, borderline false alarms; hyperparameters and task families (Sections 7.4-7.7)"]]
    table("Objectives and how they were met.", ["#", "Objective", "Outcome"], rows, "lp{6cm}p{8.4cm}", size="footnotesize")
    P("A frozen WavLM-large representation with a sparse logistic regression separates people with Parkinson's disease from elderly controls on held-out people with %s%% AUROC and %d errors in 47. "
      "Changing the classifier, widening its settings or fine-tuning the speech model does not improve this, and every model misses the same patient. "
      "The honest conclusion is that this corpus is too easy and too small to rank methods, and that recording conditions may explain part of the separation. The most valuable next steps are therefore about the data: removing shortcuts, holding out sessions, and testing on another corpus." % (pct(lr["auroc"]), lr["errors"]))

    # appendices --------------------------------------------------------
    H(0, "Appendix A. Checks performed during the build")
    P("The build stops and writes nothing if any check fails.")
    table("Checks run on the saved inputs.", ["Check", "Result"], [[n, r] for n, r, _ in checks_log], "p{13.3cm}l", size="scriptsize")
    H(0, "Appendix B. Chosen settings per fold")
    rows = []
    for f in FOLDS:
        rf_s = SEL[("RF", f)][0]
        xg_s = SEL[("XGB", f)][0]
        rows.append([str(f + 1), "C=%s %s" % (SEL[("LR", f)][0]["C"], SEL[("LR", f)][0]["penalty"].upper()),
                     "depth %s, leaf %s, weight %s" % ("none" if rf_s["max_depth"] is None else rf_s["max_depth"], rf_s["min_samples_leaf"], rf_s["class_weight"] or "none"),
                     "depth %s, rate %s, %s trees" % (xg_s["max_depth"], xg_s["learning_rate"], xg_s["n_estimators"]),
                     "k=%d, lr=%.0e, %d epochs" % (SEL[("FT", f)][0]["k"], SEL[("FT", f)][0]["lr"], SEL[("FT", f)][0]["epochs"])])
    table("Settings chosen by the inner folds in each outer fold.", ["Fold", "LR", "RF", "XGB", "Fine-tuning"], rows, "lllll", size="footnotesize")
    H(0, "Appendix C. Where to find everything")
    bullets(["Report, LaTeX source and Overleaf zip: `reports/WAVLM_BASELINE_SUBSET/versions/v1.2/`.",
             "Generator: `src/final_v1/wavlm_subset_report.py` (copy in the version folder).",
             "Pre-declarations: `supplementary/PREDECLARED_RF_GRID.md` (A2) and `supplementary/PREDECLARED_A3.md` (A3).",
             "Code: `supplementary/code/s11_random_forest.py`, `s12_hparams.py`, `s13_cache.py`, `s13_finetune.py`, `s14_param_counts.py`.",
             "Results: `supplementary/results/S11_*`, `S12_*`, `S13_*`, `S14_*`; official FINAL_V1 predictions in `predictions/FINAL_V1/`."])
    H(0, "References")
    A("\\begin{thebibliography}{99}\\setlength{\\itemsep}{1pt}\\small")
    A("".join("\\bibitem{r%d} %s\n" % (i + 1, esc(r)) for i, r in enumerate(REFS)) + "\\end{thebibliography}")


def build():
    if (VOUT / (STEM + ".pdf")).exists():
        print("version %s already exists; it is never overwritten." % REPORT_VERSION); sys.exit(10)
    compute()
    errs = sorted(set().union(*[set(SPK[n][(SPK[n].score >= THRESHOLD) != SPK[n].label.astype(bool)].index) for n, _ in MAIN]))
    make_figures(errs[:3])
    B.clear(); title_page(); title_block = "\n".join(B)
    B.clear(); write_body(); body = "\n".join(B)
    build_dir = VOUT / "_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    (build_dir / (STEM + ".tex")).write_text(PRE + title_block + body + "\n\\end{document}\n", encoding="utf-8")
    for fn in FIG.glob("*.pdf"):
        shutil.copy2(fn, build_dir / fn.name)
    for _ in range(3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", STEM + ".tex"], cwd=build_dir, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); sys.exit(8)
    for suffix in [".pdf", ".tex"]:
        shutil.copy2(build_dir / (STEM + suffix), VOUT / (STEM + suffix))
    shutil.copy2(build_dir / (STEM + ".log"), VOUT / (STEM + ".build.log"))
    src = VOUT / "latex_source"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir()
    for fn in build_dir.glob("*.pdf"):
        if fn.name != STEM + ".pdf":
            shutil.copy2(fn, src / fn.name)
    shutil.copy2(build_dir / (STEM + ".tex"), src / (STEM + ".tex"))
    with zipfile.ZipFile(VOUT / (STEM + "_latex.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(src.iterdir()):
            z.write(fn, STEM + "/" + fn.name)
    gen = VOUT / "generator"
    gen.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), gen / ("wavlm_subset_report_v" + REPORT_VERSION + ".py"))
    shutil.copy2(Path(__file__).resolve().parent / "wavlm_subset_refs.py", gen / "wavlm_subset_refs.py")
    shutil.rmtree(build_dir)
    info = subprocess.run(["pdfinfo", str(VOUT / (STEM + ".pdf"))], capture_output=True, text=True).stdout
    pages = [l.split()[-1] for l in info.splitlines() if l.startswith("Pages")][0]
    print("wrote", VOUT / (STEM + ".pdf"), "| pages", pages, "| checks passed", sum(1 for c in checks_log if c[1] == "PASS"))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "build":
        print("usage: wavlm_subset_report.py build"); sys.exit(2)
    build()
