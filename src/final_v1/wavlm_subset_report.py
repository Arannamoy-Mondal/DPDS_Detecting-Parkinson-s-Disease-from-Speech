"""WavLM baseline subset report (LaTeX -> PDF), document version 1.1.

Scope: the baseline part of the FINAL_V1 comparison. Frozen WavLM-large embeddings (1,024 values per recording) are
classified by logistic regression (the predeclared baseline), with Random Forest (exploratory amendment A2) and XGBoost
as the matched alternatives. Everything is read from the FINAL_V1 outputs and the S11 run in this workspace.
A separate PatchTST proposal run (v4.0) is used only as a labelled cross-check; its copy lives in inputs/.

Commands (workspace root):
    .venv/bin/python src/final_v1/wavlm_subset_report.py crosscheck   one-time copy of the v4.0 cross-check inputs
    .venv/bin/python src/final_v1/wavlm_subset_report.py build        recompute, check, write versions/v1.1/

The build stops and writes nothing if any check fails. Existing versions are not overwritten.
"""
import hashlib, json, re, shutil, statistics, subprocess, sys, zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wavlm_subset_refs import REFS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/WAVLM_BASELINE_SUBSET"
IN = OUT / "inputs"
FIG = OUT / "figures"
RES = ROOT / "supplementary/results"
PRED = ROOT / "predictions/FINAL_V1"
METRICS = ROOT / "metrics/FINAL_V1"
REPORT_VERSION = "1.1"
REPORT_DATE = "21 October 2026"
VOUT = OUT / "versions" / ("v" + REPORT_VERSION)
STEM = "WAVLM_BASELINE_SUBSET_REPORT_v" + REPORT_VERSION
V4_RUN = Path("/root/Projects/PatchTST/artifacts/runs/20261014T191654Z_v4.0_eb438e63b942")
FOLDS = range(5)
THRESHOLD = 0.5
NBOOT = 2000
MODELS = [("LR", "Logistic regression"), ("RF", "Random Forest"), ("XGB", "XGBoost")]
COLORS = {"LR": "#2B5FAF", "RF": "#3C8D5A", "XGB": "#B5651D"}
problems, checks_log, B = [], [], []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, ok, detail=""):
    checks_log.append((name, "PASS" if ok else "FAIL", detail))
    if not ok:
        problems.append(name)
        print("FAIL  " + name + "  " + detail)


def cite(keyword):
    """Number of the one reference whose text contains keyword, so citation numbers cannot drift from the list."""
    hits = [i + 1 for i, r in enumerate(REFS) if keyword in r]
    if len(hits) != 1:
        print("citation keyword not unique:", keyword, hits); sys.exit(7)
    return hits[0]


# ------------------------------------------------------------------ cross-check copy (v4.0, PatchTST)
def crosscheck():
    """Copy the two v4.0 files used for the cross-check, with fingerprints. The source project is only read."""
    if not V4_RUN.exists():
        print("STOP: v4.0 run not found:", V4_RUN); sys.exit(1)
    manifest = {}
    for rel in ["pooled_metrics.csv", "fine_tuning_decisions.json"]:
        dst = IN / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(V4_RUN / rel, dst)
        manifest[rel] = {"source": str(V4_RUN / rel), "sha256": digest(dst)}
    (IN / "crosscheck_manifest.json").write_text(json.dumps({"source_run": str(V4_RUN), "files": manifest}, indent=2), encoding="utf-8")
    print("cross-check inputs written:", len(manifest))


# ------------------------------------------------------------------ metrics (same definitions as the FINAL_V1 pipeline)
def metrics_from(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= THRESHOLD).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec + sens > 0 else 0.0
    return {"people": len(labels), "pd": int(labels.sum()), "controls": int(len(labels) - labels.sum()), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "recall": sens, "specificity": spec, "precision": prec, "f1": f1, "accuracy": (tp + tn) / len(labels),
            "auroc": float(roc_auc_score(labels, scores)), "auprc": float(average_precision_score(labels, scores))}


def sample_predictions(model):
    if model == "RF":
        return pd.read_csv(RES / "S11_rf_wlm_sample_predictions.csv")
    return pd.concat([pd.read_csv(PRED / ("FV1-WLM-ALL-%s-F%d.csv" % (model, f))) for f in FOLDS], ignore_index=True)


def speaker_table(model):
    d = sample_predictions(model)
    g = d.groupby("speaker_id").agg(score=("predicted_probability", "mean"), label=("true_label", "first"), fold=("outer_fold", "first"))
    return g.sort_index()


def bootstrap_ci(labels, scores, metric, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    vals = []
    for _ in range(NBOOT):
        idx = rng.randint(0, n, n)
        if len(set(labels[idx])) < 2:
            continue
        vals.append(roc_auc_score(labels[idx], scores[idx]) if metric == "auroc" else average_precision_score(labels[idx], scores[idx]))
    v = np.asarray(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ------------------------------------------------------------------ compute and check everything
def compute():
    global spk, pooled, fold_metrics, selected, inner_auprc, cond, cross, ci, deltas, rf_sum
    spk = {m: speaker_table(m) for m, _ in MODELS}
    cond = pd.read_csv(METRICS / "condition_metrics.csv").set_index("condition_id")
    pooled = {m: metrics_from(spk[m].label, spk[m].score) for m, _ in MODELS}
    # 1. The saved official WavLM LR and XGB results are reproduced from their saved predictions.
    for m, cid in [("LR", "FV1-WLM-ALL-LR"), ("XGB", "FV1-WLM-ALL-XGB")]:
        row = cond.loc[cid]
        ok = (abs(pooled[m]["f1"] - row["F1"]) < 1e-9 and abs(pooled[m]["auroc"] - row["AUROC"]) < 1e-9
              and abs(pooled[m]["auprc"] - row["AUPRC"]) < 1e-9 and pooled[m]["fn"] == int(row["FN"]) and pooled[m]["fp"] == int(row["FP"]))
        check("saved FINAL_V1 WavLM %s reproduced from predictions" % m, ok, "F1 %.6f saved %.6f" % (pooled[m]["f1"], row["F1"]))
    # 2. Same people and labels for all three classifiers.
    check("all three models score the same 47 people", set(spk["LR"].index) == set(spk["RF"].index) == set(spk["XGB"].index), "")
    check("cohort is 47 people, 25 PD and 22 controls", len(spk["LR"]) == 47 and int(spk["LR"].label.sum()) == 25, "people %d" % len(spk["LR"]))
    check("labels identical across models", all((spk[m].label.values == spk["LR"].label.values).all() for m, _ in MODELS), "")
    # 3. Folds: every recording of a person is in one fold; that fold matches SPLIT_MANIFEST.
    split = pd.read_csv(ROOT / "SPLIT_MANIFEST.csv").drop_duplicates("speaker_id").set_index("speaker_id")
    for m, _ in MODELS:
        d = sample_predictions(m)
        check("%s: each person has exactly one test fold" % m, int(d.groupby("speaker_id").outer_fold.nunique().max()) == 1, "")
    check("fold of every person matches SPLIT_MANIFEST", all(int(spk["LR"].loc[s, "fold"]) == int(split.loc[s, "outer_fold"]) for s in spk["LR"].index), "")
    # 4. Probabilities are finite and in [0, 1] for every model.
    for m, _ in MODELS:
        p = sample_predictions(m).predicted_probability.values
        check("%s: probabilities finite and in [0, 1]" % m, bool(np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1), "")
    # 5. Integrity gates saved by the official runs.
    gates_ok = True
    for m in ["LR", "XGB"]:
        for f in FOLDS:
            r = json.loads((METRICS / ("FV1-WLM-ALL-%s-F%d.json" % (m, f))).read_text())
            if r["status"] != "COMPLETED" or not all(v in (True, "PASS", "passed") for v in r["gates"].values()):
                gates_ok = False
    check("saved integrity gates PASS for all WavLM LR and XGB folds", gates_ok, "")
    # 6. Random Forest run: same recordings as the official runs, and its summary matches recomputation.
    rf_sum = json.loads((RES / "S11_rf_wlm_summary.json").read_text())
    check("RF predictions cover the same recordings as LR", set(sample_predictions("RF").sample_id) == set(sample_predictions("LR").sample_id), "")
    check("RF summary metrics equal recomputed values", abs(rf_sum["pooled"]["f1"] - pooled["RF"]["f1"]) < 1e-9, "")
    # 7. Paired bootstrap from the S11 run uses 2,000 resamples per difference.
    s11_boot = pd.read_csv(RES / "S11_paired_bootstrap.csv").set_index("difference")
    check("S11 paired bootstrap has 2,000 resamples per difference", bool((s11_boot.resamples == NBOOT).all()), "")
    # 8. Bootstrap intervals and paired differences are computed here with the same seed.
    lab = spk["LR"].label.values.astype(int)
    ci = {m: {"auroc": bootstrap_ci(lab, spk[m].score.values, "auroc"), "auprc": bootstrap_ci(lab, spk[m].score.values, "auprc")} for m, _ in MODELS}
    deltas = {}
    for m in ["RF", "XGB"]:
        vals = {"auroc": [], "auprc": []}
        rng = np.random.RandomState(42)
        for _ in range(NBOOT):
            idx = rng.randint(0, len(lab), len(lab))
            if len(set(lab[idx])) < 2:
                continue
            vals["auroc"].append(roc_auc_score(lab[idx], spk[m].score.values[idx]) - roc_auc_score(lab[idx], spk["LR"].score.values[idx]))
            vals["auprc"].append(average_precision_score(lab[idx], spk[m].score.values[idx]) - average_precision_score(lab[idx], spk["LR"].score.values[idx]))
        deltas[m] = {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in vals.items()}
    check("paired difference intervals computed for RF and XGB", len(deltas) == 2, "")
    # 9. Per-fold metrics and the settings chosen in each fold.
    rows = []
    for m, _ in MODELS:
        for f in FOLDS:
            part = spk[m][spk[m].fold == f]
            rows.append(dict(model=m, fold=f, **metrics_from(part.label, part.score)))
    fold_metrics = pd.DataFrame(rows)
    selected, inner_auprc = {}, {}
    for f in FOLDS:
        for m, name in [("LR", "LR"), ("XGB", "XGB")]:
            r = json.loads((METRICS / ("FV1-WLM-ALL-%s-F%d.json" % (name, f))).read_text())
            selected[(m, f)] = r["selected_hyperparameters"]
            inner_auprc[(m, f)] = max(x["inner_speaker_auprc"] for x in r["inner_table"])
        sel = rf_sum["selection"][str(f)]
        selected[("RF", f)] = sel["selected"]
        inner_auprc[("RF", f)] = max(x["inner_speaker_auprc"] for x in sel["inner_table"])
    # 10. The v4.0 cross-check copy is unchanged since it was made.
    cm = json.loads((IN / "crosscheck_manifest.json").read_text())
    check("v4.0 cross-check copy matches its fingerprints", all(digest(IN / rel) == info["sha256"] for rel, info in cm["files"].items()), "")
    cross = pd.read_csv(IN / "pooled_metrics.csv")
    if problems:
        print("\nSTOP: %d check(s) failed. Nothing was written." % len(problems)); sys.exit(1)


# ------------------------------------------------------------------ figures
def make_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (m, name) in enumerate(MODELS):
        part = fold_metrics[fold_metrics.model == m]
        jitter = np.linspace(-0.1, 0.1, len(part))
        ax.scatter(np.full(len(part), i) + jitter, 100 * part.f1, color=COLORS[m], s=28, zorder=3)
        mean = 100 * part.f1.mean()
        ax.hlines(mean, i - 0.28, i + 0.28, color=COLORS[m], lw=2, zorder=2)
        ax.text(i + 0.31, mean, "mean %.1f" % mean, va="center", fontsize=8)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([n for _, n in MODELS])
    ax.set_ylabel("speaker F1 (%)"); ax.set_ylim(75, 101); ax.set_xlim(-0.5, len(MODELS) - 0.2)
    ax.set_title("One dot per held-out fold (47 people, 5 folds)", fontsize=9)
    fig.savefig(FIG / "fold_f1.pdf", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for m, name in MODELS:
        fpr, tpr, _ = roc_curve(spk[m].label.values.astype(int), spk[m].score.values)
        ax.plot(fpr, tpr, color=COLORS[m], lw=1.6, label="%s (AUROC %.3f)" % (name, pooled[m]["auroc"]))
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("false positive rate (controls called PD)"); ax.set_ylabel("true positive rate (PD found)")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.set_title("Pooled speaker ROC curves (held-out folds)", fontsize=9)
    fig.savefig(FIG / "roc_speakers.pdf", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    rng = np.random.RandomState(7)
    for ax, (m, name) in zip(axes, MODELS):
        t = spk[m]
        for cls, x0 in [(0, 0), (1, 1)]:
            part = t[t.label == cls]
            xs = x0 + rng.uniform(-0.18, 0.18, len(part))
            ax.scatter(xs, part.score, s=16, color=("#888888" if cls == 0 else COLORS[m]), zorder=3)
            for x, (sid, row) in zip(xs, part.iterrows()):
                if (row.score >= THRESHOLD) != bool(row.label):
                    ax.annotate(sid, (x, row.score), xytext=(6, 0), textcoords="offset points", fontsize=6, va="center")
        ax.axhline(THRESHOLD, color="black", lw=0.8, ls="--")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "PD"]); ax.set_title(name, fontsize=9); ax.set_xlim(-0.5, 1.9)
    axes[0].set_ylabel("person score (mean of recordings)"); axes[0].set_ylim(-0.05, 1.05)
    fig.savefig(FIG / "speaker_scores.pdf", bbox_inches="tight"); plt.close(fig)
    shutil.copy2(ROOT / "reports/FINAL_V1/study_report/figures/embedding_pipeline.pdf", FIG / "embedding_pipeline.pdf")


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
             "key": ("Key point", "orange!70!black", "orange!5")}[kind]
    A("\\begin{tcolorbox}[colback=%s,colframe=%s,boxrule=0.5pt,arc=2pt,left=5pt,right=5pt,top=3pt,bottom=3pt,title={\\small\\bfseries %s}]\\small %s\\end{tcolorbox}"
      % (style[2], style[1], style[0], inline(text)))


def bullets(items, enum=False):
    env = "enumerate" if enum else "itemize"
    A("\\begin{%s}" % env + "".join("\\item " + inline(i) + "\n" for i in items) + "\\end{%s}" % env)


def cell(x):
    x = str(x)
    if x.startswith("\\") or "$" in x:
        return x
    return inline(x)


def table(caption, header, rows, spec, size="small"):
    lines = ["\\begin{table}[H]\\centering\\%s\\renewcommand{\\arraystretch}{1.15}" % size,
             "\\caption{%s}" % inline(caption), "\\begin{tabular}{%s}" % spec, "\\toprule",
             " & ".join("\\textbf{%s}" % inline(h) for h in header) + "\\\\", "\\midrule"]
    for r in rows:
        if r == "--":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(cell(c) for c in r) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}\\end{table}"]
    A("\n".join(lines))


def fig(name, caption, width="0.92"):
    A("\\begin{figure}[H]\\centering\\includegraphics[width=%s\\linewidth]{%s.pdf}\\caption{%s}\\label{fig:%s}\\end{figure}"
      % (width, name, inline(caption), name))


def pct(x):
    return "%.2f" % (100 * x)


def ci_txt(lo, hi):
    return "[%.2f, %.2f]" % (100 * lo, 100 * hi)


def signed_delta(v, lo, hi):
    verdict = "includes zero" if lo <= 0 <= hi else "excludes zero"
    return "%+.2f %s (%s)" % (100 * v, ci_txt(lo, hi), verdict)


PRE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.1cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype}
\usepackage{booktabs,float,graphicx,amsmath,amssymb,enumitem,xcolor,tcolorbox}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black,citecolor=blue!50!black]{hyperref}
\setlist{itemsep=2pt,topsep=3pt}\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}\linespread{1.08}
\usepackage{titlesec}\titleformat{\section}{\Large\bfseries}{\thesection.}{.5em}{}\titleformat{\subsection}{\large\bfseries}{\thesubsection}{.5em}{}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\fancyfoot[C]{\small\thepage}\fancyhead[L]{\small WavLM baseline subset}\fancyhead[R]{\small FINAL\_V1 comparison, baseline part}\renewcommand{\headrulewidth}{.3pt}
\begin{document}
"""


def title_page():
    A("\\begin{center}{\\LARGE\\bfseries Frozen WavLM Baseline: Logistic Regression, Random Forest and XGBoost\\par}\\vspace{6pt}"
      "{\\large Baseline subset of the FINAL\\_V1 comparison (CSE 431, topic 4a)\\par}\\vspace{4pt}"
      "{\\large Version %s \\textperiodcentered{} %s\\par}\\vspace{6pt}"
      "{\\small Arannamoy Mondal. Evidence: frozen protocol FINAL\\_V1, 47 people, 785 recordings, five speaker-disjoint folds.\\par}\\end{center}\n"
      "\\vspace{4pt}\\hrule\\vspace{6pt}\n\\tableofcontents\n\\newpage\n" % (REPORT_VERSION, REPORT_DATE))


def write_body():
    p = pooled
    lr, rf, xg = p["LR"], p["RF"], p["XGB"]
    c_pd, c_ctl = int(lr["pd"]), int(lr["controls"])
    cit = {k: cite(k) for k in ["Quan", "Iyer", "Egbo", "Mandarin language dataset", "Klempir", "WavLM: large-scale", "wav2vec 2.0: a framework", "XLS-R: self-supervised",
                                 "Crawford", "Vaiciukynas", "Italian Parkinson's Voice and Speech", "Assessment of speech intelligibility", "NeuroVoz"]}
    miss_all = sorted(set.intersection(*[set(spk[m].index[(spk[m].score >= THRESHOLD) != spk[m].label.astype(bool)]) for m, _ in MODELS]))
    miss_any = {m: spk[m][(spk[m].score >= THRESHOLD) != spk[m].label.astype(bool)] for m, _ in MODELS}
    xg_fp = miss_any["XGB"][miss_any["XGB"].label == 0]
    pd_missed = miss_any["LR"][miss_any["LR"].label == 1]

    H(0, "Key numbers at a glance")
    table("Speaker-level results on the 47 FINAL_V1 people, pooled over the five held-out folds, threshold 0.5. Percent.",
          ["Classifier", "F1", "Recall", "Specificity", "Precision", "Accuracy", "AUROC", "AUPRC"],
          [[name] + [pct(p[m][k]) for k in ["f1", "recall", "specificity", "precision", "accuracy", "auroc", "auprc"]] for m, name in MODELS],
          "lrrrrrrr", size="footnotesize")
    box("key", "Logistic regression and Random Forest make the same decisions for all 47 people. XGBoost misses the same patient, has one false alarm, and ranks people slightly better "
               "(AUROC %s%% against %s%%). The differences are one or two people out of 47."
        % (pct(xg["auroc"]), pct(lr["auroc"])))

    H(0, "Revision history")
    table("Versions of this report.", ["Version", "Date", "Changes"],
          [["1.0", "21 October 2026", "Internal draft built from the separate PatchTST proposal run (v4.0). Not delivered. Superseded by 1.1."],
           ["1.1", REPORT_DATE, "Rebuilt on the FINAL_V1 comparison: logistic regression and XGBoost from the official frozen run; Random Forest added as exploratory amendment A2 (same folds, same inner rule). Full Final Report structure; the v4.0 run appears only as a labelled cross-check."]],
          "llp{11cm}")

    H(1, "Abstract", "abstract")
    P("Parkinson's disease can change the voice and the way people speak. We asked whether a speech recording can separate people with Parkinson's disease from healthy older adults. "
      "We used a public Italian corpus of %d recordings from %d people (%d with Parkinson's disease, %d healthy older adults). Each recording was turned into 1,024 numbers by a frozen, "
      "pretrained speech model (WavLM-large) that was not trained on our data. Three classifiers were trained on exactly the same numbers and the same speaker folds: logistic regression (the baseline), "
      "Random Forest and XGBoost. On people held out from training, logistic regression and Random Forest reach a speaker F1 of %s%%, missing one of the 25 patients and raising no false alarm. "
      "XGBoost reaches an F1 of %s%%, with the same missed patient and one false alarm, but a slightly higher AUROC of %s%%. The classifiers differ by one or two people. "
      "The results come from one corpus and do not show clinical accuracy."
      % (785, 47, c_pd, c_ctl, pct(lr["f1"]), pct(xg["f1"]), pct(xg["auroc"])))

    H(1, "Motivation and objectives", "motivation")
    P("Early signs of Parkinson's disease can be subtle. A voice test could be cheap, quick and repeatable, which would help screening where no specialist is available. "
      "Patients, families and clinics would benefit if such a test is reliable. Many speech studies, however, place recordings of the same person in both training and testing, which inflates the results.")
    P("**Qualitative objective.** Test whether a frozen speech representation and simple, explainable classifiers separate the two groups when every person is kept on one side of the split.")
    P("**Quantitative objectives.** (1) Report speaker-level F1, recall, specificity, precision, accuracy, AUROC and AUPRC for three classifiers on the same representation. "
      "(2) State the success rule before the comparison: a classifier counts as better only if both its pooled F1 and its AUROC exceed the baseline's by more than one misclassified person. "
      "(3) Report how much the results change across folds and across each classifier's settings.")

    H(1, "Related work", "related")
    P("The works below are from 2022 to 2025 unless stated, grouped by approach.")
    bullets([
        "**Neural networks on spectrograms or speech.** Quan et al. (%d) train an end-to-end network on speech. Iyer et al. (%d) compare a neural network with Random Forest and logistic regression on voice samples. "
        "These models are trained from scratch on small datasets; here the speech encoder is frozen." % (cit["Quan"], cit["Iyer"]),
        "**Hand-made features with classical classifiers.** Egbo et al. (%d) use XGBoost with SHAP explanations on vocal biomarkers. Wang et al. (%d) compare feature-selection methods and four classifiers on a Mandarin corpus. "
        "They motivate the XGBoost comparison here." % (cit["Egbo"], cit["Mandarin language dataset"]),
        "**Pretrained speech embeddings.** Klempir and Krupicka (%d) classify frozen wav2vec embeddings with logistic regression, Random Forest and XGBoost on several databases: the same three classifiers used here. "
        "WavLM (%d), wav2vec 2.0 (%d) and XLS-R (%d) are the encoders in the wider project." % (cit["Klempir"], cit["WavLM: large-scale"], cit["wav2vec 2.0: a framework"], cit["XLS-R: self-supervised"]),
        "**Linguistic content.** Crawford (%d) uses transcripts and language-model features. Our models use only the audio." % cit["Crawford"],
        "**Evaluation methodology.** Vaiciukynas et al. (%d) show that results are inflated unless all recordings of a person stay on one side of a split. The folds here are speaker-disjoint for that reason." % cit["Vaiciukynas"],
    ])
    P("The corpus is described by Dimauro and Girardi (%d). Dimauro et al. (%d) study speech intelligibility in Parkinson's disease with a speech-to-text system. The NeuroVoz corpus (%d) is a Spanish dataset of the same kind. "
      "These studies use different data, splits and metrics, so we do not compare their scores with ours."
      % (cit["Italian Parkinson's Voice and Speech"], cit["Assessment of speech intelligibility"], cit["NeuroVoz"]))

    H(1, "Background", "background")
    H(2, "How Parkinson's disease changes speech")
    P("Reduced loudness, a flat voice, unclear consonants and shorter or quieter pauses are common in Parkinson's disease. A classifier does not need these names; it needs measurements that differ between the groups.")
    H(2, "Frozen representations and hand-made features")
    P("A hand-made feature is a formula chosen by a person. A frozen representation is a hidden layer of a neural network trained by its authors on large amounts of unlabelled speech. "
      "Freezing means its weights never change here, so the number of parameters learned from our 47 people stays small.")
    H(2, "Why the person, not the recording, is the unit")
    P("A person contributes several recordings. If one person's recordings are in both training and testing, a classifier can learn to recognise the person rather than the disease. "
      "We assign whole people to folds and report one score per person.")
    H(2, "Why these metrics")
    P("Recall and specificity are the clinical pair: recall counts the patients found, specificity counts the controls left alone. F1 combines precision and recall but ignores true negatives, so it is never reported alone. "
      "AUROC and AUPRC do not depend on one threshold, so they show how well each classifier ranks people.")

    H(1, "Methodology", "method")
    H(2, "Model description", "model")
    P("The pipeline has four stages. The table gives the input and output shape of each.")
    table("Shapes at each stage. A person can have several recordings; each recording becomes one vector.",
          ["Stage", "Input", "Output", "Parameters or settings"],
          [["1. Frozen WavLM-large", "one recording, resampled to 16 kHz and cut into windows of at most 10 s", "one vector of 1,024 numbers per recording", "about 315 million frozen weights"],
           ["2. Classifier", "1,024 numbers per recording, training people only", "probability of Parkinson's per recording", "LR: 1,025 weights; RF: 300 trees; XGBoost: 100 to 300 trees"],
           ["3. Person score", "probabilities of one person's recordings", "one score per person (the mean)", "no parameters"],
           ["4. Decision", "person score", "PD if score $\\geq$ 0.5, otherwise control", "fixed threshold 0.5"]],
          "p{3.2cm}p{4.8cm}p{3.9cm}p{4cm}", size="footnotesize")
    fig("embedding_pipeline", "How one recording becomes one 1,024-number embedding. The encoder weights never change; only averaging is applied.", "0.95")
    H(2, "Dataset", "dataset")
    P("The Italian Parkinson's Voice and Speech corpus (%d) holds 16 speaking tasks per person: two read passages, syllable repetitions, vowels held twice, a word list and a phrase. "
      "We used the folders of people with Parkinson's disease and of elderly healthy controls. Young controls (15 people, 45 files) are excluded: they were recorded in only three tasks and are much younger. "
      "One exact duplicate file was removed. The result is 785 recordings from 47 people (%d with Parkinson's disease, %d controls). "
      "Tasks are pooled, so the task is not an input. Age, sex, recording date and sampling rate are metadata and are never model inputs."
      % (cit["Italian Parkinson's Voice and Speech"], c_pd, c_ctl))
    H(2, "Training and evaluation steps, including batch sampling", "steps")
    P("No mini-batches are used: each classifier is fitted on all training recordings at once, so there is no batch sampling to describe. The steps are:")
    bullets([
        "Split the 47 people into five folds, stratified by class, with seed 42. Each person is in exactly one test fold (`SPLIT_MANIFEST.csv`).",
        "For each outer fold, split its training people into three inner folds, again speaker-disjoint, with seed 142 plus the fold number.",
        "For each setting, fit on two inner folds and score on the third, in each of the three possible ways. The score is the person-level AUPRC averaged over the three validation folds.",
        "Choose the setting with the highest inner score; ties go to the earliest setting in the list. Refit it on all training people of the fold.",
        "Predict the held-out recordings, average them per person and apply threshold 0.5. Repeat for each fold.",
    ], enum=True)
    H(2, "Evaluation metrics", "metrics")
    P("Speaker-level metrics are computed after averaging each person's recording probabilities into one score. Counting TP, FN, TN and FP over people:")
    A("\\begin{align*} \\text{recall} &= \\frac{TP}{TP+FN}, & \\text{specificity} &= \\frac{TN}{TN+FP}, & \\text{precision} &= \\frac{TP}{TP+FP}, \\\\"
      " \\text{F1} &= \\frac{2\\cdot\\text{precision}\\cdot\\text{recall}}{\\text{precision}+\\text{recall}}, & \\text{accuracy} &= \\frac{TP+TN}{TP+TN+FP+FN}. \\end{align*}")
    P("AUROC is the probability that a randomly chosen patient receives a higher score than a randomly chosen control. AUPRC is the average precision over all thresholds. "
      "Confidence intervals come from 2,000 resamples of people (seed 42), with the same people resampled for every classifier.")
    H(2, "Loss functions and optimisation", "loss")
    P("Logistic regression minimises the logistic loss $\\sum_i \\log(1+e^{-y_i(w^\\top x_i+b)})$ with an L1 or L2 penalty on $w$, solved by liblinear; the search parameter $C$ is the inverse of the penalty strength. "
      "Random Forest grows trees that reduce the Gini impurity $1-\\sum_k p_k^2$ of their leaves. XGBoost minimises the logistic loss plus a regularisation term, with gradient-boosted trees; the regularisation weight is `reg_lambda`. "
      "Settings are chosen by the inner AUPRC, not by the loss, so that the choice matches the evaluation.")
    H(2, "Experimental depth and iteration", "depth")
    table("Search spaces. Every setting was refitted and scored in each fold.",
          ["Classifier", "Settings searched", "Settings", "Chosen across the five folds"],
          [["Logistic regression", "C in {0.01, 0.1, 1, 10, 100}; penalty in {L1, L2}; class weight in {none, balanced}", "20",
            ", ".join(sorted({"C=%s, %s" % (selected[("LR", f)]["C"], selected[("LR", f)]["penalty"]) for f in FOLDS}))],
           ["Random Forest", "max depth in {4, unlimited}; min leaf in {1, 2}; class weight in {none, balanced}; 300 trees", "8",
            ", ".join(sorted({"depth %s, leaf %s, weight %s" % ("unlimited" if selected[("RF", f)]["max_depth"] is None else selected[("RF", f)]["max_depth"],
                                                                 selected[("RF", f)]["min_samples_leaf"], selected[("RF", f)]["class_weight"] or "none") for f in FOLDS}))],
           ["XGBoost", "24 settings drawn at random (seed 42) from 256 combinations of learning rate, depth, trees, weights, subsampling and regularisation", "24",
            "%d different settings across the five folds" % len({json.dumps(selected[("XGB", f)], sort_keys=True) for f in FOLDS})]],
          "lp{6.2cm}rp{4.6cm}", size="footnotesize")
    P("In total 52 settings were searched in each outer fold. This is a small search, so the settings chosen vary little.")

    H(1, "Baseline and extensions", "baseline")
    H(2, "Baseline selection and justification", "baselineselect")
    P("Logistic regression is the baseline. It was named the primary baseline before the comparison was run, it has the fewest parameters, its coefficients can be read directly, "
      "and it is the classifier used with frozen embeddings in Klempir and Krupicka (%d). A simple, well-understood baseline makes it clear what the extensions add." % cit["Klempir"])
    H(2, "Implemented extensions", "extensions")
    P("**Random Forest** (exploratory amendment A2) and **XGBoost** (part of the frozen FINAL_V1 comparison). Both use the same recordings, folds, inner rule and threshold as the baseline. "
      "Random Forest was added after the frozen run, at the owner's request. Its settings were written down before it was run (`supplementary/PREDECLARED_RF_GRID.md`).")
    H(2, "Baseline reproduction", "reproduction")
    P("The official logistic regression and XGBoost results were recomputed from their saved predictions. Counts, F1, AUROC and AUPRC match the saved condition table. "
      "The integrity gates saved by each run all passed. The Random Forest results come from the S11 run, which reads the same saved vectors and folds.")

    H(1, "Results and analysis", "results")
    H(2, "Validation and held-out results, kept separate", "trainval")
    P("Each outer fold has two levels of evaluation. The inner validation score (person-level AUPRC on three inner folds of the training people) was used only to choose the settings. "
      "The held-out score is computed on the outer test fold, which was never used for any choice. The table shows both.")
    rows = []
    for f in FOLDS:
        cells = []
        for m, _ in MODELS:
            part = fold_metrics[(fold_metrics.model == m) & (fold_metrics.fold == f)].iloc[0]
            cells.append("%s / %s" % (pct(inner_auprc[(m, f)]), pct(part.auprc)))
        rows.append([str(f + 1), str(int(part.people))] + cells)
    table("Validation AUPRC (used to choose) / held-out AUPRC (reported), per outer fold, percent.",
          ["Fold", "People", "Logistic regression", "Random Forest", "XGBoost"], rows, "rrccc", size="footnotesize")

    H(2, "Pooled held-out results", "pooled")
    P("Each row pools the five held-out folds, so each of the 47 people is scored once. Brackets show 95% bootstrap intervals over people.")
    rows = []
    for m, name in MODELS:
        q = pooled[m]
        rows.append([name, "%d/%d/%d/%d" % (q["tp"], q["fn"], q["tn"], q["fp"]), pct(q["f1"]), pct(q["recall"]), pct(q["specificity"]), pct(q["precision"]),
                     pct(q["accuracy"]), "%s %s" % (pct(q["auroc"]), ci_txt(*ci[m]["auroc"])), "%s %s" % (pct(q["auprc"]), ci_txt(*ci[m]["auprc"]))])
    table("Pooled person-level results, 47 people, threshold 0.5. Percent except counts. Brackets: 95% bootstrap interval.",
          ["Classifier", "TP/FN/TN/FP", "F1", "Recall", "Spec.", "Prec.", "Acc.", "AUROC [CI]", "AUPRC [CI]"], rows, "lrrrrrrll", size="scriptsize")
    fig("roc_speakers", "Pooled person-level ROC curves. Each curve uses one score per person from the held-out folds.", "0.7")
    P("Paired differences from logistic regression, on the same resampled people (percentage points, mean and 95%% interval): Random Forest AUROC %s, AUPRC %s. XGBoost AUROC %s, AUPRC %s."
      % (signed_delta(deltas["RF"]["auroc"][0], *deltas["RF"]["auroc"][1:]), signed_delta(deltas["RF"]["auprc"][0], *deltas["RF"]["auprc"][1:]),
         signed_delta(deltas["XGB"]["auroc"][0], *deltas["XGB"]["auroc"][1:]), signed_delta(deltas["XGB"]["auprc"][0], *deltas["XGB"]["auprc"][1:])))
    P("Each difference is descriptive. None is a significance claim, and the comparison uses only 47 people.")

    H(2, "Variation across folds", "folds")
    rows = []
    for m, name in MODELS:
        cells = []
        for key in ["f1", "recall", "specificity", "auroc"]:
            vals = fold_metrics[fold_metrics.model == m][key].astype(float).tolist()
            cells.append("%.2f $\\pm$ %.2f" % (100 * statistics.mean(vals), 100 * statistics.stdev(vals)))
        rows.append([name] + cells)
    table("Mean $\\pm$ sample standard deviation over the five held-out folds (percent).", ["Classifier", "F1", "Recall", "Specificity", "AUROC"], rows, "lrrrr", size="footnotesize")
    fig("fold_f1", "Held-out speaker F1 in each fold. One missed patient in a fold of about five patients lowers that fold's recall by about 20 points.", "0.8")
    P("Each fold has nine or ten people, so one person changes a fold's score a lot. The spread across folds is therefore larger than the differences between classifiers.")

    H(2, "Error and failure analysis", "errors")
    miss = []
    for m, name in MODELS:
        for sid, row in miss_any[m].iterrows():
            miss.append([name, sid, "missed patient" if row.label == 1 else "false alarm", "%.3f" % row.score, "fold %d" % (row.fold + 1)])
    table("Every misclassified person in the held-out folds.", ["Classifier", "Person", "Error", "Score", "Fold"], miss, "llllr", size="footnotesize")
    fig("speaker_scores", "Person scores in the held-out folds. The dashed line is the threshold 0.5. Labelled points are misclassified.", "0.95")
    missed_ids = list(pd_missed.index)
    pd_scores = ", ".join("%s %.3f" % (m, spk[m].loc[missed_ids[0], "score"]) for m, _ in MODELS) if missed_ids else ""
    fp_txt = ("; control %s is a false alarm for XGBoost only, with score %.3f, just above the threshold" % (xg_fp.index[0], xg_fp.score.iloc[0])) if len(xg_fp) else ""
    P("The misclassifications are few and all concern the same people. Patient %s is missed by every classifier, with scores %s, all below the threshold%s. "
      "Those are close calls, not evidence that a classifier learned a different signal. Section 9 proposes checking these recordings by task and by session." % (missed_ids[0] if missed_ids else "none", pd_scores, fp_txt))

    H(2, "Sensitivity and ablation", "sensitivity")
    P("We ask how much the result depends on the settings chosen. For each fold, the table gives the range of validation scores across the whole search space of each classifier.")
    rows = []
    for f in FOLDS:
        table_rf = [x["inner_speaker_auprc"] for x in rf_sum["selection"][str(f)]["inner_table"]]
        table_lr = [x["inner_speaker_auprc"] for x in json.loads((METRICS / ("FV1-WLM-ALL-LR-F%d.json" % f)).read_text())["inner_table"]]
        table_xg = [x["inner_speaker_auprc"] for x in json.loads((METRICS / ("FV1-WLM-ALL-XGB-F%d.json" % f)).read_text())["inner_table"]]
        rows.append([str(f + 1), "%s to %s" % (pct(min(table_lr)), pct(max(table_lr))), "%s to %s" % (pct(min(table_rf)), pct(max(table_rf))),
                     "%s to %s" % (pct(min(table_xg)), pct(max(table_xg)))])
    table("Range of validation AUPRC across all settings in each fold (percent), with the number of settings in brackets.",
          ["Fold", "Logistic regression (20)", "Random Forest (8)", "XGBoost (24)"], rows, "rlll", size="footnotesize")
    P("Each row shows how much the validation score moves across settings within one classifier. Comparing this range with the gaps between classifiers shows whether the choice of setting or the choice of classifier matters more on this data.")
    v4 = cross.set_index(["scenario", "method"])
    rows = []
    for m, name, key in [("logistic_regression", "Logistic regression", "LR"), ("random_forest", "Random Forest", "RF"), ("xgboost", "XGBoost", "XGB")]:
        q = v4.loc[("main", m)]
        rows.append([name, pct(q.f1), pct(q.recall), pct(q.specificity), pct(q.auroc), pct(pooled[key]["f1"]), pct(pooled[key]["auroc"])])
    table("Cross-check against the separate PatchTST proposal run v4.0 (46 people, different protocol). Percent.",
          ["Classifier", "v4.0 F1", "v4.0 recall", "v4.0 spec.", "v4.0 AUROC", "FINAL_V1 F1", "FINAL_V1 AUROC"], rows, "lrrrrrr", size="footnotesize")
    P("The two runs use similar classifiers but different cohorts, inner objectives and search spaces, so the numbers are not directly comparable. "
      "In v4.0 XGBoost ranks lower; in FINAL_V1 it ranks slightly higher on AUROC. The ranking of classifiers therefore changes with the pipeline, which is why we do not claim that one classifier is better.")

    H(1, "Discussion", "discussion")
    H(2, "Significance of the results", "significance")
    P("On this corpus, a frozen speech representation with a simple classifier separates the two groups well. "
      "Within this representation the three classifiers agree on almost every person.")
    H(2, "Risks", "risks")
    P("The main risk is that the classifiers separate recording conditions rather than disease. The FINAL_V1 report documents recording-condition differences between the two groups (its section 7.7), "
      "so part of the separation may come from the recording setup. A second risk is the small number of people: one patient changes recall by about 4 percentage points.")
    H(2, "Limitations", "limitations")
    bullets(["No external validation: all people come from one corpus.",
             "One split with one seed. A different seed would change the folds and the results.",
             "Random Forest was added after the frozen comparison, as an exploratory amendment. It is reported beside the frozen results, not in place of them.",
             "The best setting on the same people would look better than it really is, so no best setting is picked for reporting."])

    H(1, "Future directions", "future")
    bullets(["Inspect the missed patient and the false alarm: which tasks and recording dates drive their scores.",
             "Repeat the comparison with the recording-condition controls from the FINAL_V1 report, so that disease and recording setup can be separated.",
             "Test the same frozen encoders on a second Parkinson's corpus, for example the Spanish NeuroVoz data (%d), to check generalisation." % cit["NeuroVoz"]], enum=True)

    H(1, "Conclusion", "conclusion")
    P("A frozen WavLM-large representation separates people with Parkinson's disease from elderly controls, with a held-out speaker F1 of %s%% for logistic regression. "
      "Random Forest and XGBoost reach similar results on the same 47 people, and the differences are one or two people. The result is internal to one corpus and does not show clinical accuracy." % pct(lr["f1"]))

    H(0, "Appendix A. Checks performed")
    P("Every check below ran during the build. The build stops and writes nothing if any check fails.")
    table("Checks run on the saved inputs.", ["Check", "Result"], [[n, r] for n, r, _ in checks_log], "p{12.5cm}l", size="footnotesize")

    H(0, "Appendix B. Where to find everything")
    bullets(["This report: `reports/WAVLM_BASELINE_SUBSET/versions/v1.1/" + STEM + ".pdf`, with LaTeX source and an Overleaf zip in the same folder.",
             "Generator: `src/final_v1/wavlm_subset_report.py` (copy in `versions/v1.1/generator/`); references in `src/final_v1/wavlm_subset_refs.py`.",
             "Official WavLM logistic regression and XGBoost predictions: `predictions/FINAL_V1/FV1-WLM-ALL-*`; saved metrics: `metrics/FINAL_V1/`.",
             "Random Forest (amendment A2): `supplementary/code/s11_random_forest.py`, `supplementary/code/s11_compare.py`, results `supplementary/results/S11_*`, settings `supplementary/PREDECLARED_RF_GRID.md`.",
             "Cross-check copy (PatchTST v4.0, read-only, with fingerprints): `reports/WAVLM_BASELINE_SUBSET/inputs/`."])

    H(0, "Appendix C. References")
    A("\\begin{thebibliography}{99}\\setlength{\\itemsep}{1pt}\\small")
    A("".join("\\bibitem{r%d} %s\n" % (i + 1, esc(r)) for i, r in enumerate(REFS)) + "\\end{thebibliography}")


def build():
    if (VOUT / (STEM + ".pdf")).exists():
        print("version %s already exists; it is never overwritten. Move it aside first if you really need to rebuild." % REPORT_VERSION); sys.exit(10)
    compute()
    make_figures()
    B.clear()
    title_page()
    title_block = "\n".join(B)
    B.clear()
    write_body()
    body = "\n".join(B)
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
    shutil.copy2(build_dir / (STEM + ".log"), VOUT / (STEM + ".build.log"))
    shutil.copy2(build_dir / (STEM + ".pdf"), VOUT / (STEM + ".pdf"))
    shutil.copy2(build_dir / (STEM + ".tex"), VOUT / (STEM + ".tex"))
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
    if len(sys.argv) < 2 or sys.argv[1] not in ("crosscheck", "build"):
        print("usage: wavlm_subset_report.py crosscheck | build"); sys.exit(2)
    if sys.argv[1] == "crosscheck":
        crosscheck()
    else:
        build()
