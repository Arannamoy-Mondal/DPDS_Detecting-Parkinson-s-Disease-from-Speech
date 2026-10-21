"""Frozen WavLM baseline subset report (LaTeX -> PDF), document 1.0.

The report covers only the baseline part of the separate WavLM proposal experiment v4.0: one frozen WavLM-large
embedding per recording, classified by logistic regression (the predeclared baseline), Random Forest and XGBoost.
It is a subset of that experiment, not a new one, and it does not replace the FINAL_V1 project report.

Two commands:
    snapshot  one-time, read-only copy of the v4.0 inputs into reports/WAVLM_BASELINE_SUBSET/inputs/ with SHA-256 fingerprints
    build     reads only that snapshot, recomputes every number, stops if any check fails, writes versions/v1.0/

Run from the workspace root:
    .venv/bin/python src/final_v1/wavlm_subset_report.py snapshot
    .venv/bin/python src/final_v1/wavlm_subset_report.py build
"""
import hashlib, json, re, shutil, statistics, subprocess, sys, zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/WAVLM_BASELINE_SUBSET"
IN = OUT / "inputs"
FIG = OUT / "figures"
REPORT_VERSION = "1.0"
REPORT_DATE = "21 October 2026"
VOUT = OUT / "versions" / ("v" + REPORT_VERSION)
STEM = "WAVLM_BASELINE_SUBSET_REPORT_v" + REPORT_VERSION
SOURCE_RUN = Path("/root/Projects/PatchTST/artifacts/runs/20261014T191654Z_v4.0_eb438e63b942")
SOURCE_LABEL = "v4.0 WavLM proposal experiment (run 20261014T191654Z_v4.0_eb438e63b942)"
METHODS = [("logistic_regression", "Logistic regression"), ("random_forest", "Random Forest"), ("xgboost", "XGBoost")]
SCENARIOS = [("main", "Main cohort"), ("without_sheet_flag", "Without flagged people"), ("native16k", "Native 16 kHz only")]
FOLDS = range(5)
THRESHOLD = 0.5
TOL = 1e-9
COLORS = {"logistic_regression": "#2B5FAF", "random_forest": "#3C8D5A", "xgboost": "#B5651D"}
problems = []
checks_log = []
B = []


# ------------------------------------------------------------------ snapshot
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    """Copy only the files this report reads. Run once; re-running overwrites the copy with identical content."""
    if not SOURCE_RUN.exists():
        print("STOP: source run not found:", SOURCE_RUN); sys.exit(1)
    manifest = {}

    def copy(rel_src, rel_dst):
        src = SOURCE_RUN / rel_src
        dst = IN / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[str(rel_dst)] = {"source": str(rel_src), "sha256": digest(dst)}

    copy("pooled_metrics.csv", "pooled_metrics.csv")
    copy("fine_tuning_decisions.json", "fine_tuning_decisions.json")
    copy("config.yaml", "config.yaml")
    for folder in ["main", "native16k"]:
        for method, _ in METHODS:
            copy("%s/%s/speaker_predictions.csv" % (folder, method), "%s/%s/speaker_predictions.csv" % (folder, method))
            for fold in FOLDS:
                copy("%s/%s/fold_%d/recording_predictions.csv" % (folder, method, fold),
                     "%s/%s/fold_%d/recording_predictions.csv" % (folder, method, fold))
                if folder == "main":
                    copy("main/%s/fold_%d/selection.json" % (method, fold), "main/%s/fold_%d/selection.json" % (method, fold))
    # Split files are large; keep only the speaker identifiers and labels needed for the checks.
    for fold in FOLDS:
        for part in ["outer_train", "outer_test"]:
            src = SOURCE_RUN / ("main/splits/fold_%d/%s.csv" % (fold, part))
            dst = IN / ("splits/fold_%d_%s_speakers.csv" % (fold, part))
            dst.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(src, usecols=["speaker_id", "label"]).drop_duplicates()
            df.to_csv(dst, index=False)
            manifest[str(dst.relative_to(IN))] = {"source": "main/splits/fold_%d/%s.csv (columns speaker_id, label, unique rows)" % (fold, part),
                                                 "sha256": digest(dst), "source_sha256": digest(src)}
    (IN / "snapshot_manifest.json").write_text(json.dumps({"source_run": str(SOURCE_RUN), "files": manifest}, indent=2), encoding="utf-8")
    print("snapshot written:", len(manifest), "files ->", IN)


# ------------------------------------------------------------------ metrics
def metrics_from(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= THRESHOLD).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    p_curve, r_curve, _ = precision_recall_curve(labels, scores)
    return {"people": len(labels), "pd": int(labels.sum()), "controls": int(len(labels) - labels.sum()),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp, "recall": recall, "specificity": specificity, "precision": precision,
            "f1": f1, "accuracy": (tp + tn) / len(labels), "auroc": roc_auc_score(labels, scores), "auprc": auc(r_curve, p_curve),
            "average_precision": average_precision_score(labels, scores)}


def data_dir(scenario):
    """'without_sheet_flag' reuses the main-cohort predictions; it only drops the flagged people when scoring."""
    return "native16k" if scenario == "native16k" else "main"


def speakers(method, scenario):
    df = pd.read_csv(IN / data_dir(scenario) / method / "speaker_predictions.csv")
    if scenario == "without_sheet_flag":
        df = df[df.sheet_flag.astype(str) == "False"]
    return df.reset_index(drop=True)


def recordings(method, scenario):
    frames = [pd.read_csv(IN / data_dir(scenario) / method / ("fold_%d" % f) / "recording_predictions.csv") for f in FOLDS]
    return pd.concat(frames, ignore_index=True)


def check(name, ok, detail=""):
    checks_log.append((name, "PASS" if ok else "FAIL", detail))
    if not ok:
        problems.append(name)
        print("FAIL  " + name + "  " + detail)


# ------------------------------------------------------------------ build: compute and check
def compute():
    global pooled, folds_df, selected, reference, main_spk, fine
    reference = pd.read_csv(IN / "pooled_metrics.csv")
    pooled = {}
    for method, _ in METHODS:
        for scenario, _ in SCENARIOS:
            spk = speakers(method, scenario)
            m = metrics_from(spk.label, spk.probability)
            pooled[(method, scenario)] = m
            row = reference[(reference.method == method) & (reference.scenario == scenario)]
            if len(row) != 1:
                check("saved row exists %s %s" % (method, scenario), False, "found %d" % len(row)); continue
            row = row.iloc[0]
            counts_ok = (m["tp"], m["fn"], m["tn"], m["fp"]) == (int(row.tp), int(row.fn), int(row.tn), int(row.fp))
            check("confusion counts match saved: %s, %s" % (method, scenario), counts_ok)
            worst = max(abs(m[k] - row[k]) for k in ["f1", "recall", "specificity", "accuracy", "auroc", "auprc"])
            check("metrics match saved: %s, %s" % (method, scenario), worst < 1e-6, "largest difference %g" % worst)
    for method, _ in METHODS:
        for scenario in ["main", "native16k"]:
            rec = recordings(method, scenario).groupby("speaker_id").probability.mean()
            spk = speakers(method, scenario).set_index("speaker_id").probability
            diff = float(np.max(np.abs(rec.reindex(spk.index).to_numpy() - spk.to_numpy())))
            tolerance = 1e-6 if method == "xgboost" else TOL  # XGBoost stores float32 probabilities
            check("speaker score equals mean of recordings: %s, %s" % (method, scenario), diff < tolerance, "max difference %g" % diff)
    main_spk = speakers("logistic_regression", "main")
    all_test = []
    for fold in FOLDS:
        test_ids = set(pd.read_csv(IN / ("splits/fold_%d_outer_test_speakers.csv" % fold)).speaker_id)
        train_ids = set(pd.read_csv(IN / ("splits/fold_%d_outer_train_speakers.csv" % fold)).speaker_id)
        all_test += list(test_ids)
        check("fold %d: no speaker in both training and test" % (fold + 1), len(test_ids & train_ids) == 0, "overlap %d" % len(test_ids & train_ids))
        in_fold = set(main_spk[main_spk.fold == fold].speaker_id)
        check("fold %d: saved predictions match the split" % (fold + 1), in_fold == test_ids, "difference %d" % len(in_fold ^ test_ids))
    check("every main person is in exactly one test fold", len(all_test) == len(set(all_test)) == len(main_spk),
          "tested %d, unique %d, people %d" % (len(all_test), len(set(all_test)), len(main_spk)))
    fine = json.loads((IN / "fine_tuning_decisions.json").read_text())
    triggered = sum(1 for d in fine if d["triggered"])
    check("conditional fine-tuning never triggered", triggered == 0, "triggered %d" % triggered)

    rows = []
    for method, _ in METHODS:
        for scenario, _ in SCENARIOS:
            spk = speakers(method, scenario)
            for fold in FOLDS:
                part = spk[spk.fold == fold]
                rows.append(dict(method=method, scenario=scenario, fold=fold, **metrics_from(part.label, part.probability)))
    folds_df = pd.DataFrame(rows)
    selected = {}
    for method, _ in METHODS:
        for fold in FOLDS:
            sel = json.loads((IN / "main" / method / ("fold_%d" % fold) / "selection.json").read_text())
            selected[(method, fold)] = (sel["selected"]["parameters"], sel["selected"]["mean_inner_speaker_auroc"])


def fold_stat(method, scenario, key):
    values = folds_df[(folds_df.method == method) & (folds_df.scenario == scenario)][key].astype(float).tolist()
    return statistics.mean(values), statistics.stdev(values)


# ------------------------------------------------------------------ figures
def make_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, (method, label) in enumerate(METHODS):
        part = folds_df[(folds_df.method == method) & (folds_df.scenario == "main")]
        jitter = np.linspace(-0.12, 0.12, len(part))
        ax.scatter(np.full(len(part), i) + jitter, 100 * part.f1, color=COLORS[method], s=28, zorder=3)
        mean = 100 * part.f1.mean()
        ax.hlines(mean, i - 0.28, i + 0.28, color=COLORS[method], lw=2, zorder=2)
        ax.text(i + 0.31, mean, "mean %.1f" % mean, va="center", fontsize=8)
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([label for _, label in METHODS])
    ax.set_ylabel("speaker F1 (%)"); ax.set_ylim(75, 101); ax.set_xlim(-0.5, len(METHODS) - 0.3)
    ax.set_title("Each dot is one outer test fold (main cohort, 46 people)", fontsize=9)
    fig.savefig(FIG / "fold_f1.pdf", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True)
    rng = np.random.RandomState(7)
    for ax, (method, label) in zip(axes, METHODS):
        spk = speakers(method, "main")
        for cls, x0 in [(0, 0), (1, 1)]:
            part = spk[spk.label == cls]
            xs = x0 + rng.uniform(-0.18, 0.18, len(part))
            ax.scatter(xs, part.probability, s=16, color=("#888888" if cls == 0 else COLORS[method]), zorder=3)
            for x, (_, row) in zip(xs, part.iterrows()):
                if (row.probability >= THRESHOLD) != bool(row.label):
                    ax.annotate(row.speaker_id, (x, row.probability), xytext=(6, 0), textcoords="offset points", fontsize=6, va="center")
        ax.axhline(THRESHOLD, color="black", lw=0.8, ls="--")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "PD"]); ax.set_title(label, fontsize=9); ax.set_xlim(-0.5, 1.9)
    axes[0].set_ylabel("speaker score (mean of recordings)"); axes[0].set_ylim(-0.05, 1.05)
    fig.savefig(FIG / "speaker_scores.pdf", bbox_inches="tight"); plt.close(fig)
    # The embedding schematic is the one drawn for FINAL_V1 report 1.3; it uses the same WavLM-large pooling.
    shutil.copy2(ROOT / "reports/FINAL_V1/study_report/figures/embedding_pipeline.pdf", FIG / "embedding_pipeline.pdf")


# ------------------------------------------------------------------ latex helpers
def esc(text):
    return (str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#").replace("$", r"\$"))


def pct(x):
    return "%.2f" % (100 * x)


def inline(s):
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"`(.+?)`", lambda m: "\\texttt{" + m.group(1) + "}", s)
    return s


def A(x):
    B.append(x)


def P(text):
    A(inline(text) + "\n")


def H(level, title, key=None):
    if level == 0:  # unnumbered part (key numbers, glossary), still listed in the contents
        A("\\section*{%s}\\addcontentsline{toc}{section}{%s}" % (inline(title), inline(title)))
        return
    A("\\%s{%s}%s" % ({1: "section", 2: "subsection", 3: "subsubsection"}[level], inline(title), ("\\label{sec:%s}" % key) if key else ""))


def box(kind, text):
    style = {"plain": ("In plain words", "green!40!black", "green!3"), "read": ("What this tells us", "blue!50!black", "blue!3"),
             "key": ("Key point", "orange!70!black", "orange!5")}[kind]
    A("\\begin{tcolorbox}[colback=%s,colframe=%s,boxrule=0.5pt,arc=2pt,left=5pt,right=5pt,top=3pt,bottom=3pt,title={\\small\\bfseries %s}]\\small %s\\end{tcolorbox}"
      % (style[2], style[1], style[0], inline(text)))


def bullets(items):
    A("\\begin{itemize}" + "".join("\\item " + inline(i) + "\n" for i in items) + "\\end{itemize}")


def table(caption, header, rows, spec, size="small"):
    lines = ["\\begin{table}[H]\\centering\\%s\\renewcommand{\\arraystretch}{1.15}" % size,
             "\\caption{%s}" % inline(caption), "\\begin{tabular}{%s}" % spec, "\\toprule",
             " & ".join("\\textbf{%s}" % inline(h) for h in header) + "\\\\", "\\midrule"]
    for r in rows:
        if r == "--":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(str(c) if str(c).startswith("\\") else inline(c) for c in r) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}\\end{table}"]
    A("\n".join(lines))


def fig(name, caption, width="0.92"):
    A("\\begin{figure}[H]\\centering\\includegraphics[width=%s\\linewidth]{%s.pdf}\\caption{%s}\\label{fig:%s}\\end{figure}"
      % (width, name, inline(caption), name))


PRE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.1cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype}
\usepackage{booktabs,float,graphicx,amsmath,amssymb,enumitem,xcolor,tcolorbox}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black,citecolor=blue!50!black]{hyperref}
\setlist{itemsep=2pt,topsep=3pt}\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}\linespread{1.08}
\usepackage{titlesec}\titleformat{\section}{\Large\bfseries}{\thesection.}{.5em}{}\titleformat{\subsection}{\large\bfseries}{\thesubsection}{.5em}{}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\fancyfoot[C]{\small\thepage}\fancyhead[L]{\small Frozen WavLM baseline subset}\fancyhead[R]{\small Subset of experiment v4.0}\renewcommand{\headrulewidth}{.3pt}
\begin{document}
"""


def title_page():
    A("\\begin{center}{\\LARGE\\bfseries Frozen WavLM Baseline Subset\\par}\\vspace{6pt}"
      "{\\Large Logistic regression, Random Forest and XGBoost on the same embeddings\\par}\\vspace{4pt}"
      "{\\large Subset of the WavLM proposal experiment v4.0 \\textperiodcentered{} Version %s \\textperiodcentered{} %s\\par}\\vspace{6pt}"
      "{\\small Arannamoy Mondal (report author). Evidence: %s, saved predictions only, no model retrained.\\par}\\end{center}\n"
      "\\vspace{4pt}\\hrule\\vspace{6pt}\n\\tableofcontents\n\\newpage\n" % (REPORT_VERSION, REPORT_DATE, esc(SOURCE_LABEL)))


def write_body():
    p = pooled
    lr, rf, xg = p[("logistic_regression", "main")], p[("random_forest", "main")], p[("xgboost", "main")]
    nat = p[("logistic_regression", "native16k")]

    H(0, "Key numbers at a glance")
    table("Main cohort (46 people): pooled speaker results at threshold 0.5, percent.",
          ["Classifier", "F1", "Recall", "Specificity", "Accuracy", "AUROC"],
          [[name, pct(p[(m, "main")]["f1"]), pct(p[(m, "main")]["recall"]), pct(p[(m, "main")]["specificity"]),
            pct(p[(m, "main")]["accuracy"]), pct(p[(m, "main")]["auroc"])] for m, name in METHODS],
          "lrrrrr")
    P("Logistic regression (the predeclared baseline) and Random Forest reach the same pooled result. XGBoost is lower. "
      "The native 16 kHz cohort is smaller (39 people) and all three classifiers are perfect there, so it does not separately confirm any of them.")

    H(0, "Revision history")
    table("Versions of this report.", ["Version", "Date", "Changes"],
          [["1.0", REPORT_DATE, "First edition. Subset of experiment v4.0: frozen WavLM-large embeddings with logistic regression, Random Forest and XGBoost; conditional fine-tuning excluded because its gate never triggered."]],
          "llp{11cm}")

    H(1, "Summary", "summary")
    P("Each recording is turned into one 1,024-value vector by a frozen WavLM-large speech model. The model is never trained here. "
      "Three classifiers are trained on the same vectors, the same speaker folds and the same speech tasks. The question is whether the "
      "classifier choice changes the result. On the main 46 people, logistic regression and Random Forest both reach a speaker F1 of %s%%, "
      "with one missed patient and no false alarms. XGBoost reaches %s%%, with the same missed patient and one false alarm. These are differences "
      "of one or two people out of 46, so they are descriptive, not evidence that one classifier is better. Nothing here is an external validation."
      % (pct(lr["f1"]), pct(xg["f1"])))

    H(1, "Scope and what is left out", "scope")
    P("The full v4.0 experiment has three parts: the frozen classifier comparison, a conditional partial fine-tuning branch, and a native-rate check. "
      "This report covers the frozen classifier comparison and its native-rate repeat. The fine-tuning branch is not included. Its gate was never "
      "triggered: in all five outer folds the training-side inner AUROC was above the 0.80 trigger, so no fine-tuned model exists.")
    box("plain", "This report is a reading of saved predictions. It adds no new model, no new fold and no new threshold. Every number is recomputed "
                 "from the saved files and checked against the values the experiment saved (section 'Checks performed').")

    H(1, "Motivation and question", "motivation")
    P("Parkinson's disease changes how people speak. The aim of the experiment is to tell people with Parkinson's disease apart from elderly healthy "
      "controls using speech recordings only. The baseline question is simple: if the speech representation is fixed and frozen, does the choice "
      "of classifier change the result? Logistic regression is the baseline chosen before any result was seen. Random Forest and XGBoost are the two "
      "alternatives proposed in the same plan.")

    H(1, "The input: a frozen WavLM embedding", "input")
    P("WavLM-large is a speech model trained by its authors on large amounts of unlabelled audio. Here it is used only as a fixed feature extractor. "
      "Each recording is resampled to 16 kHz and cut into windows of at most ten seconds. The last hidden layer is averaged over time inside each window, "
      "and the window vectors are averaged, giving one 1,024-value vector per recording. The model revision is pinned, and its weights are never changed.")
    fig("embedding_pipeline", "How one recording becomes one 1,024-number embedding. The encoder weights never change; only averaging is applied.", "0.95")
    box("plain", "Think of the frozen model as a listener that writes down 1,024 numbers for each recording. The classifier only learns which "
                 "numbers go with Parkinson's disease and which with controls. The listener does not know the diagnosis.")

    H(1, "Cohort", "cohort")
    cohort_rows = []
    for scenario, name in SCENARIOS:
        m = p[("logistic_regression", scenario)]
        ids = set(speakers("logistic_regression", scenario).speaker_id)
        files = int(recordings("logistic_regression", scenario).speaker_id.isin(ids).sum())
        cohort_rows.append([name, str(m["people"]), str(m["pd"]), str(m["controls"]), str(files)])
    table("Who is scored in each scenario. Young controls are excluded from the whole study.",
          ["Scenario", "People", "PD", "Elderly controls", "Recordings"], cohort_rows, "lrrrr")
    P("Each person is one unit. A person's score is the mean of that person's recording scores. A person is called Parkinson's when the score is at least 0.5. "
      "Recordings from one person stay in one fold, so no person is both trained on and tested on. The native 16 kHz scenario keeps only recordings "
      "originally sampled at 16 kHz, and all three classifiers are refitted on that set. Vitantonio D has no passage recordings and is outside the shared "
      "46-person cohort.")

    H(1, "Classifiers and how their settings were chosen", "classifiers")
    table("Settings searched. Each classifier is trained inside the outer training data only.",
          ["Classifier", "Settings searched (frozen grid)"],
          [["Logistic regression", "C in {0.01, 0.1, 1, 10}; liblinear solver; features standardised using training data only"],
           ["Random Forest", "300 trees; maximum depth in {4, unlimited}; minimum leaf size in {1, 2}; maximum features = sqrt"],
           ["XGBoost", "200 trees; maximum depth in {1, 2, 3}; learning rate in {0.03, 0.1}; subsample 1.0; column sample 0.5; lambda 1.0"]],
          "lp{11.5cm}")
    P("For every outer fold, the settings are chosen by a three-fold search that uses only the training people. The score used for choosing is the "
      "mean inner speaker AUROC. The chosen settings are then refitted on all training people of that fold. Each training person has equal total weight "
      "and each class has equal total weight, so the larger group does not dominate.")

    def pshort(method, params):
        if method == "logistic_regression":
            return "C = %s" % params["C"]
        if method == "random_forest":
            d = "unlimited" if params["max_depth"] is None else params["max_depth"]
            return "depth %s, min leaf %s" % (d, params["min_samples_leaf"])
        return "depth %s, rate %s" % (params["max_depth"], params["learning_rate"])
    setting_rows = []
    for fold in FOLDS:
        cells = []
        for method, _ in METHODS:
            params, inner = selected[(method, fold)]
            cells.append("%s (inner %.3f)" % (pshort(method, params), inner))
        setting_rows.append([str(fold + 1)] + cells)
    table("Settings chosen in each outer fold, with the inner-search AUROC that selected them.",
          ["Fold", "Logistic regression", "Random Forest", "XGBoost"], setting_rows, "lp{4.6cm}p{4.6cm}p{4.6cm}", size="footnotesize")

    H(1, "Results", "results")
    H(2, "Pooled speaker results", "pooled")
    P("Each row pools the five outer test folds, so every person is scored exactly once. Counts are true positives (TP), false negatives (FN), true negatives (TN) "
      "and false positives (FP). All values are percentages except the counts.")
    rows = []
    for scenario, sname in SCENARIOS:
        for i, (method, mname) in enumerate(METHODS):
            m = p[(method, scenario)]
            rows.append([sname if i == 0 else "", mname, str(m["people"]), "%d/%d/%d/%d" % (m["tp"], m["fn"], m["tn"], m["fp"]),
                         pct(m["f1"]), pct(m["recall"]), pct(m["specificity"]), pct(m["precision"]), pct(m["accuracy"]), pct(m["auroc"]), pct(m["auprc"])])
        if scenario != "native16k":
            rows.append("--")
    table("Pooled speaker results at threshold 0.5 (percent; AUPRC is trapezoidal).",
          ["Scenario", "Classifier", "People", "TP/FN/TN/FP", "F1", "Recall", "Spec.", "Prec.", "Acc.", "AUROC", "AUPRC"],
          rows, "llrlrrrrrrr", size="scriptsize")
    box("key", "The clinical pair is recall and specificity. Recall says how many patients were found; specificity says how many controls were left alone. "
               "Logistic regression and Random Forest have recall 96% and specificity 100%. XGBoost has recall 96% and specificity 95.24%.")

    H(2, "Mean and spread across the five folds", "folds")
    P("Each fold has 9 or 10 test people and about five patients. One missed patient lowers a fold's recall by about 20 points, so the spread across folds is large. "
      "The table gives the mean and sample standard deviation over the five folds of the main cohort.")
    spread_rows = []
    for method, mname in METHODS:
        cells = []
        for key in ["f1", "recall", "specificity", "precision", "accuracy", "auroc"]:
            mean, sd = fold_stat(method, "main", key)
            cells.append("%.2f $\\pm$ %.2f" % (100 * mean, 100 * sd))
        spread_rows.append([mname] + cells)
    table("Mean $\\pm$ sample standard deviation over the five main-cohort folds (percent).",
          ["Classifier", "F1", "Recall", "Specificity", "Precision", "Accuracy", "AUROC"], spread_rows, "lrrrrrr", size="footnotesize")
    fig("fold_f1", "Speaker F1 in each of the five outer test folds, main cohort. The line marks the mean.", "0.85")
    P("AUROC is 1.00 in folds 1 to 4 for all three classifiers. In fold 5 each classifier ranks one patient below one control, which gives 0.95 "
      "(19 of 20 correctly ordered pairs). That is why the AUROC columns match: each classifier makes the same single ranking error.")

    H(2, "Errors by fold", "errors")
    per_fold = []
    for fold in FOLDS:
        cells = []
        people = None
        for method, _ in METHODS:
            m = folds_df[(folds_df.method == method) & (folds_df.scenario == "main") & (folds_df.fold == fold)].iloc[0]
            people = int(m.people)
            cells.append("%s (%d/%d)" % (pct(m.f1), m.fn, m.fp))
        per_fold.append([str(fold + 1), str(people)] + cells)
    table("F1 per fold, with missed patients and false alarms in brackets (FN/FP).",
          ["Fold", "People", "Logistic regression", "Random Forest", "XGBoost"], per_fold, "rrlll")

    H(2, "Which people were misclassified", "misclassified")
    missed = []
    for method, mname in METHODS:
        spk = speakers(method, "main")
        for _, row in spk.iterrows():
            if (row.probability >= THRESHOLD) != bool(row.label):
                missed.append([mname, row.speaker_id, "missed patient" if row.label == 1 else "false alarm", "%.3f" % row.probability, "fold %d" % (row.fold + 1)])
    if not missed:
        missed.append(["--", "none", "--", "--", "--"])
    table("Every misclassified person in the main cohort.", ["Classifier", "Speaker", "Error", "Score", "Fold"], missed, "llllr")
    fig("speaker_scores", "Speaker scores in the main cohort, pooled over the five outer folds. The dashed line is the 0.5 threshold. Labelled points are misclassified people.", "0.95")
    P("The same patient is missed by all three classifiers. One control is a false alarm for XGBoost only.")

    H(1, "Answer to the question", "answer")
    bullets(["Random Forest ties logistic regression on pooled F1, AUROC and AUPRC in the main cohort.",
             "XGBoost scores lower than logistic regression on F1, specificity, precision, accuracy, AUROC and AUPRC in the main cohort.",
             "In the native 16 kHz cohort all three classifiers are perfect. That cohort is smaller and does not independently confirm any of them."])
    P("The three classifiers share the same people, folds and embeddings, so the differences come from the classifiers themselves. "
      "They rest on one or two misclassified people out of 46, so none of them shows a classifier is better.")

    H(1, "Limitations", "limitations")
    bullets(["No external validation. Every person comes from the same corpus.",
             "Disease is not separated from recording conditions. Earlier checks in this project found that recording dates and levels differ between the two groups, "
             "so the classifiers may partly use acquisition differences. This remains unresolved.",
             "The native 16 kHz scenario is retrained on fewer people. Its perfect scores are not independent confirmation.",
             "Pooled figures are not best-fold figures. Choosing the best of several settings on the same pooled scores would be optimistic, so no setting is picked for reporting."])

    H(1, "Checks performed", "checks")
    P("Every check below ran during the build. The build stops and writes nothing if any check fails.")
    table("Checks run on the saved inputs.", ["Check", "Result", "Detail"],
          [[n, r, d] for n, r, d in checks_log], "p{9.5cm}ll", size="footnotesize")

    H(1, "Where to find everything", "where")
    bullets(["This report: `reports/WAVLM_BASELINE_SUBSET/versions/v1.0/" + STEM + ".pdf`.",
             "Editable LaTeX and Overleaf zip: the same folder, `" + STEM + ".tex` and `" + STEM + "_latex.zip`.",
             "Generator used for this version: `src/final_v1/wavlm_subset_report.py`, copy in `versions/v1.0/generator/`.",
             "Saved inputs (read-only copies with SHA-256 fingerprints): `reports/WAVLM_BASELINE_SUBSET/inputs/`.",
             "Original run: `artifacts/runs/20261014T191654Z_v4.0_eb438e63b942/` in the separate PatchTST project."])

    H(0, "Glossary")
    table("Terms used in this report.", ["Term", "Meaning in this report"], [
        ["Speaker (person)", "One person. All of their recordings are scored together by their mean score."],
        ["Threshold 0.5", "A person is called Parkinson's when their score is at least 0.5. The threshold is fixed and never tuned on test people."],
        ["TP / FN / TN / FP", "True positive, false negative, true negative, false positive."],
        ["Recall", "Share of patients who were correctly found: TP / (TP + FN)."],
        ["Specificity", "Share of controls who were correctly left alone: TN / (TN + FP)."],
        ["Precision", "Share of people called Parkinson's who really are: TP / (TP + FP)."],
        ["F1", "Balance of precision and recall: 2 x precision x recall / (precision + recall)."],
        ["AUROC", "Chance that a random patient gets a higher score than a random control (1.00 is perfect)."],
        ["AUPRC", "Area under the precision-recall curve, computed with the trapezoidal rule."],
        ["Inner search", "The three-fold search that chooses settings, using training people only."],
        ["Outer fold", "One of five speaker-disjoint splits. Each person is tested in exactly one."],
    ], "p{4.2cm}p{11.8cm}", size="small")


def build():
    if REPORT_VERSION == "1.0" and (VOUT / (STEM + ".pdf")).exists() and "--force" not in sys.argv:
        print("version 1.0 already exists; delete versions/v1.0 first to rebuild"); sys.exit(10)
    compute()
    if problems:
        print("\nSTOP: %d check(s) failed. Nothing was written." % len(problems)); sys.exit(1)
    make_figures()
    write_body()
    body = "\n".join(B)
    build_dir = VOUT / "_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    saved = list(B)
    B.clear()
    title_page()
    title_block = "\n".join(B)
    B.clear(); B.extend(saved)
    (build_dir / (STEM + ".tex")).write_text(PRE + title_block + body + "\n\\end{document}\n", encoding="utf-8")
    for fn in FIG.glob("*.pdf"):
        shutil.copy2(fn, build_dir / fn.name)
    for _ in range(3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", STEM + ".tex"], cwd=build_dir, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); sys.exit(8)
    VOUT.mkdir(parents=True, exist_ok=True)
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
    shutil.rmtree(build_dir)
    pages = subprocess.run(["pdfinfo", str(VOUT / (STEM + ".pdf"))], capture_output=True, text=True).stdout
    page_count = [l.split()[-1] for l in pages.splitlines() if l.startswith("Pages")][0]
    print("wrote", VOUT / (STEM + ".pdf"), "pages", page_count, "checks passed", sum(1 for c in checks_log if c[1] == "PASS"))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("snapshot", "build"):
        print("usage: wavlm_subset_report.py snapshot | build")
        sys.exit(2)
    if sys.argv[1] == "snapshot":
        snapshot()
    else:
        build()
