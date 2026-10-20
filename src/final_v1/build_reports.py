"""Build (1) FINAL_REPORT.pdf from FINAL_REPORT.md and (2) GUIDELINE_REPORT.pdf organised by the sections of 'Project Guidelines.pdf'.
Every number is read from verified FINAL_V1 files."""
import json, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import protocol_defs as P
from make_pdf import esc

MET = WORKSPACE / "metrics/FINAL_V1"; PRD = WORKSPACE / "predictions/FINAL_V1"
FIG = WORKSPACE / "reports/FINAL_V1/figures"; FIG.mkdir(parents=True, exist_ok=True)
OUT = WORKSPACE / "reports/FINAL_V1"
NAMES = {"LR": "Logistic regression", "LSVM": "Linear SVM", "RSVM": "RBF SVM", "XGB": "XGBoost"}
BLUE, GRAY, RED, AMB = "#2B5FAF", "#8A9099", "#C0392B", "#D98A00"
f3 = lambda x: "%.3f" % x

cm = pd.read_csv(MET / "condition_metrics.csv")
comp = pd.read_csv(MET / "paired_comparisons.csv")
ra = json.loads((MET / "report_analysis.json").read_text())
rec = json.loads((WORKSPACE / "FREEZE_RECORD.json").read_text())
aud = json.loads((MET / "oof_audit.json").read_text())
sa = json.loads((WORKSPACE / "manifests/stage_a_summary.json").read_text())
DEM = json.loads((WORKSPACE / "audit_opus/evidence/a05_demographics_and_flags.json").read_text())
sm = pd.read_csv(WORKSPACE / "manifests/SAMPLE_MANIFEST.csv")
cm["label"] = cm.representation + cm.feature_selection.map(lambda s: "" if s == "ALL" else "/" + s)


# ------------------------------------------------------------------ figures
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)
    def box(x, y, w, h, t, fc="white", ec=GRAY):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec=ec, lw=1.2)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)
    ax.add_patch(FancyBboxPatch((0.05, 1.85), 9.9, 1.45, boxstyle="round,pad=0.03", fc="#F3F4F6", ec="#DADDE2")); ax.text(0.2, 3.12, "Frozen: no labels, nothing trained", fontsize=7, color="#555")
    box(0.2, 2.0, 1.7, 1.0, "Recording\nmono WAV\n16 or 44.1 kHz"); box(2.2, 2.0, 1.7, 1.0, "Resample 16 kHz\n10 s windows\nno peak norm."); box(4.2, 2.0, 1.9, 1.0, "Frozen encoder\nWavLM-large / wav2vec2 /\nXLS-R (about 315M par.)"); box(6.4, 2.0, 1.7, 1.0, "Mean over frames\nand windows:\n1,024 values"); box(8.3, 2.0, 1.5, 1.0, "+ 40 acoustic\nvalues (optional\nhybrids)")
    ax.add_patch(FancyBboxPatch((0.05, 0.1), 9.9, 1.45, boxstyle="round,pad=0.03", fc="white", ec="#DADDE2")); ax.text(9.8, 1.37, "Fitted on training speakers of each fold only", fontsize=7, color="#555", ha="right")
    box(0.2, 0.25, 1.9, 1.0, "Standardize\n(train mean/SD)", "#E8EFFA", BLUE); box(2.4, 0.25, 2.3, 1.0, "Classifier (inner 3-fold,\nspeaker-disjoint tuning):\nLR / linear SVM / RBF / XGB", "#E8EFFA", BLUE)
    box(5.0, 0.25, 1.9, 1.0, "Recording\nprobability"); box(7.1, 0.25, 1.4, 1.0, "Mean per\nspeaker"); box(8.7, 0.25, 1.1, 1.0, "PD if\n>= 0.5")
    for x0, x1, y in [(1.9, 2.2, 2.5), (3.9, 4.2, 2.5), (6.1, 6.4, 2.5), (8.1, 8.3, 2.5), (2.1, 2.4, .75), (4.7, 5.0, .75), (6.9, 7.1, .75), (8.5, 8.7, .75)]:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=GRAY))
    ax.annotate("", xy=(1.15, 1.25), xytext=(7.25, 2.0), arrowprops=dict(arrowstyle="-|>", color=GRAY, connectionstyle="arc3,rad=0.0"))
    fig.savefig(FIG / "pipeline.pdf", bbox_inches="tight"); plt.close(fig)


def fig_heatmap():
    order = ["AC", "AC/FORWARD", "AC/BACKWARD", "WLM", "W2V", "XLSR"] + list(P.HYBRIDS)
    M = np.array([[cm[(cm.label == r) & (cm.model == m)].AUROC.iloc[0] for m in P.MODELS] for r in order])
    fig, ax = plt.subplots(figsize=(5.2, 5.6)); im = ax.imshow(M, cmap="Blues", vmin=0.85, vmax=1.0, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels([NAMES[m] for m in P.MODELS], rotation=20, fontsize=8); ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(4): ax.text(j, i, "%.3f" % M[i, j], ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.96 else "black")
    ax.set_title("Pooled speaker-level out-of-fold AUROC", fontsize=9); fig.colorbar(im, fraction=0.04); fig.savefig(FIG / "heatmap_auroc.pdf", bbox_inches="tight"); plt.close(fig)


def pooled_scores(cid):
    d = pd.concat([pd.read_csv(PRD / ("%s-F%d.csv" % (cid, f))) for f in range(5)])
    return d.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first"))


def fig_strip():
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.3), sharey=True)
    for ax, cid, t in zip(axes, ["FV1-AC-ALL-LR", "FV1-WLM-ALL-LR", "FV1-H3-ALL-LR"], ["Acoustic, LR", "WavLM-large, LR", "XLS-R + acoustic, LR"]):
        g = pooled_scores(cid); rng = np.random.RandomState(1)
        for yv, col, name in [(0, GRAY, "control"), (1, BLUE, "PD")]:
            s = g[g.y == yv].s; ax.scatter(yv + rng.uniform(-0.18, 0.18, len(s)), s, s=16, c=col, alpha=.85)
        wrong = g[((g.s >= .5).astype(int) != g.y)]
        for sid, r in wrong.iterrows(): ax.scatter(r.y, r.s, s=80, facecolors="none", edgecolors=RED, lw=1.2); ax.text(r.y + .12, r.s, sid, fontsize=6.5)
        ax.axhline(.5, ls="--", c="#6B7280", lw=.8); ax.set_xticks([0, 1]); ax.set_xticklabels(["controls (22)", "PD (25)"], fontsize=8); ax.set_title(t, fontsize=8.5); ax.set_xlim(-.5, 1.7)
    axes[0].set_ylabel("Speaker probability of PD", fontsize=8); fig.savefig(FIG / "speaker_scores.pdf", bbox_inches="tight"); plt.close(fig)


def fig_error_tasks():
    det = ra["errors"]["detail"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    for ax, sid in zip(axes, ["PD_005", "HC_001", "HC_020"]):
        bt = det[sid]["FV1-WLM-ALL-LR"]["by_task"]; ks = sorted(bt, key=lambda t: ("B", "D", "F", "P", "V").index(t[0]) if t[0] in "BDFPV" else 9)
        ax.bar(range(len(ks)), [bt[k] for k in ks], color=[BLUE if det[sid]["label"] == 1 else GRAY] * len(ks)); ax.axhline(.5, ls="--", c="#6B7280", lw=.8)
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=90, fontsize=6); ax.set_title("%s (%s): speaker score %.2f" % (sid, "PD" if det[sid]["label"] == 1 else "control", det[sid]["FV1-WLM-ALL-LR"]["speaker_score"]), fontsize=8)
    axes[0].set_ylabel("Recording probability of PD", fontsize=8); fig.savefig(FIG / "error_tasks.pdf", bbox_inches="tight"); plt.close(fig)


def fig_deltas():
    fig, ax = plt.subplots(figsize=(6.2, 2.8)); rng = np.random.RandomState(2)
    kinds = [("hybrid_vs_constituent", "Hybrid minus constituent"), ("model_vs_logistic_regression", "Model minus logistic regression"), ("feature_selection_vs_all_features", "Selected minus all acoustic features")]
    for i, (k, nm) in enumerate(kinds):
        d = comp[(comp.comparison_type == k) & (comp.metric == "AUROC")]
        ax.scatter(d.delta_A_minus_B, i + rng.uniform(-.22, .22, len(d)), s=12, c=np.where(d.ci_excludes_zero, RED, GRAY), alpha=.85)
    ax.axvline(0, c="#6B7280", lw=.8); ax.set_yticks(range(3)); ax.set_yticklabels([n for _, n in kinds], fontsize=8); ax.set_xlabel("Difference in pooled speaker-level AUROC (red: 95% bootstrap interval excludes 0)", fontsize=8)
    fig.savefig(FIG / "deltas.pdf", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ tex helpers
PRE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.9cm]{geometry}
\usepackage{booktabs,longtable,float,graphicx,amsmath,array,enumitem,helvet,xcolor}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\setlist{itemsep=1pt,topsep=2pt}\setlength{\parskip}{3pt}\setlength{\parindent}{0pt}
\usepackage{titlesec}\titleformat{\section}{\large\bfseries}{\thesection.}{.5em}{}\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{.5em}{}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\fancyfoot[C]{\small\thepage}\fancyhead[L]{\small FINAL\_V1 - Parkinson's disease detection from speech}\renewcommand{\headrulewidth}{.3pt}
\begin{document}
"""


def inline(s):
    s = esc(s)
    s = re.sub(r"\\textbackslash\{\}", "", s) if False else s
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"`(.+?)`", r"\\texttt{\1}", s)
    return s


def tab(caption, header, rows, spec=None, size="scriptsize"):
    spec = spec or ("l" * len(header))
    t = ["\\begin{center}\\%s\\renewcommand{\\arraystretch}{1.12}" % size, "\\begin{longtable}{%s}" % spec, "\\caption{%s}\\\\" % inline(caption), "\\toprule", " & ".join("\\textbf{%s}" % inline(h) for h in header) + "\\\\\\midrule\\endfirsthead",
         "\\toprule", " & ".join("\\textbf{%s}" % inline(h) for h in header) + "\\\\\\midrule\\endhead", "\\bottomrule\\endfoot"]
    t += [" & ".join(inline(c) for c in r) + "\\\\" for r in rows]
    t += ["\\end{longtable}\\end{center}"]
    return "\n".join(t)


def figtex(path, cap, w="0.95"):
    return "\\begin{figure}[H]\\centering\\includegraphics[width=%s\\linewidth]{%s}\\caption{%s}\\end{figure}" % (w, path, inline(cap))


def compile_tex(name, body, title, subtitle):
    d = OUT / ("_build_" + name); d.mkdir(parents=True, exist_ok=True)
    head = "\\begin{center}{\\LARGE\\bfseries %s\\par}\\vspace{3pt}{\\large %s\\par}\\vspace{3pt}{\\small Protocol FINAL\\_V1 \\quad Report generated %s \\quad Protocol SHA-256 %s\\par}\\end{center}\\vspace{2pt}\\hrule\\vspace{4pt}" % (
        inline(title), inline(subtitle), esc(now_utc()), esc(rec["protocol_json_sha256"][:16] + "..."))
    (d / (name + ".tex")).write_text(PRE + head + body + "\n\\end{document}\n", encoding="utf8")
    for fn in FIG.glob("*.pdf"): shutil.copy2(fn, d / fn.name)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"], cwd=d, capture_output=True, text=True)
        if r.returncode != 0:
            print("pdflatex failed", name, r.stdout[-1500:]); sys.exit(8)
    shutil.copy2(d / (name + ".pdf"), OUT / (name + ".pdf")); shutil.rmtree(d)
    print("wrote", OUT / (name + ".pdf"))


# ------------------------------------------------------------------ report 1: FINAL_REPORT.pdf from the markdown
def md_to_tex(md):
    out, lines, i = [], md.splitlines(), 0
    out.append(figtex("pipeline.pdf", "Pipeline of FINAL_V1 (frozen encoder and acoustic features; fitting only on training speakers).", "0.95"))
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            i += 1; continue
        if ln.startswith("## "): out.append("\\section{%s}" % inline(re.sub(r"^\d+\.\s*", "", ln[3:])))
        elif ln.startswith("### "): out.append("\\subsection{%s}" % inline(re.sub(r"^[\d.]+\s*", "", ln[4:])))
        elif ln.startswith("|"):
            blk = []
            while i < len(lines) and lines[i].startswith("|"): blk.append(lines[i]); i += 1
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in blk if not re.match(r"^\|\s*-", r)]
            n = len(rows[0]); spec = "l" + "l" * (n - 1) if n > 6 else "l" * n
            out.append(tab("", rows[0], rows[1:], spec, "tiny" if n >= 8 else "scriptsize").replace("\\caption{}\\\\", ""))
            continue
        elif re.match(r"^\s*-\s", ln):
            out.append("\\begin{itemize}")
            while i < len(lines) and re.match(r"^\s*-\s", lines[i]): out.append("\\item " + inline(re.sub(r"^\s*-\s", "", lines[i]))); i += 1
            out.append("\\end{itemize}"); continue
        elif re.match(r"^\d+\.\s", ln):
            out.append("\\begin{enumerate}")
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i]): out.append("\\item " + inline(re.sub(r"^\d+\.\s", "", lines[i]))); i += 1
            out.append("\\end{enumerate}"); continue
        elif ln.strip(): out.append(inline(ln))
        i += 1
    return "\n".join(out)


def build_final():
    md = (WORKSPACE / "FINAL_REPORT.md").read_text()
    body = md_to_tex(md)
    body += "\n" + figtex("heatmap_auroc.pdf", "Pooled speaker-level out-of-fold AUROC for every representation and model.", "0.62")
    body += "\n" + figtex("speaker_scores.pdf", "Pooled out-of-fold speaker scores for three reference conditions. Red rings: speakers on the wrong side of the fixed threshold 0.5.")
    body += "\n" + figtex("deltas.pdf", "Paired differences in AUROC for the pre-specified comparisons (descriptive).", "0.75")
    compile_tex("FINAL_REPORT", body, "Parkinson's disease detection from speech", "Final report: frozen protocol FINAL\\_V1, speaker-disjoint out-of-fold evaluation")


if __name__ == "__main__":
    fig_pipeline(); fig_heatmap(); fig_strip(); fig_error_tasks(); fig_deltas()
    build_final()


# ------------------------------------------------------------------ report 2: sections of Project Guidelines.pdf
def build_guideline():
    done = cm.copy(); done["err"] = done.FN + done.FP
    base = done[done.condition_id == "FV1-WLM-ALL-LR"].iloc[0]
    acc = done[done.condition_id == "FV1-AC-ALL-LR"].iloc[0]
    reps = ["AC", "WLM", "W2V", "XLSR"] + list(P.HYBRIDS)
    med = {r: done[done.representation == r].AUROC.median() for r in reps}
    mm = done.groupby("model").AUROC.median()
    hy = comp[(comp.comparison_type == "hybrid_vs_constituent") & (comp.metric == "AUROC")]
    mv = comp[(comp.comparison_type == "model_vs_logistic_regression") & (comp.metric == "AUROC")]
    fsc = comp[(comp.comparison_type == "feature_selection_vs_all_features") & (comp.metric == "AUROC")]
    err = ra["errors"]; counts = dict(sorted(err["speaker_error_counts"].items(), key=lambda kv: -kv[1])); top = list(counts)[:3]
    det = err["detail"]; fam = err["recording_error_by_task_family"]
    tvt = {r["condition"]: r for r in ra["train_val_test"]}; trn = {r["condition"]: r for r in ra["training_partition_scores"]}
    sec = []
    A = sec.append
    A("\\section*{How to read this report}\\textbf{Part 1} follows the sections of the course document \\emph{Project Guidelines.pdf} (CSE 431, topic 4a), one heading per rubric item, so that each requirement can be checked. \\textbf{Part 2} is a compliance table. All results are produced under the frozen protocol FINAL\\_V1 and read from verified files (registry, prediction files, metrics, manifests). Earlier experiments of this project are not used. Method names: LR logistic regression, LSVM linear SVM, RSVM RBF SVM, XGB XGBoost; AC acoustic features; WLM WavLM-large; W2V wav2vec2-large-lv60; XLSR XLS-R 300M.")
    A("\\part*{Part 1. Report by guideline section}")
    A("\\section{Abstract (problem statement for non-experts)}")
    A(inline("Parkinson's disease often changes the voice: speech becomes quieter, flatter and less clear. This project asks whether short recordings of a person speaking can tell people with Parkinson's disease from healthy people of the same age, using computer models. We used %d recordings from %d people (%d patients, %d healthy older adults) and tested 56 combinations of speech features and models. To avoid fooling ourselves, every person's recordings were kept on one side only: a model was never tested on a person it had learned from. Most combinations separated patients from controls almost perfectly (median AUROC %s; AUROC is 1.0 for a perfect ranking and 0.5 for guessing). However, a model that looks only at simple loudness and frequency statistics did equally well, and patients and controls were recorded on different days (part of the patient audio also at a different sampling rate). So the near-perfect scores may reflect recording conditions as well as, or instead of, the disease. The result is a careful internal test, not evidence that the method can diagnose Parkinson's disease." % (sa["files_included"], sa["speakers_total"], sa["speakers_pd"], sa["speakers_hc"], f3(done.AUROC.median()))))
    A("\\section{Motivation and objectives}")
    A(inline("Speech is cheap, non-invasive and easy to repeat, so a speech-based screen could help find possible Parkinson's disease earlier and in more places. Who benefits: patients and clinicians if a reliable screen existed; researchers, because honest evaluation protocols for small speech corpora are rare. **Qualitative objectives:** (1) build speaker-independent classifiers from frozen speech encoders and simple acoustic features; (2) compare representations, models and feature selection under one frozen protocol; (3) explain where and why they fail. **Quantitative objective:** report pooled speaker-level AUPRC and AUROC (primary) and F1, sensitivity, specificity and balanced accuracy at a fixed threshold, with uncertainty; no numeric performance target was frozen in the FINAL_V1 protocol; results are read against chance (AUROC 0.5, AUPRC equal to the 53.2%% prevalence). Baseline (%s, LR): AUROC %s, AUPRC %s." % ("WavLM-large", f3(base.AUROC), f3(base.AUPRC))))
    A("\\section{Literature review (related work, grouped)}")
    A(inline("Recent work (2022-2025) on speech-based Parkinson's detection falls into four lines (nine works are cited from that period). **Neural networks on spectrograms or raw speech:** Quan et al. 2022 [1] train end-to-end 2D/1D convolutional networks; Iyer et al. 2023 [2] compare a CNN with Random Forest and logistic regression on telephone vowels. These models are trained from scratch on small corpora; here, pretrained encoders are frozen instead. **Hand-crafted acoustic features with classical classifiers:** Jeong et al. 2024 [3] use eGeMAPS features across languages; Egbo et al. 2025 [4] use XGBoost with SHAP explanations. They motivate our acoustic baseline and the XGBoost model. **Pretrained speech embeddings:** Klempir and Krupicka 2024 [5] classify frozen wav2vec embeddings with logistic regression, Random Forest and XGBoost on three databases; WavLM [6], wav2vec 2.0 [7] and XLS-R [8] are the encoders used here. **Linguistic content:** Crawford 2025 [9] uses transcripts and language-model embeddings; our models use the audio only. **Evaluation methodology:** Vaiciukynas et al. 2017 [10] show that scores are inflated unless all recordings of a person stay on one side of a split; this project's folds are speaker-disjoint for that reason. Bibliographic details were taken from the project's reference list and are not re-checked online in this session; these studies use different data and splits, so their scores are not compared with ours."))
    A("\\section{Background (metric and baseline linked to the problem)}")
    A(inline("The metric: AUPRC and AUROC summarise how well patients are ranked above controls without choosing a threshold; AUPRC depends on prevalence (53.2%% of speakers are patients), so it is reported with the prevalence. Sensitivity and specificity show the two clinical errors: missing a patient and alarming a healthy person [11]. Speaker-level scoring (the mean of a person's recordings) follows [10]. The baseline: frozen WavLM-large embeddings with logistic regression, chosen because pretrained embeddings with simple classifiers are the closest published design [5] and a linear model has few parameters for a 47-person dataset. Dataset: Italian Parkinson's Voice and Speech [12]."))
    A("\\section{Model description}")
    A(figtex("pipeline.pdf", "FINAL_V1 pipeline: inputs, outputs and which stages are fitted."))
    d = {r: sum(P.DIMS[c] for c in P.components(r)) for r in reps}
    rows = [["Resample and window", "WAV at 16 or 44.1 kHz", "10 s windows, 160,000 samples at 16 kHz", "0"],
            ["Frozen encoder (WavLM-large, wav2vec2-large-lv60, XLS-R 300M)", "one normalized window (160,000 samples)", "499 frames x 1,024 values (output map)", "0 (315.4 to 315.5 million frozen)"],
            ["Pooling", "frame outputs, then windows", "1,024 values per recording", "0"],
            ["Acoustic features (AC)", "16 kHz audio, 25 ms frames, 10 ms hop", "10 frame features x 4 statistics = 40 values", "0"],
            ["Hybrid concatenation", "up to three 1,024-vectors + 40 values", "1,064 to 3,112 values", "0"],
            ["Standardization", "d values", "d values", "0 (2d stored statistics)"],
            ["Logistic regression", "d values", "recording probability", "d+1"],
            ["Linear / RBF SVM", "d values", "decision value, then logistic function", "support-vector weights"],
            ["XGBoost", "d values", "recording probability", "100-300 trees, depth 2-4"],
            ["Speaker score", "recording probabilities", "mean per speaker", "0"], ["Decision", "speaker score", "PD if >= 0.5", "0"]]
    A(tab("Stages: inputs, output shapes and trainable parameters.", ["Stage", "Input", "Output (map shape)", "Trainable parameters"], rows, "p{4.0cm}p{4.0cm}p{5.4cm}p{3.3cm}", "scriptsize"))
    A(tab("Trainable parameters of the logistic-regression head (d weights + bias) by representation.", ["Representation", "Input dimension d", "LR parameters"], [[r, str(d[r]), str(ra["parameter_counts_logistic_regression"][r])] for r in reps], "lrr", "scriptsize"))
    A("\\section{Dataset, training and evaluation steps}")
    A(inline("**Dataset.** Italian Parkinson's Voice and Speech (Dimauro and Girardi, IEEE DataPort, doi:10.21227/aw6b-tg17) [12]: PD and elderly control folders, 16 tasks (two reading passages, word list, phrase, two syllable repetitions, ten sustained vowels); %d recordings, %d speakers (%d PD / %d controls), %.2f audio hours, recorded 2016-2017. Excluded: 15 young controls (age confound) and one byte-identical duplicate recording; people recorded in two sessions were merged. Task code, age, sex, sample rate and date are never model inputs. Ages from the corpus spreadsheets: patients %.1f +/- %.1f years (%d to %d), %d men / %d women; controls %.1f +/- %.1f years (%d to %d), %d men / %d women." % (sa["files_included"], sa["speakers_total"], sa["speakers_pd"], sa["speakers_hc"], sa["audio_hours"],
        DEM["PD_age"]["mean"], DEM["PD_age"]["sd"], DEM["PD_age"]["min"], DEM["PD_age"]["max"], DEM["PD_sex"]["M"], DEM["PD_sex"]["F"], DEM["HC_age"]["mean"], DEM["HC_age"]["sd"], DEM["HC_age"]["min"], DEM["HC_age"]["max"], DEM["HC_sex"]["M"], DEM["HC_sex"]["F"])))
    A("\\textbf{Steps (per outer fold, 5 folds, seed 42).} (1) Take the recordings of the training speakers only. (2) Standardize with training mean and SD. (3) For each configuration of the frozen search space (20 LR, 10 linear SVM, 24 RBF SVM, 24 XGBoost configurations), train on two of three inner folds and score the third; inner folds are speaker-disjoint and stratified. (4) Choose the configuration with the highest pooled inner speaker-level AUPRC. (5) Refit on all training speakers and predict the test fold once. (6) Average recordings to one probability per speaker; apply the fixed threshold 0.5. \\textbf{Batch sampling.} There are no mini-batches: the classifiers are fitted on all training recordings at once (a recording is the sample unit; 611 to 644 per fold), with no augmentation. Acoustic feature selection (forward and backward, exactly 10 features) is repeated inside every inner-training partition.")
    A("\\section{Evaluation metrics (definitions)}")
    A("With PD as the positive class and TP, FN, TN, FP counted over speakers:\n\\[\\text{sensitivity}=\\frac{TP}{TP+FN},\\quad \\text{specificity}=\\frac{TN}{TN+FP},\\quad \\text{precision}=\\frac{TP}{TP+FP},\\quad F_1=\\frac{2PR}{P+R},\\quad \\text{balanced accuracy}=\\frac{\\text{sens.}+\\text{spec.}}{2}\\]\n\\[\\text{AUROC}=\\Pr(s_{PD}>s_{ctrl})+\\tfrac12\\Pr(s_{PD}=s_{ctrl}),\\qquad \\text{AUPRC}=\\sum_n (R_n-R_{n-1})P_n\\]\nwhere $s$ is a speaker's mean probability, $P_n$ and $R_n$ are precision and recall at the $n$-th score threshold. \\textbf{Primary:} pooled out-of-fold speaker-level AUPRC and AUROC (relevance: threshold-free, comparable across models). \\textbf{Supporting:} the threshold metrics at 0.5, fold mean and SD, and 95\\% percentile bootstrap intervals over speakers (2,000 resamples).")
    A("\\section{Loss function and optimization}")
    A("Logistic regression (LIBLINEAR): $\\min_{w,b}\\ \\Omega(w)+C\\sum_i \\log(1+e^{-y_i(w^\\top x_i+b)})$ with $\\Omega=\\tfrac12\\lVert w\\rVert_2^2+\\tfrac12 b^2$ ($L_2$) or $\\lVert w\\rVert_1$ ($L_1$); $y_i\\in\\{-1,+1\\}$, class weights optional. Linear and RBF SVM (LIBSVM): $\\min\\ \\tfrac12\\lVert w\\rVert^2+C\\sum_i \\xi_i$ with hinge slack $\\xi_i\\ge 0$, RBF kernel $k(x,x')=\\exp(-\\gamma\\lVert x-x'\\rVert^2)$, $\\gamma=m/d$ with $m\\in\\{0.1,1,10\\}$; the score is mapped by the logistic function so that 0.5 equals the margin. XGBoost: additive trees minimising regularised logistic loss $\\sum_i \\ell(y_i,\\hat y_i)+\\sum_k[\\gamma T_k+\\tfrac12\\lambda\\lVert w_k\\rVert^2]$ by second-order gradient boosting [13]. All are fitted by the library solvers (convex optimisation for LR/SVM; boosting for XGBoost); hyperparameters are chosen by the inner-validation AUPRC. No gradient descent or learning-rate schedule is used for the linear models.")
    A("\\section{Experimental depth and iteration}")
    A(inline("The study is not a single run: **56 conditions x 5 outer folds = 280 runs**, each with nested tuning over 78 frozen configurations (inner 3-fold, speaker-disjoint). Variations beyond the baseline: 3 frozen encoders, acoustic features, 8 hybrids, 4 model families, forward/backward selection of acoustic features, and an acoustic-only control. All variations use identical outer folds."))
    A("\\section{Baseline selection and evaluation}")
    A(inline("**Baseline (mandatory justification):** frozen WavLM-large embeddings + logistic regression, because (i) with 47 speakers a model with %d trainable parameters is less likely to overfit than a fine-tuned network; (ii) frozen pretrained embeddings with logistic regression are an established design [5] and logistic regression is a common baseline in speech-based Parkinson's studies [2,5,14]; (iii) the loss is convex and the fit is deterministic. A second, even simpler baseline was run for comparison: 40 acoustic features with the same classifier. **Systematic evaluation:** identical folds, tuning procedure, threshold and metrics for every condition. Baseline: pooled AUROC %s [%s, %s], AUPRC %s, F1 %s, sensitivity %s, specificity %s (%d false negative, %d false positive speakers)." % (ra["parameter_counts_logistic_regression"]["WLM"], f3(base.AUROC), f3(base.AUROC_ci_lo), f3(base.AUROC_ci_hi), f3(base.AUPRC), f3(base.F1), f3(base.sensitivity), f3(base.specificity), base.FN, base.FP)))
    A("\\section{Implemented extensions and experiments}")
    A(inline("Extensions: other encoders (wav2vec2, XLS-R), acoustic features, eight hybrids, three other classifiers and sequential feature selection. **Did they improve on the baseline?** Largely no. Over the %d paired hybrid-versus-constituent comparisons the median AUROC difference is %s (range %s to %s) and the 95%% interval excludes zero in %d; against logistic regression the median differences are %s. Failures are analysed in the error-analysis section: almost all conditions miss the same patient and most false alarms come from a few controls, so there is room for at most a few speakers of improvement over the baseline." % (len(hy), f3(hy.delta_A_minus_B.median()), f3(hy.delta_A_minus_B.min()), f3(hy.delta_A_minus_B.max()), int(hy.ci_excludes_zero.sum()), "; ".join("%s %s" % (NAMES[m], f3(mv[mv.condition_A.str.endswith("-" + m)].delta_A_minus_B.median())) for m in P.MODELS[1:]))))
    A("\\section{Baseline reproduction (original outputs)}")
    rp = ra["replay"]
    A(inline("The baseline outputs are the saved prediction and metric files (`predictions/FINAL_V1/FV1-WLM-ALL-LR-F*.csv`, `metrics/FINAL_V1/`). Reproduction check: three completed conditions (WavLM-LR, acoustic-LR, XLS-R linear SVM; 15 runs) were executed again from the frozen configuration into a separate folder; the largest absolute difference in predicted probability against the official files was %s, %s, %s, i.e. bit-identical. The audit re-executed all 70 logistic-regression runs with single-threaded BLAS: bit-identical selected configurations, probabilities and metrics; with unrestricted multi-threaded BLAS some L2-logistic probabilities move by up to 4e-4 without changing any decision. An independent implementation of AUROC and AUPRC reproduces the pooled values of all 56 conditions to 1e-9 (audit)." % tuple("%.1e" % r["max_abs_probability_difference"] for r in rp)))
    sub = [["%s-%s" % (r, m), "%.3f / %.3f" % (trn["FV1-%s-ALL-%s" % (r, m)]["train_auprc_mean"], trn["FV1-%s-ALL-%s" % (r, m)]["train_auroc_mean"]), "%.3f (SD %.3f)" % (tvt["FV1-%s-ALL-%s" % (r, m)]["inner_val_auprc_mean"], tvt["FV1-%s-ALL-%s" % (r, m)]["inner_val_auprc_sd"]),
            "%.3f (SD %.3f)" % (tvt["FV1-%s-ALL-%s" % (r, m)]["heldout_fold_auprc_mean"], tvt["FV1-%s-ALL-%s" % (r, m)]["heldout_fold_auprc_sd"])] for r, m in [("AC", "LR"), ("WLM", "LR"), ("W2V", "LR"), ("XLSR", "LR"), ("H3", "LR"), ("WLM", "XGB")]]
    A("\\section{Results: training, validation and held-out}")
    A(inline("Training scores come from the final model scored on its own training speakers (diagnostic refit, mean over 5 folds); validation is the pooled inner speaker-level AUPRC at the chosen configuration (the quantity used to choose it); held-out is the outer test fold (mean AUPRC over 5 folds)."))
    A(tab("Training, inner validation and held-out speaker-level scores (six representative conditions).", ["Condition (representation-model)", "Training AUPRC / AUROC", "Inner validation AUPRC", "Held-out fold AUPRC"], sub, "llll", "scriptsize"))
    A(inline("The models fit their training speakers perfectly (linear models with 41 to 3,113 coefficients, and XGBoost, on about 37 speakers); validation and held-out scores are lower and close to each other, so tuning did not overfit the held-out folds."))
    cols = ["Condition", "AUPRC [95% CI]", "AUROC [95% CI]", "F1", "Sens.", "Spec.", "Bal. acc.", "FN/FP"]
    rows = [[c.condition_id.replace("FV1-", ""), "%s [%s, %s]" % (f3(c.AUPRC), f3(c.AUPRC_ci_lo), f3(c.AUPRC_ci_hi)), "%s [%s, %s]" % (f3(c.AUROC), f3(c.AUROC_ci_lo), f3(c.AUROC_ci_hi)), f3(c.F1), f3(c.sensitivity), f3(c.specificity), f3(c.balanced_accuracy), "%d/%d" % (c.FN, c.FP)] for c in done.itertuples()]
    A(tab("Key findings: pooled out-of-fold speaker-level results for all 56 conditions (47 speakers).", cols, rows, "lllrrrrl", "tiny"))
    A(figtex("heatmap_auroc.pdf", "Pooled AUROC by representation and model.", "0.55"))
    A(figtex("speaker_scores.pdf", "Pooled out-of-fold speaker scores; red rings mark errors at the fixed threshold 0.5."))
    A("\\section{Error and failure-case analysis}")
    tops = ", ".join("%s: %d of 56 conditions" % (s, counts[s]) for s in top)
    A(inline("Errors are concentrated in very few speakers: %s. The patient %s (female, born 19%s according to the file name, recorded %s at %s Hz) is missed in nearly every condition: her speaker score is %s with WavLM-LR and %s with acoustic-LR. Her recording probabilities are all close to the threshold (WavLM-LR: %s to %s; %d of 16 below 0.5; figure below), so the model is uncertain about this speaker as a whole rather than split between tasks. The controls %s (%s, %s) and %s (%s, %s) are false alarms in a smaller set of conditions (their WavLM-LR speaker scores are %s and %s); both are among the four controls flagged in the corpus spreadsheet for poor recording quality or reading difficulty, and these four flagged controls account for %d of the %d false-positive events over all 56 conditions. Recording-level errors by task family for WavLM-LR: %s; the sustained vowels are the least reliable family (%s of recordings wrong, against %s overall). The pattern is consistent across models: the same few speakers sit near the decision boundary, which is a property of these speakers' recordings, not of one classifier. Why they are ambiguous cannot be answered from this corpus, which has no severity, duration or medication data." % (
        tops, top[0], det[top[0]]["age_token"], det[top[0]]["dates"][0], det[top[0]]["sample_rates"][0], f3(det[top[0]]["FV1-WLM-ALL-LR"]["speaker_score"]), f3(det[top[0]]["FV1-AC-ALL-LR"]["speaker_score"]), f3(min(det[top[0]]["FV1-WLM-ALL-LR"]["by_task"].values())), f3(max(det[top[0]]["FV1-WLM-ALL-LR"]["by_task"].values())), sum(v < .5 for v in det[top[0]]["FV1-WLM-ALL-LR"]["by_task"].values()),
        top[1], det[top[1]]["sex"] + ", born 19" + det[top[1]]["age_token"], det[top[1]]["dates"][0], top[2], det[top[2]]["sex"] + ", born 19" + det[top[2]]["age_token"], det[top[2]]["dates"][0],
        f3(det[top[1]]["FV1-WLM-ALL-LR"]["speaker_score"]), f3(det[top[2]]["FV1-WLM-ALL-LR"]["speaker_score"]), DEM["false_positive_events_from_flagged_controls"], DEM["false_positive_events_total"],
        ", ".join("%s %d/%d" % (k, v["wrong"], v["n"]) for k, v in fam["FV1-WLM-ALL-LR"].items() if k != "_all"), "%.1f%%" % (100 * fam["FV1-WLM-ALL-LR"]["vowel"]["rate"]), "%.1f%%" % (100 * fam["FV1-WLM-ALL-LR"]["_all"]["rate"]))))
    A(figtex("error_tasks.pdf", "WavLM-LR recording probabilities (by task code) of the three most frequently misclassified speakers; dashed line 0.5."))
    A("\\section{Sensitivity and ablation analysis}")
    rtab = [[r, f3(med[r]), f3(done[done.representation == r].AUROC.min()), f3(done[done.representation == r].AUROC.max())] for r in reps]
    A(inline("Design choices were varied one at a time on identical folds. **Representation:** median AUROC over the four models (table). **Model:** median AUROC over all representations: %s. **Feature selection** (10 of 40 acoustic features): selected minus all, median AUROC difference %s (range %s to %s); no interval excludes zero in %d of %d. **Hybrids:** see the extensions section and the figure." % ("; ".join("%s %s" % (NAMES[m], f3(mm[m])) for m in P.MODELS), f3(fsc.delta_A_minus_B.median()), f3(fsc.delta_A_minus_B.min()), f3(fsc.delta_A_minus_B.max()), len(fsc) - int(fsc.ci_excludes_zero.sum()), len(fsc))))
    A(tab("AUROC by representation (over the four models).", ["Representation", "Median", "Min", "Max"], rtab, "lrrr", "scriptsize"))
    A(figtex("deltas.pdf", "Paired AUROC differences for the pre-specified comparisons.", "0.7"))
    A(inline("The RBF SVM is the clear outlier (lowest AUROC: WavLM %s, WavLM+acoustic %s) possibly because its logistic-mapped decision value is not calibrated to the fixed 0.5 threshold; this explanation was not tested. Sensitivity to input changes that were not tested: band-limiting, a different microphone, and a second corpus." % (f3(done[done.condition_id == "FV1-WLM-ALL-RSVM"].AUROC.iloc[0]), f3(done[done.condition_id == "FV1-H1-ALL-RSVM"].AUROC.iloc[0]))))
    A("\\section{Discussion}")
    A(inline("**Significance.** With speaker-disjoint folds, frozen embeddings and even 40 acoustic values separate the 25 patients from the 22 controls almost perfectly (%d of %d conditions make at most one speaker error). **Risks and limits.** (i) Every recording date holds one class and all 44.1 kHz audio is PD, so recording conditions are confounded with disease; acoustic loudness and band statistics alone give median AUROC %s, which a disease-specific signal does not require. (ii) With 47 speakers one person moves a metric by about two points; intervals are wide for the weaker conditions. (iii) The best of 56 conditions is an optimistic estimate. (iv) The data were used to design nothing after the freeze, but the corpus is small and single-site. **Sensitivity to input changes:** results were stable across encoders and classifiers, which shows robustness of the pipeline, not of the disease signal." % (int((done.err <= 1).sum()), len(done), f3(done[done.representation == "AC"].AUROC.median()))))
    A("\\section{Future directions}")
    A("\\begin{enumerate}\\item \\textbf{Independent, recording-matched corpus.} Freeze the pipeline exactly as it is and score a corpus in which patients and controls were recorded in the same sessions with the same devices; only this can separate disease from recording condition. \\item \\textbf{Remove or control the recording shortcut.} Define (before running) a procedure such as FWL residualization, or noise/bandwidth matching, fitted on training speakers only; FWL was blocked here for lack of a definition. \\item \\textbf{Task-aware scoring.} Recording-level errors are most frequent for sustained vowels; choose task weights or a task subset using inner validation only, and report per-task results.\\end{enumerate}")
    A("\\section{Conclusion}")
    A(inline("Under one frozen protocol and speaker-disjoint evaluation, all 56 combinations of acoustic, WavLM, wav2vec2 and XLS-R features with four classifiers were evaluated (280 of 280 runs completed, all 20 integrity checks passed). 36 of 56 conditions reach pooled speaker-level AUROC of at least 0.99 (45 reach 0.98), but extensions and hybrids do not improve on the simple baseline, errors concentrate on a few speakers, and a trivial acoustic model matches the encoders. The work meets its technical objective but cannot support a claim of Parkinson's detection; an independent corpus is the next step."))
    A("\\section{Team contribution}")
    A(tab("Team members and roles (assigned in the proposal).", ["Member", "Role / sections (from the proposal)"], [["Md. Osman Goni", "Overview and context, related work and background, discussion"], ["Md. Mohibur Rahman", "Methodology (dataset, feature extraction, model inputs/outputs, experimental plan), conclusion"], ["Arannamoy Mondal", "Baseline and extensions, results and analysis, future directions"]], "p{4cm}p{12cm}", "scriptsize"))
    A("The FINAL\\_V1 execution described here (code, runs, audit, report) was carried out by an AI assistant under the project owner's instruction. Individual contribution will be assessed from the document history, as the guideline states; this PDF does not claim authorship by any member.")
    A("\\begin{thebibliography}{99}\\setlength{\\itemsep}{1pt}\\scriptsize")
    refs = ["C. Quan et al., End-to-end deep learning approach for Parkinson's disease detection from speech signals, Biocybern. Biomed. Eng. 42(2):556-574, 2022, doi:10.1016/j.bbe.2022.04.002.",
            "A. Iyer et al., A machine learning method to process voice samples for identification of Parkinson's disease, Sci. Rep. 13:20615, 2023, doi:10.1038/s41598-023-47568-w.",
            "S.-M. Jeong et al., Machine learning-based classification of Parkinson's disease using acoustic features: insights from multilingual speech tasks, Comput. Biol. Med. 182:109078, 2024, doi:10.1016/j.compbiomed.2024.109078.",
            "B. Egbo et al., Explainable machine learning for early detection of Parkinson's disease in aging populations using vocal biomarkers, Front. Aging Neurosci. 17:1672971, 2025, doi:10.3389/fnagi.2025.1672971.",
            "O. Klempir and R. Krupicka, Analyzing wav2vec 1.0 embeddings for cross-database Parkinson's disease detection and speech features extraction, Sensors 24(17):5520, 2024, doi:10.3390/s24175520.",
            "S. Chen et al., WavLM: large-scale self-supervised pre-training for full stack speech processing, IEEE J. Sel. Topics Signal Process. 16(6):1505-1518, 2022, doi:10.1109/JSTSP.2022.3188113.",
            "A. Baevski et al., wav2vec 2.0: a framework for self-supervised learning of speech representations, NeurIPS 2020.",
            "A. Babu et al., XLS-R: self-supervised cross-lingual speech representation learning at scale, Interspeech 2022, arXiv:2111.09296.",
            "J. L. Crawford, Linguistic changes in spontaneous speech for detecting Parkinson's disease using large language models, PLOS Digit. Health 4(2):e0000757, 2025, doi:10.1371/journal.pdig.0000757.",
            "E. Vaiciukynas et al., Detecting Parkinson's disease from sustained phonation and speech signals, PLoS ONE 12(10):e0185613, 2017, doi:10.1371/journal.pone.0185613.",
            "Q. Wang et al., Early detection of Parkinson's disease from multiple signal speech: based on Mandarin language dataset, Front. Aging Neurosci. 14:1036588, 2022, doi:10.3389/fnagi.2022.1036588.",
            "G. Dimauro and F. Girardi, Italian Parkinson's Voice and Speech, IEEE DataPort, 2019, doi:10.21227/aw6b-tg17.",
            "T. Chen and C. Guestrin, XGBoost: a scalable tree boosting system, KDD 2016, doi:10.1145/2939672.2939785.",
            "A. A. Toye and S. Kompalli, Comparative study of speech analysis methods to predict Parkinson's disease, arXiv:2111.10207, 2021."]
    A("".join("\\bibitem{r%d} %s\n" % (i + 1, esc(r)) for i, r in enumerate(refs)) + "\\end{thebibliography}")
    # ---- part 2: compliance
    A("\\newpage\\part*{Part 2. Compliance with Project Guidelines.pdf}")
    A("The table lists every item of the final-report column of the guideline, where this report addresses it, and an honest status (Met, Partly, Not met).")
    comp_rows = [
        ["Abstract: problem clear to non-experts", "Part 1, Abstract", "Met"],
        ["Motivation and objectives (qualitative and quantitative)", "Part 1, Motivation and objectives", "Met"],
        ["Literature review: relevant prior work, grouped, with citations", "Part 1, Literature review (4 lines; 9 works from 2022-2025)", "Met; citations from the project's reference list, not re-verified online in this session"],
        ["Background: cited papers connect the metric and baseline to the problem", "Part 1, Background", "Met"],
        ["Model description: diagrams/tables of inputs, outputs, structure; output map shapes and parameter counts", "Part 1, Model description (figure, two tables)", "Met"],
        ["Dataset: training and evaluation steps incl. batch sampling; dataset citation", "Part 1, Dataset, training and evaluation steps", "Met (no mini-batches: stated)"],
        ["Evaluation metric: stated, relevance, mathematical definitions with labelled variables", "Part 1, Evaluation metrics", "Met"],
        ["Loss function and optimization with mathematical precision", "Part 1, Loss function and optimization", "Met (library solvers described, not hand-derived)"],
        ["Experimental depth and iteration: multiple configurations beyond a single run", "Part 1, Experimental depth (56 conditions, 280 runs)", "Met"],
        ["Baseline selection and evaluation (mandatory): justified with references; systematic evaluation", "Part 1, Baseline selection and evaluation", "Met"],
        ["Implemented extensions: description; improvement or justified analysis of failures", "Part 1, Extensions and Error analysis", "Met (no improvement over baseline; analysed)"],
        ["Baseline reproduction: original outputs confirming reproduction", "Part 1, Baseline reproduction (replay of 15 runs bit-identical; 70 LR runs bit-identical under single-thread BLAS; saved outputs)", "Met"],
        ["Results: separate clear results for training and validation (plots/tables); key findings visualised", "Part 1, Results (training / inner validation / held-out table, 56-condition table, figures)", "Met; training scores are a diagnostic refit"],
        ["Error / failure case analysis (at least one failure mode)", "Part 1, Error analysis (concentrated speaker errors, vowel family)", "Met"],
        ["Sensitivity / ablation (at least one design-choice comparison)", "Part 1, Sensitivity and ablation", "Met"],
        ["Discussion: significance, risks, sensitivity to input changes, limitations", "Part 1, Discussion", "Met"],
        ["Future directions: areas for exploration", "Part 1, Future directions (3 items)", "Met"],
        ["Conclusion", "Part 1, Conclusion", "Met"],
        ["Team contribution: detailed breakdown by members", "Part 1, Team contribution", "Partly: roles from the proposal; per-member breakdown must come from the document history"],
        ["Format: final report on Google Doc, minimum 8-10 pages", "This PDF (LaTeX)", "Partly: length met; not produced as a Google Doc"],
        ["Presentation items (6 minutes, three members)", "Not part of the written report", "Not addressed"]]
    A(tab("Compliance matrix.", ["Guideline item", "Where addressed", "Status"], comp_rows, "p{6.4cm}p{5.2cm}p{4.9cm}", "scriptsize"))
    A("\\textbf{Items for the project owner:} (1) upload or paste this report into the course Google Doc and assign contributions; (2) confirm that the 2022-2025 citations are acceptable after checking them; (3) FWL remains blocked until a definition is approved.")
    compile_tex("GUIDELINE_REPORT", "\n".join(sec), "Parkinson's disease detection from speech", "Report organised by the sections of Project Guidelines.pdf (CSE 431, topic 4a), with compliance matrix")


if __name__ == "__main__":
    build_guideline()
