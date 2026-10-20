"""Study-edition project report for FINAL_V1 (LaTeX -> PDF).

Written to be read and understood by the project team, who will write their own submission from it. It follows the order of
'Project Guidelines.pdf' (CSE 431, topic 4a), uses only the FINAL_V1 workspace (frozen protocol, audit, exploratory amendment A1),
and reads every number from the saved files. The team-contribution section is deliberately left out (the team adds it).
"""
import json, re, shutil, subprocess, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import protocol_defs as PDEF
from make_pdf import esc

OUT = WORKSPACE / "reports/FINAL_V1/study_report"
MET = WORKSPACE / "metrics/FINAL_V1"
SUPP = WORKSPACE / "supplementary/results"
FIGDIRS = [OUT / "figures", WORKSPACE / "reports/FINAL_V1/figures", WORKSPACE / "supplementary/figures"]
f3 = lambda x: "%.3f" % x
f2 = lambda x: "%.2f" % x
pct = lambda x: "%.0f%%" % (100 * x)
NAMES = {"LR": "logistic regression", "LSVM": "linear SVM", "RSVM": "RBF SVM", "XGB": "XGBoost"}
NAMESC = {"LR": "Logistic regression", "LSVM": "Linear SVM", "RSVM": "RBF SVM", "XGB": "XGBoost"}
REPN = {"AC": "Acoustic (40 features)", "WLM": "WavLM-large", "W2V": "wav2vec2-large-lv60", "XLSR": "XLS-R 300M", "H1": "WavLM + acoustic", "H2": "wav2vec2 + acoustic",
        "H3": "XLS-R + acoustic", "H4": "WavLM + wav2vec2", "H5": "WavLM + XLS-R", "H6": "wav2vec2 + XLS-R", "H7": "All three encoders", "H8": "All three encoders + acoustic"}

# ---------------------------------------------------------------------------------------------- data
cm = pd.read_csv(MET / "condition_metrics.csv"); cm["errors"] = cm.FN + cm.FP
comp = pd.read_csv(MET / "paired_comparisons.csv")
rec = json.loads((WORKSPACE / "FREEZE_RECORD.json").read_text()); aud = json.loads((MET / "oof_audit.json").read_text())
sa = json.loads((WORKSPACE / "manifests/stage_a_summary.json").read_text()); ra = json.loads((MET / "report_analysis.json").read_text())
dem = json.loads((WORKSPACE / "audit_opus/evidence/a05_demographics_and_flags.json").read_text())
a1 = json.loads((WORKSPACE / "audit_opus/evidence/a01_reextract.json").read_text()); a3 = json.loads((WORKSPACE / "audit_opus/evidence/a03_independent_recompute.json").read_text())
a4 = pd.read_csv(WORKSPACE / "audit_opus/evidence/a04_thread_sensitivity.csv")
S0 = pd.read_csv(SUPP / "S0_univariate_features.csv"); S1 = pd.read_csv(SUPP / "S1_ablation.csv"); S2 = pd.read_csv(SUPP / "S2_acoustic_importance.csv"); S2b = pd.read_csv(SUPP / "S2_block_importance.csv")
S3s = pd.read_csv(SUPP / "S3_selected_and_ties.csv"); S3L = pd.read_csv(SUPP / "S3_landscape.csv"); S5 = pd.read_csv(SUPP / "S5_exact_intervals.csv"); S6 = pd.read_csv(SUPP / "S6_session_predictability.csv")
S7 = pd.read_csv(SUPP / "S7_task_families.csv"); S8 = pd.read_csv(SUPP / "S8_compute.csv"); S9 = json.loads((SUPP / "S9_digital_silence_summary.json").read_text())
sm = pd.read_csv(WORKSPACE / "manifests/SAMPLE_MANIFEST.csv"); split = pd.read_csv(WORKSPACE / "SPLIT_MANIFEST.csv")
ck = json.loads((WORKSPACE / "features/manifests/encoder_checkpoints.json").read_text()); wman = json.loads((WORKSPACE / "features/manifests/WLM_manifest.json").read_text())
base = cm[cm.condition_id == "FV1-WLM-ALL-LR"].iloc[0]; bcp = S5[S5.condition_id == "FV1-WLM-ALL-LR"].iloc[0]
hy = comp[(comp.comparison_type == "hybrid_vs_constituent") & (comp.metric == "AUROC")]
mv = comp[(comp.comparison_type == "model_vs_logistic_regression") & (comp.metric == "AUROC")]
fsc = comp[(comp.comparison_type == "feature_selection_vs_all_features") & (comp.metric == "AUROC")]
task_med = sm.groupby("task_code").seconds.median()
spk_fold = split.drop_duplicates("speaker_id").groupby("outer_fold").agg(n=("speaker_id", "size"), pd_=("label", "sum"))
train_rec = [int((split.outer_fold != f).sum()) for f in range(5)]
runs = {f.stem: json.loads(f.read_text()) for f in sorted(MET.glob("FV1-*-F[0-4].json"))}
reg = pd.read_csv(WORKSPACE / "EXPERIMENT_REGISTRY.csv", keep_default_na=False)
z = lambda s, m: S1[(S1.subset == s) & (S1.model == m)].iloc[0]

# ---------------------------------------------------------------------------------------------- latex helpers
def inline(s):
    maths = []
    def keep(m):
        maths.append(m.group(1)); return "MATHTOKEN%dZ" % (len(maths) - 1)
    s = re.sub("\u27ea(.+?)\u27eb", keep, s)
    s = esc(s)
    s = s.replace(">=", "$\\geq$").replace("<=", "$\\leq$").replace("+/-", "$\\pm$").replace(" x ", " $\\times$ ")
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r"`(.+?)`", r"\\texttt{\1}", s)
    for i, m in enumerate(maths):
        s = s.replace("MATHTOKEN%dZ" % i, "$" + m + "$")
    return s


B = []
def A(x): B.append(x)
def P(text): A(inline(text) + "\n")
def H(level, title, key=None):
    if level == 0:
        A("\\section*{%s}\\addcontentsline{toc}{section}{%s}" % (inline(title), inline(title))); return
    A("\\%s{%s}%s" % ({1: "section", 2: "subsection", 3: "subsubsection"}[level], inline(title), "\\label{sec:%s}" % key if key else ""))
def box(kind, text):
    style = {"plain": ("In plain words", "green!40!black", "green!3"), "read": ("What this tells us", "blue!50!black", "blue!3"),
             "key": ("Key point", "orange!70!black", "orange!5"), "note": ("Note", "gray!60!black", "gray!5")}[kind]
    A("\\begin{tcolorbox}[colback=%s,colframe=%s,boxrule=0.5pt,arc=2pt,left=5pt,right=5pt,top=3pt,bottom=3pt,title={\\small\\bfseries %s}]\\small %s\\end{tcolorbox}" % (style[2], style[1], style[0], inline(text)))
def bullets(items, enum=False):
    env = "enumerate" if enum else "itemize"
    A("\\begin{%s}" % env + "".join("\\item " + inline(i) + "\n" for i in items) + "\\end{%s}" % env)
def fit_spec(spec, wide=False, minw=1.55):
    """Turn a column spec into fixed-width columns that always fit the text width (headers wrap instead of overflowing)."""
    toks = re.findall(r"p\{([\d.]+)cm\}|([lrc])", spec)
    total = 24.0 if wide else 16.5
    n = len(toks); avail = total - 0.42 * n
    fixed = sum(float(a) for a, b in toks if a); k = sum(1 for a, b in toks if b)
    scale, w = 1.0, None
    if k:
        w = (avail - fixed) / k
        if w < minw:
            w = minw; scale = max(0.3, (avail - k * w) / fixed) if fixed else 1.0
    elif fixed > avail:
        scale = avail / fixed
    out = []
    for a, b in toks:
        if a:
            out.append(">{\\raggedright\\arraybackslash}p{%.2fcm}" % (float(a) * scale))
        else:
            out.append(">{\\%s\\arraybackslash}p{%.2fcm}" % ("raggedleft" if b == "r" else ("centering" if b == "c" else "raggedright"), w))
    return "".join(out)


def tab(caption, header, rows, spec, size="small", wide=False, minw=1.55):
    spec = fit_spec(spec, wide, minw)
    cell = lambda x: "\\hspace{0pt}" + inline(str(x)).replace("/", "/\\allowbreak{}").replace("+", "+\\allowbreak{}").replace("\\_", "\\_\\allowbreak{}")
    head = " & ".join("\\textbf{%s}" % cell(h) for h in header)
    t = ["\\begin{center}\\%s\\renewcommand{\\arraystretch}{1.15}" % size, "\\begin{longtable}{%s}" % spec, "\\caption{%s}\\\\" % inline(caption), "\\toprule",
         head + "\\\\\\midrule\\endfirsthead", "\\toprule", head + "\\\\\\midrule\\endhead",
         "\\bottomrule\\endfoot"] + [" & ".join(cell(c) for c in r) + "\\\\" for r in rows] + ["\\end{longtable}\\end{center}"]
    A("\n".join(t))
def fig(name, cap, w="0.95"):
    A("\\begin{figure}[H]\\centering\\includegraphics[width=%s\\linewidth]{%s.pdf}\\caption{%s}\\label{fig:%s}\\end{figure}" % (w, name, inline(cap), name))
def math(s):
    A(s)


# ================================================================================================ FRONT
box("note", "This report covers **one study only**: the FINAL_V1 experiment in the isolated workspace `Project4a_FINAL_V1_ISOLATED`. It follows the order of the course guideline (Project Guidelines.pdf), so each part can be carried into the final report. Every number is produced by code from the saved result files. The team-contribution section is intentionally left for the team to write. Boxes called *In plain words* restate a technical idea simply; boxes called *What this tells us* explain how to read a result.")
H(0, "Key numbers at a glance")
rows = [["Question", "Parkinson's disease (PD) versus age-comparable healthy controls, from speech"],
        ["Data", "%d recordings from %d speakers (%d PD, %d controls), 16 speech tasks, %.1f hours of audio" % (sa["files_included"], sa["speakers_total"], sa["speakers_pd"], sa["speakers_hc"], sa["audio_hours"])],
        ["Evaluation", "5-fold cross-validation in which no person is ever in both training and test data; tuning inside each training part only"],
        ["Experiments", "56 configurations (12 representations x 4 classifiers, plus feature selection), 280 training runs, all completed"],
        ["Baseline", "WavLM-large embeddings + logistic regression: AUROC %s, AUPRC %s, sensitivity %s, specificity %s" % (f3(base.AUROC), f3(base.AUPRC), f2(base.sensitivity), f2(base.specificity))],
        ["All configurations", "median AUROC %s (range %s to %s); %d of 56 make at most one speaker error" % (f3(cm.AUROC.median()), f3(cm.AUROC.min()), f3(cm.AUROC.max()), int((cm.errors <= 1).sum()))],
        ["Simplest model", "40 hand-made acoustic numbers + logistic regression: AUROC %s" % f3(cm[cm.condition_id == "FV1-AC-ALL-LR"].AUROC.iloc[0])],
        ["Integrity", "%d of %d leakage and integrity checks passed; independent audit found no result-changing bug" % (sum(c["status"] == "PASS" for c in aud["checks"]), len(aud["checks"]))],
        ["Main caveat", "patients and controls were recorded in different sessions, and their audio files differ in technical details (sampling rate, digital silence), so the high scores cannot be read as proof that the models detect the disease"]]
tab("The study in nine lines.", ["Item", "Value"], rows, "p{3.2cm}p{12.6cm}")

H(0, "How this report follows the course guideline")
P("The headings follow the *Final Report* column of Project Guidelines.pdf (CSE 431), in the same order. The table shows where each rubric item is answered. The team-contribution item is left for the team. Analyses that go beyond the minimum (feature importance, tuning, tasks, recording conditions) sit inside *Results and analysis*.")
rows = [["Overview and context", "Abstract", "section [[abstract]]"], ["", "Motivation and objectives", "section [[motivation]]"],
        ["Related work and background", "Literature review", "section [[related]]"], ["", "Background (metric and baseline linked to the problem)", "section [[background]]"],
        ["Methodology", "Model description (diagrams, output shapes, parameter counts)", "section [[model]]"], ["", "Dataset (citation; training and evaluation steps incl. batch sampling)", "section [[dataset]]"],
        ["", "Evaluation metric (mathematical definitions)", "section [[metrics]]"], ["", "Loss function and optimisation", "section [[loss]]"], ["", "Experimental depth and iteration", "section [[experiments]]"],
        ["Baseline and extensions", "Baseline selection and evaluation (mandatory)", "section [[baseline]]"], ["", "Implemented extensions / experiments", "section [[extensions]]"], ["", "Baseline outputs confirming reproduction", "section [[reproduction]]"],
        ["Results and analysis", "Results (training and validation separately; key findings visualised)", "section [[results]]"], ["", "Error / failure-case analysis", "section [[errors]]"], ["", "Sensitivity / ablation analysis", "section [[sensitivity]]"],
        ["Discussion", "Significance, risks, sensitivity to input changes, limitations", "section [[discussion]]"], ["Future directions", "Areas for future work", "section [[future]]"], ["Conclusion", "Summary of findings and objectives", "section [[conclusion]]"],
        ["Team contribution", "Breakdown by members", "to be added by the team"]]
tab("Guideline rubric item and the section of this report that answers it.", ["Guideline part", "Rubric item", "Where"], rows, "p{4.0cm}p{8.6cm}p{3.0cm}")

# ================================================================================================ 1 ABSTRACT
H(1, "Abstract", "abstract")
P("Parkinson's disease often changes the way people speak: the voice can become quieter, flatter and less precise. Because speech is cheap to record, a computer that recognises these changes could one day help doctors find the disease earlier. In this project we asked whether short speech recordings can tell people with Parkinson's disease apart from healthy people of the same age, when the computer is always tested on people it has never heard before.")
P("We used a public Italian dataset with %d recordings from %d people (%d with Parkinson's, %d healthy older adults) performing 16 short speaking tasks such as reading, repeating syllables and holding a vowel. We turned each recording into numbers in two ways: 40 simple acoustic measurements (loudness, spectral shape and the energy in different frequency bands), and the outputs of three large pretrained speech models (WavLM-large, wav2vec2 and XLS-R). We then trained four kinds of classifiers on these numbers and tested 56 combinations under one fixed, pre-registered plan." % (sa["files_included"], sa["speakers_total"], sa["speakers_pd"], sa["speakers_hc"]))
P("Almost every combination separated the two groups very well: the median area under the ROC curve (AUROC) was %s, and the baseline (WavLM-large with logistic regression) missed one patient and raised no false alarm. However, 40 simple acoustic numbers did just as well as the large models, the tree-based models relied most on numbers that describe the quiet background between words, and the two groups were recorded on different days, with audio files that differ in technical details such as digital silence at the start or end. The scores are therefore real for this dataset but cannot be taken as proof that the models hear the disease. A test on independent recordings made under the same conditions for both groups is the necessary next step." % f3(cm.AUROC.median()))

# ================================================================================================ 2 MOTIVATION
H(1, "Motivation and objectives", "motivation")
P("About 11.8 million people were living with Parkinson's disease worldwide in 2021 [1]. Changes in speech and voice affect around 89%% of people with the disease, yet only a small minority receive speech treatment [2]. Diagnosis still relies mainly on clinical examination, and it is hardest in the early stages and outside specialist clinics [3]. Speech offers something rare in medicine: a signal that can be recorded in a few minutes with any phone, at home, as often as needed. If a reliable speech marker existed it could support screening, referral decisions and the follow-up of patients in clinical trials.")
P("The danger in this field is over-optimism. Many published speech studies report very high accuracies on small datasets, and those numbers often do not survive a stricter test. So our project had two aims at once: build good classifiers, and measure them honestly.")
P("**Qualitative objectives.**")
bullets(["Build a pipeline that predicts Parkinson's disease from a person's speech recordings and is always evaluated on unseen speakers.",
         "Compare different ways of describing speech (hand-made acoustic features versus pretrained deep speech models, and their combinations), different classifiers and feature selection under identical conditions.",
         "Understand *why* the models work or fail: which features matter, which recordings are misclassified, how sensitive the results are to design choices, and whether recording conditions could explain the results."])
P("**Quantitative objectives.**")
bullets(["Estimate, for each of the 56 configurations, the pooled speaker-level AUROC and AUPRC (primary metrics) and the sensitivity, specificity, F1 score and balanced accuracy at a fixed threshold of 0.5, with 95% confidence intervals.",
         "Test whether any extension (other encoders, hybrids, other classifiers, feature selection) improves on the baseline by more than the statistical uncertainty, using paired bootstrap comparisons on the same people.",
         "Verify that no information from test speakers reached training (zero speaker overlap in all 280 runs) and that every reported number can be reproduced."])

# ================================================================================================ 3 RELATED WORK
H(1, "Related work (literature review)", "related")
P("Speech-based detection of Parkinson's disease has been studied with very different tools. We group recent work (2022 to 2025) into four lines and explain how each connects to our project. The accuracies quoted below come from different datasets, languages and evaluation designs, so they cannot be compared with our numbers directly.")
H(2, "Deep networks trained on spectrograms or raw speech")
P("Quan et al. (2022) [4] built an end-to-end network in which time-distributed 2D convolutional blocks turn the speech spectrogram into a sequence of features and a 1D convolutional block learns how they change over time; they tested it on Chinese and Spanish recordings. Iyer et al. (2023) [5] recorded the sustained vowel /a/ over the telephone from 50 people with Parkinson's and 50 controls and classified spectrogram images with a pretrained convolutional network (Inception V3) using transfer learning. **Connection:** both train deep networks on relatively small datasets. With only 47 speakers we chose instead to *reuse* large pretrained speech models without training them, which needs far less data.")
H(2, "Hand-crafted acoustic features with classical classifiers")
P("Wang et al. (2022) [6] collected a Mandarin dataset, compared three feature-selection methods (LASSO, mRMR and Relief-F) with four classifiers, and reported their best result as 75.8% accuracy with 82.4% sensitivity and 73.2% specificity. Egbo et al. (2025) [7] used XGBoost to select the ten most useful acoustic features from sustained phonations of 31 people and then a tuned XGBoost classifier, reporting 98% accuracy on a held-out test set, while noting the small sample. **Connection:** these studies motivate three parts of our design: hand-crafted acoustic features, forward and backward feature selection, and XGBoost as one of our classifiers. Wang et al. also show why sensitivity and specificity must be reported side by side.")
H(2, "Pretrained self-supervised speech embeddings")
P("Self-supervised speech models learn from very large amounts of unlabelled audio. wav2vec 2.0 [8] learns by predicting masked parts of the speech signal; XLS-R [9] applies the same idea to many languages; WavLM [10] adds denoising and was trained on about 94,000 hours of speech. Klempir and Krupicka (2024) [11] used *non-fine-tuned* (frozen) wav2vec embeddings on three Parkinson's datasets in different languages and found them better than classical MFCC features when a model trained on one database was tested on another. **Connection:** this is the closest design to ours. We use frozen embeddings from three newer models (WavLM-large, wav2vec2-large and XLS-R) and compare them with each other, with acoustic features and with their combinations.")
H(2, "Linguistic content")
P("Crawford (2025) [12] analysed *what* people say rather than *how* they sound: spontaneous speech was transcribed and language-model features were used to detect Parkinson's disease with up to 78% accuracy. **Connection:** our models use only the audio signal and ignore the words, so linguistic cues are outside our scope; they are a possible future extension.")
H(2, "Recording conditions")
P("Vaiciukynas et al. (2017) [13] recorded the same people simultaneously with two different microphones and found that the error rate depended on the microphone (for the fused system, about 19% with a cardioid microphone versus 23% with a smartphone). **Connection:** recording conditions change the signal that a classifier sees. This turned out to be central in our project (section [[confound]]).")

# ================================================================================================ 4 BACKGROUND
H(1, "Background: the ideas behind our design", "background")
H(2, "How Parkinson's disease changes speech")
P("The speech disorder typical of Parkinson's disease is called hypokinetic dysarthria. People may speak more softly, with less variation in pitch and loudness, with less precise consonants and sometimes with a breathy or hoarse voice. Such changes are reported in most people with the disease [2]. This is why loudness, spectral shape and the fine structure of the voice are reasonable things to measure, and why we did not normalise the loudness of the recordings.")
H(2, "Two ways to turn sound into numbers")
P("**Hand-crafted acoustic features** are formulas chosen by people: for example the energy of each short frame, how often the waveform crosses zero, or where the centre of mass of the spectrum lies. They are easy to interpret but describe only what we thought of measuring. **Learned embeddings** come from large neural networks trained on thousands of hours of speech. They describe the sound with 1,024 numbers per moment, capturing patterns no one designed by hand, but the numbers have no individual meaning. We use both and compare them.")
box("plain", "Think of the pretrained speech model as a very experienced listener who has heard tens of thousands of hours of speech. We never teach this listener anything about Parkinson's disease; we only ask it to describe each recording in its own language (1,024 numbers). A small, simple classifier then learns which descriptions belong to patients.")
H(2, "Why the person, not the recording, is the unit")
P("Each person contributes about 16 recordings. If some recordings of a person were used for training and others for testing, the model could recognise *the person's voice* rather than the disease and still look accurate. We therefore always keep all recordings of a person on the same side of every split (speaker-disjoint evaluation), and we score people rather than recordings: a person's score is the average of the probabilities of their recordings. Our estimate is thus about **new people**, not about new recordings of people the model already knows.")
H(2, "Why these metrics")
P("A screening tool can make two different mistakes: missing a patient (low sensitivity) or alarming a healthy person (low specificity). Studies in this field report both [6], and so do we. AUROC and AUPRC summarise how well the model ranks patients above controls over all possible thresholds; they are our primary metrics because they do not depend on one chosen threshold. We also report F1, which balances precision and sensitivity, and balanced accuracy, the average of sensitivity and specificity.")
H(2, "Why this baseline")
P("Our baseline is the simplest strong design: frozen WavLM-large embeddings and logistic regression. With only 47 people, a model with few trainable parameters is far less likely to overfit than a deep network trained from scratch; frozen pretrained embeddings with simple classifiers are an established design for Parkinson's speech [11]; WavLM is a state-of-the-art general speech model [10]; and logistic regression has a convex training problem, so it always reaches the same solution. Every other configuration is compared with this baseline on exactly the same people and folds.")

# ================================================================================================ METHODOLOGY
H(1, "Methodology", "method")
H(2, "Model description", "model")
H(3, "Overview of the pipeline")
fig("pipeline", "The pipeline. The grey part is frozen and never sees a label; only the standardisation and the classifier are fitted, and only on the training speakers of each fold.")
P("Each recording goes through the same steps: resampling, conversion into a fixed-length vector of numbers (the *representation*), standardisation, a classifier that outputs the probability of Parkinson's disease, and finally the average over all recordings of a person. A person is called a patient if this average is at least 0.5.")
H(3, "Representation 1: forty acoustic features")
P("We cut each recording into short frames of 25 ms (400 samples) that start every 10 ms, and compute ten measurements per frame. Over all frames of a recording we then take four statistics: the mean, the standard deviation, the 10th percentile and the 90th percentile. This gives 10 x 4 = 40 numbers per recording.")
rows = [["log energy (dB)", "loudness of the frame", "softer voice in PD; also microphone gain and distance"],
        ["zero-crossing rate", "how often the waveform changes sign; high for noisy or high-frequency sounds", "breathiness, consonants, background hiss"],
        ["spectral centroid (Hz)", "'centre of mass' of the spectrum; a brightness measure", "changes in voice quality and articulation; microphone colouring"],
        ["spectral roll-off 85% (Hz)", "frequency below which 85% of the energy lies", "similar to the centroid"],
        ["spectral flatness", "how noise-like (flat) versus tonal (peaky) the spectrum is", "breathy or hoarse voice; noise"],
        ["band energy 0-500 Hz", "energy of low frequencies (pitch region)", "voicing, hum, room noise"],
        ["band energy 0.5-1 kHz", "first-formant region", "vowel quality"],
        ["band energy 1-2 kHz", "second-formant region", "articulation"],
        ["band energy 2-4 kHz", "upper formants and consonants", "articulation, clarity"],
        ["band energy 4-8 kHz", "high frequencies (fricatives, hiss)", "consonants, noise"]]
tab("The ten frame-level measurements and why they could matter.", ["Measurement", "What it measures", "Possible link to PD or to recording conditions"], rows, "p{3.4cm}p{6.0cm}p{6.2cm}", "small")
box("plain", "The four statistics describe different parts of a recording. The **mean** describes the typical frame, the **standard deviation** how much the frames vary, the **90th percentile** the loudest or brightest moments (usually the voice), and the **10th percentile** the quietest moments, which are mostly pauses and background. This last point becomes important in section [[importance]].")
H(3, "Representation 2: three frozen speech models")
P("The three encoders share the same architecture: a stack of seven convolutional layers turns the 16-kHz waveform into one vector every 20 ms, and 24 transformer layers with 1,024 hidden units refine these vectors using the context of the whole window. We cut each recording into 10-second windows (the last window is aligned with the end of the file), pass each window through the encoder, average the 499 output vectors over time and then average the windows. Every recording therefore becomes one vector of 1,024 numbers. The encoders are never trained in this project.")
rows = [[REPN[k], ck[k]["model_id"], "%s" % format(int(S8[S8.encoder == k].parameters.iloc[0]), ","), "499 x 1,024 per 10-s window", "1,024", "%.2f s" % S8[(S8.encoder == k) & (S8.threads == 8)].seconds_per_10s_window_median.iloc[0]] for k in ["WLM", "W2V", "XLSR"]]
tab("The frozen encoders: size, output map shape and CPU time per 10-s window (8 threads).", ["Encoder", "Checkpoint", "Parameters (frozen)", "Output map", "Vector", "CPU time"], rows, "p{2.6cm}p{4.0cm}rp{2.8cm}rr", "small")
P("WavLM-large was trained on about 94,000 hours of English speech with a denoising objective [10]; wav2vec2-large-lv60 on about 60,000 hours of English audiobooks [8]; XLS-R 300M on speech in many languages [9]. Using three different encoders lets us see whether the result depends on one particular model.")
H(3, "Representation 3: hybrids")
P("A hybrid simply places two or more representations side by side in one longer vector (for example WavLM's 1,024 numbers followed by the 40 acoustic features). We built the eight hybrids listed in table [[tab-dims]]. Because every column is standardised before the classifier, each feature starts on the same scale.")
rows = [[k, REPN[k], "+".join(PDEF.components(k)), "%d" % sum(PDEF.DIMS[c] for c in PDEF.components(k)), "%d" % (sum(PDEF.DIMS[c] for c in PDEF.components(k)) + 1)] for k in PDEF.BASE_REPS + list(PDEF.HYBRIDS)]
A("\\label{tab-dims}")
tab("All twelve representations, their size and the number of trainable parameters of a logistic-regression classifier on top (one weight per input plus a bias).", ["Key", "Representation", "Components", "Input size d", "LR parameters (d+1)"], rows, "llllr")
H(3, "Classifiers")
P("**Logistic regression** draws a straight boundary (a weighted sum of the inputs) and turns its distance from the boundary into a probability. **Linear SVM** also draws a straight boundary but places it to maximise the margin between the classes. **RBF SVM** can draw curved boundaries by comparing each recording with the training recordings through a Gaussian similarity. **XGBoost** builds many small decision trees one after another, each correcting the errors of the previous ones. The first three need standardised inputs; XGBoost does not.")
rows = [["Resampling and windowing", "WAV file at 16 or 44.1 kHz", "10-s windows of 160,000 samples at 16 kHz", "0"],
        ["Acoustic feature extractor", "16-kHz audio", "40 numbers per recording", "0"],
        ["Frozen encoder", "one 10-s window", "499 x 1,024 output map", "0 of about 315 million"],
        ["Pooling", "output maps of all windows", "1,024 numbers per recording", "0"],
        ["Standardisation", "d numbers", "d numbers", "0 (2d statistics stored)"],
        ["Logistic regression", "d numbers", "probability of PD", "d + 1 (41 to 3,113)"],
        ["Linear / RBF SVM", "d numbers", "score mapped to (0, 1)", "one weight per support vector"],
        ["XGBoost", "d numbers", "probability of PD", "100 or 300 trees of depth 2 or 4"],
        ["Person score", "probabilities of one person's recordings", "their mean", "0"],
        ["Decision", "person score", "PD if at least 0.5", "0"]]
tab("Model description: inputs, outputs and trainable parameters of every stage.", ["Stage", "Input", "Output", "Trainable parameters"], rows, "p{3.2cm}p{4.2cm}p{4.4cm}p{3.6cm}", "small")
# ================================================================================================ 5 DATASET
H(2, "Dataset", "dataset")
H(3, "Source and content")
P("We used the *Italian Parkinson's Voice and Speech* corpus by Dimauro and Girardi, published on IEEE DataPort (doi:10.21227/aw6b-tg17) [14] and described in Dimauro et al. (2017) [15]. It contains recordings of people with Parkinson's disease, elderly healthy controls and young healthy controls, made in 2016 and 2017. Every person performed up to 16 short tasks (table below).")
desc = {"B1": "reading a phonetically balanced text (first reading)", "B2": "reading the same text again", "D1": "repeating the syllable /pa/ for about 5 s", "D2": "repeating the syllable /ta/ for about 5 s",
        "FB1": "reading phrases", "PR1": "reading a list of words", "VA1": "sustained vowel /a/ (first)", "VA2": "sustained vowel /a/ (second)", "VE1": "sustained vowel /e/ (first)",
        "VE2": "sustained vowel /e/ (second)", "VI1": "sustained vowel /i/ (first)", "VI2": "sustained vowel /i/ (second)", "VO1": "sustained vowel /o/ (first)", "VO2": "sustained vowel /o/ (second)",
        "VU1": "sustained vowel /u/ (first)", "VU2": "sustained vowel /u/ (second)"}
rows = [[t, desc[t], "%d" % (sm.task_code == t).sum(), "%.1f" % task_med[t]] for t in sorted(task_med.index)]
tab("The 16 speech tasks in the cohort we analysed (recordings and median duration).", ["Code", "Task", "Recordings", "Median seconds"], rows, "lp{8.6cm}rr")
H(3, "Who is included")
P("We kept the people with Parkinson's disease and the elderly healthy controls, and excluded the 15 young controls (aged 19 to 29), because their age alone would separate them from the patients and they recorded only three tasks. One recording was a byte-for-byte copy of another recording of the same person and was removed. Three people with Parkinson's were recorded in two sessions (under the same or a slightly reordered name code); we merged each of them into one speaker so that a person can never appear on both sides of a split. The final cohort has **%d recordings from %d speakers (%d PD, %d controls)**, about %.1f hours of audio." % (sa["files_included"], sa["speakers_total"], sa["speakers_pd"], sa["speakers_hc"], sa["audio_hours"]))
rows = [["People", "%d" % sa["speakers_pd"], "%d" % sa["speakers_hc"]], ["Recordings", "%d" % sa["recordings_pd"], "%d" % sa["recordings_hc"]],
        ["Age in years, mean +/- SD (range)", "%.1f +/- %.1f (%d to %d)" % (dem["PD_age"]["mean"], dem["PD_age"]["sd"], dem["PD_age"]["min"], dem["PD_age"]["max"]), "%.1f +/- %.1f (%d to %d)" % (dem["HC_age"]["mean"], dem["HC_age"]["sd"], dem["HC_age"]["min"], dem["HC_age"]["max"])],
        ["Men / women", "%d / %d" % (dem["PD_sex"]["M"], dem["PD_sex"]["F"]), "%d / %d" % (dem["HC_sex"]["M"], dem["HC_sex"]["F"])],
        ["Recordings at 44.1 kHz / 16 kHz", "%d / %d" % ((sm[(sm.label == 1)].sample_rate == 44100).sum(), (sm[(sm.label == 1)].sample_rate == 16000).sum()), "%d / %d" % ((sm[(sm.label == 0)].sample_rate == 44100).sum(), (sm[(sm.label == 0)].sample_rate == 16000).sum())],
        ["Recording dates", ", ".join(sorted(sm[sm.label == 1].date.unique())), ", ".join(sorted(sm[sm.label == 0].date.unique()))]]
tab("The analysed cohort. Ages and sex are from the corpus spreadsheets.", ["", "Parkinson's disease", "Controls"], rows, "p{4.4cm}p{5.6cm}p{5.6cm}")
P("Two details matter later. First, the corpus spreadsheet marks four of the controls with an asterisk and explains that their recordings are of poor quality or that they had reading difficulties (one of them has no reading-passage or phrase recordings). We kept them, as planned, and report their effect in the error analysis. Second, the two-digit number inside the file names is the year of birth, not the age; ages come from the spreadsheets and are never given to the models.")
H(3, "How the recording sessions are arranged")
fig("wp_confound_timeline", "Recording sessions by date. Every date contains only one group; all audio recorded at 44.1 kHz belongs to patients.", "0.85")
P("This is the single most important property of the dataset. The patients were recorded on four dates (two in 2016, two in early 2017) and the controls on five other dates in March and April 2017. No recording session contains both groups, and all recordings at 44.1 kHz are from patients. Anything that differs between recording sessions (room, microphone placement, background noise, recording software) is therefore perfectly aligned with the diagnosis.")
H(3, "What the recordings look like")
fig("dataset_examples", "Representative recordings: the sustained vowel /a/ and the phrase task for one person with Parkinson's disease and one control, as waveforms (left) and spectrograms (right; brighter = more energy).")
P("The examples show the kind of material the models receive. Both people produce a clear vowel and clearly separated phrases. Differences in overall level and in the background between words are visible, which is exactly the kind of information that can come from the disease (softer voice) *or* from the recording set-up.")
H(3, "Preparation of the audio")
bullets(["All recordings were converted to mono and resampled to 16 kHz with a polyphase filter, so that the original sample rate (which differs between the groups) is not directly visible.",
         "We did **not** normalise the loudness and did **not** cut silences, because a softer voice is a symptom of the disease.",
         "No statistic from other recordings, other people or the labels is used during this step."])

H(3, "Training and evaluation steps (including batch sampling)", "training")
fig("nested_cv", "Nested, speaker-disjoint cross-validation: tuning happens inside the training part of each outer fold, and the test fold is touched only once.")
P("The 47 speakers were split once, with a fixed random seed, into five outer folds of 10, 10, 9, 9 and 9 speakers, each containing both patients and controls (stratified). For each outer fold the procedure was:")
bullets(["Take the recordings of the speakers in the four training folds (between %d and %d recordings)." % (min(train_rec), max(train_rec)),
         "Split these training speakers again into three inner folds (also speaker-disjoint and stratified).",
         "For every hyperparameter configuration of the classifier, train on two inner folds and predict the third, rotating until each inner fold has been predicted once. Average the predictions per speaker and compute the speaker-level AUPRC. Keep the configuration with the highest value (ties go to the first configuration in the list).",
         "If the configuration uses feature selection, repeat the selection inside every inner training part, and once more on the full training part.",
         "Retrain the chosen configuration on all training speakers, predict the recordings of the test fold, average them per speaker and apply the threshold 0.5.",
         "After all five folds, every speaker has exactly one out-of-fold prediction; we compute the metrics once on all 47 speakers (pooled) and also per fold."], enum=True)
P("**Batch sampling.** None of our classifiers uses mini-batches or stochastic gradient descent. Each model is fitted on all training recordings at once (full-batch), with no data augmentation and no resampling of the data. Class imbalance (55% of recordings are from patients) is handled through the hyperparameter *class weight*, which is part of every search space.")
box("read", "Because the test speakers are never used for scaling, feature selection, tuning or training, the scores estimate how the pipeline would do on new people recorded under the conditions of this corpus.")
H(2, "Evaluation metrics", "metrics")
P("Let TP be the number of patients predicted as patients, FN the patients predicted as controls, TN the controls predicted as controls and FP the controls predicted as patients (all counted over speakers). Then:")
math(r"""\begin{align*}
\text{sensitivity (recall)} &= \frac{TP}{TP+FN}, & \text{specificity} &= \frac{TN}{TN+FP},\\[4pt]
\text{precision} &= \frac{TP}{TP+FP}, & F_1 &= \frac{2\cdot\text{precision}\cdot\text{sensitivity}}{\text{precision}+\text{sensitivity}},\\[4pt]
\text{balanced accuracy} &= \frac{\text{sensitivity}+\text{specificity}}{2}. &&
\end{align*}""")
P("For the threshold-free metrics, let ⟪s_i⟫ be the score of speaker ⟪i⟫, ⟪P⟫ the set of patients and ⟪C⟫ the set of controls:")
math(r"""\begin{align*}
\text{AUROC} &= \frac{1}{|P|\,|C|}\sum_{i\in P}\sum_{j\in C}\Big(\mathbf{1}[s_i>s_j]+\tfrac12\,\mathbf{1}[s_i=s_j]\Big), &
\text{AUPRC} &= \sum_{n}\,(R_n-R_{n-1})\,P_n,
\end{align*}""")
P("where ⟪R_n⟫ and ⟪P_n⟫ are the recall and precision when everyone with a score of at least the ⟪n⟫-th highest score is called a patient. AUROC is the probability that a randomly chosen patient gets a higher score than a randomly chosen control (0.5 = guessing, 1 = perfect). AUPRC is the area under the precision-recall curve; its guessing level equals the share of patients, 25/47 = 0.53.")
P("**Uncertainty.** With 47 people every estimate is uncertain. For AUROC and AUPRC we resampled the 47 speakers with replacement 2,000 times and took the 2.5th and 97.5th percentiles (bootstrap interval). For sensitivity and specificity we also give exact (Clopper-Pearson) binomial intervals. When AUROC is exactly 1, every resample is also perfect, so the bootstrap interval collapses to [1, 1]; this does not mean there is no uncertainty.")
H(2, "Loss functions and optimisation", "loss")
P("**Logistic regression** (solver LIBLINEAR [16]). With inputs ⟪x_i⟫, labels ⟪y_i \\in \\{-1,+1\\}⟫ (+1 = Parkinson's) and optional class weights ⟪c_i⟫, the weights ⟪w⟫ and the bias ⟪b⟫ minimise")
math(r"""\[\min_{w,b}\;\; \Omega(w,b) \;+\; C\sum_{i=1}^{n} c_i\,\log\!\big(1+e^{-y_i\,(w^\top x_i+b)}\big),\qquad \Omega=\tfrac12\lVert (w,b)\rVert_2^2\;\;\text{(L2)}\;\;\text{or}\;\;\lVert (w,b)\rVert_1\;\;\text{(L1)},\]""")
P("and the probability of Parkinson's disease is ⟪\\hat p(x) = 1/\\big(1+e^{-(w^\\top x+b)}\\big)⟫. The first term keeps the weights small (regularisation; L1 can set weights exactly to zero), the second penalises wrong or uncertain predictions, and ⟪C⟫ sets the balance between them. The problem is convex, so there is a single best solution and no random initialisation.")
P("**Support vector machines** (LIBSVM [17]) minimise the hinge loss with a margin term,")
math(r"""\[\min_{w,b,\xi}\;\tfrac12\lVert w\rVert^2 + C\sum_i c_i\,\xi_i\quad\text{s.t.}\quad y_i\big(w^\top\phi(x_i)+b\big)\ge 1-\xi_i,\;\;\xi_i\ge 0,\]""")
P("where ⟪\\xi_i⟫ measures how far recording ⟪i⟫ is on the wrong side of the margin, ⟪\\phi⟫ is the identity for the linear SVM, and for the RBF SVM it is defined implicitly by the kernel ⟪k(x,x') = \\exp(-\\gamma\\lVert x-x'\\rVert^2)⟫ with ⟪\\gamma = m/d⟫, ⟪m \\in \\{0.1, 1, 10\\}⟫ and ⟪d⟫ the number of inputs. The SVM score ⟪f(x)⟫ is mapped to ⟪1/(1+e^{-f(x)})⟫, so the threshold 0.5 corresponds to the SVM margin ⟪f(x)=0⟫; this mapping is not a calibrated probability.")
P("**XGBoost** [18] adds trees ⟪f_k⟫ one at a time and minimises the regularised logistic loss")
math(r"""\[\mathcal{L}=\sum_i \ell\big(y_i,\hat y_i\big)+\sum_k\Big(\gamma T_k+\tfrac12\lambda\lVert w_k\rVert^2\Big),\qquad \hat y_i=\sum_k f_k(x_i),\]""")
P("where ⟪\\ell⟫ is the logistic loss, ⟪T_k⟫ the number of leaves and ⟪w_k⟫ the leaf weights of tree ⟪k⟫; each new tree is found with a second-order (gradient and Hessian) approximation of the loss. The learning rate, the depth, the number of trees, the minimum child weight, the row and column subsampling, the L2 penalty ⟪\\lambda⟫ and the positive-class weight are the tuned hyperparameters.")
# ================================================================================================ 7 EXPERIMENTS
H(2, "Experimental depth and iteration", "experiments")
P("The study is not a single training run. We varied four design choices on identical folds:")
bullets(["**Representation** - 4 single representations (acoustic, WavLM, wav2vec2, XLS-R) and 8 hybrids.",
         "**Classifier** - logistic regression, linear SVM, RBF SVM and XGBoost, each with its own frozen search space (20, 10, 24 and 24 configurations).",
         "**Feature selection** - for the 40 acoustic features only: all features, forward selection to 10 features, backward elimination to 10 features.",
         "**Hyperparameters** - tuned in every run by nested cross-validation."])
P("Together this gives 12 acoustic configurations (3 selection settings x 4 classifiers) plus 44 encoder and hybrid configurations (11 x 4), i.e. **56 configurations and 280 outer-fold runs**, each with its own inner tuning (several thousand model fits in total). One further planned comparison, an adjustment method called Frisch-Waugh-Lovell (FWL), was not run because its exact procedure had not been defined before the experiments; we recorded it as blocked rather than inventing it afterwards.")
rows = [[NAMESC[m], "%d" % len(PDEF.grid(m)), {"LR": "C in {0.01, 0.1, 1, 10, 100}; penalty L1 or L2; class weight none or balanced", "LSVM": "C in {0.01, 0.1, 1, 10, 100}; class weight", "RSVM": "C in {0.1, 1, 10, 100}; gamma = m/d with m in {0.1, 1, 10}; class weight",
         "XGB": "24 fixed configurations drawn from a 256-point grid: learning rate {0.03, 0.1}, depth {2, 4}, trees {100, 300}, min child weight {1, 5}, subsample {0.7, 1}, column sample {0.3, 1}, lambda {1, 10}, positive weight {1, balanced}"}[m]] for m in PDEF.MODELS]
tab("Frozen hyperparameter search spaces.", ["Classifier", "Configurations", "Search space"], rows, "p{3.0cm}rp{10.8cm}", "small")
P("To understand the results better we also ran a set of clearly labelled **exploratory analyses** inside the same workspace (amendment A1): statistics of every acoustic feature, 66 feature-family ablations, feature importance, a map of how every hyperparameter configuration would have performed, an analysis of the speech tasks, a check of whether recording sessions can be recognised, and a timing benchmark. They use the same folds, never change the main results and are reported as exploratory.")

H(2, "Protecting the evaluation from leakage")
bullets(["**Frozen plan.** The protocol (representations, models, search spaces, folds, metrics, threshold) was written and fixed with a SHA-256 hash before any model was trained; nothing was changed after seeing results.",
         "**Gates on every run.** Each of the 280 runs was checked automatically: no speaker in both training and test, every test recording predicted exactly once, valid probabilities, correct labels and folds, unchanged protocol and split files. A run failing any check would have been quarantined; none failed.",
         "**Code tests.** Before the real runs, 25 tests on synthetic data proved, for example, that every fit sees only its own training rows and that random features give chance-level results while an informative feature gives near-perfect results.",
         "**Audit.** After the runs, %d further checks and an independent audit recomputed the features, the nested cross-validation and every metric with separate code; all agreed." % len(aud["checks"])])

# ================================================================================================ BASELINE AND EXTENSIONS
H(1, "Baseline and extensions", "baseline-ext")
# ================================================================================================ 8 BASELINE
H(2, "Baseline selection and evaluation", "baseline")
P("**Choice.** Frozen WavLM-large embeddings with logistic regression (section [[background]]). It is justified by the small number of people, by prior work with frozen embeddings for Parkinson's speech [11], by WavLM's strength as a general speech model [10] and by the convex, deterministic training of logistic regression.")
P("**Systematic evaluation.** The baseline went through exactly the same pipeline as every other configuration: the same five outer folds, the same nested tuning, the same threshold and metrics. The table shows its result on each held-out fold and overall.")
rows = []
for f in range(5):
    r = runs["FV1-WLM-ALL-LR-F%d" % f]; sl = r["speaker_level"]; hp = r["selected_hyperparameters"]
    rows.append(["Fold %d" % f, "%d" % r["n_test_speakers"], "C=%s, %s, class weight %s" % (hp["C"], hp["penalty"].upper(), hp["class_weight"] or "none"), f3(sl["auprc"]), f3(sl["auroc"]), f2(sl["sensitivity"]), f2(sl["specificity"]), "%d/%d" % (sl["fn"], sl["fp"])])
rows.append(["**Pooled (47 people)**", "47", "", f3(base.AUPRC), f3(base.AUROC), f2(base.sensitivity), f2(base.specificity), "%d/%d" % (base.FN, base.FP)])
tab("Baseline (WavLM-large + logistic regression) on each held-out fold and pooled. FN/FP = missed patients / false alarms.", ["Fold", "Test people", "Chosen configuration", "AUPRC", "AUROC", "Sens.", "Spec.", "FN/FP"], rows, "p{2.2cm}rp{4.4cm}rrrrr", "small")
P("Pooled over all 47 people, the baseline reaches **AUROC %s** (bootstrap 95%% interval %s to %s), **AUPRC %s**, **sensitivity %s** (exact interval %s to %s), **specificity %s** (exact interval %s to %s) and F1 %s. It misses one patient and raises no false alarm." % (
    f3(base.AUROC), f3(base.AUROC_ci_lo), f3(base.AUROC_ci_hi), f3(base.AUPRC), f2(bcp.sensitivity), f2(bcp.sens_cp_lo), f2(bcp.sens_cp_hi), f2(bcp.specificity), f2(bcp.spec_cp_lo), f2(bcp.spec_cp_hi), f3(base.F1)))
# ================================================================================================ 10 EXTENSIONS
H(2, "Implemented extensions and experiments", "extensions")
P("Beyond the baseline we implemented four kinds of extensions, all evaluated on the same people and folds:")
bullets(["**Other encoders**: wav2vec2-large-lv60 (English) and XLS-R 300M (multilingual), to see whether the result depends on WavLM.",
         "**Hand-crafted features**: 40 acoustic measurements, as an interpretable and very cheap alternative.",
         "**Hybrids**: eight combinations of encoders with each other and with the acoustic features.",
         "**Classifiers and feature selection**: linear SVM, RBF SVM and XGBoost; forward and backward selection of 10 acoustic features."])
P("**Did they improve on the baseline?** No extension improved on the baseline by more than the uncertainty (the numbers are in section [[results]]). This is not a failure of the extensions but a consequence of the ceiling: the baseline already makes a single error out of 47 people, so the largest possible gain is one or two people (about 2 to 4 points of F1), which is within the statistical noise of a 47-person test. The error analysis (section [[errors]]) adds a second explanation: almost all configurations fail on the same few people, so the remaining errors seem to come from these people's recordings rather than from the choice of model.")

H(2, "Baseline outputs and reproduction", "reproduction")
P("All original outputs are saved: the prediction files (one row per test recording, with the run, fold, configuration hash and probability), the per-run metric files with the full tuning log, and the run configurations. We confirmed that the baseline can be reproduced in three independent ways: (1) re-executing the saved configurations gave bit-identical predictions; (2) a separately written implementation of the whole nested cross-validation chose the same configuration in all five folds and produced the same predictions (largest difference %.1e); (3) re-executing all 70 logistic-regression runs with a single BLAS thread reproduced every selected configuration, probability and metric exactly. (With many BLAS threads, the L2-regularised solver can move probabilities in the fourth decimal place without changing any decision.)" % a3["independent_nested_cv_WLM_LR"]["max_abs_probability_difference"])

# ================================================================================================ RESULTS AND ANALYSIS
H(1, "Results and analysis", "results-analysis")
# ================================================================================================ 9 RESULTS
H(2, "Results", "results")
H(3, "Training, validation and held-out results")
tv = {r["condition"]: r for r in ra["train_val_test"]}; trs = {r["condition"]: r for r in ra["training_partition_scores"]}
rows = [[c.replace("FV1-", "").replace("-ALL-", " "), "%s / %s" % (f3(min(1.0, trs[c]["train_auprc_mean"])), f3(trs[c]["train_auroc_mean"])), "%s (SD %s)" % (f3(tv[c]["inner_val_auprc_mean"]), f3(tv[c]["inner_val_auprc_sd"])), "%s (SD %s)" % (f3(tv[c]["heldout_fold_auprc_mean"]), f3(tv[c]["heldout_fold_auprc_sd"]))] for c in tv]
tab("Training, inner-validation and held-out scores, kept separate (means over the five outer folds).", ["Configuration", "Training AUPRC / AUROC", "Inner-validation AUPRC", "Held-out AUPRC"], rows, "llll")
fig("train_val_test", "Training (grey), inner-validation (amber) and held-out (blue) speaker-level AUPRC for six representative configurations.", "0.8")
box("read", "Every model fits its own training speakers perfectly, which is expected when there are many more features than people. The useful comparison is between validation and held-out scores: they are both high and close to each other, so the tuning did not overfit the test folds. A perfect training score on its own tells us nothing about generalisation.")
H(3, "All 56 configurations")
fig("heatmap_auroc", "Pooled speaker-level AUROC of every representation (rows) and classifier (columns).", "0.58")
P("The heatmap shows the main result in one picture: almost every configuration reaches an AUROC between 0.97 and 1.00. The median over the 56 configurations is %s, and %d of them make at most one speaker error at the threshold 0.5. The clear exceptions are some RBF SVM configurations (for example WavLM with RBF SVM: %s), which we explain in section [[tuning]]. The full table with all metrics is in appendix [[app-full]]." % (f3(cm.AUROC.median()), int((cm.errors <= 1).sum()), f3(cm[cm.condition_id == "FV1-WLM-ALL-RSVM"].AUROC.iloc[0])))
H(3, "Comparison by feature combination")
rows = []
for r in ["AC", "WLM", "W2V", "XLSR"] + ["H%d" % i for i in range(1, 9)]:
    x = cm[(cm.representation == r) & (cm.feature_selection == "ALL")]
    rows.append([REPN[r], "%d" % sum(PDEF.DIMS[c] for c in PDEF.components(r)), f3(x.AUROC.median()), "%s to %s" % (f3(x.AUROC.min()), f3(x.AUROC.max())), f3(x.AUPRC.median()), "%d" % x.errors.min(), "%d" % x.errors.median()])
tab("Feature combinations ranked by their median AUROC over the four classifiers.", ["Representation", "Inputs", "Median AUROC", "Range", "Median AUPRC", "Fewest errors", "Median errors"], rows, "lrrrrrr")
fig("wp_forest_auroc", "AUROC with 95% bootstrap intervals for each representation, with logistic regression (left) and XGBoost (right).")
ex = hy[hy.ci_excludes_zero]
P("Combining representations did not help in a reliable way. Over all %d paired comparisons of a hybrid with each of its parts (same classifier, same people) the median change in AUROC was %s, ranging from %s to %s; the 95%% interval of the difference excluded zero in only %d comparisons (%d improvements, %d losses), and these are differences of about one person." % (
    len(hy), f3(hy.delta_A_minus_B.median()), f3(hy.delta_A_minus_B.min()), f3(hy.delta_A_minus_B.max()), len(ex), int((ex.delta_A_minus_B > 0).sum()), int((ex.delta_A_minus_B < 0).sum())))
fig("deltas", "Paired AUROC differences for all planned comparisons; red points have a 95% interval that excludes zero.", "0.7")
H(3, "Comparison by classifier")
rows = []
for m in PDEF.MODELS:
    d = cm[cm.model == m]; x = mv[mv.condition_A.str.endswith("-" + m)]
    rows.append([NAMESC[m], "%d" % len(d), f3(d.AUROC.median()), "%s to %s" % (f3(d.AUROC.min()), f3(d.AUROC.max())), "%d" % d.errors.median(), (f3(x.delta_A_minus_B.median()) if len(x) else "reference")])
tab("Classifiers compared over all representations.", ["Classifier", "Configurations", "Median AUROC", "Range", "Median errors", "Median difference to LR"], rows, "lrrrrr")
P("Logistic regression, the linear SVM and XGBoost perform almost identically; the RBF SVM is lower on average and much less stable. More complex classifiers did not help because the simple ones already separate the people nearly perfectly.")
H(3, "Person-level scores and ROC curves")
fig("speaker_scores", "Out-of-fold scores of all 47 people for three configurations. Red rings mark people on the wrong side of the threshold 0.5.")
fig("wp_roc", "Speaker-level ROC curves of six configurations.", "0.5")
box("read", "In most configurations the patients cluster near 1 and the controls near 0, with one patient (PD_005) consistently on the wrong side. The ROC curves rise almost vertically: at very low false-alarm rates the models already find nearly all patients.")

# ================================================================================================ 11 FEATURES
H(2, "Understanding the acoustic features", "features")
H(3, "Every feature on its own")
n90 = int((S0.AUROC_abs >= .9).sum())
fig("wp_univariate_features", "How well each of the 40 acoustic features separates the groups on its own (speaker averages). Bars: all recordings; black ticks: 16-kHz recordings only.", "0.75")
P("We first looked at each feature alone, averaged per speaker (a descriptive analysis on all 47 people, not used to choose anything). %d of the 40 features separate the groups with an AUROC of at least 0.90 by themselves, and all of them are spectral-shape features: the zero-crossing rate, the spectral centroid and the roll-off, each of which is **lower in the patients**. The separation stays as strong when only recordings made at 16 kHz are used, so it is not simply a side effect of resampling. The complete list with means, effect sizes and session effects is in appendix [[app-features]]." % n90)
fam = S0.groupby("family").agg(mean=("AUROC_abs", "mean"), best=("AUROC_abs", "max"), dpd=("eta2_date_within_PD", "mean"), dhc=("eta2_date_within_controls", "mean")).sort_values("mean", ascending=False)
rows = [[f.replace("_log", " (log)").replace("_hz", " (Hz)").replace("log_energy_db", "log energy (dB)").replace("rolloff85", "roll-off 85%"), f3(r["mean"]), f3(r["best"]), f2(r["dpd"]), f2(r["dhc"])] for f, r in fam.iterrows()]
tab("Feature families: average and best single-feature AUROC (direction-free), and the share of variance explained by the recording date within each group (eta-squared).", ["Family", "Mean AUROC", "Best AUROC", "Date effect within PD", "Date effect within controls"], rows, "lrrrr")
box("key", "The same features that separate patients from controls also differ strongly **between recording dates of the same group** (eta-squared up to %s within the controls). A feature that changes from one session to another inside a group is partly measuring the session. This is the first sign of the recording-condition problem discussed in section [[confound]]." % f2(S0.eta2_date_within_controls.max()))
H(3, "Which feature families are needed (ablation)")
fig("wp_ablation", "Exploratory ablations with the full nested protocol: one family alone (left), all families except one (middle) and one statistic of all families (right).")
P("We retrained logistic regression and XGBoost on subsets of the 40 features. Three findings stand out. First, no family is necessary: removing any single family leaves AUROC at %s or higher. Second, many small subsets are sufficient: the four zero-crossing statistics alone give %s with logistic regression, and a single band (0.5 to 1 kHz) gives %s with XGBoost. Third, loudness alone is weak for a linear model (%s) but strong for XGBoost (%s), which can use non-linear thresholds on the quiet frames. In short, the information is spread over many redundant features (46 pairs of features have a correlation of at least 0.9)." % (
    f3(S1[S1.subset.str.startswith("drop_") & ~S1.subset.str.startswith("drop_stat_")].AUROC.min()), f3(z("only_zcr", "LR").AUROC), f3(z("only_band1_log", "XGB").AUROC), f3(z("only_log_energy_db", "LR").AUROC), f3(z("only_log_energy_db", "XGB").AUROC)))
H(3, "Forward and backward feature selection")
fsr = cm[cm.representation == "AC"]
P("In the main experiment we also selected exactly 10 of the 40 features inside each training part: forward selection adds the feature that improves a 3-fold speaker-level AUPRC most, one at a time; backward elimination starts with all 40 and removes the least useful one at a time. The selector is a logistic regression with C = 1, so the selected features do not depend on the final classifier. Selection gave AUROC between %s and %s (forward) and %s and %s (backward), against %s to %s with all features; the median difference to all features was %s, and no difference was outside the uncertainty." % (
    f3(fsr[fsr.feature_selection == "FORWARD"].AUROC.min()), f3(fsr[fsr.feature_selection == "FORWARD"].AUROC.max()), f3(fsr[fsr.feature_selection == "BACKWARD"].AUROC.min()), f3(fsr[fsr.feature_selection == "BACKWARD"].AUROC.max()),
    f3(fsr[fsr.feature_selection == "ALL"].AUROC.min()), f3(fsr[fsr.feature_selection == "ALL"].AUROC.max()), f3(fsc.delta_A_minus_B.median())))
sel_rows = []
for fs in ["FORWARD", "BACKWARD"]:
    cnt = Counter()
    for f in range(5):
        cnt.update(runs["FV1-AC-%s-LR-F%d" % (fs, f)]["selected_features"])
    sel_rows.append([fs.lower(), "%d" % sum(1 for v in cnt.values() if v == 5), "%d" % len(cnt), ", ".join("%s (%d)" % (k.replace("ac_", "").replace("_", " "), v) for k, v in cnt.most_common(6))])
tab("Stability of the selected features over the five outer folds (selected in all 5 folds / distinct features ever selected / the six most frequent).", ["Method", "In all 5", "Distinct", "Most frequently selected (folds)"], sel_rows, "lrrp{11.0cm}", "small")
P("The selected sets change from fold to fold, which is typical when many features carry the same information: different but equivalent features win in different training sets.")
H(3, "Feature importance", "importance")
fig("wp_importance_acoustic", "Importance in the acoustic models: increase in log-loss when a family or a statistic is shuffled in the test folds (left, middle), and the XGBoost gain of the ten most used features (right).")
lr = S2[S2.condition == "FV1-AC-ALL-LR"]; xg = S2[S2.condition == "FV1-AC-ALL-XGB"]
top_xg = xg[xg.group.str.startswith("ac_")].sort_values("gain_share_mean", ascending=False).head(3)
P("We measured importance in three ways on the fitted models of the main runs: the size of the logistic-regression coefficients, the share of XGBoost's total gain, and permutation importance (how much worse the predictions become when one feature or one group of features is randomly shuffled within the test folds). Logistic regression spreads its weight over many correlated features; its most important families are loudness and the spectral centroid, and some coefficients even change sign between folds, a sign of redundancy. XGBoost is much more concentrated: three **10th-percentile** features (%s) carry %s of its total gain, and shuffling the 10th-percentile statistic of all families hurts it more than shuffling any family." % (", ".join(g.replace("ac_", "") for g in top_xg.group), pct(top_xg.gain_share_mean.sum())))
box("plain", "The 10th percentile of a frame measurement describes the quietest tenth of the recording: pauses, breathing and the background noise of the room. When a model relies mostly on these values, it is listening more to the recording conditions than to the voice.")
fig("wp_block_importance", "Which part of a hybrid carries the model: logistic regression (share of coefficient size) versus XGBoost (share of gain).", "0.8")
bl = S2b
P("In the hybrids the two classifier families disagree about what matters. Logistic regression puts %s to %s of its weight on the encoder part, whereas XGBoost puts %s to %s of its gain on the 40 acoustic features. When the three encoders are combined, logistic regression relies mostly on WavLM (%s of the weight), then XLS-R (%s), then wav2vec2 (%s); shuffling the wav2vec2 part changes nothing, so it is redundant once the other two are present." % (
    pct(bl[(bl.condition.isin(["FV1-H1-ALL-LR", "FV1-H2-ALL-LR", "FV1-H3-ALL-LR"])) & (bl.group != "AC")].coef_mass_share.min()), pct(bl[(bl.condition.isin(["FV1-H1-ALL-LR", "FV1-H2-ALL-LR", "FV1-H3-ALL-LR"])) & (bl.group != "AC")].coef_mass_share.max()),
    pct(bl[(bl.condition.isin(["FV1-H1-ALL-XGB", "FV1-H2-ALL-XGB", "FV1-H3-ALL-XGB"])) & (bl.group == "AC")].gain_share.min()), pct(bl[(bl.condition.isin(["FV1-H1-ALL-XGB", "FV1-H2-ALL-XGB", "FV1-H3-ALL-XGB"])) & (bl.group == "AC")].gain_share.max()),
    pct(bl[(bl.condition == "FV1-H7-ALL-LR") & (bl.group == "WLM")].coef_mass_share.iloc[0]), pct(bl[(bl.condition == "FV1-H7-ALL-LR") & (bl.group == "XLSR")].coef_mass_share.iloc[0]), pct(bl[(bl.condition == "FV1-H7-ALL-LR") & (bl.group == "W2V")].coef_mass_share.iloc[0])))

# ================================================================================================ 12 TUNING
H(2, "Hyperparameter tuning", "tuning")
tie = S3s.groupby("model").agg(runs=("run_id", "size"), tied=("n_tied_at_best", lambda x: int((x > 1).sum())), med=("n_tied_at_best", "median"))
P("Hyperparameters were tuned in every run by the nested procedure of section [[training]]. Looking at all 280 tuning logs taught us something important about tuning on small data.")
P("**The tuning objective saturates.** Because the inner folds contain only about 12 people each, many configurations reach the same, often perfect, inner score. In %s, more than one configuration tied for the best score. The choice then falls back to the order of the list, so in practice tuning mostly chose the first, most strongly regularised configurations: logistic regression usually ended with C = 0.1 or 0.01 and an L2 penalty, the linear SVM with C = 0.01, the RBF SVM with C = 1 and a narrow kernel, and XGBoost with the first configuration of its list." % "; ".join("%s %d of %d runs (median %g tied configurations)" % (NAMES[m], tie.loc[m, "tied"], tie.loc[m, "runs"], tie.loc[m, "med"]) for m in PDEF.MODELS))
sel_rows = []
for m in PDEF.MODELS:
    vc = S3s[S3s.model == m].selected.value_counts()
    for cfg, n in vc.head(3).items():
        sel_rows.append([NAMESC[m], ", ".join("%s = %s" % (k, v) for k, v in json.loads(cfg).items()), "%d of %d" % (n, int((S3s.model == m).sum()))])
tab("The three most frequently chosen configurations of each classifier (over 70 runs each).", ["Classifier", "Configuration", "Runs"], sel_rows, "lp{11.2cm}r", "small")
g = S3L.groupby(["representation", "model"]).pooled_auroc.agg(["min", "median", "max"]).reset_index()
g["nested"] = [cm[(cm.representation == r) & (cm.model == m) & (cm.feature_selection == "ALL")].AUROC.iloc[0] for r, m in zip(g.representation, g.model)]
worst = g.assign(gap=g["median"] - g["nested"]).sort_values("gap", ascending=False).iloc[0]
fig("wp_tuning", "Left: number of configurations tied at the best inner score in each run. Right: AUROC of every configuration of the grid on the outer folds (grey range, black median) and the configuration chosen by nested tuning (coloured).")
P("**What the grid looks like.** As an exploratory check we trained every configuration of the grid on each outer training part and scored it on the test fold (this is only for description; it was never used to choose anything). For the linear models almost every configuration performs alike. The RBF SVM is different: its results range widely (for wav2vec2 from %s to %s), and the configuration chosen by nested tuning can be a poor one. The clearest case is %s with the %s, where nested tuning chose a configuration with AUROC %s although the median configuration of the grid reaches %s." % (
    f3(g[(g.representation == "W2V") & (g.model == "RSVM")]["min"].iloc[0]), f3(g[(g.representation == "W2V") & (g.model == "RSVM")]["max"].iloc[0]), REPN[worst.representation], NAMES[worst.model], f3(worst.nested), f3(worst["median"])))
box("read", "The weak RBF SVM results are a tuning problem, not proof that curved boundaries are worse: with saturated inner scores, the tie-break picked a narrow kernel that does not generalise. On small datasets it is safer to use simple, strongly regularised defaults and to spend data on external testing rather than on tuning.")

# ================================================================================================ 13 TASKS
H(2, "Which speech tasks carry the signal", "tasks")
fig("wp_tasks", "Speaker-level AUROC when each person's score uses only one family of tasks (the same out-of-fold models, no retraining).", "0.92")
P("Using the existing out-of-fold predictions, we recomputed each person's score from one task family at a time. Every family on its own still separates the groups well (speaker AUROC of at least %s in every case). The phrase (about 30 s) and the word list (about 56 s) reach 1.000 with wav2vec2 and XLS-R. Sustained vowels are the least reliable family when single recordings are judged, which matters because they make up 10 of a person's 16 recordings." % f3(S7[S7.task_family != "all tasks"].speaker_auroc.min()))

# ================================================================================================ 14 ERRORS
H(2, "Error and failure-case analysis", "errors")
det = ra["errors"]["detail"]
P("Errors are rare and concentrated. Across all 56 configurations, every missed patient is the same person: **PD_005** is missed in %d of the 56 configurations. Her recordings are not split between clearly 'healthy' and clearly 'Parkinson's' tasks; instead almost all of them sit just below the threshold (WavLM with logistic regression gives values between %s and %s for her 16 recordings), so the models are uncertain about her as a whole. The corpus gives no information on disease stage or medication that could explain this." % (
    dem["false_negative_events_by_speaker"]["PD_005"], f2(min(det["PD_005"]["FV1-WLM-ALL-LR"]["by_task"].values())), f2(max(det["PD_005"]["FV1-WLM-ALL-LR"]["by_task"].values()))))
P("False alarms come from a few controls. The four controls that the corpus spreadsheet flags for poor recording quality or reading difficulty produce **%d of the %d** false-alarm events counted over all configurations, although they are only 4 of 22 controls. The two most frequent false alarms, HC_001 and HC_020, are both flagged; HC_020 has no reading-passage or phrase recordings at all." % (dem["false_positive_events_from_flagged_controls"], dem["false_positive_events_total"]))
fig("error_tasks", "Recording probabilities (WavLM + logistic regression) of the three most frequently misclassified people.", "0.95")
fam = ra["errors"]["recording_error_by_task_family"]["FV1-WLM-ALL-LR"]
rows = [[k, "%d" % v["n"], "%d" % v["wrong"], pct(v["rate"])] for k, v in fam.items() if k != "_all"] + [["all recordings", "%d" % fam["_all"]["n"], "%d" % fam["_all"]["wrong"], pct(fam["_all"]["rate"])]]
tab("Recording-level errors of the baseline by task family (a person's decision uses the average of all their recordings).", ["Task family", "Recordings", "Wrong", "Error rate"], rows, "lrrr")
box("read", "Two failure modes appear: (1) a patient whose recordings all look borderline, and (2) controls whose recordings have known quality problems. A real product should therefore check recording quality and abstain rather than give a confident answer on poor recordings.")

# ================================================================================================ 15 SENSITIVITY
H(2, "Sensitivity and ablation analysis", "sensitivity")
P("The table brings together how much the result changes when one design choice is changed while everything else is kept fixed.")
rows = [["Representation", "4 single representations and 8 hybrids", "median AUROC per representation between %s and %s" % (f3(min(cm[(cm.representation == r) & (cm.feature_selection == "ALL")].AUROC.median() for r in PDEF.BASE_REPS + list(PDEF.HYBRIDS))), f3(max(cm[(cm.representation == r) & (cm.feature_selection == "ALL")].AUROC.median() for r in PDEF.BASE_REPS + list(PDEF.HYBRIDS)))), "small"],
        ["Classifier", "LR, linear SVM, RBF SVM, XGBoost", "median difference to LR: " + "; ".join("%s %s" % (NAMES[m], f3(mv[mv.condition_A.str.endswith('-' + m)].delta_A_minus_B.median())) for m in PDEF.MODELS[1:]), "small except RBF SVM"],
        ["Feature selection", "all 40 versus forward/backward 10", "median difference %s" % f3(fsc.delta_A_minus_B.median()), "none detectable"],
        ["Feature families", "drop one family / keep one family", "drop: AUROC at least %s; keep ZCR only (LR): %s" % (f3(S1[S1.subset.str.startswith("drop_") & ~S1.subset.str.startswith("drop_stat_")].AUROC.min()), f3(z("only_zcr", "LR").AUROC)), "information is redundant"],
        ["Statistics", "keep only mean, SD, p10 or p90", "AUROC at least %s" % f3(S1[S1.subset.str.startswith("only_stat_")].AUROC.min()), "small"],
        ["Hyperparameters", "every grid configuration on the outer folds", "linear models stable; RBF SVM from %s to %s (wav2vec2)" % (f3(g[(g.representation == "W2V") & (g.model == "RSVM")]["min"].iloc[0]), f3(g[(g.representation == "W2V") & (g.model == "RSVM")]["max"].iloc[0])), "large for RBF SVM only"],
        ["Speech task", "score from one task family only", "speaker AUROC at least %s" % f3(S7[S7.task_family != "all tasks"].speaker_auroc.min()), "small"],
        ["Numerical environment", "1 versus many BLAS threads", "probabilities change by at most 4e-4; no decision changes", "none"]]
tab("Sensitivity of the results to single design choices.", ["Design choice", "What was varied", "Effect", "Size of effect"], rows, "p{2.8cm}p{4.4cm}p{6.0cm}p{2.4cm}", "small")
P("The results are remarkably stable to every choice we could vary inside the dataset. What we could **not** vary is the recording situation, and the next section shows why that matters.")

# ================================================================================================ 16 CONFOUND
H(2, "The recording-condition problem", "confound")
P("Because each recording session contains only one group, a classifier could separate the groups by recognising the *session* instead of the *disease*. We looked for evidence of this in three ways.")
H(3, "Can the models recognise the session inside one group?")
s6 = S6[S6.model == "LR"]; t1 = s6[s6.task.str.startswith("T1")]; t2 = s6[s6.task.str.startswith("T2")]; t3 = s6[s6.task.str.startswith("T3")]
fig("wp_session", "Can the same representations tell recording sessions apart within one group, for people they have not seen? (logistic regression, nested, speaker-disjoint)", "0.9")
P("We trained the same pipeline to predict the recording date *within* a single group, again always testing on unseen people. Among the controls, early-March and late-March sessions could be told apart with an AUROC between %s and %s. Among the patients, the 2016 sessions (44.1 kHz) and the 2017 sessions (16 kHz) were separated with an AUROC between %s and %s. Only the two 2017 patient sessions were hard to separate (%s to %s). So the representations clearly carry information about the session, even when the diagnosis is the same." % (f3(t2.unit_auroc.min()), f3(t2.unit_auroc.max()), f3(t3.unit_auroc.min()), f3(t3.unit_auroc.max()), f3(t1.unit_auroc.min()), f3(t1.unit_auroc.max())))
H(3, "The two groups' audio files were produced differently")
P("Looking at the raw samples, we found exact digital silence (samples that are exactly zero) in different places for the two groups. %d of %d control recordings begin with a block of about 6 ms of exact zeros, which never occurs in patient recordings; %d of %d patient recordings (from the 2016 sessions) end with at least half a second of exact zeros, which never occurs in control recordings. The leading zero block alone would separate the speakers with an AUROC of %s. The 6-ms block is too short to change the 25-ms frame features much, but the trailing silence is not: in the 2016 patient sessions it adds frames of pure digital silence, which directly change the quietest-frame (10th-percentile) statistics. Both details show that the files of the two groups went through different recording or export processes." % (
    S9["recordings_with_leading_zero_block_ge_5ms"]["0"], S9["recordings_per_label"]["0"], S9["recordings_with_trailing_zeros_ge_0.5s"]["1"], S9["recordings_per_label"]["1"], f3(1 - S9["speaker_AUROC_leading_zero_ms"])))
H(3, "What the models rely on")
P("Section [[importance]] showed that the most important features for XGBoost describe the quietest frames, and section [[features]] that many features vary strongly between sessions of the same group. Together with the session-recognition results, this means that a large part of the separation can be explained by recording conditions.")
box("key", "The high scores are genuine for this dataset, but this dataset cannot tell us how much of them comes from the disease and how much from the recording sessions. Some signal may be real (the spectral-shape differences remain in 16-kHz-only data, and speech changes in Parkinson's disease are well documented [2]), but its size is unknown. Only recordings in which patients and controls are recorded under the same conditions can answer this.")

# ================================================================================================ 17 DISCUSSION
H(1, "Discussion", "discussion")
H(2, "Significance of the results")
P("Technically, the project shows that a carefully controlled pipeline - frozen pretrained speech models or even 40 simple acoustic measurements, with a simple classifier - separates people with Parkinson's disease from age-comparable controls in this corpus almost perfectly, and that this result is stable across encoders, classifiers, feature subsets, tasks and hyperparameters. Methodologically, the project shows how to evaluate such a system honestly: speaker-disjoint nested cross-validation, a frozen plan, automatic integrity checks and an independent audit. Scientifically, the most important result is negative: in this dataset the diagnosis cannot be separated from the recording session, so near-perfect scores are not evidence that speech reveals the disease.")
H(2, "Risks")
bullets(["**Shortcut learning.** The models may learn the session, not the disease (section [[confound]]); this is the main risk.",
         "**Small sample.** With 47 people, one person changes a speaker-level metric by about two percentage points, and fold-level results (9 or 10 people per fold) are very noisy.",
         "**Selection optimism.** Choosing the best of 56 configurations after seeing their scores would overstate performance; we therefore present the baseline and the full distribution rather than the best configuration.",
         "**Label and data quality.** Clinical diagnosis itself is not perfect [3], four controls have documented recording problems, and the corpus has no information on disease stage or medication."])
H(2, "Sensitivity of the results to input changes")
P("Within the dataset, the results hardly change when the representation, classifier, feature subset, statistic, task or hyperparameters change (section [[sensitivity]]). But the session-recognition experiment shows that the inputs carry strong recording-condition information. We could not test the most important input change, a different recording setup (microphone, room, device or language), because the corpus does not contain it. Results on new recordings could therefore be much lower.")
H(2, "Limitations")
bullets(["One corpus, one language (Italian), one site and 47 people.",
         "Recording session and diagnosis are perfectly aligned; part of the patient audio was recorded at a different sample rate.",
         "Sex balance differs between the groups (64% of patients and 45% of controls are men), and the patients' ages are more spread out (40 to 80) than the controls' (60 to 77).",
         "No trained neural network: the deep models are used only as frozen feature extractors.",
         "The FWL adjustment could not be run because it had not been defined in advance."])

# ================================================================================================ 18 FUTURE
H(1, "Future directions", "future")
bullets(["**Test on independent data without retraining.** Freeze the current pipeline and apply it to a different corpus, for example NeuroVoz, a Castilian Spanish Parkinson's speech corpus with about 110 speakers [19]. Decide the success criterion in advance (for example an AUROC whose lower 95% bound is above 0.80).",
         "**Collect or use recordings with matched conditions.** Patients and controls should be recorded in the same sessions, rooms and devices, interleaved in time, with clinical information (disease stage, medication). A classifier that sees only the background must then perform at chance level; only after that check is a disease score meaningful.",
         "**Remove recording-condition cues during training.** Options include trimming or masking pauses, matching the background noise across recordings, and training with a penalty for predicting the recording session.",
         "**Task-aware scoring and quality control.** Weight or select tasks using inner validation only (sustained vowels are the least reliable), and add a recording-quality check that abstains on poor recordings.",
         "**Trained deep models once the data allow it.** With a larger, matched dataset, fine-tuning the upper layers of the speech encoders or training a small network on spectrograms becomes realistic."], enum=True)

# ================================================================================================ 19 CONCLUSION
H(1, "Conclusion", "conclusion")
P("We built and evaluated, under one frozen and audited plan, 56 ways of detecting Parkinson's disease from speech, combining hand-crafted acoustic features and three frozen pretrained speech models with four classifiers. Tested always on unseen people, almost all of them separate the 25 patients from the 22 controls in the corpus nearly perfectly; the WavLM-plus-logistic-regression baseline reaches AUROC %s with one missed patient, and no extension improves on it beyond the uncertainty. The detailed analyses explain why: the information is highly redundant, the simplest acoustic measurements are enough, the models rely heavily on the quiet background of the recordings, and the recording sessions - which coincide with the diagnosis - can be recognised from the same features. We therefore met our technical objectives and our objective of honest measurement, but the central question - does the model hear the disease? - cannot be answered with this dataset. The next step is a frozen test on independent recordings in which patients and controls share the same recording conditions." % f3(base.AUROC))

# ================================================================================================ APPENDICES
A("\\appendix")
H(1, "Full results of the 56 configurations", "app-full")
A("\\begin{landscape}")
rows = []
for c in cm.itertuples():
    e = S5[S5.condition_id == c.condition_id].iloc[0]
    rows.append([REPN[c.representation] + ("" if c.feature_selection == "ALL" else " / " + c.feature_selection.lower()), NAMESC[c.model], f3(c.AUPRC), "%s [%s-%s]" % (f3(c.AUROC), f3(c.AUROC_ci_lo), f3(c.AUROC_ci_hi)),
                 "%s (%s-%s)" % (f2(e.sensitivity), f2(e.sens_cp_lo), f2(e.sens_cp_hi)), "%s (%s-%s)" % (f2(e.specificity), f2(e.spec_cp_lo), f2(e.spec_cp_hi)), f3(c.F1), f3(c.balanced_accuracy), "%d/%d" % (c.FN, c.FP), f3(c.sample_AUROC)])
tab("All configurations, pooled over the 47 speakers. AUROC with bootstrap interval; sensitivity and specificity with exact interval; FN/FP = missed patients / false alarms; last column = recording-level AUROC.",
    ["Representation", "Classifier", "AUPRC", "AUROC [CI]", "Sensitivity", "Specificity", "F1", "Bal. acc.", "FN/FP", "Rec. AUROC"], rows, "p{3.6cm}p{2.2cm}rp{3.3cm}p{2.3cm}p{2.3cm}rrrr", "footnotesize", wide=True, minw=1.15)
A("\\end{landscape}")
H(1, "The 40 acoustic features one by one", "app-features")
A("\\begin{landscape}")
rows = [[r.feature.replace("ac_", ""), r.direction, f3(r.mean_PD), f3(r.mean_control), f3(r.AUROC_class), f3(r.AUROC_class_native16k_only), f2(r.cohens_d), f2(r.eta2_date_within_PD), f2(r.eta2_date_within_controls)] for r in S0.itertuples()]
tab("Speaker-level statistics of every acoustic feature (AUROC above 0.5 means higher in PD). Cohen's d: standardised mean difference. Date effect: share of variance explained by the recording date within a group.",
    ["Feature", "Direction", "Mean PD", "Mean controls", "AUROC", "AUROC (16 kHz only)", "Cohen d", "Date effect PD", "Date effect controls"], rows, "p{4.2cm}lrrrrrrr", "footnotesize", wide=True)
A("\\end{landscape}")
H(1, "Feature-family ablations", "app-ablation")
A1 = S1[~S1.subset.isin(["only_energy", "no_level_features"])]
rows = [[r.subset.replace("only_stat_", "only statistic: ").replace("drop_stat_", "without statistic: ").replace("only_", "only: ").replace("drop_", "without: "), NAMESC[r.model], "%d" % r.n_features, f3(r.AUROC), f3(r.AUPRC), f3(r.F1), "%d/%d" % (r.FN, r.FP)] for r in A1.itertuples()]
tab("Exploratory ablations with the full nested protocol (same folds).", ["Feature subset", "Classifier", "Features", "AUROC", "AUPRC", "F1", "FN/FP"], rows, "p{5.2cm}lrrrrl", "scriptsize")
H(1, "Integrity and audit checks", "app-audit")
rows = [[str(i), c["check"], c["status"]] for i, c in enumerate(aud["checks"], 1)]
tab("Out-of-fold leakage and integrity checks run after the experiments.", ["#", "Check", "Result"], rows, "rp{13.2cm}l", "scriptsize")
P("The independent audit additionally re-derived all labels and speakers from the raw folders, re-extracted embeddings in a fresh process (largest difference %.1e), re-implemented the acoustic features and the nested cross-validation, recomputed all 336 reported metric values (no disagreement) and re-ran all logistic-regression runs. It found no bug that changes a prediction or a metric; it corrected documentation (for example, the two-digit number in file names is the year of birth) and tightened one automatic check." % max(x["max_abs_diff"] for v in a1.values() for x in v))
H(1, "Where to find everything", "app-files")
rows = [["Protocol and its hash", "FINAL_V1_PROTOCOL.md, configs/FINAL_V1/protocol.json, FREEZE_RECORD.json, PROTOCOL_ERRATA.md"],
        ["Data and splits", "DATA_INVENTORY.csv, DATA_AUDIT.md, manifests/SAMPLE_MANIFEST.csv, SPLIT_MANIFEST.csv"],
        ["Features", "features/acoustic_features.csv, features/*_embeddings.npy, features/manifests/"],
        ["Predictions and metrics", "predictions/FINAL_V1/ (280 files), metrics/FINAL_V1/ (run files, condition_metrics.csv, paired_comparisons.csv)"],
        ["Registry and logs", "EXPERIMENT_REGISTRY.csv, FAILURE_LOG.csv, DECISION_LOG.md, logs/FINAL_V1/"],
        ["Audit", "OOF_LEAKAGE_AUDIT.md, audit_opus/OPUS_AUDIT.md, tests/test_pipeline.py"],
        ["Exploratory analyses", "supplementary/SUPPLEMENTARY_REGISTRY.csv, supplementary/results/, supplementary/code/"],
        ["Code", "src/final_v1/ (pipeline.py, protocol_defs.py, splits.py, acoustic.py, extract_embeddings.py, runner.py, ...)"]]
tab("Files in the workspace Project4a_FINAL_V1_ISOLATED.", ["What", "Where"], rows, "p{3.6cm}p{12.2cm}", "small")
H(1, "Glossary", "app-glossary")
rows = [["AUROC", "Probability that a random patient scores higher than a random control; 0.5 = guessing, 1 = perfect."],
        ["AUPRC", "Area under the precision-recall curve; guessing level = share of patients (0.53 here)."],
        ["Speaker-disjoint", "No person has recordings in both the training and the test data of a fold."],
        ["Nested cross-validation", "Hyperparameters are chosen with an inner cross-validation inside the training part of each outer fold."],
        ["Embedding", "A fixed-length vector of numbers that a neural network uses to describe an input."],
        ["Frozen model", "A pretrained network whose weights are not changed; it is only used to compute embeddings."],
        ["Self-supervised learning", "Training on unlabelled data by solving an artificial task, such as predicting masked parts of speech."],
        ["Bootstrap interval", "Uncertainty interval obtained by resampling the people with replacement many times."],
        ["Clopper-Pearson interval", "Exact confidence interval for a proportion such as sensitivity."],
        ["Confounding", "When a third factor (here the recording session) is linked to both the input and the label."],
        ["Permutation importance", "How much a model's predictions get worse when one feature is randomly shuffled."],
        ["10th percentile (p10)", "The value below which the quietest tenth of a recording's frames lie; mostly pauses and background."]]
tab("Terms used in this report.", ["Term", "Meaning"], rows, "p{3.6cm}p{12.2cm}", "small")

refs = ["Global, regional, national epidemiology and trends of Parkinson's disease from 1990 to 2021: findings from the Global Burden of Disease Study 2021; Frontiers in Aging Neuroscience, 2025. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11757241/",
        "M. Trail et al., Speech treatment for Parkinson's disease, NeuroRehabilitation 20(3), 2005. https://content.iospress.com/articles/neurorehabilitation/nre00283",
        "A. Raty et al., Stability and accuracy of a diagnosis of Parkinson disease over 10 years, Neurology, 2025. https://research.utu.fi/converis/portal/detail/Publication/491935225",
        "C. Quan, K. Ren, Z. Luo, Z. Chen, Y. Ling, End-to-end deep learning approach for Parkinson's disease detection from speech signals, Biocybernetics and Biomedical Engineering 42(2):556-574, 2022. doi:10.1016/j.bbe.2022.04.002",
        "A. Iyer, A. Kemp et al., A machine learning method to process voice samples for identification of Parkinson's disease, Scientific Reports 13:20615, 2023. doi:10.1038/s41598-023-47568-w",
        "Q. Wang, Y. Fu, B. Shao, L. Chang, K. Ren, Z. Chen, Y. Ling, Early detection of Parkinson's disease from multiple signal speech: based on Mandarin language dataset, Frontiers in Aging Neuroscience 14:1036588, 2022. doi:10.3389/fnagi.2022.1036588",
        "B. Egbo, Z. Nigmetolla, N. A. Khan, P. K. Jamwal, Explainable machine learning for early detection of Parkinson's disease in aging populations using vocal biomarkers, Frontiers in Aging Neuroscience 17:1672971, 2025. doi:10.3389/fnagi.2025.1672971",
        "A. Baevski, H. Zhou, A. Mohamed, M. Auli, wav2vec 2.0: a framework for self-supervised learning of speech representations, NeurIPS 2020.",
        "A. Babu et al., XLS-R: self-supervised cross-lingual speech representation learning at scale, Interspeech 2022. arXiv:2111.09296",
        "S. Chen et al., WavLM: large-scale self-supervised pre-training for full stack speech processing, IEEE Journal of Selected Topics in Signal Processing 16(6):1505-1518, 2022. doi:10.1109/JSTSP.2022.3188113",
        "O. Klempir, R. Krupicka, Analyzing wav2vec 1.0 embeddings for cross-database Parkinson's disease detection and speech features extraction, Sensors 24(17):5520, 2024. doi:10.3390/s24175520",
        "J. L. Crawford, Linguistic changes in spontaneous speech for detecting Parkinson's disease using large language models, PLOS Digital Health 4(2):e0000757, 2025. doi:10.1371/journal.pdig.0000757",
        "E. Vaiciukynas, A. Verikas, A. Gelzinis, M. Bacauskiene, Detecting Parkinson's disease from sustained phonation and speech signals, PLoS ONE 12(10):e0185613, 2017. doi:10.1371/journal.pone.0185613",
        "G. Dimauro, F. Girardi, Italian Parkinson's Voice and Speech, IEEE DataPort, 2019. doi:10.21227/aw6b-tg17",
        "G. Dimauro, V. Di Nicola, V. Bevilacqua, D. Caivano, F. Girardi, Assessment of speech intelligibility in Parkinson's disease using a speech-to-text system, IEEE Access 5:22199-22208, 2017. doi:10.1109/ACCESS.2017.2762475",
        "R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, C.-J. Lin, LIBLINEAR: a library for large linear classification, Journal of Machine Learning Research 9:1871-1874, 2008.",
        "C.-C. Chang, C.-J. Lin, LIBSVM: a library for support vector machines, ACM Transactions on Intelligent Systems and Technology 2(3):27, 2011.",
        "T. Chen, C. Guestrin, XGBoost: a scalable tree boosting system, KDD 2016. doi:10.1145/2939672.2939785",
        "L. Mendes-Laureano et al., NeuroVoz: a Castillian Spanish corpus of parkinsonian speech, Scientific Data 11:1367, 2024. https://zenodo.org/records/10777657",
        "F. Pedregosa et al., Scikit-learn: machine learning in Python, Journal of Machine Learning Research 12:2825-2830, 2011."]
A("\\begin{thebibliography}{99}\\setlength{\\itemsep}{1pt}\\small")
A("".join("\\bibitem{r%d} %s\n" % (i + 1, esc(r)) for i, r in enumerate(refs)) + "\\end{thebibliography}")

PRE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.1cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype}
\usepackage{booktabs,longtable,float,graphicx,amsmath,amssymb,array,enumitem,xcolor,tcolorbox,pdflscape}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black,citecolor=blue!50!black]{hyperref}
\setlist{itemsep=2pt,topsep=3pt}\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}\linespread{1.08}
\usepackage{titlesec}\titleformat{\section}{\Large\bfseries}{\thesection.}{.5em}{}\titleformat{\subsection}{\large\bfseries}{\thesubsection}{.5em}{}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\fancyfoot[C]{\small\thepage}\fancyhead[L]{\small Detecting Parkinson's disease from speech}\fancyhead[R]{\small FINAL\_V1 project report}\renewcommand{\headrulewidth}{.3pt}
\begin{document}
\begin{center}{\LARGE\bfseries Detecting Parkinson's Disease from Speech\par}\vspace{6pt}
{\Large Final project report (FINAL\_V1, study edition)\par}\vspace{6pt}
{\normalsize CSE 431 Machine Learning and Deep Learning, topic 4a: detecting neurodegenerative disorders from speech\par}\vspace{3pt}
{\small Evidence: frozen protocol FINAL\_V1 (SHA-256 %s\ldots), 280 runs, independent audit \textperiodcentered{} %s\par}\end{center}
\vspace{4pt}\hrule\vspace{6pt}
\tableofcontents
\newpage
""" % (rec["protocol_json_sha256"][:16], esc(now_utc()[:10]))


def build():
    body = "\n".join(B)
    body = re.sub(r"([Ss]ection|[Ss]ections) \[\[([\w-]+)\]\]", lambda m: "%s~\\ref{sec:%s}" % (m.group(1), m.group(2)), body)
    body = re.sub(r"([Aa]ppendix) \[\[([\w-]+)\]\]", lambda m: "%s~\\ref{sec:%s}" % (m.group(1), m.group(2)), body)
    body = re.sub(r"table \[\[([\w-]+)\]\]", lambda m: "the table below", body)
    body = body.replace("\\label{tab-dims}\n", "")
    if "[[" in body:
        print("unresolved reference token:", body[body.index("[["):body.index("[[") + 40]); sys.exit(9)
    build_dir = OUT / "_build"; build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "PROJECT_REPORT_FINAL_V1.tex").write_text(PRE + body + "\n\\end{document}\n", encoding="utf8")
    for d in FIGDIRS:
        for fn in d.glob("*.pdf"):
            shutil.copy2(fn, build_dir / fn.name)
    for _ in range(3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "PROJECT_REPORT_FINAL_V1.tex"], cwd=build_dir, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); sys.exit(8)
    shutil.copy2(build_dir / "PROJECT_REPORT_FINAL_V1.pdf", OUT / "PROJECT_REPORT_FINAL_V1.pdf")
    shutil.copy2(build_dir / "PROJECT_REPORT_FINAL_V1.tex", OUT / "PROJECT_REPORT_FINAL_V1.tex")
    srcfig = OUT / "latex_source"; srcfig.mkdir(exist_ok=True)
    for fn in build_dir.glob("*.pdf"):
        if fn.name != "PROJECT_REPORT_FINAL_V1.pdf":
            shutil.copy2(fn, srcfig / fn.name)
    shutil.copy2(build_dir / "PROJECT_REPORT_FINAL_V1.tex", srcfig / "PROJECT_REPORT_FINAL_V1.tex")
    shutil.rmtree(build_dir)
    print("wrote", OUT / "PROJECT_REPORT_FINAL_V1.pdf")


build()
