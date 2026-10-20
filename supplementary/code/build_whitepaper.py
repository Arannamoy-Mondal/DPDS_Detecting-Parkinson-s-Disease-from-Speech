"""Build the venture-capital whitepaper (LaTeX -> PDF). Every number is read from verified FINAL_V1 files, the audit evidence or the
supplementary results; market facts are cited to external sources listed in the references."""
import json, re, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
from make_pdf import esc

OUT = ROOT / "reports/FINAL_V1"
MET = ROOT / "metrics/FINAL_V1"
f3 = lambda x: "%.3f" % x
f2 = lambda x: "%.2f" % x
pct = lambda x: "%.0f%%" % (100 * x)
NAMES = {"LR": "Logistic regression", "LSVM": "Linear SVM", "RSVM": "RBF SVM", "XGB": "XGBoost"}
REPN = {"AC": "Acoustic (40)", "WLM": "WavLM-large", "W2V": "wav2vec2-large-lv60", "XLSR": "XLS-R 300M", "H1": "WavLM + acoustic", "H2": "wav2vec2 + acoustic", "H3": "XLS-R + acoustic",
        "H4": "WavLM + wav2vec2", "H5": "WavLM + XLS-R", "H6": "wav2vec2 + XLS-R", "H7": "three encoders", "H8": "three encoders + acoustic"}

# ------------------------------------------------------------------ data
cm = pd.read_csv(MET / "condition_metrics.csv"); comp = pd.read_csv(MET / "paired_comparisons.csv")
rec = json.load(open(ROOT / "FREEZE_RECORD.json")); aud = json.load(open(MET / "oof_audit.json")); sa = json.load(open(ROOT / "manifests/stage_a_summary.json"))
ra = json.load(open(MET / "report_analysis.json")); dem = json.load(open(ROOT / "audit_opus/evidence/a05_demographics_and_flags.json"))
a1 = json.load(open(ROOT / "audit_opus/evidence/a01_reextract.json")); a3 = json.load(open(ROOT / "audit_opus/evidence/a03_independent_recompute.json"))
a4 = pd.read_csv(ROOT / "audit_opus/evidence/a04_thread_sensitivity.csv")
S0 = pd.read_csv(RES / "S0_univariate_features.csv"); S1 = pd.read_csv(RES / "S1_ablation.csv"); S2 = pd.read_csv(RES / "S2_acoustic_importance.csv"); S2b = pd.read_csv(RES / "S2_block_importance.csv")
S3s = pd.read_csv(RES / "S3_selected_and_ties.csv"); S3L = pd.read_csv(RES / "S3_landscape.csv"); S5 = pd.read_csv(RES / "S5_exact_intervals.csv"); S5n = pd.read_csv(RES / "S5_sample_size_plan.csv")
S5a = pd.read_csv(RES / "S5_auroc_precision.csv"); S6 = pd.read_csv(RES / "S6_session_predictability.csv"); S7 = pd.read_csv(RES / "S7_task_families.csv"); S8 = pd.read_csv(RES / "S8_compute.csv")
sm = pd.read_csv(ROOT / "manifests/SAMPLE_MANIFEST.csv"); ckp = json.load(open(ROOT / "features/manifests/encoder_checkpoints.json"))
wlm_man = json.load(open(ROOT / "features/manifests/WLM_manifest.json"))
cm["errors"] = cm.FN + cm.FP
base = cm[cm.condition_id == "FV1-WLM-ALL-LR"].iloc[0]; bcp = S5[S5.condition_id == "FV1-WLM-ALL-LR"].iloc[0]
ac_med = cm[cm.representation == "AC"].AUROC.median()
best = cm.sort_values(["AUPRC", "AUROC", "errors"], ascending=[False, False, True]).iloc[0]
hy = comp[(comp.comparison_type == "hybrid_vs_constituent") & (comp.metric == "AUROC")]; mv = comp[(comp.comparison_type == "model_vs_logistic_regression") & (comp.metric == "AUROC")]
fsc = comp[(comp.comparison_type == "feature_selection_vs_all_features") & (comp.metric == "AUROC")]
wlm8 = S8[(S8.encoder == "WLM") & (S8.threads == 8)].iloc[0]; ac_t = S8[S8.encoder.str.startswith("AC")].iloc[0].seconds_per_minute_of_audio
win_per_spk = wlm_man["total_windows"] / sa["speakers_total"]
task_med = sm.groupby("task_code").seconds.median(); protocol_minutes = task_med.sum() / 60


# ------------------------------------------------------------------ latex helpers
def inline(s):
    s = esc(s); s = s.replace(">=", "$\\geq$").replace("<=", "$\\leq$").replace(">", "$>$").replace("<", "$<$").replace("|", "$|$"); s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s); s = re.sub(r"`(.+?)`", r"\\texttt{\1}", s); return s


def tab(caption, header, rows, spec, size="scriptsize"):
    t = ["\\begin{center}\\%s\\renewcommand{\\arraystretch}{1.12}" % size, "\\begin{longtable}{%s}" % spec, "\\caption{%s}\\\\" % inline(caption), "\\toprule",
         " & ".join("\\textbf{%s}" % inline(h) for h in header) + "\\\\\\midrule\\endfirsthead", "\\toprule", " & ".join("\\textbf{%s}" % inline(h) for h in header) + "\\\\\\midrule\\endhead",
         "\\bottomrule\\endfoot"] + [" & ".join(inline(str(c)) for c in r) + "\\\\" for r in rows] + ["\\end{longtable}\\end{center}"]
    return "\n".join(t)


def fig(name, cap, w="0.95"):
    return "\\begin{figure}[H]\\centering\\includegraphics[width=%s\\linewidth]{%s.pdf}\\caption{%s}\\end{figure}" % (w, name, inline(cap))


def box(title, body):
    return "\\begin{tcolorbox}[colback=blue!3,colframe=blue!45!black,boxrule=0.5pt,arc=2pt,left=5pt,right=5pt,top=3pt,bottom=3pt,title={\\small\\bfseries %s}]\\small %s\\end{tcolorbox}" % (inline(title), body)


def bullets(items):
    return "\\begin{itemize}" + "".join("\\item " + inline(i) + "\n" for i in items) + "\\end{itemize}"


B = []
A = B.append
# ------------------------------------------------------------------ title and executive summary
A(box("Status and reading guide", inline("Research-stage technology assessment prepared for investor due diligence. The product described is **not a medical device** and has **not** been clinically validated. All performance numbers come from one frozen, audited internal study (protocol FINAL_V1, %d speakers, one Italian corpus). Market and regulatory statements are cited to public sources (reference list at the end) and must be re-verified before any filing or investment decision." % sa["speakers_total"])))
A("\\section{Executive summary}\\label{sec:exec}")
A(inline("**The opportunity.** About 11.8 million people lived with Parkinson's disease (PD) worldwide in 2021 [1], nearly one million in the United States with about 90,000 new US diagnoses per year [2], and the US economic burden was estimated at about $52 billion per year [3]. Clinical diagnosis is wrong in a meaningful minority of cases, especially early and outside specialist centres (pooled clinicopathologic accuracy about 74-80%) [4]. Speech is affected in about 89% of people with PD [5], it is cheap to record on any phone, and it can be repeated at home. A validated speech screen could triage referrals, support earlier diagnosis and provide digital endpoints for clinical trials."))
A(inline("**What was built.** A speaker-independent pipeline that turns a short multi-task speech recording (about %.1f minutes of median task durations) into a PD probability: 40 hand-crafted acoustic features and three frozen self-supervised speech encoders (WavLM-large, wav2vec2-large-lv60, XLS-R 300M; about 315 million parameters each), four classifier families, feature selection and eight hybrid representations - 56 configurations, each evaluated with nested, speaker-disjoint five-fold cross-validation under a frozen, hash-locked protocol and an independent audit." % protocol_minutes))
A(inline("**What the data show.** On %d speakers (%d PD, %d age-comparable controls; %d recordings) the median pooled speaker-level AUROC over the 56 configurations is %s (range %s to %s); %d of 56 make at most one speaker error. The reference model (WavLM-large + logistic regression, %d trainable parameters) reaches AUROC %s [95%% CI %s-%s], sensitivity %s (exact 95%% CI %s-%s) and specificity %s (exact CI %s-%s)." % (
    sa["speakers_total"], sa["speakers_pd"], sa["speakers_hc"], sa["files_included"], f3(cm.AUROC.median()), f3(cm.AUROC.min()), f3(cm.AUROC.max()), int((cm.errors <= 1).sum()),
    ra["parameter_counts_logistic_regression"]["WLM"], f3(base.AUROC), f3(base.AUROC_ci_lo), f3(base.AUROC_ci_hi), f2(bcp.sensitivity), f2(bcp.sens_cp_lo), f2(bcp.sens_cp_hi), f2(bcp.specificity), f2(bcp.spec_cp_lo), f2(bcp.spec_cp_hi))))
A(box("The critical caveat (read before the numbers)", inline("In this corpus every recording session contains only one class (patients and controls were recorded on different days; part of the patient audio was recorded at a different sampling rate). Four simple statistics of the zero-crossing rate already give AUROC %s, the most important features are statistics of the **quietest 10%% of frames** (background), and the same representations can tell recording sessions apart **inside one class** for unseen speakers (controls early versus late March: AUROC up to %s). The near-perfect separation is therefore **not evidence of disease detection**; it is at least partly a recording-condition effect. This is the single largest technical risk and the first thing a funded programme must remove." % (
    f3(S1[(S1.subset == "only_zcr") & (S1.model == "LR")].AUROC.iloc[0]), f2(S6[(S6.task.str.startswith("T2")) & (S6.model == "LR")].unit_auroc.max())))))
A(inline("**Investment thesis.** The asset today is a rigorously engineered and audited evaluation pipeline, a feature-level understanding of where the signal comes from, and a concrete plan to obtain confound-free evidence. It is **pre-validation (proof of concept on a confounded dataset)**. A staged programme (section [[roadmap]]) converts it into an investable clinical asset only if a recording-matched, multi-site cohort reproduces the separation with an AUROC lower confidence bound above 0.80 while a background-only model stays at chance. Milestones are designed so that each tranche buys a decisive answer."))
A("\\begin{center}\\small\\begin{tabular}{p{7.4cm}p{7.4cm}}\\toprule\\textbf{What we can claim} & \\textbf{What we cannot claim}\\\\\\midrule " +
  "A leakage-controlled, reproducible pipeline (280 runs, 20 audit checks, independent re-implementation). & That the models detect Parkinson's disease. \\\\ " +
  "Near-perfect speaker-independent separation on this corpus with simple and with large models. & Any clinical accuracy, sensitivity or specificity for a real screening population. \\\\ " +
  "Which features, tasks and models carry the separation, and how fragile tuning is. & Generalisation to other languages, devices, rooms or disease stages. \\\\ " +
  "Low compute cost per test (CPU seconds). & Regulatory readiness, reimbursement or clinical utility. \\\\\\bottomrule\\end{tabular}\\end{center}")

# ------------------------------------------------------------------ problem and market
A("\\section{Problem and market}\\label{sec:market}")
A(bullets([
    "Prevalence: about 11.77 million people with PD worldwide in 2021, age-standardised prevalence 138.6 per 100,000, rising (GBD 2021 based analysis) [1].",
    "United States: nearly 1 million people living with PD (earlier estimates about 650,000) and about 90,000 new diagnoses per year [2]; total economic burden about $52 billion per year (about $25.4 billion direct medical, $26.5 billion non-medical) [3].",
    "Diagnostic gap: pooled clinicopathologic accuracy of the clinical diagnosis about 74-80%, lower in early disease and with non-specialists; in a 10-year cohort 10.6% of diagnoses were revised [4].",
    "Speech: about 89% of people with PD have a speech or voice disorder, but only 3-4% receive speech treatment [5]; voice is therefore both a candidate biomarker and an under-served symptom.",
    "Where a speech screen could create value: (1) triage of referrals to movement-disorder clinics; (2) case finding in primary care or tele-medicine; (3) remote monitoring and digital endpoints in clinical trials; (4) speech-therapy follow-up."]))
A(inline("**Why now.** (1) Self-supervised speech foundation models (wav2vec 2.0, WavLM, XLS-R) give strong frozen representations without disease labels [11]; (2) smartphones make repeated, remote recordings cheap; (3) regulators have begun to process speech-based software devices (section [[regulatory]]); (4) public corpora such as NeuroVoz [8] allow frozen external tests."))
rows = []
for adopt in [0.01, 0.05, 0.10]:
    for price in [100, 300]:
        rows.append(["%d%%" % (100 * adopt), "$%d" % price, "%s" % format(int(1_000_000 * adopt), ","), "$%.1f M" % (1_000_000 * adopt * price / 1e6)])
A(tab("Illustrative, assumption-driven sizing of one segment only (remote monitoring of people already diagnosed in the US; base of about 1 million people [2]). Adoption and annual price per patient are assumptions to be tested, not forecasts.", ["Assumed adoption", "Assumed annual price", "Patients", "Annual revenue"], rows, "rrrr"))
ppv_ = lambda se, sp, pr: se * pr / (se * pr + (1 - sp) * (1 - pr))
A(inline("**Why predictive value matters commercially.** A screen is used where prevalence is low. With sensitivity 0.96, a positive result means PD in only about %s of cases at 1%% prevalence if specificity is 0.80, %s at specificity 0.95 and %s at specificity 0.99 (figure below). Any go-to-market therefore needs either very high specificity or an enriched population (for example people already referred for motor symptoms)." % (pct(ppv_(.96, .80, .01)), pct(ppv_(.96, .95, .01)), pct(ppv_(.96, .99, .01)))))
A(fig("wp_ppv", "Positive predictive value as a function of prevalence for sensitivity 0.96 and four specificities (formula, no study data).", "0.6"))

# ------------------------------------------------------------------ product concept
A("\\section{Product concept}\\label{sec:product}")
A(inline("A guided recording session on a smartphone or web browser (sustained vowels, /pa/ and /ta/ repetitions, a reading passage, a word list and a phrase; median durations in this corpus give about %.1f minutes of speech), automatic recording-quality checks, server-side or on-device scoring and a clinician report with a risk score and confidence. The recording-quality and **background-consistency gate** is part of the core product, not an add-on: section [[confound]] shows that background differences can dominate the score." % protocol_minutes))
rows = [[t, "%d" % (sm.task_code == t).sum(), "%.1f" % task_med[t]] for t in sorted(task_med.index)]
A(tab("Recording protocol in the corpus (16 task codes; median seconds per recording).", ["Task code", "Recordings", "Median s"], rows, "lrr"))

# ------------------------------------------------------------------ technology
A("\\section{Technology}\\label{sec:tech}")
A(fig("pipeline", "Pipeline: frozen feature extraction (no labels) and the only fitted stages (standardisation and classifier), fitted on training speakers only."))
A(inline("**Representations.** (i) 40 acoustic features: 10 frame features (log energy, zero-crossing rate, spectral centroid, roll-off at 85%%, spectral flatness, log energy in 0-0.5, 0.5-1, 1-2, 2-4 and 4-8 kHz) x 4 statistics over 25-ms frames (mean, SD, 10th and 90th percentile); (ii) frozen encoders: each 10-s window gives a 499 x 1,024 output map, averaged over time and windows to 1,024 values; (iii) hybrids: concatenations up to 3,112 values. **Models.** Logistic regression (20 configurations), linear SVM (10), RBF SVM (24), XGBoost (24 of a 256-point grid), class weighting inside every grid. **Decision.** Speaker score = mean recording probability, fixed threshold 0.5."))
rows = [[k, ckp[k]["model_id"], "%d" % S8[(S8.encoder == k)].parameters.iloc[0], "%.0f" % S8[(S8.encoder == k)].weights_MB.iloc[0],
         "%.2f / %.2f / %.2f" % tuple(S8[(S8.encoder == k) & (S8.threads == t)].seconds_per_10s_window_median.iloc[0] for t in [1, 4, 8]),
         {"WLM": "not declared on model card", "W2V": "Apache-2.0", "XLSR": "Apache-2.0"}[k]] for k in ["WLM", "W2V", "XLSR"]]
A(tab("Frozen encoders: size, CPU cost per 10-s window (1 / 4 / 8 threads, desktop CPU) and licence (model hub metadata).", ["Key", "Checkpoint", "Parameters", "Weights MB", "s per window", "Licence"], rows, "llrrll"))
A(inline("**Cost per test.** The corpus averages %.1f encoder windows per speaker; at %.2f s per window with 8 threads the WavLM pass costs about %.0f s of wall time (about %.2f core-hours) per test. The 40 acoustic features take %.2f s per minute of audio on one thread. At an assumed cloud price of p dollars per vCPU-hour the encoder cost per test is about %.2f x p dollars (for example about $%.3f at an assumed p = $0.05), i.e. compute is not a cost driver; clinical validation, regulatory work and data acquisition are." % (
    win_per_spk, wlm8.seconds_per_10s_window_median, win_per_spk * wlm8.seconds_per_10s_window_median, win_per_spk * wlm8.seconds_per_10s_window_median * 8 / 3600, ac_t,
    win_per_spk * wlm8.seconds_per_10s_window_median * 8 / 3600, 0.05 * win_per_spk * wlm8.seconds_per_10s_window_median * 8 / 3600)))
A(fig("wp_compute", "Measured CPU inference cost of the three encoders.", "0.55"))

# ------------------------------------------------------------------ data
A("\\section{Data asset and its limits}\\label{sec:data}")
A(inline("Italian Parkinson's Voice and Speech [6] (IEEE DataPort, listed as Creative Commons Attribution [7]): after exclusions **%d recordings, %d speakers (%d PD, %d elderly controls), %.2f audio hours, 16 task codes**. Spreadsheet demographics: PD %.1f +/- %.1f years (%d-%d), %d men / %d women; controls %.1f +/- %.1f years (%d-%d), %d men / %d women. Exclusions: 15 young controls (age confound) and one byte-identical duplicate recording; three people recorded in two sessions were merged into one speaker each. Four controls are flagged in the corpus spreadsheet for poor recording quality or reading difficulty." % (
    sa["files_included"], sa["speakers_total"], sa["speakers_pd"], sa["speakers_hc"], sa["audio_hours"], dem["PD_age"]["mean"], dem["PD_age"]["sd"], dem["PD_age"]["min"], dem["PD_age"]["max"], dem["PD_sex"]["M"], dem["PD_sex"]["F"],
    dem["HC_age"]["mean"], dem["HC_age"]["sd"], dem["HC_age"]["min"], dem["HC_age"]["max"], dem["HC_sex"]["M"], dem["HC_sex"]["F"])))
A(fig("wp_confound_timeline", "The structural confound: recording dates, class and sampling rate. No date contains both classes; all 44.1-kHz audio is PD.", "0.85"))
A(bullets(["No disease severity, duration or medication data; no clinical scale (for example MDS-UPDRS) to relate scores to disease stage.",
           "Single site, single language, small sample: one speaker changes a speaker-level metric by about two percentage points.",
           "Sex balance differs between groups (PD 64% men, controls 45% men); age distributions overlap (60-80 for most).",
           "The dataset licence permits attribution-based reuse, but clinical-grade development requires new, consented, protocol-controlled data (section [[roadmap]])."]))

# ------------------------------------------------------------------ evaluation design
A("\\section{Evaluation design and integrity}\\label{sec:eval}")
A(bullets(["Frozen protocol (SHA-256 %s...) written before any model saw the data; 56 conditions x 5 outer folds = 280 runs; FWL adjustment blocked because it was not defined in advance." % rec["protocol_json_sha256"][:16],
           "Stratified, speaker-disjoint 5-fold outer CV; nested speaker-disjoint 3-fold inner CV for tuning and feature selection; every learned step fitted on training speakers only; fixed threshold 0.5.",
           "Integrity gates on every run (speaker overlap, coverage, probability validity, labels, folds, hashes) and an out-of-fold audit: %d of %d checks passed, 0 failed runs." % (sum(c["status"] == "PASS" for c in aud["checks"]), len(aud["checks"])),
           "Independent audit (this document's appendix): independent re-parse of the raw data, re-extraction of embeddings (max difference %.1e), independent nested CV (same selected configurations), recomputation of all metrics (0 disagreements), bit-identical re-execution of all 70 logistic-regression runs under single-threaded BLAS. No result-changing bug; documentation errata recorded." % max(x["max_abs_diff"] for v in a1.values() for x in v),
           "Primary metrics: pooled speaker-level AUPRC and AUROC (prevalence 53.2% PD); supporting: F1, sensitivity, specificity, balanced accuracy, bootstrap and exact intervals."]))

# ------------------------------------------------------------------ results by combination
A("\\section{Results: metrics by feature combination and model}\\label{sec:results}")
A(fig("heatmap_auroc", "Pooled speaker-level AUROC for every representation and model (56 configurations).", "0.55"))
A(fig("wp_forest_auroc", "AUROC with 95% speaker-bootstrap intervals for logistic regression and XGBoost. Intervals collapse to a point when AUROC = 1 (percentile bootstrap at the boundary).", "0.98"))
rows = []
for c in cm.itertuples():
    e = S5[S5.condition_id == c.condition_id].iloc[0]
    rows.append([REPN[c.representation] + ("" if c.feature_selection == "ALL" else " / " + c.feature_selection.lower()), NAMES[c.model], f3(c.AUPRC), "%s [%s, %s]" % (f3(c.AUROC), f3(c.AUROC_ci_lo), f3(c.AUROC_ci_hi)),
                 "%s (%s-%s)" % (f2(e.sensitivity), f2(e.sens_cp_lo), f2(e.sens_cp_hi)), "%s (%s-%s)" % (f2(e.specificity), f2(e.spec_cp_lo), f2(e.spec_cp_hi)), f3(c.F1), f3(c.balanced_accuracy), "%d/%d" % (c.FN, c.FP)])
A(tab("All 56 configurations: pooled out-of-fold speaker-level metrics (47 speakers). Sensitivity and specificity with exact Clopper-Pearson 95% intervals; FN/FP = missed patients / false alarms at threshold 0.5.",
      ["Representation", "Model", "AUPRC", "AUROC [bootstrap CI]", "Sensitivity (exact CI)", "Specificity (exact CI)", "F1", "Bal. acc.", "FN/FP"], rows, "p{3.0cm}p{2.0cm}rlllrrl", "tiny"))
rows = []
for r in ["AC", "WLM", "W2V", "XLSR"] + ["H%d" % i for i in range(1, 9)]:
    x = cm[(cm.representation == r) & (cm.feature_selection == "ALL")]
    rows.append([REPN[r], "%d" % sum(P.DIMS[c] for c in P.components(r)), f3(x.AUROC.median()), f3(x.AUROC.min()), f3(x.AUROC.max()), f3(x.AUPRC.median()), "%d" % x.errors.min(), "%d" % x.errors.median()])
A(tab("Feature-combination leaderboard (median over the four models).", ["Combination", "Dim.", "Median AUROC", "Min", "Max", "Median AUPRC", "Fewest errors", "Median errors"], rows, "lrrrrrrr"))
ex = hy[hy.ci_excludes_zero]
A(inline("**Do combinations help?** Over %d paired comparisons of a hybrid with each of its components (same model, same folds) the median AUROC difference is %s (range %s to %s); the 95%% interval excludes zero in %d comparisons (%d up, %d down). **Do non-linear models help?** Median AUROC difference against logistic regression: %s. **Does feature selection help?** Ten selected acoustic features minus all 40: median %s, interval excluding zero in %d of %d. In short, at this ceiling no combination, model or selection beats the simplest configurations; the differences are within one or two speakers." % (
    len(hy), f3(hy.delta_A_minus_B.median()), f3(hy.delta_A_minus_B.min()), f3(hy.delta_A_minus_B.max()), len(ex), int((ex.delta_A_minus_B > 0).sum()), int((ex.delta_A_minus_B < 0).sum()),
    "; ".join("%s %s" % (NAMES[m], f3(mv[mv.condition_A.str.endswith("-" + m)].delta_A_minus_B.median())) for m in P.MODELS[1:]), f3(fsc.delta_A_minus_B.median()), int(fsc.ci_excludes_zero.sum()), len(fsc))))
A(fig("deltas", "Paired AUROC differences for all pre-specified comparisons (red: interval excludes zero).", "0.7"))
A(fig("wp_roc", "Pooled speaker-level ROC curves for six configurations.", "0.5"))


fold_rows = []
for cid in ["FV1-AC-ALL-LR", "FV1-AC-ALL-XGB", "FV1-WLM-ALL-LR", "FV1-WLM-ALL-XGB", "FV1-W2V-ALL-LR", "FV1-XLSR-ALL-LR", "FV1-H3-ALL-LR", "FV1-H8-ALL-XGB"]:
    for f in range(5):
        r = json.load(open(MET / ("%s-F%d.json" % (cid, f)))); sl = r["speaker_level"]
        fold_rows.append([cid.replace("FV1-", "").replace("-ALL-", " "), "%d" % f, "%d" % r["n_test_speakers"], f3(sl["auprc"]), f3(sl["auroc"]), f3(sl["f1"]), "%d/%d" % (sl["fn"], sl["fp"])])
A(tab("Fold-by-fold held-out results for eight representative configurations (9 or 10 test speakers per fold).", ["Configuration", "Fold", "Test speakers", "AUPRC", "AUROC", "F1", "FN/FP"], fold_rows, "lrrrrrl", "tiny"))
tv = {r["condition"]: r for r in ra["train_val_test"]}; tr_ = {r["condition"]: r for r in ra["training_partition_scores"]}
rows = [[c.replace("FV1-", "").replace("-ALL-", " "), "%s / %s" % (f3(tr_[c]["train_auprc_mean"]), f3(tr_[c]["train_auroc_mean"])), "%s (SD %s)" % (f3(tv[c]["inner_val_auprc_mean"]), f3(tv[c]["inner_val_auprc_sd"])),
         "%s (SD %s)" % (f3(tv[c]["heldout_fold_auprc_mean"]), f3(tv[c]["heldout_fold_auprc_sd"]))] for c in tv]
A(tab("Training versus inner-validation versus held-out scores (means over the five outer folds; training scores from a diagnostic refit). Perfect training fit with lower but close validation and held-out scores: no sign that tuning overfitted the held-out folds.", ["Configuration", "Training AUPRC / AUROC", "Inner-validation AUPRC", "Held-out fold AUPRC"], rows, "llll"))
rows = [[REPN[c.representation] + ("" if c.feature_selection == "ALL" else " / " + c.feature_selection.lower()), NAMES[c.model], f3(c.sample_AUROC), f3(c.sample_AUPRC), f3(c.AUROC), f3(c.AUPRC)] for c in cm.itertuples()]
A(tab("Recording-level versus speaker-level discrimination: averaging a speaker's recordings raises AUROC in almost every configuration.", ["Representation", "Model", "Recording AUROC", "Recording AUPRC", "Speaker AUROC", "Speaker AUPRC"], rows, "p{3.6cm}p{2.4cm}rrrr", "tiny"))

# ------------------------------------------------------------------ every feature
A("\\section{Every acoustic feature}\\label{sec:features}")
n90 = int((S0.AUROC_abs >= .9).sum())
fam = S0.groupby("family").AUROC_abs.mean().sort_values(ascending=False)
A(inline("Speaker-level univariate analysis of all 40 features (descriptive, whole cohort, not used for any model decision). %d features separate the groups with direction-free AUROC of at least 0.90 on their own; all of them are spectral-shape features (zero-crossing rate, spectral centroid, roll-off), and **they are lower in PD**. The effect remains when only 16-kHz recordings are used, so it is not only a resampling artefact. But the share of variance explained by the recording date **within one class** is large for many features (within-control eta-squared up to %s), i.e. the same features also vary strongly between sessions of healthy people." % (n90, f2(S0.eta2_date_within_controls.max()))))
A(fig("wp_univariate_features", "Univariate speaker-level separation of each of the 40 acoustic features (bars) and the same analysis on 16-kHz recordings only (ticks).", "0.78"))
rows = [[r.feature.replace("ac_", ""), r.direction, f3(r.mean_PD), f3(r.mean_control), f3(r.AUROC_class), f3(r.AUROC_class_native16k_only), f2(r.cohens_d), f2(r.eta2_class), f2(r.eta2_date_within_PD), f2(r.eta2_date_within_controls)] for r in S0.itertuples()]
A(tab("Feature dictionary and univariate statistics (speaker means; AUROC > 0.5 means higher in PD). eta2: share of variance explained by class, or by recording date within PD / within controls.",
      ["Feature", "Direction", "Mean PD", "Mean ctrl", "AUROC", "AUROC 16 kHz", "Cohen d", "eta2 class", "eta2 date|PD", "eta2 date|ctrl"], rows, "lllrrrrrrrr", "tiny"))
rows = [[f.replace("_log", " (log)").replace("_hz", " (Hz)"), f3(v), f3(S0[S0.family == f].AUROC_abs.max()), f2(S0[S0.family == f].eta2_date_within_PD.mean()), f2(S0[S0.family == f].eta2_date_within_controls.mean())] for f, v in fam.items()]
A(tab("Feature families (mean and best direction-free AUROC of their four statistics; mean within-class date effect).", ["Family", "Mean AUROC", "Best AUROC", "Mean eta2 date|PD", "Mean eta2 date|ctrl"], rows, "lrrrr"))
hp = pd.read_csv(RES / "S0_highly_correlated_pairs.csv")
A(inline("Redundancy: %d feature pairs have an absolute Spearman correlation of at least 0.9 at speaker level (for example the mean and the percentiles of the same frame feature), so single-feature importance is diluted and family-level analysis is more informative." % len(hp)))

# ------------------------------------------------------------------ ablation and selection
A("\\section{Feature families, statistics and feature selection}\\label{sec:ablation}")
A(fig("wp_ablation", "Exploratory ablations with the same nested protocol: one family alone, all but one family, and one statistic of all families.", "0.98"))
A1 = S1[~S1.subset.isin(["only_energy", "no_level_features"])]
rows = [[r.subset.replace("only_stat_", "only statistic: ").replace("drop_stat_", "drop statistic: ").replace("only_", "only: ").replace("drop_", "drop: ").replace("_log", "").replace("_hz", ""), NAMES[r.model], "%d" % r.n_features, f3(r.AUROC), f3(r.AUPRC), f3(r.F1), "%d/%d" % (r.FN, r.FP)] for r in A1.itertuples()]
A(tab("Ablation results (66 configurations, exploratory amendment A1, same folds and nested tuning).", ["Feature subset", "Model", "Features", "AUROC", "AUPRC", "F1", "FN/FP"], rows, "llrrrrl", "tiny"))
z = lambda s, m: S1[(S1.subset == s) & (S1.model == m)].iloc[0]
A(inline("Reading: no family is necessary (dropping any single family leaves AUROC >= %s), and several are sufficient on their own with XGBoost (for example the 500-1000 Hz band: AUROC %s). With logistic regression, the four zero-crossing statistics alone reach %s and energy alone only %s, so the linear signal is mostly spectral shape. Any single statistic over all families (10 values) reaches at least %s." % (
    f3(S1[S1.subset.str.startswith("drop_") & ~S1.subset.str.startswith("drop_stat_")].AUROC.min()), f3(z("only_band1_log", "XGB").AUROC), f3(z("only_zcr", "LR").AUROC), f3(z("only_log_energy_db", "LR").AUROC), f3(S1[S1.subset.str.startswith("only_stat_")].AUROC.min()))))
sel_rows = []
for fs in ["FORWARD", "BACKWARD"]:
    from collections import Counter
    cnt = Counter()
    for f in range(5):
        cnt.update(json.load(open(MET / ("FV1-AC-%s-LR-F%d.json" % (fs, f))))["selected_features"])
    for k, v in cnt.most_common():
        sel_rows.append([fs.lower(), k.replace("ac_", ""), "%d/5" % v])
A(inline("**Sequential selection (frozen FINAL_V1 conditions).** Forward and backward selection of exactly 10 of the 40 features, repeated inside every training partition with a logistic-regression selector (selection is identical for the four final models). Forward: AUROC %s-%s across models; backward: %s-%s; all 40: %s-%s. Stability is moderate: the table lists how often each feature was selected over the five outer folds." % (
    f3(cm[(cm.representation == "AC") & (cm.feature_selection == "FORWARD")].AUROC.min()), f3(cm[(cm.representation == "AC") & (cm.feature_selection == "FORWARD")].AUROC.max()),
    f3(cm[(cm.representation == "AC") & (cm.feature_selection == "BACKWARD")].AUROC.min()), f3(cm[(cm.representation == "AC") & (cm.feature_selection == "BACKWARD")].AUROC.max()),
    f3(cm[(cm.representation == "AC") & (cm.feature_selection == "ALL")].AUROC.min()), f3(cm[(cm.representation == "AC") & (cm.feature_selection == "ALL")].AUROC.max()))))
A(tab("Selected features and selection frequency over the five outer folds.", ["Method", "Feature", "Selected"], sel_rows, "lll", "tiny"))

# ------------------------------------------------------------------ importance
A("\\section{Feature importance}\\label{sec:importance}")
A(fig("wp_importance_acoustic", "Acoustic models: permutation importance (increase in recording-level log-loss when a family or a statistic is shuffled within the outer-test folds) and XGBoost gain shares.", "0.98"))
lr = S2[S2.condition == "FV1-AC-ALL-LR"]; xg = S2[S2.condition == "FV1-AC-ALL-XGB"]
top_lr = lr[lr.group.str.startswith("ac_")].sort_values("logloss_increase", ascending=False).head(8)
top_xg = xg[xg.group.str.startswith("ac_")].sort_values("gain_share_mean", ascending=False).head(8)
A(inline("**Logistic regression** relies on many correlated features (top by permutation: %s); its coefficients change sign across folds for some energy features, a sign of collinearity. **XGBoost** concentrates on **10th-percentile statistics**: %s alone carry %s of the total gain, and shuffling the p10 statistic of all families costs more than any family. The 10th percentile of a frame feature describes the quietest frames, i.e. pauses and background, not the voice." % (
    ", ".join(g.replace("ac_", "") for g in top_lr.group[:5]), ", ".join(g.replace("ac_", "") for g in top_xg.group[:3]), pct(top_xg.gain_share_mean[:3].sum()))))
rows = [[r.group.replace("ac_", ""), f3(r.logloss_increase), f3(r.auroc_drop), f2(r.coef_mean) if pd.notna(r.coef_mean) else "", "yes" if r.coef_sign_consistent is True else ("no" if r.coef_sign_consistent is False else "")] for r in lr[lr.group.str.startswith("ac_")].sort_values("logloss_increase", ascending=False).head(15).itertuples()]
A(tab("Logistic regression on acoustic features: top 15 features by permutation importance (mean of 20 shuffles) and the mean standardised coefficient over the five outer folds.", ["Feature", "Log-loss increase", "AUROC drop", "Mean coefficient", "Sign stable"], rows, "lrrrl", "tiny"))
A(fig("wp_block_importance", "Hybrid models: which representation block carries the weight.", "0.8"))
bl = S2b
A(inline("**In hybrids the two model families disagree about what matters.** Logistic regression puts %s-%s of its coefficient mass on the encoder block, whereas XGBoost puts %s-%s of its gain on the 40 acoustic features. When all three encoders are combined, logistic regression weights WavLM most (%s), then XLS-R (%s), then wav2vec2 (%s). Shuffling the wav2vec2 block in the three-encoder models changes nothing, i.e. it is redundant given the other two." % (
    pct(bl[(bl.condition.isin(["FV1-H1-ALL-LR", "FV1-H2-ALL-LR", "FV1-H3-ALL-LR"])) & (bl.group != "AC")].coef_mass_share.min()), pct(bl[(bl.condition.isin(["FV1-H1-ALL-LR", "FV1-H2-ALL-LR", "FV1-H3-ALL-LR"])) & (bl.group != "AC")].coef_mass_share.max()),
    pct(bl[(bl.condition.isin(["FV1-H1-ALL-XGB", "FV1-H2-ALL-XGB", "FV1-H3-ALL-XGB"])) & (bl.group == "AC")].gain_share.min()), pct(bl[(bl.condition.isin(["FV1-H1-ALL-XGB", "FV1-H2-ALL-XGB", "FV1-H3-ALL-XGB"])) & (bl.group == "AC")].gain_share.max()),
    pct(bl[(bl.condition == "FV1-H7-ALL-LR") & (bl.group == "WLM")].coef_mass_share.iloc[0]), pct(bl[(bl.condition == "FV1-H7-ALL-LR") & (bl.group == "XLSR")].coef_mass_share.iloc[0]), pct(bl[(bl.condition == "FV1-H7-ALL-LR") & (bl.group == "W2V")].coef_mass_share.iloc[0]))))
rows = [[r.condition.replace("FV1-", "").replace("-ALL-", " "), r.group, f3(r.auroc_drop), f3(r.logloss_increase), pct(r.coef_mass_share) if pd.notna(r.coef_mass_share) else pct(r.gain_share)] for r in bl.itertuples()]
A(tab("Block permutation importance and weight share in hybrids (LR: coefficient mass; XGBoost: gain).", ["Model", "Block", "AUROC drop", "Log-loss increase", "Weight share"], rows, "llrrr", "tiny"))

# ------------------------------------------------------------------ hyperparameters
A("\\section{Hyperparameter tuning}\\label{sec:tuning}")
rows = [[NAMES[m], "%d" % len(P.grid(m)), {"LR": "C {0.01,0.1,1,10,100} x penalty {L1,L2} x class weight {none, balanced}", "LSVM": "C {0.01,0.1,1,10,100} x class weight", "RSVM": "C {0.1,1,10,100} x gamma = m/d, m {0.1,1,10} x class weight",
         "XGB": "24 fixed of 256: learning rate {0.03,0.1}, depth {2,4}, trees {100,300}, min child weight {1,5}, subsample {0.7,1}, colsample {0.3,1}, lambda {1,10}, pos. weight {1, balanced}"}[m]] for m in P.MODELS]
A(tab("Frozen search spaces (inner objective: pooled speaker-level AUPRC over three speaker-disjoint inner folds; ties to the first configuration).", ["Model", "Configs", "Grid"], rows, "lrp{11.5cm}"))
tie = S3s.groupby("model").agg(runs=("run_id", "size"), tied=("n_tied_at_best", lambda x: int((x > 1).sum())), med=("n_tied_at_best", "median"))
A(inline("**Finding 1 - the inner objective saturates.** In most runs several configurations tie at the best inner score (%s). Selection then falls back to the order of the grid, so 'tuning' is largely tie-breaking: logistic regression most often ends at C = 0.1 or 0.01 with L2, the linear SVM at C = 0.01, the RBF SVM at C = 1 with a narrow kernel (m = 10), XGBoost at the first configuration of its list." % "; ".join("%s %d of %d runs (median %g tied)" % (NAMES[m], tie.loc[m, "tied"], tie.loc[m, "runs"], tie.loc[m, "med"]) for m in P.MODELS)))

sel_rows = []
for m in P.MODELS:
    vc = S3s[S3s.model == m].selected.value_counts()
    for cfg, n in vc.head(5).items():
        sel_rows.append([NAMES[m], ", ".join("%s=%s" % (k, v) for k, v in json.loads(cfg).items()), "%d of %d" % (n, int((S3s.model == m).sum()))])
A(tab("Most frequently selected configurations over the 70 runs of each model family.", ["Model", "Configuration", "Runs"], sel_rows, "lp{11.5cm}r", "tiny"))
L = S3L.copy()
g = L.groupby(["representation", "model"]).pooled_auroc.agg(["max", "median", "min"]).reset_index()
g["nested"] = [cm[(cm.representation == r) & (cm.model == m) & (cm.feature_selection == "ALL")].AUROC.iloc[0] for r, m in zip(g.representation, g.model)]
worst = g.assign(gap=g["median"] - g["nested"]).sort_values("gap", ascending=False).iloc[0]
A(inline("**Finding 2 - when tuning goes wrong it is the RBF SVM.** Refitting every configuration on the outer folds (post hoc, descriptive) shows that most configurations of the linear models perform alike, while the RBF SVM spans a wide range (for wav2vec2 from %s to %s). The largest loss from the nested choice is %s with %s: nested choice %s against a grid median of %s. The RBF SVM's poor scores are therefore a tuning artefact of a saturated objective (plus an uncalibrated score at the fixed threshold), not evidence against the model family." % (
    f3(g[(g.representation == "W2V") & (g.model == "RSVM")]["min"].iloc[0]), f3(g[(g.representation == "W2V") & (g.model == "RSVM")]["max"].iloc[0]), REPN[worst.representation], NAMES[worst.model], f3(worst.nested), f3(worst["median"]))))
A(fig("wp_tuning", "Left: number of configurations tied at the best inner score per run. Right: outer-fold AUROC of every grid configuration (range, median) and the nested choice.", "0.98"))
rows = [[REPN[r.representation], NAMES[r.model], f3(r["min"]), f3(r["median"]), f3(r["max"]), f3(r.nested)] for _, r in g.iterrows()]
A(tab("Tuning landscape: pooled AUROC of every configuration refitted on the outer folds (descriptive) versus the nested selection.", ["Representation", "Model", "Worst config", "Median config", "Best config", "Nested choice"], rows, "llrrrr", "tiny"))
A(inline("**Implication for product development.** With few speakers per fold, classical grid search cannot discriminate between configurations. A product model should use simple, strongly regularised defaults, report sensitivity to them, and spend data on external validation rather than on tuning."))

# ------------------------------------------------------------------ tasks
A("\\section{Which speech tasks carry the signal}\\label{sec:tasks}")
A(fig("wp_tasks", "Speaker-level AUROC when each speaker's score is computed from one task family only (no retraining).", "0.9"))
rows = [[r.condition_id.replace("FV1-", "").replace("-ALL-", " "), r.task_family, "%d" % r.recordings, "%.1f" % r.median_seconds_per_recording, f3(r.speaker_auroc), "%d" % r["speaker_errors_at_0.5"], f3(r.recording_auroc)] for _, r in S7.iterrows()]
A(tab("Task-family analysis.", ["Model", "Task family", "Recordings", "Median s", "Speaker AUROC", "Errors", "Recording AUROC"], rows, "llrrrrr", "tiny"))
A(inline(("Every task family alone separates the groups well (speaker AUROC %s or higher in every cell of the table); the 30-s phrase and the 56-s word list reach 1.000 with wav2vec2 and XLS-R, and sustained vowels have the lowest (or tied lowest) recording-level AUROC in every model. For a product this suggests a protocol of about two minutes (word list + phrase + a few vowels) - but the same caveat applies: a short recording also carries the session background.") % f3(S7[S7.task_family != "all tasks"].speaker_auroc.min())))

# ------------------------------------------------------------------ confounding
A("\\section{The confounding problem - evidence and implications}\\label{sec:confound}")
s6 = S6[S6.model == "LR"]
t2 = s6[s6.task.str.startswith("T2")]; t1 = s6[s6.task.str.startswith("T1")]; t3 = s6[s6.task.str.startswith("T3")]
A(inline("Pre-specified diagnostic (amendment A1): train the same pipeline to recognise the **recording session inside one class**, for unseen speakers. If sessions are recognisable within a class, a PD-versus-control classifier can succeed simply by recognising sessions, because sessions and class coincide in this corpus."))
A(fig("wp_session", "Session predictability within one class (unit = speaker x session; nested, speaker-disjoint).", "0.9"))
A(bullets(["Controls, early versus late March 2017 (same class, same sampling rate): AUROC %s to %s across representations - sessions are clearly recognisable." % (f3(t2.unit_auroc.min()), f3(t2.unit_auroc.max())),
           "PD, 2016 (44.1 kHz) versus 2017 (16 kHz): AUROC %s to %s - recognisable (session and resampling)." % (f3(t3.unit_auroc.min()), f3(t3.unit_auroc.max())),
           "PD, 24 January versus 10 February 2017 (same rate): AUROC %s to %s - near chance for three of four representations; not every session pair differs." % (f3(t1.unit_auroc.min()), f3(t1.unit_auroc.max())),
           "Feature importance (section [[importance]]) points at background statistics (10th percentiles), and the four flagged controls (poor recording quality, reading difficulty) produce %d of %d false alarms." % (dem["false_positive_events_from_flagged_controls"], dem["false_positive_events_total"])]))
A(box("Interpretation for investors", inline("The current numbers cannot separate 'detects Parkinson's speech' from 'detects which session a recording came from'. Part of the signal may be genuine (spectral-shape features stay discriminative in 16-kHz-only data, and literature documents real PD speech changes [5]), but the size of the genuine effect is **unknown**. The programme in section [[roadmap]] is designed to measure it.")))

# ------------------------------------------------------------------ errors
A("\\section{Error analysis}\\label{sec:errors}")
A(inline("All missed patients are one person: PD_005 is missed in %d of 56 configurations; her recording probabilities sit just below the threshold in every task (WavLM-LR %s-%s), so the models are uncertain about her as a whole. False alarms concentrate in a few controls; the four controls flagged in the corpus spreadsheet produce %d of %d false-alarm events. Sustained vowels have the highest recording-level error rate. In a product, recording-quality flags should abstain rather than score." % (
    dem["false_negative_events_by_speaker"]["PD_005"], f2(min(ra["errors"]["detail"]["PD_005"]["FV1-WLM-ALL-LR"]["by_task"].values())), f2(max(ra["errors"]["detail"]["PD_005"]["FV1-WLM-ALL-LR"]["by_task"].values())),
    dem["false_positive_events_from_flagged_controls"], dem["false_positive_events_total"])))
A(fig("error_tasks", "Recording probabilities (WavLM-LR) of the three most frequently misclassified speakers.", "0.95"))

# ------------------------------------------------------------------ roadmap
A("\\section{De-risking programme and validation plan}\\label{sec:roadmap}")
rows = [["0 (done)", "Frozen-protocol internal study and independent audit", "Pipeline, 56 configurations, audit, this whitepaper", "Completed"],
        ["1: confound control on existing data", "Background-consistency features and gate; re-analysis with background removed or matched; acquisition of a second public corpus (e.g. NeuroVoz, Castilian Spanish, 108-112 speakers [8]) for frozen external testing", "A background-only model at chance on matched data; frozen model AUROC on an external corpus with pre-set success criterion (lower 95% bound > 0.80)", "Go/no-go 1"],
        ["2: prospective recording-matched cohort", "Multi-site study: PD and age/sex-matched controls recorded in the same sessions, rooms and devices, interleaved; clinical scales (MDS-UPDRS), medication state, disease duration", "Sample size from the tables in section [[roadmap]]; pre-registered endpoints; independent statistician", "Go/no-go 2"],
        ["3: clinical validation", "Intended-use population (for example referrals with suspected parkinsonism), differential diagnoses (essential tremor, atypical parkinsonism), multiple languages/devices", "Regulatory-grade performance study; quality-management system", "Regulatory submission"],
        ["4: product", "App, clinician report, monitoring, integration, post-market surveillance", "Real-world performance monitoring", "Commercial launch"]]
A(tab("Staged programme with decision gates (no budget figures are given; each stage is sized by the founders).", ["Stage", "Work", "Evidence produced", "Gate"], rows, "p{2.6cm}p{6.2cm}p{5.0cm}p{2.0cm}"))
rows = [[f2(r.assumed_true_sensitivity_or_specificity), f2(r.required_lower_95_bound), "%d" % r.people_needed_in_that_class] for r in S5n.itertuples()]
A(tab("Approximate number of people needed in one class so that the exact 95% lower bound of sensitivity (or specificity) reaches a target, if the observed proportion equals the assumed true value.", ["Assumed true value", "Required lower bound", "People in that class"], rows, "rrr"))
rows = [[f2(r.true_auroc), "%d" % r.people_per_class, "+/- %s" % f3(r.ci_halfwidth)] for r in S5a.itertuples()]
A(tab("Approximate 95% half-width of an AUROC estimate (Hanley-McNeil) for equal group sizes.", ["True AUROC", "People per class", "Half-width"], rows, "rrr"))

# ------------------------------------------------------------------ regulatory, competition, business, risks
A("\\section{Regulatory pathway (to be confirmed with regulatory counsel)}\\label{sec:regulatory}")
A(bullets(["The intended product is software as a medical device (SaMD) for decision support; in the US the likely routes are 510(k) if a predicate exists or De Novo otherwise; in the EU, MDR rule 11 typically places diagnostic decision-support software in class IIa or higher.",
           "Precedents in speech: Aural Analytics' Speech Vitals (speech analysis, ALS) received FDA breakthrough designation, and reports describe registration of its product as a class II software device; Kintsugi has described a De Novo submission for a voice model for depression and anxiety [9]. Status changes quickly and must be checked in FDA databases.",
           "Required before any submission: quality-management system (ISO 13485), software life cycle (IEC 62304), risk management (ISO 14971), clinical evaluation, cybersecurity, data protection (voice is personal and health data; GDPR special-category processing)."]))
A("\\section{Intellectual property and freedom to operate}\\label{sec:ip}")
A(bullets(["Own assets today: the evaluation pipeline, protocol, audit chain and the feature-level findings (copyright/know-how); no patent has been filed.",
           "Patent landscape: patents on speech-based diagnosis of neurological disorders exist (for example EP 4 346 558 A1, 'software-based, speech-operated and objective diagnostic tool for use in diagnosing a chronic neurological disorder' [12]); a freedom-to-operate analysis is required before product development.",
           "Model licences: wav2vec2-large-lv60 and XLS-R 300M are Apache-2.0; the WavLM-large model card declares no licence (to be clarified with the publisher or replaced); the Italian corpus is listed as Creative Commons Attribution [7].",
           "Defensibility is expected to come from proprietary, protocol-controlled clinical data and validation evidence rather than from model architecture."]))
A("\\section{Competitive landscape}\\label{sec:competition}")
rows = [["Canary Speech (US)", "Voice platform for neurological and mental-health conditions; Parkinson's listed among targets; about $22M raised (CB Insights)", "[10]"],
        ["Winterlight Labs (CA)", "Speech-based cognitive assessment (dementia, Alzheimer's)", "[10]"],
        ["Sonde Health (US)", "Voice-based mental, cognitive and respiratory health monitoring", "[10]"],
        ["Aural Analytics (US)", "Speech Vitals (ALS, cognition); FDA breakthrough designation", "[9]"],
        ["Kintsugi (US)", "Voice model for depression/anxiety; De Novo submission described", "[9]"],
        ["Novoic, Vocalis Health", "Speech/voice AI (claims about Parkinson's for Vocalis rest on a single aggregator)", "[9][10]"]]
A(tab("Selected companies working on voice biomarkers (public sources; to be verified).", ["Company", "Focus", "Source"], rows, "p{3.6cm}p{10.2cm}l"))
A(inline("**Defensible differentiation** must come from evidence, not from the model: confound-controlled, protocol-matched clinical data; a published, auditable validation chain (frozen protocols, hash-locked artefacts, independent audits such as the one in the appendix); multi-task recording with quality gating; and low-cost CPU inference."))
A("\\section{Business model hypotheses (to be tested)}\\label{sec:business}")
A(bullets(["Clinical-trial services (pharma, CROs): speech as a remote, frequent digital endpoint - shortest path to revenue, lower regulatory burden if used as exploratory endpoint.",
           "Referral triage / tele-neurology decision support: per-test or per-site subscription; requires regulatory clearance.",
           "Primary-care or population screening: largest volume but lowest prevalence, hence the hardest predictive-value problem (section [[market]]).",
           "Licensing of a validated scoring engine to device or platform partners.",
           "Unit cost of compute is negligible (section [[tech]]); value and cost sit in clinical evidence, regulatory status, integration and reimbursement."]))
A("\\section{Risk register}\\label{sec:risks}")
rows = [["Recording-condition confound", "Critical", "Scores may reflect sessions, not disease", "Recording-matched data; background gate; background-only model must be at chance; external frozen test"],
        ["Generalisation (language, device, room)", "High", "Performance drop outside the corpus", "Multi-site, multi-device, multilingual cohorts; domain-shift monitoring"],
        ["Small and narrow data", "High", "Wide uncertainty; no severity data", "Prospective cohorts sized as in section [[roadmap]]; clinical scales"],
        ["Label quality", "Medium", "Clinical PD diagnosis is 74-80% accurate [4]", "Specialist-confirmed diagnoses, follow-up, differential diagnoses"],
        ["Tuning instability", "Medium", "Saturated objectives choose arbitrary configurations", "Fixed regularised defaults; external validation instead of tuning"],
        ["Low prevalence in screening", "High", "Low positive predictive value", "Enriched intended-use population; high-specificity operating point"],
        ["Regulatory / reimbursement", "High", "Time and cost to clearance; payment unclear", "Early regulatory consultation; trial-services revenue first"],
        ["Privacy and consent", "Medium", "Voice is biometric and health data", "Consent design, on-device features, data minimisation"],
        ["Model and data licences", "Medium", "WavLM-large licence not declared on the model card; dataset CC BY", "Licence clearance or replacement by Apache-2.0 encoders (wav2vec2, XLS-R perform comparably here)"]]
A(tab("Main risks and mitigations.", ["Risk", "Severity", "Consequence", "Mitigation"], rows, "p{3.4cm}p{1.4cm}p{4.4cm}p{6.6cm}"))

# ------------------------------------------------------------------ team, use of funds, conclusion
A("\\section{Team and use of funds}\\label{sec:team}")
A(inline("Project team (CSE 431, topic 4a): Md. Osman Goni, Md. Mohibur Rahman, Arannamoy Mondal. The FINAL_V1 execution, audit and this document were produced with an AI assistant under the team's direction. Roles that a funded programme must add: a clinical lead (movement-disorder neurologist), a speech-language pathologist, regulatory and quality lead, data engineering, and clinical-study operations. Use of funds should follow the stages in section [[roadmap]], each tranche tied to its gate; the largest cost items are expected to be prospective data collection, clinical study management and regulatory work, not computing."))
A("\\section{Conclusion}\\label{sec:conclusion}")
A(inline("The technology pipeline is sound, reproducible and cheap to run, and on the available data even simple acoustic statistics separate patients from controls almost perfectly. That is exactly why the result cannot yet be trusted: in this corpus the recording session coincides with the diagnosis, the decisive features describe the background, and sessions are recognisable within a class. The investable proposition is a disciplined de-risking programme whose first milestone - a frozen model that still separates patients from controls on recording-matched or external data while a background-only model fails - is cheap to test and decisive either way."))

# ------------------------------------------------------------------ appendix
A("\\appendix\\section{Audit summary}")
A(inline("An independent audit (Opus 5.5, max effort) re-derived the data from raw files, re-extracted embeddings, re-implemented the acoustic features and the nested cross-validation, recomputed all metrics and bootstrap intervals, and re-executed all 70 logistic-regression runs. Verdict: no critical bug and no change to any reported prediction or metric; two major documentation errors (the file-name number is the birth year, not the age; spreadsheet quality flags were not reported) and eight minor glitches were fixed and recorded as errata. Details: `audit_opus/OPUS_AUDIT.md`."))
rows = [[r.condition.replace("FV1-", ""), "%d/5" % r.folds_same_config, "%.1e" % r.max_abs_prob_diff, f3(r.AUROC_official), f3(r.AUROC_rerun)] for r in a4.itertuples()]
A(tab("Re-execution of all logistic-regression runs with single-threaded BLAS.", ["Condition", "Same config", "Max |dp|", "AUROC official", "AUROC re-run"], rows, "llrrr", "tiny"))
A("\\section{Glossary}")
rows = [["AUROC", "Probability that a random patient scores higher than a random control (0.5 chance, 1 perfect)"], ["AUPRC", "Area under the precision-recall curve; chance level equals the share of patients (0.53 here)"],
        ["Speaker-disjoint", "No person appears in both training and test data of a fold"], ["Nested CV", "Tuning happens inside the training data of each outer fold"],
        ["Confound", "A variable related to both the input and the label (here: recording session)"], ["p10 statistic", "10th percentile of a frame feature over a recording - the quietest frames"],
        ["Clopper-Pearson", "Exact binomial confidence interval for a proportion"], ["SaMD", "Software as a medical device"]]
A(tab("Glossary.", ["Term", "Meaning"], rows, "p{3.2cm}p{12.8cm}"))
A("\\begin{thebibliography}{99}\\setlength{\\itemsep}{1pt}\\scriptsize")
refs = ["Global, regional, national epidemiology and trends of Parkinson's disease from 1990 to 2021: findings from the Global Burden of Disease Study 2021 (secondary analysis), Frontiers in Aging Neuroscience 2025; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11757241/ and https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2024.1498756/full",
        "Parkinson's Foundation: incidence and prevalence (npj Parkinson's Disease 2022 study, coverage); https://thedailyherald.sx/us/us-parkinson-s-disease-incidence-far-higher-than-earlier-estimates",
        "Parkinson's Foundation: Economic burden of Parkinson's disease is $52 billion; https://parkinson.org/blog/research/economic-burden ; https://www.parkinson.org/about-us/news/economic-burden-of-parkinsons-disease-study",
        "Raty et al., Stability and accuracy of a diagnosis of Parkinson disease over 10 years, Neurology 2025; https://research.utu.fi/converis/portal/detail/Publication/491935225 ; Rizzo et al. review, https://pmc.ncbi.nlm.nih.gov/articles/PMC5877503",
        "Trail M. et al., Speech treatment for Parkinson's disease, NeuroRehabilitation 2005 (89% speech/voice disorder; 3-4% treated); https://content.iospress.com/articles/neurorehabilitation/nre00283",
        "G. Dimauro, F. Girardi, Italian Parkinson's Voice and Speech, IEEE DataPort 2019, doi:10.21227/aw6b-tg17; G. Dimauro et al., IEEE Access 5:22199-22208, 2017, doi:10.1109/ACCESS.2017.2762475",
        "IEEE DataPort dataset page (licence listed as Creative Commons Attribution); https://ieee-dataport.org/node/1482",
        "Mendes-Laureano et al., NeuroVoz: a Castillian Spanish corpus of parkinsonian speech, Scientific Data 11:1367, 2024; https://zenodo.org/records/10777657 ; https://arxiv.org/abs/2403.02371v3",
        "Voice-biomarker regulatory news: https://pharmaphorum.com/news/speech-analysis-tech-als-gets-fda-breakthrough-tag ; https://www.speechtechmag.com/Articles/News/Speech-Technology-News/Aural-Analytics-Gets-FDA-Approval-155602.aspx ; https://healthy-innovations.beehiiv.com/p/voice-biomarkers",
        "Company profiles: https://www.cbinsights.com/company/winterlight-labs ; https://www.cbinsights.com/company/sonde-health ; https://www.thesilab.com/vendor-directory/canary-speech ; https://www.statnews.com/2023/11/07/voice-biomarkers-detecting-disease-ai-privacy-concerns/",
        "S. Chen et al., WavLM, IEEE JSTSP 16(6):1505-1518, 2022; A. Baevski et al., wav2vec 2.0, NeurIPS 2020; A. Babu et al., XLS-R, Interspeech 2022 (arXiv:2111.09296); T. Chen, C. Guestrin, XGBoost, KDD 2016.",
        "European patent application EP4346558A1 (patent listing); https://data.epo.org/gpi/EP4346558A1"]
A("".join("\\bibitem{w%d} %s\n" % (i + 1, esc(r)) for i, r in enumerate(refs)) + "\\end{thebibliography}")

PRE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.9cm]{geometry}
\usepackage{booktabs,longtable,float,graphicx,amsmath,array,enumitem,helvet,xcolor,tcolorbox}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\setlist{itemsep=1pt,topsep=2pt}\setlength{\parskip}{3.5pt}\setlength{\parindent}{0pt}
\usepackage{titlesec}\titleformat{\section}{\large\bfseries}{\thesection.}{.5em}{}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}\fancyfoot[C]{\small\thepage}\fancyhead[L]{\small Speech biomarkers for Parkinson's screening - venture whitepaper (research stage)}\fancyhead[R]{\small Confidential draft}\renewcommand{\headrulewidth}{.3pt}
\begin{document}
\begin{center}{\LARGE\bfseries Speech Biomarkers for Parkinson's Disease Screening\par}\vspace{4pt}{\large Technology assessment and venture whitepaper (research stage)\par}\vspace{4pt}
{\small Md.~Osman Goni \textperiodcentered{} Md.~Mohibur Rahman \textperiodcentered{} Arannamoy Mondal\par}\vspace{2pt}{\small Evidence base: frozen protocol FINAL\_V1 (audited) \textperiodcentered{} %s\par}\end{center}\vspace{2pt}\hrule\vspace{4pt}
\tableofcontents\newpage
""" % esc(now_utc()[:10])


def build():
    d = OUT / "_build_whitepaper"; d.mkdir(parents=True, exist_ok=True)
    body = re.sub(r"\[\[(\w+)\]\]", r"\\ref{sec:\1}", "\n".join(B))
    (d / "WHITEPAPER.tex").write_text(PRE + body + "\n\\end{document}\n", encoding="utf8")
    for fn in list(FIGS.glob("*.pdf")) + list((OUT / "figures").glob("*.pdf")):
        shutil.copy2(fn, d / fn.name)
    for _ in range(3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "WHITEPAPER.tex"], cwd=d, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2500:]); sys.exit(8)
    shutil.copy2(d / "WHITEPAPER.pdf", OUT / "WHITEPAPER_VC.pdf"); shutil.copy2(d / "WHITEPAPER.tex", SUPP / "WHITEPAPER_VC.tex"); shutil.rmtree(d)
    print("wrote", OUT / "WHITEPAPER_VC.pdf")


build()
