"""Extra explanatory figures for the study-edition report (FINAL_V1 only): dataset examples, nested-CV diagram,
training / inner-validation / held-out comparison."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

FIG = WORKSPACE / "reports/FINAL_V1/study_report/figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, GRAY, RED, AMB, GRN = "#2B5FAF", "#8A9099", "#C0392B", "#D98A00", "#2E7D4F"


def dataset_examples():
    sm = pd.read_csv(WORKSPACE / "manifests/SAMPLE_MANIFEST.csv")
    flagged = {"HC_001", "HC_007", "HC_008", "HC_020"}   # controls flagged in the corpus spreadsheet: not used as examples
    def pick(label, task):
        x = sm[(sm.label == label) & (sm.task_code == task) & (sm.sample_rate == 16000) & ~sm.speaker_id.isin(flagged)].sort_values("speaker_id")
        return x.iloc[0]
    rows = [(pick(1, "VA1"), "PD, sustained vowel /a/"), (pick(0, "VA1"), "Control, sustained vowel /a/"), (pick(1, "FB1"), "PD, phrase"), (pick(0, "FB1"), "Control, phrase")]
    fig, axes = plt.subplots(4, 2, figsize=(9.6, 7.6), gridspec_kw={"width_ratios": [1, 1.25]})
    for i, (r, title) in enumerate(rows):
        x = load_audio_16k(DATA_INPUT / r.relative_path); t = np.arange(len(x)) / SAMPLE_RATE
        axes[i, 0].plot(t, x, lw=.3, color=BLUE if r.label == 1 else GRAY); axes[i, 0].set_ylim(-1, 1)
        axes[i, 0].set_title("%s (%s, %.1f s): waveform" % (title, r.speaker_id, r.seconds), fontsize=8); axes[i, 0].set_xlabel("seconds")
        axes[i, 1].specgram(x, NFFT=512, Fs=SAMPLE_RATE, noverlap=352, cmap="magma", vmin=-140, vmax=-20)
        axes[i, 1].set_ylim(0, 8000); axes[i, 1].set_ylabel("Hz"); axes[i, 1].set_title("spectrogram (dark = quiet)", fontsize=8); axes[i, 1].set_xlabel("seconds")
    fig.tight_layout(); fig.savefig(FIG / "dataset_examples.pdf", bbox_inches="tight"); plt.close(fig)


def nested_cv():
    fig, ax = plt.subplots(figsize=(8.0, 4.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.text(0.05, 3.75, "Outer loop: 47 speakers in 5 folds (one fold = 9 or 10 people, both classes)", fontsize=8, weight="bold")
    for k in range(5):
        c = RED if k == 2 else BLUE
        ax.add_patch(Rectangle((0.1 + 1.25 * k, 3.0), 1.15, 0.5, fc=c, alpha=.25 if k != 2 else .45, ec=c))
        ax.text(0.675 + 1.25 * k, 3.25, "fold %d%s" % (k, "\nTEST" if k == 2 else "\ntrain"), ha="center", va="center", fontsize=7)
    ax.text(6.5, 3.25, "repeat 5 times, each fold\nis the test fold once", fontsize=7.5, va="center")
    ax.text(0.05, 2.55, "Inner loop (inside the 4 training folds only): 3 speaker-disjoint inner folds", fontsize=8, weight="bold")
    for k in range(3):
        ax.add_patch(Rectangle((0.1 + 1.7 * k, 1.75), 1.6, 0.5, fc=AMB if k == 1 else GRN, alpha=.3, ec=AMB if k == 1 else GRN))
        ax.text(0.9 + 1.7 * k, 2.0, "inner %d%s" % (k, " (validate)" if k == 1 else " (fit)"), ha="center", va="center", fontsize=7)
    steps = ["1. For every configuration of the grid: fit on 2 inner folds, score the 3rd; rotate.",
             "2. Pool the inner predictions per speaker; compute speaker-level AUPRC; keep the best configuration.",
             "3. Refit that configuration on all training speakers (and redo feature selection if used).",
             "4. Predict the untouched test fold once; average each test speaker's recordings; threshold 0.5."]
    for i, s in enumerate(steps):
        ax.text(5.4, 2.35 - 0.42 * i, s, fontsize=7.3)
    ax.text(0.05, 0.75, "Nothing learned (scaling, feature selection, hyperparameters, model) ever sees a test speaker.", fontsize=8, color=RED)
    fig.savefig(FIG / "nested_cv.pdf", bbox_inches="tight"); plt.close(fig)


def train_val_test():
    ra = json.loads((WORKSPACE / "metrics/FINAL_V1/report_analysis.json").read_text())
    tv = {r["condition"]: r for r in ra["train_val_test"]}; tr = {r["condition"]: r for r in ra["training_partition_scores"]}
    conds = list(tv)
    fig, ax = plt.subplots(figsize=(7.6, 2.8)); x = np.arange(len(conds)); w = .26
    ax.bar(x - w, [min(1.0, tr[c]["train_auprc_mean"]) for c in conds], w, color=GRAY, label="training (own speakers)")
    ax.bar(x, [tv[c]["inner_val_auprc_mean"] for c in conds], w, yerr=[tv[c]["inner_val_auprc_sd"] for c in conds], color=AMB, label="inner validation", ecolor="#555", capsize=2)
    ax.bar(x + w, [tv[c]["heldout_fold_auprc_mean"] for c in conds], w, yerr=[tv[c]["heldout_fold_auprc_sd"] for c in conds], color=BLUE, label="held-out test fold", ecolor="#555", capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([c.replace("FV1-", "").replace("-ALL-", " ") for c in conds]); ax.set_ylim(.9, 1.01); ax.set_ylabel("speaker-level AUPRC (mean over 5 folds)")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12)); ax.set_title("Training, validation and held-out scores are reported separately", fontsize=8.5)
    fig.savefig(FIG / "train_val_test.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    dataset_examples(); nested_cv(); train_val_test(); print("figures written to", FIG)
