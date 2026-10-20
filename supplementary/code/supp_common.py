"""Supplementary (post-freeze, exploratory) analyses. They never touch FINAL_V1 official outputs; everything is written under supplementary/.
Amendment A1 (owner request of 2026-10-20): detailed feature, importance and tuning analyses for the whitepaper."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path("/root/Projects/Project4a_FINAL_V1_ISOLATED")
sys.path.insert(0, str(ROOT / "src/final_v1"))
from common import *  # noqa
import protocol_defs as P
import pipeline as pl
SUPP = ROOT / "supplementary"
RES, FIGS = SUPP / "results", SUPP / "figures"
FAMILIES = ["log_energy_db", "zcr", "centroid_hz", "rolloff85_hz", "flatness", "band0_log", "band1_log", "band2_log", "band3_log", "band4_log"]
STATS = ["mean", "std", "p10", "p90"]
BAND_NAMES = ["band0_0-500Hz", "band1_500-1000Hz", "band2_1-2kHz", "band3_2-4kHz", "band4_4-8kHz"]


def load_ac():
    sm = pl.load_manifest()
    X, names = pl.load_representation("AC", sm)
    return sm, X, names


def outer_idx(sm):
    split = pd.read_csv(ROOT / "SPLIT_MANIFEST.csv").set_index("sample_id").loc[sm.sample_id]
    return split.outer_fold.values


def pooled_metrics_from_pred(frames):
    d = pd.concat(frames)
    g = d.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first"))
    m = pl.binary_metrics(g.s.values, g.y.values.astype(int))
    return m, d, g
