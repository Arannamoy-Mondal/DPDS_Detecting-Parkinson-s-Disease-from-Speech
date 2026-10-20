"""Training-partition scores (diagnostic only, for the 'training versus validation' table): refit each fold's selected configuration on the outer-training rows and score those same rows."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import pipeline as pl

MET = WORKSPACE / "metrics/FINAL_V1"
sm = pl.load_manifest()
split = pd.read_csv(WORKSPACE / "SPLIT_MANIFEST.csv").set_index("sample_id").loc[sm.sample_id]
of = split.outer_fold.values
rows = []
for cid in ["FV1-AC-ALL-LR", "FV1-WLM-ALL-LR", "FV1-W2V-ALL-LR", "FV1-XLSR-ALL-LR", "FV1-H3-ALL-LR", "FV1-WLM-ALL-XGB"]:
    rep, model = cid.split("-")[1], cid.split("-")[3]
    X, names = pl.load_representation(rep, sm)
    au, ap = [], []
    for f in range(5):
        r = json.loads((MET / ("%s-F%d.json" % (cid, f))).read_text())
        tr = np.where(of != f)[0]
        p = pl.fit_predict(model, r["selected_hyperparameters"], X[tr], sm.label.values[tr].astype(int), X[tr])
        sc, y, _ = pl.speaker_scores(p, sm.speaker_id.values[tr], sm.label.values[tr].astype(int))
        m = pl.binary_metrics(sc, y)
        au.append(m["auroc"]); ap.append(m["auprc"])
    rows.append({"condition": cid, "train_auprc_mean": float(np.mean(ap)), "train_auroc_mean": float(np.mean(au))})
ra = json.loads((MET / "report_analysis.json").read_text())
ra["training_partition_scores"] = rows
write_json(MET / "report_analysis.json", ra)
print(rows)
