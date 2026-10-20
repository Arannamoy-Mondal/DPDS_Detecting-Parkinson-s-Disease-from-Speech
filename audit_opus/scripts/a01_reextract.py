"""Audit A01: re-extract embeddings for a few recordings in a fresh process and compare with the cached FINAL_V1 vectors."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "4"; os.environ["MKL_NUM_THREADS"] = "4"
sys.path.insert(0, "/root/Projects/Project4a_FINAL_V1_ISOLATED/src/final_v1")
from common import *  # noqa
os.environ["HF_HOME"] = str(WORKSPACE / "cache/hf"); os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, pandas as pd
import extract_embeddings as ee
sm = pd.read_csv(WORKSPACE / "manifests/SAMPLE_MANIFEST.csv")
order = sm.sort_values("seconds", ascending=False).reset_index(drop=True)
pick = [order.sample_id[0], order.sample_id[1], order.sample_id[2], order.sample_id[400], order.sample_id[784]]  # parts 0,1,2 (longest) + middle + shortest
out = {}
for key in ["WLM", "W2V", "XLSR"]:
    ext, model, ck = ee.load_encoder(key, 4)
    arr = np.load(WORKSPACE / "features" / (key + "_embeddings.npy"))
    pos = {s: i for i, s in enumerate(pd.read_csv(WORKSPACE / "features/manifests" / (key + "_sample_order.csv")).sample_id)}
    res = []
    for sid in pick:
        r = sm[sm.sample_id == sid].iloc[0]
        v, nwin = ee.embed(DATA_INPUT / r.relative_path, ext, model)
        c = arr[pos[sid]]
        res.append({"sample_id": sid, "seconds": float(r.seconds), "windows": nwin, "max_abs_diff": float(np.abs(v - c).max()), "cosine": float(v @ c / np.linalg.norm(v) / np.linalg.norm(c)), "mean_abs_value": float(np.abs(c).mean())})
    out[key] = res
    print(key, [(round(x["max_abs_diff"], 7), round(x["cosine"], 9)) for x in res], flush=True)
json.dump(out, open(WORKSPACE / "audit_opus/evidence/a01_reextract.json", "w"), indent=1)
