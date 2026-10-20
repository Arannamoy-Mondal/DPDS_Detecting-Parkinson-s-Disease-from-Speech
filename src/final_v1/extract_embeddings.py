"""Frozen-encoder embeddings (no label, speaker or fold information is used).

Recipe (identical for the three encoders): audio -> mono 16 kHz (no peak normalization) -> consecutive 10-second windows plus one final window
aligned to the file end (a window shorter than 400 samples is padded by repeating the last sample) -> the checkpoint's own feature
extractor normalizes every window to zero mean and unit variance -> mean over time of last_hidden_state (1,024 values) -> mean over windows.

Usage:  python extract_embeddings.py KEY part parts threads      (KEY = WLM, W2V or XLSR; processes recordings part, part+parts, ...)
        python extract_embeddings.py KEY collect
"""
import json
import os
import sys
from pathlib import Path

if len(sys.argv) > 4 and sys.argv[2] != "collect":
    os.environ["OMP_NUM_THREADS"] = sys.argv[4]
    os.environ["MKL_NUM_THREADS"] = sys.argv[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
os.environ["HF_HOME"] = str(WORKSPACE / "cache/hf")
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import pandas as pd

WINDOW = 160000
MIN_SAMPLES = 400


def load_encoder(key, threads):
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    torch.set_num_threads(threads)
    ck = json.loads((WORKSPACE / "features/manifests/encoder_checkpoints.json").read_text())[key]
    path = ck["snapshot"]
    ext = AutoFeatureExtractor.from_pretrained(path)
    if not ext.do_normalize or ext.sampling_rate != SAMPLE_RATE:
        print("FATAL: unexpected feature extractor settings", ext)
        sys.exit(3)
    model = AutoModel.from_pretrained(path, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return ext, model, ck


def windows_of(n):
    if n <= WINDOW:
        return [0]
    pos = list(range(0, n - WINDOW + 1, WINDOW))
    if pos[-1] != n - WINDOW:
        pos.append(n - WINDOW)
    return pos


def embed(path, ext, model):
    import torch
    x = load_audio_16k(path)
    if len(x) == 0 or not np.isfinite(x).all():
        print("FATAL: bad audio", path); sys.exit(4)
    vecs = []
    for s in windows_of(len(x)):
        w = x[s:s + WINDOW]
        if len(w) < MIN_SAMPLES:
            w = np.pad(w, (0, MIN_SAMPLES - len(w)), mode="edge")
        inp = ext(w, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            h = model(inp.input_values).last_hidden_state.mean(dim=1).squeeze(0).numpy().astype(np.float32)
        vecs.append(h)
    return np.mean(np.stack(vecs), axis=0).astype(np.float32), len(vecs)


def sample_order():
    sm = pd.read_csv(WORKSPACE / "manifests/SAMPLE_MANIFEST.csv")
    return sm


def run_part(key, part, parts, threads):
    sm = sample_order().sort_values("seconds", ascending=False).reset_index(drop=True)
    mine = sm.iloc[part::parts]
    cache = safe_path(WORKSPACE / "features/cache" / key)
    cache.mkdir(parents=True, exist_ok=True)
    loaded = None
    done = 0
    for i, r in enumerate(mine.itertuples(), 1):
        out = cache / (r.sample_id + ".npy")
        meta = cache / (r.sample_id + ".json")
        if out.exists() and meta.exists():
            continue
        if loaded is None:
            loaded = load_encoder(key, threads)
        ext, model, ck = loaded
        vec, nwin = embed(DATA_INPUT / r.relative_path, ext, model)
        tmp = out.with_name(out.name + ".tmp%d" % os.getpid())
        with open(tmp, "wb") as f:
            np.save(f, vec, allow_pickle=False)
        tmp.replace(out)
        meta.write_text(json.dumps({"sample_id": r.sample_id, "speaker_audio_sha256": r.sha256_original, "windows": nwin, "dim": int(vec.shape[0]),
                                    "revision": ck["revision"], "npy_sha256": sha256_file(out), "created_utc": now_utc()}))
        done += 1
        if done % 20 == 0:
            print(key, "part", part, "done", done, "of", len(mine), flush=True)
    print(key, "part", part, "finished", done, flush=True)


def collect(key):
    sm = sample_order()
    cache = WORKSPACE / "features/cache" / key
    ck = json.loads((WORKSPACE / "features/manifests/encoder_checkpoints.json").read_text())[key]
    vecs, rows = [], []
    for r in sm.itertuples():
        out, meta = cache / (r.sample_id + ".npy"), cache / (r.sample_id + ".json")
        if not (out.exists() and meta.exists()):
            print("FATAL: missing", r.sample_id); sys.exit(5)
        m = json.loads(meta.read_text())
        if sha256_file(out) != m["npy_sha256"] or m["speaker_audio_sha256"] != r.sha256_original or m["revision"] != ck["revision"]:
            print("FATAL: cache entry inconsistent", r.sample_id); sys.exit(6)
        v = np.load(out, allow_pickle=False)
        if v.shape != (1024,) or not np.isfinite(v).all():
            print("FATAL: bad vector", r.sample_id); sys.exit(7)
        vecs.append(v); rows.append({"sample_id": r.sample_id, "windows": m["windows"]})
    arr = np.stack(vecs)
    p = safe_path(WORKSPACE / "features" / (key + "_embeddings.npy"))
    np.save(p, arr, allow_pickle=False)
    pd.DataFrame(rows).to_csv(safe_path(WORKSPACE / "features/manifests" / (key + "_sample_order.csv")), index=False)
    write_json(WORKSPACE / "features/manifests" / (key + "_manifest.json"), {
        "key": key, "model_id": ck["model_id"], "revision": ck["revision"], "checkpoint_file_sha256": ck["file_sha256"], "shape": list(arr.shape),
        "dtype": "float32", "pooling": "mean over time of last_hidden_state, then mean over 10-s windows", "window_seconds": 10.0,
        "code_sha256": sha256_file(__file__), "file_sha256": sha256_file(p), "created_utc": now_utc(), "validation": "all %d samples present, finite, hashes verified" % len(arr),
        "total_windows": int(sum(r["windows"] for r in rows))})
    print("collected", key, arr.shape)


if __name__ == "__main__":
    key = sys.argv[1]
    if sys.argv[2] == "collect":
        collect(key)
    else:
        run_part(key, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
