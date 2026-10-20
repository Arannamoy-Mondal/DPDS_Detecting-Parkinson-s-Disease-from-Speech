"""Shared helpers for the FINAL_V1 pipeline. Every write goes through safe_path()."""
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

WORKSPACE = Path("/root/Projects/Project4a_FINAL_V1_ISOLATED").resolve()
ORIGINAL = Path("/root/Projects/PatchTST").resolve()
ORIGINAL_DATA = ORIGINAL / "Italian Parkinson's Voice and speech"
DATA_INPUT = WORKSPACE / "data_input"
PROTOCOL_VERSION = "FINAL_V1"
SEED = 42
SAMPLE_RATE = 16000


def safe_path(path):
    """Resolve a destination and refuse to write outside the isolated workspace."""
    p = Path(path).resolve()
    if WORKSPACE != p and WORKSPACE not in p.parents:
        print("FATAL: refusing to write outside the isolated workspace:", p)
        sys.exit(2)
    return p


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def write_text(path, text):
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp%d" % os.getpid())
    tmp.write_text(text, encoding="utf8")
    tmp.replace(p)
    return p


def write_json(path, obj):
    return write_text(path, json.dumps(obj, indent=1, sort_keys=True, default=str))


def write_csv(path, rows, fields):
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp%d" % os.getpid())
    with open(tmp, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    tmp.replace(p)
    return p


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_audio_16k(path):
    """Read a WAV file, make it mono float32 and resample to 16 kHz (polyphase). No peak normalization."""
    import numpy as np
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(SAMPLE_RATE, sr)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype("float32")
    return x
