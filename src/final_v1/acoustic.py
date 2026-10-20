"""Hand-crafted acoustic features (frozen definition). 10 frame-level features x 4 statistics = 40 values per recording.

Frame: 25 ms Hann window, 10 ms hop, 512-point FFT, on 16-kHz audio (no peak normalization, no silence removal).
Frame features: log energy (dB of the RMS), zero-crossing rate, spectral centroid, spectral rolloff (85 %), spectral flatness,
log energy in five bands (0-500, 500-1000, 1000-2000, 2000-4000, 4000-8000 Hz).
Statistics over frames: mean, standard deviation, 10th and 90th percentile."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

FRAME, HOP, NFFT = 400, 160, 512
BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
FRAME_FEATURES = ["log_energy_db", "zcr", "centroid_hz", "rolloff85_hz", "flatness"] + ["band%d_log" % i for i in range(5)]
STATS = ["mean", "std", "p10", "p90"]
FEATURE_NAMES = ["ac_%s_%s" % (f, s) for f in FRAME_FEATURES for s in STATS]


def frame_features(x):
    if len(x) < FRAME:
        x = np.pad(x, (0, FRAME - len(x)), mode="edge")
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[idx].astype(np.float64)
    win = np.hanning(FRAME)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    log_e = 20 * np.log10(rms + 1e-10)
    zcr = (np.abs(np.diff(np.sign(fr), axis=1)) > 0).mean(axis=1)
    spec = np.abs(np.fft.rfft(fr * win, NFFT, axis=1)) ** 2
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
    tot = spec.sum(axis=1) + 1e-20
    centroid = (spec * freqs).sum(axis=1) / tot
    cum = np.cumsum(spec, axis=1)
    rolloff = freqs[(cum >= 0.85 * tot[:, None]).argmax(axis=1)]
    flat = np.exp(np.log(spec + 1e-20).mean(axis=1)) / (spec.mean(axis=1) + 1e-20)
    bands = [np.log10(spec[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) + 1e-20) for lo, hi in BANDS]
    return np.stack([log_e, zcr, centroid, rolloff, flat] + bands, axis=1)


def recording_vector(x):
    ff = frame_features(x)
    out = []
    for j in range(ff.shape[1]):
        c = ff[:, j]
        out += [c.mean(), c.std(), np.percentile(c, 10), np.percentile(c, 90)]
    return np.array(out, dtype=np.float64)


def _one(args):
    sid, rel = args
    return sid, recording_vector(load_audio_16k(DATA_INPUT / rel))


if __name__ == "__main__":
    from joblib import Parallel, delayed
    sm = pd.read_csv(WORKSPACE / "manifests/SAMPLE_MANIFEST.csv")
    res = Parallel(n_jobs=16)(delayed(_one)((r.sample_id, r.relative_path)) for r in sm.itertuples())
    df = pd.DataFrame({sid: v for sid, v in res}, index=FEATURE_NAMES).T
    df.index.name = "sample_id"
    df = df.loc[sm.sample_id]
    print("shape", df.shape, "| non-finite:", int((~np.isfinite(df.values)).sum()))
    p = safe_path(WORKSPACE / "features/acoustic_features.csv")
    df.to_csv(p)
    write_json(WORKSPACE / "features/manifests/acoustic_manifest.json", {"features": FEATURE_NAMES, "n_samples": len(df), "code_sha256": sha256_file(__file__),
                                                                           "file_sha256": sha256_file(p), "created_utc": now_utc(), "frame": FRAME, "hop": HOP, "nfft": NFFT})
