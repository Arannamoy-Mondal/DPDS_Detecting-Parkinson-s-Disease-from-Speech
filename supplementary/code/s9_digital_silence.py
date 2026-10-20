"""S9 (descriptive): exact digital silence (samples equal to 0) in the raw files and its link to the 10th-percentile acoustic features.
No model is trained; class labels are used only to describe differences."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
import soundfile as sf
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(ROOT / "src/final_v1"))
import acoustic
sm = pd.read_csv(ROOT / "manifests/SAMPLE_MANIFEST.csv")
ac = pd.read_csv(ROOT / "features/acoustic_features.csv", index_col="sample_id")
rows = []
for r in sm.itertuples():
    raw, sr = sf.read(str(ROOT / "data_input" / r.relative_path), dtype="float32", always_2d=False)
    nz = np.flatnonzero(raw != 0)
    x16 = load_audio_16k(ROOT / "data_input" / r.relative_path)
    n = 1 + (len(x16) - acoustic.FRAME) // acoustic.HOP
    fr = np.lib.stride_tricks.sliding_window_view(x16, acoustic.FRAME)[::acoustic.HOP][:n]
    rms = np.sqrt((fr.astype(np.float64) ** 2).mean(1))
    rows.append({"sample_id": r.sample_id, "speaker_id": r.speaker_id, "label": r.label, "date": r.date, "sample_rate": r.sample_rate,
                 "leading_zero_ms": 1000 * (nz[0] / sr if len(nz) else len(raw) / sr), "trailing_zero_s": (len(raw) - 1 - nz[-1]) / sr if len(nz) else 0.0,
                 "zero_sample_fraction": float((raw == 0).mean()), "near_silent_frame_fraction": float((rms < 1e-6).mean()),
                 "ac_log_energy_db_p10": ac.loc[r.sample_id, "ac_log_energy_db_p10"], "ac_centroid_hz_p10": ac.loc[r.sample_id, "ac_centroid_hz_p10"]})
d = pd.DataFrame(rows); d.to_csv(RES / "S9_digital_silence.csv", index=False)
spk = d.groupby("speaker_id").agg(label=("label", "first"), trailing=("trailing_zero_s", "mean"), leading=("leading_zero_ms", "mean"), silent=("near_silent_frame_fraction", "mean"))
summ = {"recordings_with_trailing_zeros_ge_0.5s": {str(k): int(v) for k, v in d.assign(t=d.trailing_zero_s >= 0.5).groupby("label").t.sum().items()},
        "recordings_with_leading_zero_block_ge_5ms": {str(k): int(v) for k, v in d.assign(t=d.leading_zero_ms >= 5).groupby("label").t.sum().items()},
        "recordings_per_label": {str(k): int(v) for k, v in d.groupby("label").size().items()},
        "recordings_with_ge_10pct_near_silent_frames": {str(k): int(v) for k, v in d.assign(t=d.near_silent_frame_fraction >= 0.10).groupby("label").t.sum().items()},
        "speaker_AUROC_trailing_zero_seconds": float(roc_auc_score(spk.label, spk.trailing)), "speaker_AUROC_leading_zero_ms": float(roc_auc_score(spk.label, spk.leading)),
        "speaker_AUROC_near_silent_fraction": float(roc_auc_score(spk.label, spk.silent)),
        "spearman_silent_fraction_vs_energy_p10": float(d[["near_silent_frame_fraction", "ac_log_energy_db_p10"]].corr(method="spearman").iloc[0, 1]),
        "by_date": d.groupby(["label", "date", "sample_rate"]).agg(recordings=("sample_id", "size"), median_leading_ms=("leading_zero_ms", "median"), median_trailing_s=("trailing_zero_s", "median"),
                                                                   share_trailing_ge_0p5=("trailing_zero_s", lambda s: float((s >= 0.5).mean())), median_silent_frames=("near_silent_frame_fraction", "median")).reset_index().to_dict("records")}
json.dump(summ, open(RES / "S9_digital_silence_summary.json", "w"), indent=1, default=str)
print(json.dumps(summ, indent=1, default=str)[:3000])
