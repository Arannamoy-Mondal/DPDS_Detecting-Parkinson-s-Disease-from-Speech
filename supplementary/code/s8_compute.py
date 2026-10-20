"""S8: CPU inference benchmark (deployment cost): encoder forward time for one 10-s window at 1, 4 and 8 threads, model size on disk,
acoustic-feature time per minute of audio, classifier inference time."""
import os, sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
os.environ["HF_HOME"] = str(ROOT / "cache/hf"); os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoModel, AutoFeatureExtractor
import acoustic
ck = json.load(open(ROOT / "features/manifests/encoder_checkpoints.json"))
rng = np.random.RandomState(0); x = (0.05 * rng.randn(160000)).astype(np.float32)
rows = []
for k, v in ck.items():
    ext = AutoFeatureExtractor.from_pretrained(v["snapshot"]); m = AutoModel.from_pretrained(v["snapshot"]).eval()
    size = sum(os.path.getsize(os.path.join(v["snapshot"], f)) for f in os.listdir(v["snapshot"]) if f.endswith((".bin", ".safetensors")))
    inp = ext(x, sampling_rate=16000, return_tensors="pt")
    for th in [1, 4, 8]:
        torch.set_num_threads(th)
        with torch.inference_mode():
            m(inp.input_values)  # warm-up
            ts = []
            for _ in range(5):
                t = time.perf_counter(); m(inp.input_values); ts.append(time.perf_counter() - t)
        rows.append({"encoder": k, "model_id": v["model_id"], "threads": th, "seconds_per_10s_window_median": float(np.median(ts)), "real_time_factor": float(np.median(ts) / 10.0),
                     "weights_MB": size / 1e6, "parameters": int(sum(p.numel() for p in m.parameters()))})
        print(rows[-1], flush=True)
    del m
y = (0.05 * rng.randn(16000 * 60)).astype(np.float32)
t = time.perf_counter(); acoustic.recording_vector(y); ta = time.perf_counter() - t
rows.append({"encoder": "AC (40 acoustic features)", "threads": 1, "seconds_per_minute_of_audio": ta})
pd.DataFrame(rows).to_csv(RES / "S8_compute.csv", index=False)
print("acoustic seconds per minute:", ta)
