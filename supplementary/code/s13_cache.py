"""S13 step 1 (amendment A3): cache the output of frozen WavLM-large layers 0-19 for every 10-second window.

Same audio recipe as FINAL_V1 (extract_embeddings.py). The cache is float16 and lives under cache/ (not a result).
Check required before training (PREDECLARED_A3.md): running layers 20-23 and the final layer norm on the cache must
reproduce the saved FV1 WLM embeddings (max |diff| < 1e-2, cosine > 0.9999).
"""
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
os.environ["HF_HOME"] = str(ROOT / "cache/hf")
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoFeatureExtractor, AutoModel
from extract_embeddings import windows_of, WINDOW, MIN_SAMPLES

SPLIT_AT = 20
CACHE = ROOT / "cache/ft_wlm_L20"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_model(device):
    ck = json.loads((ROOT / "features/manifests/encoder_checkpoints.json").read_text())["WLM"]
    ext = AutoFeatureExtractor.from_pretrained(ck["snapshot"])
    model = AutoModel.from_pretrained(ck["snapshot"], dtype=torch.float32).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return ext, model, ck


def raw_bias(model, T, bsz):
    att = model.encoder.layers[0].attention
    b = att.compute_bias(T, T)
    return b.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * att.num_heads, T, T)


def prefix(model, input_values):
    f = model.feature_extractor(input_values).transpose(1, 2)
    h, _ = model.feature_projection(f)
    enc = model.encoder
    h = enc.dropout(h + enc.pos_conv_embed(h))
    pb = None
    for i in range(SPLIT_AT):
        h, pb = enc.layers[i](h, position_bias=pb)
    return h


def suffix(model, h, start=SPLIT_AT):
    pb = raw_bias(model, h.shape[1], h.shape[0])
    for i in range(start, len(model.encoder.layers)):
        h, pb = model.encoder.layers[i](h, position_bias=pb)
    return model.encoder.layer_norm(h)


def windows(x, ext):
    for s in windows_of(len(x)):
        w = x[s:s + WINDOW]
        if len(w) < MIN_SAMPLES:
            w = np.pad(w, (0, MIN_SAMPLES - len(w)), mode="edge")
        yield ext(w, sampling_rate=16000, return_tensors="pt").input_values


def main():
    t0 = time.time()
    device = "cuda"
    ext, model, ck = load_model(device)
    sm = pl.load_manifest()
    saved, _ = pl.load_component("WLM", sm)
    CACHE.mkdir(parents=True, exist_ok=True)
    chunks, index = [], []
    start = 0
    check_rows = []
    for n, r in enumerate(sm.itertuples()):
        x = load_audio_16k(DATA_INPUT / r.relative_path)
        full_means, cache_means = [], []
        for wi, inp in enumerate(windows(x, ext)):
            with torch.inference_mode():
                inp = inp.to(device)
                h = prefix(model, inp)
                hf16 = h.to(torch.float16)
                chunks.append(hf16[0].cpu().numpy())
                T = h.shape[1]
                index.append({"sample_id": r.sample_id, "speaker_id": r.speaker_id, "label": int(r.label), "task_code": r.task_code,
                              "window": wi, "start": start, "frames": T})
                start += T
                if n < 40:  # check: full fp32 path and cached fp16 path against the saved embedding
                    full_means.append(model(inp).last_hidden_state.mean(dim=1)[0].double().cpu().numpy())
                    cache_means.append(suffix(model, hf16.float()).mean(dim=1)[0].double().cpu().numpy())
        if n < 40:
            v_full, v_cache, v_saved = np.mean(full_means, 0), np.mean(cache_means, 0), saved[n]
            cos = float(v_cache @ v_saved / np.linalg.norm(v_cache) / np.linalg.norm(v_saved))
            check_rows.append({"sample_id": r.sample_id, "max_abs_full_vs_saved": float(np.abs(v_full - v_saved).max()),
                               "max_abs_cache_vs_saved": float(np.abs(v_cache - v_saved).max()), "cosine_cache_vs_saved": cos})
        if (n + 1) % 50 == 0:
            print("recordings", n + 1, "of", len(sm), "windows", len(index), "%.0f s" % (time.time() - t0), flush=True)
    arr = np.concatenate(chunks, axis=0)
    np.save(CACHE / "hidden_L20_fp16.npy", arr)
    pd.DataFrame(index).to_csv(CACHE / "window_index.csv", index=False)
    chk = pd.DataFrame(check_rows)
    chk.to_csv(RES / "S13_cache_check.csv", index=False)
    ok = bool(chk.max_abs_cache_vs_saved.max() < 1e-2 and chk.cosine_cache_vs_saved.min() > 0.9999)
    json.dump({"windows": len(index), "frames": int(arr.shape[0]), "split_at_layer": SPLIT_AT, "revision": ck["revision"],
               "check_recordings": len(chk), "max_abs_full_vs_saved": float(chk.max_abs_full_vs_saved.max()),
               "max_abs_cache_vs_saved": float(chk.max_abs_cache_vs_saved.max()), "min_cosine_cache_vs_saved": float(chk.cosine_cache_vs_saved.min()),
               "check_passed": ok, "runtime_seconds": time.time() - t0}, open(RES / "S13_cache_summary.json", "w"), indent=2)
    print("cache written", arr.shape, "check", "PASS" if ok else "FAIL", "%.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
