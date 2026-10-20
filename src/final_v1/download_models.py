"""Download the three frozen encoders into the workspace cache and record their revision hashes."""
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
os.environ["HF_HOME"] = str(WORKSPACE / "cache/hf")
from huggingface_hub import snapshot_download, HfApi
MODELS = {"WLM": "microsoft/wavlm-large", "W2V": "facebook/wav2vec2-large-lv60", "XLSR": "facebook/wav2vec2-xls-r-300m"}
out = {}
api = HfApi()
for key, name in MODELS.items():
    info = api.model_info(name)
    path = snapshot_download(name, allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt", "*.model"], cache_dir=str(WORKSPACE / "cache/hf"))
    files = {f.name: sha256_file(f) for f in Path(path).iterdir() if f.is_file() or f.is_symlink()}
    out[key] = {"model_id": name, "revision": info.sha, "snapshot": path, "file_sha256": files}
    print(key, name, info.sha, sorted(files))
write_json(WORKSPACE / "features/manifests/encoder_checkpoints.json", out)
