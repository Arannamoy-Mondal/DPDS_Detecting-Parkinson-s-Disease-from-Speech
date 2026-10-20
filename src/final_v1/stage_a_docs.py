"""Stage A: archive historical material (copy only), write the workspace manifest, audit and management files, make the Stage A PDF."""
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import make_pdf

ARCHIVE = WORKSPACE / "Archive/previous_versions"
COPY_ITEMS = ["docs", "experiments", "src", "scripts", "notebooks", "config", "artifacts", "archive_removed", "CHANGELOG.md", "PLAN.md", "README.md", "AGENTS.md",
              "claude_find_leakage.md", "pyproject.toml", "uv.lock", "Detecting Parkinson's Disease from Speech PP.pdf",
              "Detecting Parkinson's Disease from Speech_ WavLM (final layout) - Google Docs.pdf", "Project Guidelines.pdf", "Final_Prompt.md"]
REFERENCE_ONLY = ["data", "literature_review", "Italian Parkinson's Voice and speech.zip", ".venv"]


def main():
    rows = []
    n_files = 0
    for item in COPY_ITEMS:
        src = ORIGINAL / item
        if not src.exists():
            rows.append({"original_path": str(src), "isolated_path": "", "sha256": "", "bytes": "", "status": "MISSING"}); continue
        dst = safe_path(ARCHIVE / item)
        if src.is_dir():
            if not dst.exists():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            files = [p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            files = [src]
        for f in files:
            rel = f.relative_to(src) if src.is_dir() else Path(".")
            d = dst / rel if src.is_dir() else dst
            rows.append({"original_path": str(f), "isolated_path": str(d), "sha256": sha256_file(f), "bytes": f.stat().st_size,
                         "status": "COPIED_TO_ARCHIVE" if sha256_file(d) == sha256_file(f) else "COPY_HASH_MISMATCH"})
            n_files += 1
    for item in REFERENCE_ONLY:
        src = ORIGINAL / item
        rows.append({"original_path": str(src), "isolated_path": "(not copied; historical reference only)", "sha256": "", "bytes": "", "status": "REFERENCE_ONLY_NOT_COPIED"})
    # raw corpus copy already in data_input: record
    inv = pd.read_csv(WORKSPACE / "manifests/raw_file_inventory.csv", keep_default_na=False)
    for r in inv.itertuples():
        rows.append({"original_path": str(ORIGINAL_DATA / r.relative_path), "isolated_path": str(DATA_INPUT / r.relative_path), "sha256": r.sha256_original, "bytes": r.bytes,
                     "status": "COPIED_INPUT" if r.sha256_original == r.sha256_copy else "COPY_HASH_MISMATCH"})
    for p in sorted(ORIGINAL_DATA.rglob("*")):
        if p.is_file() and p.suffix.lower() != ".wav":
            dst = DATA_INPUT / p.relative_to(ORIGINAL_DATA)
            rows.append({"original_path": str(p), "isolated_path": str(dst), "sha256": sha256_file(p), "bytes": p.stat().st_size,
                         "status": "COPIED_INPUT" if sha256_file(p) == sha256_file(dst) else "COPY_HASH_MISMATCH"})
    write_csv(WORKSPACE / "WORKSPACE_MANIFEST.csv", rows, ["original_path", "isolated_path", "sha256", "bytes", "status"])
    bad = [r for r in rows if r["status"] == "COPY_HASH_MISMATCH"]
    print("manifest rows:", len(rows), "| archived files:", n_files, "| hash mismatches:", len(bad))
    return rows


main()
