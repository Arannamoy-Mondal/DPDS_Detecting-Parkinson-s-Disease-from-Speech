"""ARTIFACT_MANIFEST.csv: path, category, size, SHA-256, modification time for every produced artifact (inputs, cache and archive are listed in WORKSPACE_MANIFEST.csv)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

SKIP = (".venv", "data_input", "Archive", "cache", "__pycache__", "tmp_smoke")
CAT = {"configs": "configuration", "manifests": "manifest", "features": "features", "predictions": "predictions", "metrics": "metrics", "reports": "report", "src": "code", "tests": "test",
       "logs": "log", "models": "model"}


def main():
    rows = []
    for p in sorted(WORKSPACE.rglob("*")):
        if not p.is_file() or any(s in p.parts for s in SKIP):
            continue
        rel = p.relative_to(WORKSPACE)
        top = rel.parts[0]
        rows.append({"path": str(rel), "category": CAT.get(top, "management" if len(rel.parts) == 1 else top), "bytes": p.stat().st_size, "sha256": sha256_file(p),
                     "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))})
    write_csv(WORKSPACE / "ARTIFACT_MANIFEST.csv", rows, ["path", "category", "bytes", "sha256", "modified_utc"])
    print("artifacts:", len(rows))


main()
