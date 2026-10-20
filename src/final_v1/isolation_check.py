"""Isolation acceptance test: workspace distinct from the original, original inputs unchanged, original project file list unchanged since the start snapshot.
Usage: python isolation_check.py LABEL"""
import subprocess
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa


def snapshot():
    out = subprocess.run("cd %s && find . -path ./.venv -prune -o -path ./.git -prune -o -name __pycache__ -prune -o -type f -printf '%%p\\t%%s\\t%%T@\\n' | sort" % ORIGINAL,
                         shell=True, capture_output=True, text=True).stdout
    return out


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "check"
    res = {"label": label, "time_utc": now_utc(), "original": str(ORIGINAL), "workspace": str(WORKSPACE)}
    res["workspace_distinct_and_not_inside_original"] = bool(WORKSPACE != ORIGINAL and ORIGINAL not in WORKSPACE.parents and WORKSPACE not in ORIGINAL.parents)
    inv = pd.read_csv(WORKSPACE / "manifests/raw_file_inventory.csv", keep_default_na=False)
    changed = [r.relative_path for r in inv.itertuples() if sha256_file(ORIGINAL_DATA / r.relative_path) != r.sha256_original]
    res["original_audio_files_checked"] = int(len(inv)); res["original_audio_files_changed"] = changed
    start = (WORKSPACE / "manifests/original_project_snapshot_START.tsv").read_text().splitlines()
    now = snapshot().splitlines()
    s, n = set(start), set(now)
    res["original_project_files_added_or_changed_since_start"] = sorted(n - s)[:50]
    res["original_project_files_removed_or_changed_since_start"] = sorted(s - n)[:50]
    res["passed"] = bool(res["workspace_distinct_and_not_inside_original"] and not changed and not (n - s) and not (s - n))
    write_json(WORKSPACE / ("manifests/isolation_check_%s.json" % label), res)
    print(json.dumps(res, indent=1)[:1500])


main()
