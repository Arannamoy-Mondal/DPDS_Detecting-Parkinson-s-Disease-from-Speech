"""Stage A: copy the raw corpus into the workspace (read-only source), hash it, parse file names, list anomalies."""
import re
import shutil
import sys
import wave
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

TASK = r"(B1|B2|D1|D2|PR1|PR11|FB1|V[AEIOU][12])"
PATTERN = re.compile(r"^" + TASK + r"([A-Z]+)(\d{2})([MF])(\d{6,8})(\d{4})\.+WAV$")
GROUPS = {"28 People with Parkinson's disease": "PD", "22 Elderly Healthy Control": "HC_ELDERLY", "15 Young Healthy Control": "HC_YOUNG"}


def main():
    rows = []
    wav_files = sorted(ORIGINAL_DATA.rglob("*.wav"))
    other = sorted(p for p in ORIGINAL_DATA.rglob("*") if p.is_file() and p.suffix.lower() != ".wav")
    print("wav files found:", len(wav_files), "| other files:", len(other))
    # copy (original stays untouched)
    for src in wav_files + other:
        rel = src.relative_to(ORIGINAL_DATA)
        dst = safe_path(DATA_INPUT / rel)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    nonmatch = []
    for src in wav_files:
        rel = src.relative_to(ORIGINAL_DATA)
        dst = DATA_INPUT / rel
        group_folder = rel.parts[0]
        group = GROUPS[group_folder]
        norm = src.name.upper().replace(" ", "").replace("_", "")
        m = PATTERN.match(norm)
        with wave.open(str(dst), "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        row = {"relative_path": str(rel), "group_folder": group_folder, "person_folder": rel.parts[-2] if len(rel.parts) > 2 else "",
               "sub_folder": rel.parts[1] if len(rel.parts) > 3 else "", "group": group, "bytes": src.stat().st_size,
               "sha256_original": sha256_file(src), "sha256_copy": sha256_file(dst), "sample_rate": sr, "channels": ch,
               "sample_width_bytes": sw, "frames": n, "seconds": round(n / sr, 3)}
        if m:
            task, code, age, sex, dpart, hhmm = m.groups()
            if len(dpart) == 6: dd, mm, yyyy = dpart[:2], dpart[2:4], "20" + dpart[4:]
            else: dd, mm, yyyy = dpart[:2], dpart[2:4], dpart[4:]
            row.update({"task_code": task, "name_code": code, "name_token": code + age + sex, "age_in_name": age, "sex": sex, "date": "%s-%s-%s" % (yyyy, mm, dd), "time": hhmm, "parse_ok": 1})
        else:
            row.update({"parse_ok": 0}); nonmatch.append(str(rel))
        rows.append(row)
    fields = ["relative_path", "group_folder", "sub_folder", "person_folder", "group", "task_code", "name_code", "name_token", "age_in_name", "sex", "date", "time",
              "sample_rate", "channels", "sample_width_bytes", "frames", "seconds", "bytes", "sha256_original", "sha256_copy", "parse_ok"]
    write_csv(WORKSPACE / "manifests/raw_file_inventory.csv", rows, fields)
    print("non-matching file names:", len(nonmatch)); [print("  ", n) for n in nonmatch[:20]]
    bad_copy = [r["relative_path"] for r in rows if r["sha256_original"] != r["sha256_copy"]]
    print("copy hash mismatches:", len(bad_copy))
    # candidate speakers: folder + token
    by = defaultdict(set)
    for r in rows:
        if r["parse_ok"]:
            by[(r["group"], r["sub_folder"] if r["group"]=="PD" else "", r["person_folder"])].add(r["name_token"])
    print("person folders:", len(by), "| folders with >1 token:")
    for k, v in sorted(by.items()):
        if len(v) > 1: print("  ", k, sorted(v))
    tok = defaultdict(set)
    for r in rows:
        if r["parse_ok"]:
            if r["group"] != "HC_YOUNG": tok[r["name_code"]].add((r["group"], r["person_folder"], r["sub_folder"]))
    print("tokens present in >1 folder:")
    for k, v in sorted(tok.items()):
        if len(v) > 1: print("  ", k, sorted(v))
    cnt = defaultdict(int)
    for r in rows: cnt[(r["group"], r["sample_rate"])] += 1
    print(dict(cnt))

main()
