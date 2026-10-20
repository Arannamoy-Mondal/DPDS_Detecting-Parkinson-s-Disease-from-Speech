"""Stage A: build the audited speaker/sample manifests from the raw inventory (no model, no label-dependent step)."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

LABEL = {"PD": 1, "HC_ELDERLY": 0}


def canonical_person(df):
    """Same folder name + same sex + same age (or age typo) + letter-anagram code -> one person."""
    keyed = {}
    for (sub, folder, code, sex), g in df.groupby(["sub_folder", "person_folder", "name_code", "sex"]):
        keyed[(sub, folder, code, sex)] = g
    people = {}   # canonical key -> list of source keys
    for k in keyed:
        sub, folder, code, sex = k
        placed = False
        for ck in people:
            if ck[1] == folder and ck[3] == sex and sorted(ck[2]) == sorted(code):
                people[ck].append(k); placed = True; break
        if not placed:
            people[k] = [k]
    return people


def main():
    inv = pd.read_csv(WORKSPACE / "manifests/raw_file_inventory.csv", keep_default_na=False)
    inv["sub_folder"] = inv["sub_folder"].astype(str)
    notes = []
    excluded = inv[inv.group == "HC_YOUNG"]
    use = inv[inv.group.isin(LABEL)].copy()
    # person identity
    person_of = {}
    rows_p = []
    pd_people = canonical_person(use[use.group == "PD"])
    hc_people = canonical_person(use[use.group == "HC_ELDERLY"])
    n = {"PD": 0, "HC_ELDERLY": 0}
    for group, people in [("PD", pd_people), ("HC_ELDERLY", hc_people)]:
        for ck, srcs in sorted(people.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            n[group] += 1
            pid = "%s_%03d" % ("PD" if group == "PD" else "HC", n[group])
            for s in srcs:
                person_of[(group,) + s] = pid
            rows_p.append({"speaker_id": pid, "group": group, "label": LABEL[group], "source_keys": ";".join("/".join(x) for x in srcs),
                           "merged_sources": len(srcs), "folder_name_hash": sha256_text(ck[1])[:10]})
    use["speaker_id"] = [person_of[("%s" % r.group, r.sub_folder, r.person_folder, r.name_code, r.sex)] for r in use.itertuples()]
    use["label"] = use["group"].map(LABEL)
    use = use.sort_values("relative_path").reset_index(drop=True)
    use["sample_id"] = ["S_" + sha256_text(p)[:16] for p in use["relative_path"]]
    # audio validity + duplicate-content checks
    stats = []
    pcm_hash = {}
    for r in use.itertuples():
        x, sr = sf.read(str(DATA_INPUT / r.relative_path), dtype="float32", always_2d=False)
        ok = bool(np.isfinite(x).all()) and len(x) > 0
        pcm_hash[r.sample_id] = sha256_text(x.tobytes().hex()[:2000000]) if False else hash_bytes(x)
        stats.append({"sample_id": r.sample_id, "finite_nonempty": int(ok), "peak": float(np.abs(x).max()), "rms": float(np.sqrt((x ** 2).mean())),
                      "frac_zero": float((x == 0).mean()), "frac_clipped": float((np.abs(x) >= 0.999).mean())})
    st = pd.DataFrame(stats)
    use = use.merge(st, on="sample_id", validate="one_to_one")
    use["pcm_sha256"] = use["sample_id"].map(pcm_hash)
    dup_files = use[use.duplicated("sha256_original", keep=False)]
    dup_second = use[use.duplicated("sha256_original", keep="first")]
    dup_removed = dup_second[["sample_id", "relative_path", "speaker_id", "task_code", "sha256_original"]].to_dict("records")
    use_dups = use.copy()
    use = use[~use.duplicated("sha256_original", keep="first")].copy()
    dup_pcm = use[use.duplicated("pcm_sha256", keep=False)]
    dup_ids = use[use.duplicated("sample_id", keep=False)]
    # speaker tables
    sp = pd.DataFrame(rows_p)
    cnt = use.groupby("speaker_id").agg(n_recordings=("sample_id", "size"), tasks=("task_code", lambda s: ",".join(sorted(set(s)))),
                                         dates=("date", lambda s: ",".join(sorted(set(s)))), age_names=("age_in_name", lambda s: ",".join(sorted(set(map(str, s))))),
                                         sex=("sex", lambda s: ",".join(sorted(set(s)))), hours=("seconds", lambda s: round(s.sum() / 3600, 4)),
                                         sample_rates=("sample_rate", lambda s: ",".join(sorted(set(map(str, s)))))).reset_index()
    sp = sp.merge(cnt, on="speaker_id")
    write_csv(WORKSPACE / "manifests/SPEAKER_MANIFEST.csv", sp.to_dict("records"), list(sp.columns))
    fields = ["sample_id", "speaker_id", "label", "group", "task_code", "date", "time", "sample_rate", "seconds", "frames", "relative_path",
              "sha256_original", "pcm_sha256", "peak", "rms", "frac_zero", "frac_clipped", "finite_nonempty", "age_in_name", "sex"]
    use = use.sort_values(["speaker_id", "task_code", "relative_path"]).reset_index(drop=True)
    inv_out = use[fields].copy()
    inv_out.insert(0, "protocol_version", PROTOCOL_VERSION)
    inv_out["included"] = 1
    inv_out["inclusion_reason"] = "PD or elderly healthy control; all tasks"
    ex = excluded[["relative_path", "group", "task_code", "date", "sample_rate", "seconds", "sha256_original"]].copy()
    ex.insert(0, "protocol_version", PROTOCOL_VERSION)
    ex["included"] = 0
    ex["inclusion_reason"] = "EXCLUDED: young healthy control (age 19-29; age confound; only 3 tasks)"
    ex["sample_id"] = ["X_" + h[:16] for h in ex["sha256_original"]]
    ex["speaker_id"] = ""
    dd = dup_second.copy()
    dd["included"] = 0
    dd["inclusion_reason"] = "EXCLUDED: byte-identical duplicate of another recording of the same speaker (first copy kept)"
    ex["label"] = ""
    allrows = pd.concat([inv_out, ex, dd], ignore_index=True)
    cols = ["protocol_version", "sample_id", "speaker_id", "label", "group", "task_code", "date", "time", "sample_rate", "seconds", "frames", "relative_path",
            "sha256_original", "pcm_sha256", "peak", "rms", "frac_zero", "frac_clipped", "finite_nonempty", "age_in_name", "sex", "included", "inclusion_reason"]
    write_csv(WORKSPACE / "DATA_INVENTORY.csv", allrows.to_dict("records"), cols)
    write_csv(WORKSPACE / "manifests/SAMPLE_MANIFEST.csv", inv_out.to_dict("records"), list(inv_out.columns))
    summary = {
        "files_total": int(len(inv)), "files_included": int(len(use)), "files_excluded_young": int(len(excluded)),
        "speakers_total": int(sp.shape[0]), "speakers_pd": int((sp.label == 1).sum()), "speakers_hc": int((sp.label == 0).sum()),
        "recordings_pd": int((use.label == 1).sum()), "recordings_hc": int((use.label == 0).sum()),
        "audio_hours": round(float(use.seconds.sum() / 3600), 3), "tasks": dict(Counter(use.task_code)),
        "merged_speakers": [r for r in rows_p if r["merged_sources"] > 1],
        "age_typo_speakers": [r["speaker_id"] for r in rows_p if "," in str(sp[sp.speaker_id == r["speaker_id"]].age_names.iloc[0])],
        "duplicate_files_by_bytes": int(len(dup_files)), "duplicates_excluded": dup_removed, "duplicate_files_by_pcm": int(len(dup_pcm)), "duplicate_sample_ids": int(len(dup_ids)), "files_total_group_pd_hc": int(len(use_dups)),
        "non_finite_or_empty": int((use.finite_nonempty == 0).sum()), "sample_rates": dict(Counter(map(int, use.sample_rate))),
        "speakers_with_fewer_than_16_recordings": sp[sp.n_recordings < 16][["speaker_id", "n_recordings"]].to_dict("records"),
        "date_by_label": {str(k): sorted(set(v)) for k, v in use.groupby("label").date},
        "class_prevalence_speakers": float((sp.label == 1).mean()), "class_prevalence_recordings": float((use.label == 1).mean()),
        "dates_one_class_only": bool(all(g.label.nunique() == 1 for _, g in use.groupby("date"))),
        "sr_44100_all_pd": bool((use[use.sample_rate == 44100].label == 1).all()),
    }
    write_json(WORKSPACE / "manifests/stage_a_summary.json", summary)
    print(json.dumps(summary, indent=1, default=str)[:3500])


def hash_bytes(x):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


main()
