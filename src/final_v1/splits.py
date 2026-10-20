"""Speaker-disjoint, class-stratified fold construction (outer 5-fold seed 42; inner 3-fold seed 142+outer fold)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa


def assign_folds(speaker_ids, labels, n_folds, seed):
    """Return {speaker_id: fold}. Speakers of each class are shuffled with the seed and dealt round-robin;
    the second class continues the cycle where the first one stopped, so fold sizes differ by at most one."""
    ids = np.array(sorted(speaker_ids))
    lab = np.array([labels[s] for s in ids])
    rng = np.random.RandomState(seed)
    out, counter = {}, 0
    for cls in sorted(set(lab.tolist())):
        members = ids[lab == cls].copy()
        members = members[rng.permutation(len(members))]
        for s in members:
            out[str(s)] = counter % n_folds
            counter += 1
    return out


def outer_split():
    sm = pd.read_csv(WORKSPACE / "manifests/SAMPLE_MANIFEST.csv")
    labels = sm.drop_duplicates("speaker_id").set_index("speaker_id")["label"].to_dict()
    return sm, assign_folds(list(labels), labels, 5, SEED)


def inner_folds(train_speakers, labels, outer_fold, n_inner=3):
    return assign_folds(train_speakers, {s: labels[s] for s in train_speakers}, n_inner, 142 + outer_fold)


if __name__ == "__main__":
    sm, fold_of = outer_split()
    sm["outer_fold"] = sm.speaker_id.map(fold_of)
    rows = sm[["sample_id", "speaker_id", "label", "outer_fold", "task_code"]].copy()
    rows.insert(0, "protocol_version", PROTOCOL_VERSION)
    rows["split_seed"] = SEED
    write_csv(WORKSPACE / "SPLIT_MANIFEST.csv", rows.to_dict("records"), list(rows.columns))
    sp = sm.drop_duplicates("speaker_id")[["speaker_id", "label", "outer_fold"]]
    print(sp.groupby("outer_fold").label.agg(["count", "sum"]))
    print("split manifest sha256:", sha256_file(WORKSPACE / "SPLIT_MANIFEST.csv"))
