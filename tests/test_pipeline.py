"""Stage C code-level checks. Synthetic features only; outputs go to tests/tmp_smoke and are NOT experiment results.
Run: .venv/bin/python tests/test_pipeline.py"""
import json, sys, time, shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/root/Projects/Project4a_FINAL_V1_ISOLATED")
sys.path.insert(0, str(ROOT / "src/final_v1"))
from common import *  # noqa
import protocol_defs as P
import pipeline as pl
import runner
from splits import assign_folds

results = []
TMP = ROOT / "tests/tmp_smoke"
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True)


def check(name, ok, detail=""):
    results.append({"test": name, "passed": bool(ok), "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name, detail)


sm = pl.load_manifest()
split = pd.read_csv(ROOT / "SPLIT_MANIFEST.csv")
spk_fold = split.drop_duplicates("speaker_id").set_index("speaker_id").outer_fold
lab = sm.drop_duplicates("speaker_id").set_index("speaker_id").label

# T1 labels and positive class
pd_path = sm.relative_path.str.startswith("28 People")
check("T01 label encoding: PD folder -> 1, elderly folder -> 0", ((sm.label == 1) == pd_path).all() and set(sm.label) == {0, 1})
check("T01b positive class is PD (label 1) and counts 25/22 speakers", (lab == 1).sum() == 25 and (lab == 0).sum() == 22)

# T2 splits
folds = {f: set(spk_fold[spk_fold == f].index) for f in range(5)}
check("T02 outer folds speaker-disjoint and cover all 47 speakers", sum(len(v) for v in folds.values()) == 47 and len(set.union(*folds.values())) == 47)
check("T02b every outer test fold contains both classes", all(set(lab[list(v)]) == {0, 1} for v in folds.values()), {f: len(v) for f, v in folds.items()})
check("T02c every recording sits in its speaker's fold", (split.set_index("sample_id").loc[sm.sample_id].outer_fold.values == sm.speaker_id.map(spk_fold).values).all())
ok_inner = True
for f in range(5):
    tr = sorted(set(sm.speaker_id) - folds[f])
    im = assign_folds(tr, {s: int(lab[s]) for s in tr}, 3, 142 + f)
    for k in range(3):
        mem = [s for s in tr if im[s] == k]
        ok_inner &= set(lab[mem]) == {0, 1} and not (set(mem) & folds[f])
check("T02d inner folds speaker-disjoint, inside the training partition, both classes in each inner validation fold", ok_inner)
check("T02e split is reproducible (same seed -> same assignment)", assign_folds(list(lab.index), lab.to_dict(), 5, 42) == spk_fold.to_dict())

# T3 duplicates
check("T03 sample_ids unique, no duplicated source audio, no speaker in two groups",
      sm.sample_id.is_unique and not sm.sha256_original.duplicated().any() and sm.groupby("speaker_id").group.nunique().max() == 1)

# T4 alignment and missing values
Xac, names = pl.load_representation("AC", sm)
check("T04 acoustic matrix aligned by sample_id, 40 columns, no missing", Xac.shape == (len(sm), 40) and np.isfinite(Xac).all() and len(names) == 40)
ac = pd.read_csv(ROOT / "features/acoustic_features.csv")
check("T04b acoustic file sample_id order equals manifest", (ac.sample_id.values == sm.sample_id.values).all())

# T5 metrics on a toy example
m = pl.binary_metrics(np.array([0.9, 0.8, 0.3, 0.2, 0.6]), np.array([1, 1, 1, 0, 0]))
check("T05 metrics toy example (TP2 FN1 TN1 FP1)", (m["tp"], m["fn"], m["tn"], m["fp"]) == (2, 1, 1, 1) and abs(m["f1"] - 2 / 3) < 1e-12 and abs(m["accuracy"] - 0.6) < 1e-12
      and abs(m["balanced_accuracy"] - (2 / 3 + 0.5) / 2) < 1e-12 and abs(m["auroc"] - 5 / 6) < 1e-12, m)
check("T05b average precision toy", abs(pl.speaker_auprc(np.array([0.9, 0.1, 0.8, 0.2]), np.array(["a", "b", "c", "d"]), np.array([1, 0, 1, 0])) - 1.0) < 1e-12)
ts = pl.speaker_scores(np.array([0.2, 0.4, 0.9]), np.array(["a", "a", "b"]), np.array([0, 0, 1]))
check("T05c speaker score is the mean of recording probabilities", abs(ts[0][0] - 0.3) < 1e-12 and abs(ts[0][1] - 0.9) < 1e-12)

# T6 fit boundaries: standardization uses training rows only; test rows do not influence each other
rng = np.random.RandomState(0)
Xtr = rng.randn(60, 5); ytr = (Xtr[:, 0] + 0.5 * rng.randn(60) > 0).astype(int); Xte = rng.randn(10, 5)
p1 = pl.fit_predict("LR", {"C": 1, "penalty": "l2", "class_weight": None}, Xtr, ytr, Xte)
Xte2 = Xte.copy(); Xte2[5] *= 1000
p2 = pl.fit_predict("LR", {"C": 1, "penalty": "l2", "class_weight": None}, Xtr, ytr, Xte2)
check("T06 changing one test row leaves the other test predictions unchanged (no test statistics in scaling)", np.allclose(np.delete(p1, 5), np.delete(p2, 5)))

# T7 instrumented run: every fit sees only rows of its own training partition; selection sees only training speakers
Xs = np.hstack([np.arange(len(sm))[:, None].astype(float), np.random.RandomState(1).randn(len(sm), 39)])  # column 0 = row id, constant per row
calls, sel_calls = [], []
orig_fp, orig_sel = pl.fit_predict, pl.select_features
def spy_fp(model, params, a, y, b):
    calls.append((set(a[:, 0].astype(int)), set(b[:, 0].astype(int)))); return orig_fp(model, params, a, y, b)
def spy_sel(X, y, spk, method, seed):
    sel_calls.append(set(spk)); return orig_sel(X, y, spk, method, seed)
pl.fit_predict, pl.select_features = spy_fp, spy_sel
cond = {"model": "LR", "feature_selection": "ALL", "representation": "AC", "condition_id": "x"}
out = pl.run_fold(cond, 1, Xs, ["f%d" % i for i in range(40)], sm)
pl.fit_predict, pl.select_features = orig_fp, orig_sel
of = split.set_index("sample_id").loc[sm.sample_id].outer_fold.values
test_rows = set(np.where(of == 1)[0]); train_rows = set(np.where(of != 1)[0])
final_calls = [c for c in calls if c[1] == test_rows or (c[1] & test_rows)]
check("T07 outer-test rows appear only as the prediction target of the single final fit", len(final_calls) == 1 and final_calls[0][1] == test_rows and not (final_calls[0][0] & test_rows))
tuning = [c for c in calls if not (c[1] & test_rows) and len(c[1]) != 0 and c is not final_calls[0]]
fs_internal = sum(1 for c in tuning if len(c[0]) < 30)
check("T07b every tuning fit trains and validates on outer-training rows only; inner train/validation disjoint", all(not (a & test_rows) and not (b & test_rows) and not (a & b) for a, b in tuning), "n tuning fits %d" % len(tuning))
test_speakers = set(sm.speaker_id.values[sorted(test_rows)])
sel_calls.clear()
pl.select_features = spy_sel
pl.run_fold({"model": "LR", "feature_selection": "FORWARD", "representation": "AC", "condition_id": "x"}, 1, Xs, ["f%d" % i for i in range(40)], sm)
pl.select_features = orig_sel
check("T07c feature selection never sees an outer-test speaker (4 selections: 3 inner + 1 final)", len(sel_calls) == 4 and all(not (s & test_speakers) for s in sel_calls))
check("T07d the selector for the final model used the whole outer-training partition", len(sel_calls[-1]) == 47 - len(test_speakers), len(sel_calls[-1]))

# T8 hyperparameter configurations are valid and give probabilities
Xt = np.random.RandomState(2).randn(80, 12); yt = (Xt[:, 0] > 0).astype(int)
bad = []
for model in P.MODELS:
    for params in P.grid(model):
        try:
            p = pl.fit_predict(model, params, Xt[:60], yt[:60], Xt[60:])
            if not (np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1 and len(p) == 20):
                bad.append((model, params))
        except Exception as e:
            bad.append((model, params, repr(e)))
check("T08 all 78 frozen configurations fit and return finite probabilities in [0,1]", not bad and sum(len(P.grid(m)) for m in P.MODELS) == 78, bad[:2])

# T9 null and positive controls through the whole nested loop (synthetic features, real manifest/labels)
def pooled_auroc(X, rid_prefix):
    preds = []
    for f in range(5):
        rid = "%s-F%d" % (rid_prefix, f)
        r = runner.run_one(rid, X, ["f%d" % i for i in range(X.shape[1])], sm, TMP, *runner.frozen_hashes())
        if r["status"] != "COMPLETED":
            return None, r
        preds.append(pd.read_csv(TMP / "predictions/FINAL_V1" / (rid + ".csv")))
    p = pd.concat(preds)
    sc, yt, _ = pl.speaker_scores(p.predicted_probability.values, p.speaker_id.values, p.true_label.values)
    return pl.binary_metrics(sc, yt)["auroc"], None
Xnull = np.random.RandomState(3).randn(len(sm), 40)
a, err = pooled_auroc(Xnull, "FV1-AC-ALL-LR")
check("T09 null control (random features, true labels): pooled speaker AUROC near chance", a is not None and 0.25 < a < 0.75, a)
ysig = sm.label.values[:, None] * 1.0
Xpos = np.hstack([ysig * 2 + 0.3 * np.random.RandomState(4).randn(len(sm), 1), np.random.RandomState(5).randn(len(sm), 39)])
a2, err2 = pooled_auroc(Xpos, "FV1-AC-ALL-LSVM")
check("T09b positive control (label-informative feature): pooled AUROC high", a2 is not None and a2 > 0.95, a2)

# T10 resume: second call returns the stored result without recomputation
r1 = json.loads((TMP / "metrics/FINAL_V1/FV1-AC-ALL-LR-F0.json").read_text())
r2 = runner.run_one("FV1-AC-ALL-LR-F0", Xnull, ["f%d" % i for i in range(40)], sm, TMP, *runner.frozen_hashes())
check("T10 checkpoint/resume: completed run is reused, not recomputed", r2["started_utc"] == r1["started_utc"] and r2["status"] == "COMPLETED")

# T11 failure injection: a corrupted run (test speaker also in training) is quarantined and marked FAILED
orig_rf = pl.run_fold
def bad_run_fold(cond, fold, X, names, sm):
    o = orig_rf(cond, fold, X, names, sm)
    other = sm[split.set_index("sample_id").loc[sm.sample_id].outer_fold.values != fold].iloc[:1]
    row = o["pred"].iloc[:1].copy(); row["sample_id"] = other.sample_id.values[0]; row["speaker_id"] = other.speaker_id.values[0]
    o["pred"] = pd.concat([o["pred"], row]); return o
pl.run_fold = bad_run_fold
r3 = runner.run_one("FV1-AC-ALL-LR-F4", Xnull, ["f%d" % i for i in range(40)], sm, TMP / "inject", *runner.frozen_hashes())
pl.run_fold = orig_rf
check("T11 corrupted prediction file fails the gates and is quarantined (not repaired)", r3["status"] == "FAILED" and not r3["gates"]["no_speaker_overlap_train_test"]
      and (TMP / "inject/predictions/FINAL_V1/quarantine/FV1-AC-ALL-LR-F4.csv").exists() and not (TMP / "inject/predictions/FINAL_V1/FV1-AC-ALL-LR-F4.csv").exists(), r3.get("failure_reason"))

# T12 hashing and registry fields
h1, h2 = sha256_file(ROOT / "configs/FINAL_V1/protocol.json"), sha256_file(ROOT / "configs/FINAL_V1/protocol.json")
reg = pd.read_csv(ROOT / "EXPERIMENT_REGISTRY.csv")
check("T12 hashing is stable; registry has 280 fold runs, 56 condition rows and the blocked FWL row", h1 == h2 and (reg.row_type == "fold_run").sum() == 280 and (reg.row_type == "condition_pooled").sum() == 56
      and (reg.row_type == "blocked_experiment").sum() == 1)
check("T12b frozen hashes equal the files on disk", runner.frozen_hashes() == (sha256_file(ROOT / "configs/FINAL_V1/protocol.json"), sha256_file(ROOT / "SPLIT_MANIFEST.csv")))

write_json(ROOT / "tests/test_results.json", {"run_utc": now_utc(), "label": "SMOKE_TESTS_NOT_EXPERIMENT_RESULTS", "results": results,
                                               "all_passed": all(r["passed"] for r in results)})
print("ALL PASSED" if all(r["passed"] for r in results) else "SOME FAILED")
