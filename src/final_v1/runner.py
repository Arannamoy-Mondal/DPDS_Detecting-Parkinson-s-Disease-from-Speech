"""Run FINAL_V1 outer-fold runs: gates, atomic outputs, resumable. Usage:
   python runner.py run [workers] [only_condition_prefix]    run every planned run that has no valid result yet
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import protocol_defs as P
import pipeline as pl

GATES = ["no_speaker_overlap_train_test", "every_test_sample_predicted_exactly_once", "probabilities_finite_in_0_1", "labels_match_manifest",
         "folds_match_split_manifest", "protocol_and_split_hashes_match_freeze"]


def frozen_hashes(root=WORKSPACE):
    rec = json.loads((root / "FREEZE_RECORD.json").read_text())
    return rec["protocol_json_sha256"], rec["split_manifest_sha256"]


def run_one(rid, X, names, sm, out_root, protocol_hash, split_hash, freeze_root=WORKSPACE):
    """Execute one run id like FV1-AC-FORWARD-LR-F2. Returns the metrics dict that was written."""
    cond_id, fold = rid.rsplit("-F", 1)
    fold = int(fold)
    cond = next(c for c in P.conditions() if c["condition_id"] == cond_id)
    out_root = Path(out_root)
    pred_path = out_root / "predictions/FINAL_V1" / (rid + ".csv")
    met_path = out_root / "metrics/FINAL_V1" / (rid + ".json")
    if met_path.exists() and pred_path.exists():
        old = json.loads(met_path.read_text())
        if old.get("status") == "COMPLETED":
            return old
    cfg_path = freeze_root / "configs/FINAL_V1/runs" / (rid + ".json")
    cfg_hash = sha256_file(cfg_path)
    t0 = time.time()
    res = {"run_id": rid, "condition_id": cond_id, "outer_fold": fold, "status": "RUNNING", "started_utc": now_utc(), "configuration_hash": cfg_hash,
           "protocol_hash": protocol_hash, "split_manifest_hash": split_hash}
    try:
        out = pl.run_fold(cond, fold, X, names, sm)
    except Exception as e:
        res.update({"status": "FAILED", "failure_reason": "exception: " + repr(e), "trace": traceback.format_exc()[-1500:], "finished_utc": now_utc(), "runtime_seconds": time.time() - t0})
        write_text(met_path, json.dumps(res, indent=1)); return res
    pred = out["pred"].copy()
    split = pd.read_csv(freeze_root / "SPLIT_MANIFEST.csv").set_index("sample_id")
    allsm = sm.set_index("sample_id")
    train_speakers = set(sm.speaker_id[split.loc[sm.sample_id].outer_fold.values != fold])
    test_ids = set(sm.sample_id[split.loc[sm.sample_id].outer_fold.values == fold])
    gates = {
        "no_speaker_overlap_train_test": not (train_speakers & set(pred.speaker_id)),
        "every_test_sample_predicted_exactly_once": (len(pred) == len(test_ids)) and (set(pred.sample_id) == test_ids) and (not pred.sample_id.duplicated().any()),
        "probabilities_finite_in_0_1": bool(np.isfinite(pred.predicted_probability).all() and pred.predicted_probability.between(0, 1).all()),
        "labels_match_manifest": bool((pred.true_label.values == allsm.loc[pred.sample_id].label.values).all()),
        "folds_match_split_manifest": bool((split.loc[pred.sample_id].outer_fold.values == fold).all() and (pred.outer_fold == fold).all()),
        # audit fix (20 Oct 2026): compare the files on disk with the freeze record (the earlier form compared the record with itself)
        "protocol_and_split_hashes_match_freeze": (protocol_hash, split_hash) == frozen_hashes(freeze_root)
            and (sha256_file(freeze_root / "configs/FINAL_V1/protocol.json"), sha256_file(freeze_root / "SPLIT_MANIFEST.csv")) == frozen_hashes(freeze_root),
    }
    pred.insert(0, "protocol_version", PROTOCOL_VERSION)
    pred.insert(1, "run_id", rid)
    pred.insert(2, "task", "all_tasks_pooled")
    pred["predicted_class"] = (pred.predicted_probability >= P.THRESHOLD).astype(int)
    pred["model"] = cond["model"]
    pred["representation"] = cond["representation"] + ("/" + cond["feature_selection"] if cond["feature_selection"] != "ALL" else "")
    pred["split_manifest_hash"] = split_hash
    pred["configuration_hash"] = cfg_hash
    pred["prediction_timestamp"] = now_utc()
    res.update({"gates": gates, "selected_hyperparameters": out["best_params"], "selected_features": out["selected_features"], "n_features_used": out["n_features_used"],
                "inner_table": out["inner_table"], "runtime_seconds": time.time() - t0, "finished_utc": now_utc(), "n_train_speakers": out["n_train_speakers"], "n_test_speakers": out["n_test_speakers"]})
    if not all(gates.values()):
        res.update({"status": "FAILED", "failure_reason": "integrity gate failed: " + ",".join(k for k, v in gates.items() if not v)})
        q = out_root / "predictions/FINAL_V1/quarantine" / (rid + ".csv")
        q.parent.mkdir(parents=True, exist_ok=True); pred.to_csv(q, index=False)
        write_text(met_path, json.dumps(res, indent=1, default=str)); return res
    sc, yt, _ = pl.speaker_scores(pred.predicted_probability.values, pred.speaker_id.values, pred.true_label.values)
    res["speaker_level"] = pl.binary_metrics(sc, yt)
    res["sample_level"] = pl.binary_metrics(pred.predicted_probability.values, pred.true_label.values)
    res["status"] = "COMPLETED"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pred_path.with_name(pred_path.name + ".tmp%d" % os.getpid())
    pred.to_csv(tmp, index=False); tmp.replace(pred_path)
    write_text(met_path, json.dumps(res, indent=1, default=str))
    return res


_CACHE = {}


def _worker(rid):
    rep = rid.split("-")[1]
    if rep not in _CACHE:
        sm = pl.load_manifest()
        _CACHE.clear()
        _CACHE[rep] = (pl.load_representation(rep, sm), sm)
    (X, names), sm = _CACHE[rep]
    ph, sh = frozen_hashes()
    r = run_one(rid, X, names, sm, WORKSPACE, ph, sh)
    return rid, r["status"], round(r.get("runtime_seconds", 0), 1)


if __name__ == "__main__":
    from joblib import Parallel, delayed
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    prefix = sys.argv[3] if len(sys.argv) > 3 else ""
    order = [r for r in json.loads((WORKSPACE / "configs/FINAL_V1/planned_run_order.json").read_text()) if r.startswith(prefix)]
    todo = []
    for r in order:
        m = WORKSPACE / "metrics/FINAL_V1" / (r + ".json")
        if m.exists() and json.loads(m.read_text()).get("status") == "COMPLETED":
            continue
        todo.append(r)
    # group by representation so that each worker reloads data rarely
    todo.sort(key=lambda r: (r.split("-")[1], r))
    print("runs to do:", len(todo), "of", len(order), flush=True)
    for rid, status, rt in Parallel(n_jobs=workers, return_as="generator", batch_size=1)(delayed(_worker)(r) for r in todo):
        print(rid, status, rt, flush=True)
