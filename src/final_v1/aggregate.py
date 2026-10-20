"""Build condition-level tables, paired bootstrap comparisons and the updated registry from verified run files only."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import protocol_defs as P
import pipeline as pl
from freeze_protocol_fields import REG_FIELDS

MET = WORKSPACE / "metrics/FINAL_V1"
PRD = WORKSPACE / "predictions/FINAL_V1"


def load_runs():
    runs = {}
    for f in sorted(MET.glob("FV1-*-F[0-4].json")):
        runs[f.stem] = json.loads(f.read_text())
    return runs


def pooled_speaker_scores(cond_id):
    frames = [pd.read_csv(PRD / ("%s-F%d.csv" % (cond_id, f))) for f in range(P.N_OUTER)]
    d = pd.concat(frames)
    g = d.groupby("speaker_id").agg(score=("predicted_probability", "mean"), y=("true_label", "first"))
    return d, g.sort_index()


def boot_stats(score, y, idx):
    au = np.full(len(idx), np.nan); ap = np.full(len(idx), np.nan)
    for b, ix in enumerate(idx):
        yy = y[ix]
        if yy.min() == yy.max():
            continue
        au[b] = roc_auc_score(yy, score[ix]); ap[b] = average_precision_score(yy, score[ix])
    return au, ap


def ci(a):
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    runs = load_runs()
    conds = P.conditions()
    sm = pl.load_manifest()
    speakers = sorted(sm.speaker_id.unique())
    rng = np.random.RandomState(P.SEED)
    idx = rng.randint(0, len(speakers), size=(P.BOOTSTRAP, len(speakers)))
    rows, boots = [], {}
    for c in conds:
        cid = c["condition_id"]
        cr = [runs.get("%s-F%d" % (cid, f)) for f in range(P.N_OUTER)]
        status = ["MISSING" if r is None else r["status"] for r in cr]
        row = {"condition_id": cid, "representation": c["representation"], "feature_selection": c["feature_selection"], "model": c["model"], "folds_completed": sum(s == "COMPLETED" for s in status),
               "fold_status": ",".join(status)}
        if all(s == "COMPLETED" for s in status):
            d, g = pooled_speaker_scores(cid)
            if len(g) != len(speakers) or d.sample_id.duplicated().any():
                row["coverage_problem"] = "speaker/sample coverage mismatch"
            score = g.reindex(speakers).score.values; y = g.reindex(speakers).y.values.astype(int)
            m = pl.binary_metrics(score, y)
            au, ap = boot_stats(score, y, idx)
            boots[cid] = (au, ap)
            lo_a, hi_a = ci(au); lo_p, hi_p = ci(ap)
            sl = pl.binary_metrics(d.predicted_probability.values, d.true_label.values)
            row.update({"n_speakers": len(g), "n_samples": len(d), "AUPRC": m["auprc"], "AUPRC_ci_lo": lo_p, "AUPRC_ci_hi": hi_p, "AUROC": m["auroc"], "AUROC_ci_lo": lo_a, "AUROC_ci_hi": hi_a,
                        "F1": m["f1"], "sensitivity": m["sensitivity"], "specificity": m["specificity"], "precision": m["precision"], "accuracy": m["accuracy"],
                        "balanced_accuracy": m["balanced_accuracy"], "TP": m["tp"], "FN": m["fn"], "TN": m["tn"], "FP": m["fp"],
                        "sample_AUROC": sl["auroc"], "sample_AUPRC": sl["auprc"], "sample_balanced_accuracy": sl["balanced_accuracy"]})
            for k in ["auprc", "auroc", "f1", "sensitivity", "specificity", "balanced_accuracy"]:
                v = np.array([r["speaker_level"][k] for r in cr], dtype=float)
                row["fold_mean_" + k] = float(np.nanmean(v)); row["fold_sd_" + k] = float(np.nanstd(v, ddof=1))
            row["runtime_seconds_total"] = float(sum(r["runtime_seconds"] for r in cr))
            row["selected_hyperparameters_by_fold"] = json.dumps([r["selected_hyperparameters"] for r in cr])
            row["feature_count"] = int(np.mean([r["n_features_used"] for r in cr]))
        rows.append(row)
    cm = pd.DataFrame(rows)
    cm.to_csv(safe_path(MET / "condition_metrics.csv"), index=False)
    # paired comparisons
    comp = []
    def add(kind, a, b, note):
        if a in boots and b in boots:
            for k, nm in enumerate(["AUPRC", "AUROC"]):
                ca = cm.set_index("condition_id").loc[a][nm]; cb = cm.set_index("condition_id").loc[b][nm]
                dlt = boots[a][1 - k] - boots[b][1 - k]
                lo, hi = ci(dlt)
                comp.append({"comparison_type": kind, "condition_A": a, "condition_B": b, "metric": nm, "A": ca, "B": cb, "delta_A_minus_B": ca - cb, "ci95_lo": lo, "ci95_hi": hi,
                             "ci_excludes_zero": bool(lo > 0 or hi < 0), "note": note})
    for h, parts in P.HYBRIDS.items():
        for m in P.MODELS:
            for part in parts:
                add("hybrid_vs_constituent", "FV1-%s-ALL-%s" % (h, m), "FV1-%s-ALL-%s" % (part, m), "%s vs %s, model %s" % (h, part, m))
    for rep in P.BASE_REPS + list(P.HYBRIDS):
        for fs in (["ALL", "FORWARD", "BACKWARD"] if rep == "AC" else ["ALL"]):
            for m in P.MODELS[1:]:
                add("model_vs_logistic_regression", "FV1-%s-%s-%s" % (rep, fs, m), "FV1-%s-%s-LR" % (rep, fs), "%s vs LR (%s, %s)" % (m, rep, fs))
    for fs in ["FORWARD", "BACKWARD"]:
        for m in P.MODELS:
            add("feature_selection_vs_all_features", "FV1-AC-%s-%s" % (fs, m), "FV1-AC-ALL-%s" % m, "%s selection vs all 40 features, %s" % (fs, m))
    pd.DataFrame(comp).to_csv(safe_path(MET / "paired_comparisons.csv"), index=False)
    # registry refresh
    reg = pd.read_csv(WORKSPACE / "EXPERIMENT_REGISTRY.csv", keep_default_na=False, dtype=str)
    out = []
    cmi = cm.set_index("condition_id")
    for r in reg.to_dict("records"):
        if r["row_type"] == "fold_run":
            run = runs.get(r["run_id"])
            if run:
                r["training_status"] = run["status"]; r["tuning_status"] = "COMPLETED" if run["status"] == "COMPLETED" else run["status"]
                r["runtime_seconds"] = "%.1f" % run.get("runtime_seconds", 0)
                r["hyperparameters"] = json.dumps(run.get("selected_hyperparameters", ""))
                r["feature_count"] = str(run.get("n_features_used", ""))
                if run["status"] == "COMPLETED":
                    sl = run["speaker_level"]
                    r.update({"AUPRC": "%.6f" % sl["auprc"], "AUROC": "%.6f" % sl["auroc"], "F1": "%.6f" % sl["f1"], "sensitivity": "%.6f" % sl["sensitivity"],
                              "specificity": "%.6f" % sl["specificity"], "balanced_accuracy": "%.6f" % sl["balanced_accuracy"],
                              "prediction_file": "predictions/FINAL_V1/%s.csv" % r["run_id"], "artifact_path": "metrics/FINAL_V1/%s.json" % r["run_id"], "peak_memory": "not recorded"})
                else:
                    r["failure_reason"] = run.get("failure_reason", "")
        elif r["row_type"] == "condition_pooled":
            c = cmi.loc[r["condition_id"]]
            if c["folds_completed"] == P.N_OUTER and "AUROC" in c and pd.notna(c.get("AUROC")):
                r.update({"training_status": "COMPLETED", "tuning_status": "COMPLETED", "AUPRC": "%.6f" % c["AUPRC"], "AUROC": "%.6f" % c["AUROC"], "F1": "%.6f" % c["F1"],
                          "sensitivity": "%.6f" % c["sensitivity"], "specificity": "%.6f" % c["specificity"], "balanced_accuracy": "%.6f" % c["balanced_accuracy"],
                          "runtime_seconds": "%.1f" % c["runtime_seconds_total"], "artifact_path": "metrics/FINAL_V1/condition_metrics.csv"})
            elif c["folds_completed"] > 0:
                r["training_status"] = "RUNNING"
        out.append(r)
    write_csv(WORKSPACE / "EXPERIMENT_REGISTRY.csv", out, REG_FIELDS)
    print("conditions completed:", int((cm.folds_completed == P.N_OUTER).sum()), "of", len(cm), "| runs found:", len(runs))
    print(cm[cm.folds_completed == P.N_OUTER][["condition_id", "AUPRC", "AUROC", "F1", "sensitivity", "specificity"]].round(3).to_string())


if __name__ == "__main__":
    main()
