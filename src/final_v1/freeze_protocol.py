"""Stage B: write the machine-readable protocol, per-run configs, the planned registry and the freeze record (before any model is trained)."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import protocol_defs as P
import make_pdf

REG_FIELDS = ["run_id", "condition_id", "row_type", "protocol_version", "protocol_hash", "task", "outer_fold", "inner_cv_configuration", "seed", "representation", "feature_group",
              "feature_selection_method", "feature_count", "model", "tuning_status", "hyperparameters", "training_status", "AUPRC", "AUROC", "F1", "sensitivity", "specificity",
              "balanced_accuracy", "runtime_seconds", "peak_memory", "prediction_file", "configuration_file", "artifact_path", "failure_reason", "notes"]


def main():
    ck = json.loads((WORKSPACE / "features/manifests/encoder_checkpoints.json").read_text())
    proto = P.protocol_dict()
    proto["encoders"] = {k: {"model_id": v["model_id"], "revision": v["revision"]} for k, v in ck.items()}
    proto["split_manifest_sha256"] = sha256_file(WORKSPACE / "SPLIT_MANIFEST.csv")
    proto["sample_manifest_sha256"] = sha256_file(WORKSPACE / "manifests/SAMPLE_MANIFEST.csv")
    proto["data_inventory_sha256"] = sha256_file(WORKSPACE / "DATA_INVENTORY.csv")
    proto["acoustic_features_sha256"] = json.loads((WORKSPACE / "features/manifests/acoustic_manifest.json").read_text())["file_sha256"]
    pjson = write_json(WORKSPACE / "configs/FINAL_V1/protocol.json", proto)
    phash = sha256_file(pjson)
    md_hash = sha256_file(WORKSPACE / "FINAL_V1_PROTOCOL.md")
    code = {f.name: sha256_file(f) for f in sorted((WORKSPACE / "src/final_v1").glob("*.py"))}
    reg, order = [], []
    for c in P.conditions():
        cfg = {"protocol_hash": phash, **c, "grid": P.grid(c["model"]), "feature_count": 10 if c["feature_selection"] != "ALL" else None}
        for f in range(P.N_OUTER):
            rid = "%s-F%d" % (c["condition_id"], f)
            rcfg = {**cfg, "run_id": rid, "outer_fold": f, "seed": P.SEED, "inner_seed": 142 + f, "selection_seed": 242 + f}
            cp = write_json(WORKSPACE / "configs/FINAL_V1/runs" / (rid + ".json"), rcfg)
            reg.append({"run_id": rid, "condition_id": c["condition_id"], "row_type": "fold_run", "protocol_version": "FINAL_V1", "protocol_hash": phash, "task": "all_tasks_pooled",
                        "outer_fold": f, "inner_cv_configuration": "3-fold stratified speaker-disjoint, seed %d" % (142 + f), "seed": P.SEED, "representation": c["representation"],
                        "feature_group": "+".join(P.components(c["representation"])), "feature_selection_method": c["feature_selection"], "feature_count": "10" if c["feature_selection"] != "ALL" else "",
                        "model": c["model"], "tuning_status": "PLANNED", "training_status": "PLANNED", "configuration_file": str(cp.relative_to(WORKSPACE)), "notes": "config sha256 " + sha256_file(cp)[:16]})
            order.append(rid)
        reg.append({"run_id": c["condition_id"], "condition_id": c["condition_id"], "row_type": "condition_pooled", "protocol_version": "FINAL_V1", "protocol_hash": phash,
                    "task": "all_tasks_pooled", "outer_fold": "ALL", "seed": P.SEED, "representation": c["representation"], "feature_group": "+".join(P.components(c["representation"])),
                    "feature_selection_method": c["feature_selection"], "model": c["model"], "tuning_status": "PLANNED", "training_status": "PLANNED"})
    reg.append({"run_id": "FV1-FWL", "condition_id": "FV1-FWL", "row_type": "blocked_experiment", "protocol_version": "FINAL_V1", "protocol_hash": phash, "task": "all_tasks_pooled",
                "training_status": "BLOCKED", "failure_reason": "PROTOCOL_DECISION_REQUIRED: FWL purpose, response, nuisance covariates, residualization, statistic and stopping rule are undefined",
                "notes": "not improvised; run only after the owner approves a definition (would be a documented protocol amendment)"})
    write_csv(WORKSPACE / "EXPERIMENT_REGISTRY.csv", reg, REG_FIELDS)
    write_json(WORKSPACE / "configs/FINAL_V1/planned_run_order.json", order)
    # compliance checklist
    cond = P.conditions()
    checks = {
        "protocol_json_hash_recorded": True,
        "56_conditions_x_5_folds_equals_280_runs": len(cond) == 56 and len(order) == 280,
        "grid_sizes_20_10_24_24": [len(P.grid(m)) for m in P.MODELS] == [20, 10, 24, 24],
        "no_FWL_run_planned": all("FWL" not in c["condition_id"] for c in cond),
        "split_manifest_has_47_speakers_5_folds": pd.read_csv(WORKSPACE / "SPLIT_MANIFEST.csv").drop_duplicates("speaker_id").outer_fold.nunique() == 5,
        "all_hybrid_components_available": all(c in P.BASE_REPS for h in P.HYBRIDS.values() for c in h),
        "encoder_revisions_recorded": all(v["revision"] for v in ck.values()),
        "unresolved_methodological_decisions": ["FWL (BLOCKED, PROTOCOL_DECISION_REQUIRED)"],
        "implementation_details_open": [],
    }
    rec = {"freeze_time_utc": now_utc(), "protocol_version": "FINAL_V1", "status": "FROZEN", "protocol_json_sha256": phash, "protocol_md_sha256": md_hash,
           "split_manifest_sha256": proto["split_manifest_sha256"], "sample_manifest_sha256": proto["sample_manifest_sha256"], "data_inventory_sha256": proto["data_inventory_sha256"],
           "seeds": {"outer": 42, "inner": "142+fold", "selection": "242+fold", "xgb_grid_subset": 42, "bootstrap": 42}, "code_sha256": code, "compliance": checks,
           "environment_file": "ENVIRONMENT.md", "execution_order_file": "configs/FINAL_V1/planned_run_order.json"}
    write_json(WORKSPACE / "FREEZE_RECORD.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "code_sha256"}, indent=1))
    ok = all(v for k, v in checks.items() if isinstance(v, bool))
    pdf = make_pdf.summary_pdf("STAGE_B_protocol_verification_and_freeze", "Stage B: protocol verification and freeze", "FINAL_V1 - Parkinson's disease detection from speech",
        [("Protocol status", ["No frozen protocol file was supplied. A protocol was drafted by the agent and frozen after the owner's instruction (D-000): FINAL_V1_PROTOCOL.md and configs/FINAL_V1/protocol.json.",
                              "Freeze time %s UTC. Protocol JSON SHA-256 %s." % (rec["freeze_time_utc"], phash[:16] + "..."), "Split manifest SHA-256 %s." % (proto["split_manifest_sha256"][:16] + "...")]),
         ("Design", ["Speaker-disjoint stratified 5-fold outer CV (seed 42) and speaker-disjoint 3-fold inner CV for tuning and feature selection.",
                     "12 representations (acoustic, WavLM-large, wav2vec2-large-lv60, XLS-R 300M, 8 hybrids) x 4 models (LR, linear SVM, RBF SVM, XGBoost); acoustic features also with forward and backward selection (k=10). 56 conditions, 280 outer-fold runs.",
                     "Primary metrics: pooled speaker-level AUPRC and AUROC; fixed threshold 0.5; inner objective speaker-level AUPRC; bootstrap CIs over speakers."]),
         ("Compliance check and open items", ["All automatic compliance checks passed." if ok else "A compliance check FAILED; do not train.",
                                              "Unresolved methodological decision: FWL is BLOCKED (PROTOCOL_DECISION_REQUIRED) because no purpose, response, covariates, statistic or stopping rule exists. It is recorded, not improvised.",
                                              "No implementation detail is open. Historical scores were not used for any design choice."])],
        [("Conditions by representation", ["Representation", "Components", "Dim", "Conditions"],
          [[r, "+".join(P.components(r)), str(sum(P.DIMS[c] for c in P.components(r))), str(sum(1 for c in cond if c["representation"] == r))] for r in P.BASE_REPS + list(P.HYBRIDS)])])
    print(pdf, "all checks ok:", ok)


main()
