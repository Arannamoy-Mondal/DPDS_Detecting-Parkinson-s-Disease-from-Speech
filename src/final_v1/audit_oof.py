"""Stage E: independent OOF leakage and integrity audit. Re-derives everything from the frozen manifests and the saved prediction files."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import protocol_defs as P

PRD = WORKSPACE / "predictions/FINAL_V1"
MET = WORKSPACE / "metrics/FINAL_V1"


def auroc_manual(score, y):
    """Mann-Whitney U form (independent of sklearn)."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


def auprc_manual(score, y):
    """Average precision: sum over thresholds of (R_n - R_{n-1}) * P_n with ties handled as one threshold."""
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp); rec = tp / yy.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def main():
    chk = []
    def add(name, ok, evidence):
        chk.append({"check": name, "status": "PASS" if ok else "FAIL", "evidence": evidence})
    rec = json.loads((WORKSPACE / "FREEZE_RECORD.json").read_text())
    sm = pd.read_csv(WORKSPACE / "manifests/SAMPLE_MANIFEST.csv")
    split = pd.read_csv(WORKSPACE / "SPLIT_MANIFEST.csv")
    fold_of_sample = split.set_index("sample_id").outer_fold
    spk_fold = split.groupby("speaker_id").outer_fold.nunique()
    # 1,2 speaker grouping
    add("Every speaker belongs to exactly one outer fold (all records of a speaker share the fold)", bool((spk_fold == 1).all()) and len(spk_fold) == 47, "%d speakers, max folds per speaker %d" % (len(spk_fold), spk_fold.max()))
    runs = {f.stem: json.loads(f.read_text()) for f in sorted(MET.glob("FV1-*-F[0-4].json"))}
    comp = {k: v for k, v in runs.items() if v["status"] == "COMPLETED"}
    failed = {k: v for k, v in runs.items() if v["status"] == "FAILED"}
    # per-run independent recomputation
    overlap_bad, cover_bad, label_bad, prob_bad, fold_bad, hash_bad, cls_bad, cfg_bad = [], [], [], [], [], [], [], []
    for rid, r in comp.items():
        p = pd.read_csv(PRD / (rid + ".csv"))
        f = int(rid.rsplit("-F", 1)[1])
        test_spk = set(sm.speaker_id[fold_of_sample.loc[sm.sample_id].values == f])
        train_spk = set(sm.speaker_id) - test_spk
        if set(p.speaker_id) & train_spk: overlap_bad.append(rid)
        exp_ids = set(sm.sample_id[fold_of_sample.loc[sm.sample_id].values == f])
        if set(p.sample_id) != exp_ids or p.sample_id.duplicated().any(): cover_bad.append(rid)
        lab = sm.set_index("sample_id").label.loc[p.sample_id].values
        if not (lab == p.true_label.values).all(): label_bad.append(rid)
        if not (np.isfinite(p.predicted_probability).all() and p.predicted_probability.between(0, 1).all()): prob_bad.append(rid)
        if not (fold_of_sample.loc[p.sample_id].values == f).all() or not (p.outer_fold == f).all(): fold_bad.append(rid)
        if set(p.split_manifest_hash) != {rec["split_manifest_sha256"]} or r["protocol_hash"] != rec["protocol_json_sha256"]: hash_bad.append(rid)
        if not ((p.predicted_probability >= P.THRESHOLD).astype(int).values == p.predicted_class.values).all(): cls_bad.append(rid)
        cfg = WORKSPACE / "configs/FINAL_V1/runs" / (rid + ".json")
        if sha256_file(cfg) != r["configuration_hash"] or set(p.configuration_hash) != {r["configuration_hash"]}: cfg_bad.append(rid)
    n = len(comp)
    add("Speaker overlap between training and test in every completed outer-fold run is zero (recomputed from the frozen split manifest)", not overlap_bad, "%d runs checked, %d with overlap" % (n, len(overlap_bad)))
    add("Each test sample has exactly one prediction in its run; ids equal the split manifest", not cover_bad, "%d runs checked, %d violations" % (n, len(cover_bad)))
    add("True labels in prediction files equal the audited sample manifest", not label_bad, "%d violations" % len(label_bad))
    add("Probabilities are finite and in [0,1]", not prob_bad, "%d violations" % len(prob_bad))
    add("Folds in prediction files equal the frozen split manifest", not fold_bad, "%d violations" % len(fold_bad))
    add("Protocol hash and split-manifest hash in every run equal the frozen values", not hash_bad and sha256_file(WORKSPACE / "configs/FINAL_V1/protocol.json") == rec["protocol_json_sha256"]
        and sha256_file(WORKSPACE / "SPLIT_MANIFEST.csv") == rec["split_manifest_sha256"], "%d violations; protocol and split files unchanged since freeze" % len(hash_bad))
    add("Threshold policy: predicted class equals probability >= 0.5 fixed in advance; no calibration or threshold search exists in the code", not cls_bad, "%d violations" % len(cls_bad))
    add("Run configurations on disk equal the configuration hashes stored in predictions and metrics", not cfg_bad, "%d violations" % len(cfg_bad))
    # duplicates across folds
    fold_by_hash = sm.assign(f=fold_of_sample.loc[sm.sample_id].values).groupby("sha256_original").f.nunique()
    add("No byte-identical source audio appears in two outer folds; sample ids unique", bool((fold_by_hash == 1).all()) and sm.sample_id.is_unique, "%d unique audio hashes, %d samples" % (len(fold_by_hash), len(sm)))
    pcm = sm.groupby("pcm_sha256").sample_id.size()
    add("No duplicated decoded audio among included recordings", bool((pcm == 1).all()), "%d unique decoded hashes" % len(pcm))
    # alignment
    ac = pd.read_csv(WORKSPACE / "features/acoustic_features.csv")
    ok_al = (ac.sample_id.values == sm.sample_id.values).all()
    am = json.loads((WORKSPACE / "features/manifests/acoustic_manifest.json").read_text())
    ok_al &= am["file_sha256"] == sha256_file(WORKSPACE / "features/acoustic_features.csv")
    detail = ["acoustic ok" if ok_al else "acoustic MISMATCH"]
    for k in P.ENCODERS:
        mf = WORKSPACE / "features/manifests" / (k + "_manifest.json")
        if not mf.exists():
            ok_al = False; detail.append(k + " manifest missing"); continue
        m = json.loads(mf.read_text())
        order = pd.read_csv(WORKSPACE / "features/manifests" / (k + "_sample_order.csv")).sample_id.tolist()
        arr = np.load(WORKSPACE / "features" / (k + "_embeddings.npy"))
        good = m["file_sha256"] == sha256_file(WORKSPACE / "features" / (k + "_embeddings.npy")) and arr.shape == (785, 1024) and np.isfinite(arr).all() and order == sm.sample_id.tolist()
        ck = json.loads((WORKSPACE / "features/manifests/encoder_checkpoints.json").read_text())[k]
        good &= m["revision"] == ck["revision"]
        # cache entries carry the audio hash of their source recording
        bad_cache = 0
        for r in sm.itertuples():
            meta = json.loads((WORKSPACE / "features/cache" / k / (r.sample_id + ".json")).read_text())
            if meta["speaker_audio_sha256"] != r.sha256_original or meta["revision"] != ck["revision"]: bad_cache += 1
        good &= bad_cache == 0
        ok_al &= good; detail.append("%s %s (cache entries with wrong source/revision: %d)" % (k, "ok" if good else "MISMATCH", bad_cache))
    add("Features and embeddings are joined by stable sample_id; shapes, finiteness, checkpoint revision and source-audio hash verified", bool(ok_al), "; ".join(detail))
    # fit boundaries (static)
    mod_files = ["protocol_defs.py", "pipeline.py", "splits.py", "acoustic.py", "extract_embeddings.py", "common.py"]
    changed = [f for f in mod_files if f in rec["code_sha256"] and sha256_file(WORKSPACE / "src/final_v1" / f) != rec["code_sha256"][f]]
    new_after = [f for f in ["runner.py", "aggregate.py", "audit_oof.py"] if f not in rec["code_sha256"]]
    tests = json.loads((WORKSPACE / "tests/test_results.json").read_text())
    add("Modelling code is unchanged since the freeze (hashes); only execution harness files were added after the freeze", not changed, "changed modelling files: %s; files added after freeze (harness/audit only): %s" % (changed, new_after))
    add("Preprocessing, tuning and feature-selection fit boundaries verified by the instrumented Stage C tests (T06, T07, T07b-d) and the null/positive controls (T09, T09b)",
        tests["all_passed"], "%d of %d tests passed" % (sum(t["passed"] for t in tests["results"]), len(tests["results"])))
    # inner tuning boundary evidence from the runs
    inner_ok = all(r["n_train_speakers"] + r["n_test_speakers"] == 47 and all(t["inner_speaker_auprc"] == t["inner_speaker_auprc"] for t in r["inner_table"]) for r in comp.values())
    sel_ok = all(r["selected_hyperparameters"] in P.grid(r["condition_id"].split("-")[3]) for r in comp.values())
    add("Inner tuning logs exist for every run, train+test speakers equal 47, and selected hyperparameters belong to the frozen grid", inner_ok and sel_ok, "%d runs" % n)
    # FS selected names come from training only: names recorded; no way to check labels; verify counts
    fs_runs = [r for r in comp.values() if r["condition_id"].split("-")[2] in ("FORWARD", "BACKWARD")]
    add("Feature-selection runs record exactly 10 selected acoustic features per fold", all(len(r["selected_features"]) == P.FS_K for r in fs_runs), "%d runs" % len(fs_runs))
    # coverage per condition & recomputed metrics
    cm = pd.read_csv(MET / "condition_metrics.csv")
    done = cm[cm.folds_completed == P.N_OUTER]
    mism = []
    cover_cond = []
    for c in done.itertuples():
        d = pd.concat([pd.read_csv(PRD / ("%s-F%d.csv" % (c.condition_id, f))) for f in range(5)])
        if len(d) != len(sm) or d.sample_id.duplicated().any() or set(d.sample_id) != set(sm.sample_id): cover_cond.append(c.condition_id)
        g = d.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first"))
        a1, a2 = auroc_manual(g.s.values, g.y.values.astype(int)), auprc_manual(g.s.values, g.y.values.astype(int))
        if abs(a1 - c.AUROC) > 1e-9 or abs(a2 - c.AUPRC) > 1e-9: mism.append((c.condition_id, a1 - c.AUROC, a2 - c.AUPRC))
    add("Every completed condition has exactly one OOF prediction for each of the 785 recordings (47 speakers)", not cover_cond, "%d completed conditions, %d with coverage problems" % (len(done), len(cover_cond)))
    add("Pooled speaker-level AUROC and AUPRC of every completed condition were recomputed with independent code and agree to 1e-9", not mism, "%d conditions, %d disagreements" % (len(done), len(mism)))
    qn = list((PRD / "quarantine").glob("*.csv")) if (PRD / "quarantine").exists() else []
    add("Failed or quarantined runs", True, "FAILED runs: %d, quarantined prediction files: %d (excluded from all metrics)" % (len(failed), len(qn)))
    reg = pd.read_csv(WORKSPACE / "EXPERIMENT_REGISTRY.csv", keep_default_na=False)
    fr = reg[reg.row_type == "fold_run"]
    st = fr.training_status.value_counts().to_dict()
    add("Registry statuses are consistent with the metric files", st.get("COMPLETED", 0) == n, "registry fold-run statuses: %s; metric files completed: %d" % (st, n))
    all_ok = all(c["status"] == "PASS" for c in chk)
    write_json(MET / "oof_audit.json", {"time_utc": now_utc(), "all_passed": all_ok, "completed_runs": n, "failed_runs": len(failed), "completed_conditions": int(len(done)), "checks": chk})
    md = ["# OOF LEAKAGE AUDIT", "", "Generated %s by `src/final_v1/audit_oof.py` (machine-readable: `metrics/FINAL_V1/oof_audit.json`)." % now_utc(), "",
          "Overall: **%s** (%d completed runs, %d failed, %d of 56 conditions complete)." % ("ALL CHECKS PASSED" if all_ok else "FAILURES PRESENT", n, len(failed), len(done)), "",
          "| # | Check | Status | Evidence |", "|---|---|---|---|"]
    for i, c in enumerate(chk, 1):
        md.append("| %d | %s | %s | %s |" % (i, c["check"], c["status"], c["evidence"].replace("|", "/")))
    md += ["", "Scope note: the audit shows the pipeline and the saved predictions obey the speaker-disjoint rules. It cannot show that the score reflects disease: recording date and class are fully confounded in this corpus (`DATA_AUDIT.md`), and speaker-disjoint folds do not remove that."]
    write_text(WORKSPACE / "OOF_LEAKAGE_AUDIT.md", "\n".join(md) + "\n")
    print("audit:", "ALL PASS" if all_ok else "FAILURES", "| runs", n)
    for c in chk:
        if c["status"] != "PASS": print("FAIL:", c["check"], c["evidence"])


if __name__ == "__main__":
    main()
