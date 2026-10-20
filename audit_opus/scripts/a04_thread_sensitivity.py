"""Audit A04: re-run every logistic-regression outer-fold run (14 conditions x 5 folds) with BLAS limited to 1 thread and compare selected
configurations, probabilities and pooled metrics with the official FINAL_V1 results."""
import json, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
W = "/root/Projects/Project4a_FINAL_V1_ISOLATED"
sys.path.insert(0, W + "/src/final_v1")
OUT = W + "/audit_opus/rerun_lr_1thread"

def work(rid):
    from threadpoolctl import threadpool_limits
    import pipeline as pl, runner
    with threadpool_limits(limits=1):
        sm = pl.load_manifest(); rep = rid.split("-")[1]
        X, names = pl.load_representation(rep, sm)
        r = runner.run_one(rid, X, names, sm, OUT, *runner.frozen_hashes())
    return rid, r["status"]

if __name__ == "__main__":
    order = json.load(open(W + "/configs/FINAL_V1/planned_run_order.json"))
    rids = [r for r in order if r.split("-")[3] == "LR"]
    res = Parallel(n_jobs=14)(delayed(work)(r) for r in rids)
    import pipeline as pl
    rows = []
    for cid in sorted(set(r.rsplit("-F", 1)[0] for r in rids)):
        cfg_same = 0; mx = 0.0; fo, fn_ = [], []
        for f in range(5):
            a = json.load(open(W + "/metrics/FINAL_V1/%s-F%d.json" % (cid, f))); b = json.load(open(OUT + "/metrics/FINAL_V1/%s-F%d.json" % (cid, f)))
            cfg_same += int(a["selected_hyperparameters"] == b["selected_hyperparameters"])
            pa = pd.read_csv(W + "/predictions/FINAL_V1/%s-F%d.csv" % (cid, f)).set_index("sample_id").predicted_probability
            pb = pd.read_csv(OUT + "/predictions/FINAL_V1/%s-F%d.csv" % (cid, f)).set_index("sample_id").predicted_probability.loc[pa.index]
            mx = max(mx, float((pa - pb).abs().max()))
            fo.append(pd.read_csv(W + "/predictions/FINAL_V1/%s-F%d.csv" % (cid, f))); fn_.append(pd.read_csv(OUT + "/predictions/FINAL_V1/%s-F%d.csv" % (cid, f)))
        def pooled(fr):
            d = pd.concat(fr).groupby("speaker_id").agg(p=("predicted_probability", "mean"), y=("true_label", "first")); return pl.binary_metrics(d.p.values, d.y.values)
        A, B = pooled(fo), pooled(fn_)
        rows.append({"condition": cid, "folds_same_config": cfg_same, "max_abs_prob_diff": mx, "AUROC_official": A["auroc"], "AUROC_rerun": B["auroc"], "AUPRC_official": A["auprc"], "AUPRC_rerun": B["auprc"],
                     "errors_official": A["fn"] + A["fp"], "errors_rerun": B["fn"] + B["fp"]})
    df = pd.DataFrame(rows); df.to_csv(W + "/audit_opus/evidence/a04_thread_sensitivity.csv", index=False)
    print(df.round(6).to_string())
