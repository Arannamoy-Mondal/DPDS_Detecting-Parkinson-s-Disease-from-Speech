"""S3: hyperparameter-tuning analysis. (a) selected configurations and tie degeneracy from the 280 official runs; (b) landscape: every
configuration of the frozen grid refitted on each outer-training fold and scored on the outer-test fold (post-hoc, for describing sensitivity
only; never used to choose anything)."""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
from joblib import Parallel, delayed

def selected_table():
    rows = []
    for f in sorted((ROOT / "metrics/FINAL_V1").glob("FV1-*-F[0-4].json")):
        r = json.load(open(f)); t = [x["inner_speaker_auprc"] for x in r["inner_table"]]; mx = max(t)
        rows.append({"run_id": r["run_id"], "condition": r["condition_id"], "model": r["condition_id"].split("-")[3], "representation": r["condition_id"].split("-")[1],
                     "fs": r["condition_id"].split("-")[2], "selected": json.dumps(r["selected_hyperparameters"], sort_keys=True), "inner_best": mx,
                     "inner_median": float(np.median(t)), "inner_min": float(np.min(t)), "n_configs": len(t), "n_tied_at_best": int(sum(abs(v - mx) < 1e-12 for v in t)),
                     "outer_fold_auprc": r["speaker_level"]["auprc"], "outer_fold_auroc": r["speaker_level"]["auroc"]})
    df = pd.DataFrame(rows); df.to_csv(RES / "S3_selected_and_ties.csv", index=False); return df

def landscape_job(rep, model, ci, f):
    sm = pl.load_manifest(); of = outer_idx(sm)
    X, names = pl.load_representation(rep, sm); y = sm.label.values.astype(int)
    tr, te = np.where(of != f)[0], np.where(of == f)[0]
    p = pl.fit_predict(model, P.grid(model)[ci], X[tr], y[tr], X[te])
    return rep, model, ci, f, te, p

if __name__ == "__main__":
    df = selected_table(); print("selected table", len(df))
    reps = ["AC", "WLM", "W2V", "XLSR", "H3"]
    jobs = [(r, m, ci, f) for r in reps for m in P.MODELS for ci in range(len(P.grid(m))) for f in range(5)]
    out = Parallel(n_jobs=12, verbose=0)(delayed(landscape_job)(*j) for j in jobs)
    sm = pl.load_manifest(); y = sm.label.values.astype(int); spk = np.asarray(sm.speaker_id.astype(str).tolist())
    acc = {}
    for rep, model, ci, f, te, p in out:
        acc.setdefault((rep, model, ci), np.full(len(y), np.nan))[te] = p
    rows = []
    for (rep, model, ci), p in acc.items():
        m = pl.binary_metrics(*pl.speaker_scores(p, spk, y)[:2])
        rows.append({"representation": rep, "model": model, "config_index": ci, "params": json.dumps(P.grid(model)[ci], sort_keys=True), "pooled_auroc": m["auroc"], "pooled_auprc": m["auprc"],
                     "errors_at_0.5": m["fn"] + m["fp"], "balanced_accuracy": m["balanced_accuracy"]})
    pd.DataFrame(rows).to_csv(RES / "S3_landscape.csv", index=False); print("landscape rows", len(rows))
