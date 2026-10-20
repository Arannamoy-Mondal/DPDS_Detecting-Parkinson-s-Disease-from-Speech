"""S6: session-predictability diagnostics (pre-specified before running). Within ONE class, can the same representations tell recording
sessions apart for unseen speakers? T1: PD 24 Jan 2017 vs PD 10 Feb 2017 (same 16-kHz rate); T2: controls 17+21 Mar vs 23+30 Mar+2 Apr 2017;
T3: PD 2016 (44.1 kHz) vs PD 2017 (16 kHz). Speaker-disjoint 5-fold CV with nested 3-fold tuning over the frozen LR grid (and XGBoost for AC);
unit = speaker x session (mean of recording probabilities). If sessions are easy to tell apart inside a class, a PD-vs-control classifier
can succeed by recognising sessions, because sessions and class coincide in this corpus."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
from splits import assign_folds
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score

sm = pl.load_manifest()
TASKS = {
    "T1_PD_24Jan2017_vs_10Feb2017": (sm.label == 1) & sm.date.isin(["2017-01-24", "2017-02-10"]), "T2_controls_17-21Mar_vs_23Mar-2Apr": (sm.label == 0),
    "T3_PD_2016_44k_vs_2017_16k": (sm.label == 1)}
def target(name, d):
    if name.startswith("T1"): return (d.date == "2017-02-10").astype(int).values
    if name.startswith("T2"): return d.date.isin(["2017-03-23", "2017-03-30", "2017-04-02"]).astype(int).values
    return d.date.str.startswith("2017").astype(int).values

def unit_scores(p, unit, t):
    d = pd.DataFrame({"p": p, "u": unit, "t": t}).groupby("u").agg(p=("p", "mean"), t=("t", "first")); return d

def run(name, rep, model):
    mask = TASKS[name].values; d = sm[mask].reset_index(drop=True)
    X, _ = pl.load_representation(rep, sm); X = X[mask]; t = target(name, d)
    spk = np.asarray(d.speaker_id.astype(str).tolist()); unit = np.array([s + "|" + str(v) for s, v in zip(spk, t)])
    lab = {s: int(t[spk == s][0]) for s in np.unique(spk)}
    fo = assign_folds(list(lab), lab, 5, 42); fold = np.array([fo[s] for s in spk])
    probs = np.zeros(len(t))
    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        labi = {s: lab[s] for s in np.unique(spk[tr])}; im = assign_folds(list(labi), labi, 3, 142 + f); inner = np.array([im[s] for s in spk[tr]])
        best, bs = None, -1
        for ci, hp in enumerate(P.grid(model)):
            p = np.zeros(len(tr))
            for k in range(3):
                a, b = np.where(inner != k)[0], np.where(inner == k)[0]
                if len(set(t[tr][a])) < 2 or len(set(t[tr][b])) < 1: continue
                p[b] = pl.fit_predict(model, hp, X[tr][a], t[tr][a], X[tr][b])
            u = unit_scores(p, unit[tr], t[tr])
            sc = round(average_precision_score(u.t, u.p), 12) if u.t.nunique() == 2 else -1
            if sc > bs: best, bs = hp, sc
        probs[te] = pl.fit_predict(model, best, X[tr], t[tr], X[te])
    u = unit_scores(probs, unit, t)
    return {"task": name, "representation": rep, "model": model, "n_units": len(u), "n_units_label1": int(u.t.sum()), "n_speakers": len(lab), "n_recordings": len(t),
            "unit_auroc": roc_auc_score(u.t, u.p), "unit_auprc": average_precision_score(u.t, u.p), "chance_auprc": float(u.t.mean()),
            "recording_auroc": roc_auc_score(t, probs), "unit_balanced_accuracy": float((((u.p >= .5).astype(int) == u.t) & (u.t == 1)).sum() / max(1, (u.t == 1).sum()) / 2 + (((u.p >= .5).astype(int) == u.t) & (u.t == 0)).sum() / max(1, (u.t == 0).sum()) / 2)}

if __name__ == "__main__":
    for name, m in TASKS.items():
        d = sm[m.values]; print(name, "speakers per date:", d.groupby("date").speaker_id.nunique().to_dict())
    jobs = [(n, r, "LR") for n in TASKS for r in ["AC", "WLM", "W2V", "XLSR"]] + [(n, "AC", "XGB") for n in TASKS]
    res = Parallel(n_jobs=8)(delayed(run)(*j) for j in jobs)
    df = pd.DataFrame(res); df.to_csv(RES / "S6_session_predictability.csv", index=False); print(df.round(3).to_string())
