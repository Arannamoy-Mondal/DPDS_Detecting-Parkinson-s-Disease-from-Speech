"""Audit A03: (1) acoustic features recomputed with an independent implementation (scipy.signal) for 25 recordings;
(2) nested CV re-implemented independently (own fold builder, own grid search) for two conditions and compared with the official predictions;
(3) all threshold metrics of the 56 conditions recomputed from the prediction files and compared with condition_metrics.csv and the registry;
(4) bootstrap CIs and paired differences recomputed for three conditions; (5) feature-selection results identical across models within a fold;
(6) convergence of the final logistic-regression fits."""
import json, sys, warnings
import numpy as np, pandas as pd
from math import gcd
from scipy.signal import resample_poly
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, average_precision_score
W = "/root/Projects/Project4a_FINAL_V1_ISOLATED"
sys.path.insert(0, W + "/src/final_v1")
ev = {}
sm = pd.read_csv(W + "/manifests/SAMPLE_MANIFEST.csv")

# ---------- (1) acoustic features, independent implementation
def load(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True); x = x.mean(1)
    if sr != 16000:
        g = gcd(16000, sr); x = resample_poly(x, 16000 // g, sr // g)
    return x
def feats(x):
    L, H = 400, 160
    n = 1 + (len(x) - L) // H
    fr = np.lib.stride_tricks.sliding_window_view(x, L)[::H][:n]
    rms = np.sqrt(np.mean(fr ** 2, 1)); le = 20 * np.log10(rms + 1e-10)
    s = np.sign(fr); zc = np.mean(s[:, 1:] != s[:, :-1], 1)
    P = np.abs(np.fft.rfft(fr * np.hanning(L), 512)) ** 2; f = np.arange(257) * 16000 / 512
    tot = P.sum(1) + 1e-20; cen = (P * f).sum(1) / tot
    ro = f[np.argmax(np.cumsum(P, 1) >= 0.85 * tot[:, None], 1)]
    fl = np.exp(np.mean(np.log(P + 1e-20), 1)) / (P.mean(1) + 1e-20)
    bands = [np.log10(P[:, (f >= a) & (f < b)].sum(1) + 1e-20) for a, b in [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]]
    out = []
    for c in [le, zc, cen, ro, fl] + bands:
        out += [c.mean(), c.std(), np.percentile(c, 10), np.percentile(c, 90)]
    return np.array(out)
ac = pd.read_csv(W + "/features/acoustic_features.csv", index_col="sample_id")
rng = np.random.RandomState(7); pick = sm.sample_id.values[rng.choice(len(sm), 25, replace=False)]
rel = []
for sid in pick:
    r = sm[sm.sample_id == sid].iloc[0]
    v = feats(load(W + "/data_input/" + r.relative_path)); o = ac.loc[sid].values
    rel.append(np.max(np.abs(v - o) / (np.abs(o) + 1e-9)))
ev["acoustic_independent_max_relative_difference"] = float(np.max(rel))
print("(1) acoustic: max relative difference over 25 recordings x 40 features:", ev["acoustic_independent_max_relative_difference"])

# ---------- (2) independent nested CV for two conditions
split = pd.read_csv(W + "/SPLIT_MANIFEST.csv").set_index("sample_id").loc[sm.sample_id]
def folds_for(speakers, labels, k, seed):
    ids = np.array(sorted(speakers)); lab = np.array([labels[s] for s in ids]); r = np.random.RandomState(seed); out = {}; c = 0
    for cls in sorted(set(lab.tolist())):
        mem = ids[lab == cls][r.permutation((lab == cls).sum())]
        for s in mem: out[s] = c % k; c += 1
    return out
def lr_prob(Xa, ya, Xb, C, pen, cw):
    mu, sd = Xa.mean(0), Xa.std(0); sd[sd < 1e-12] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegression(solver="liblinear", C=C, penalty=pen, class_weight=cw, max_iter=3000, random_state=42).fit((Xa - mu) / sd, ya)
    return m.predict_proba((Xb - mu) / sd)[:, 1]
def spk_ap(p, s, y):
    d = pd.DataFrame({"p": p, "s": s, "y": y}).groupby("s").agg(p=("p", "mean"), y=("y", "first")); return average_precision_score(d.y, d.p)
grid = [(c, p, w) for c in [0.01, 0.1, 1, 10, 100] for p in ["l1", "l2"] for w in [None, "balanced"]]
spk = sm.speaker_id.astype(str).values; y = sm.label.values.astype(int); of = split.outer_fold.values
labs = dict(zip(sm.speaker_id.astype(str), sm.label.astype(int)))
for rep, key in [("WLM", "WLM"), ("XLSR", "XLSR")]:
    X = np.load(W + "/features/%s_embeddings.npy" % key).astype(np.float64)
    order = pd.read_csv(W + "/features/manifests/%s_sample_order.csv" % key).sample_id.tolist(); pos = {s: i for i, s in enumerate(order)}
    X = X[[pos[s] for s in sm.sample_id]]
    mx = 0.0; same_cfg = 0
    for f in range(5):
        tr = np.where(of != f)[0]; te = np.where(of == f)[0]
        trs = sorted(set(spk[tr])); fm = folds_for(trs, labs, 3, 142 + f); inner = np.array([fm[s] for s in spk[tr]])
        best, bscore = None, -1
        for ci, (C, pen, cw) in enumerate(grid):
            p = np.zeros(len(tr))
            for k in range(3):
                a, b = np.where(inner != k)[0], np.where(inner == k)[0]
                p[b] = lr_prob(X[tr[a]], y[tr[a]], X[tr[b]], C, pen, cw)
            sc = round(spk_ap(p, spk[tr], y[tr]), 12)
            if sc > bscore: best, bscore = (C, pen, cw), sc
        pt = lr_prob(X[tr], y[tr], X[te], *best)
        off = pd.read_csv(W + "/predictions/FINAL_V1/FV1-%s-ALL-LR-F%d.csv" % (rep, f)).set_index("sample_id").loc[sm.sample_id.values[te]]
        mx = max(mx, float(np.abs(off.predicted_probability.values - pt).max()))
        offp = json.load(open(W + "/metrics/FINAL_V1/FV1-%s-ALL-LR-F%d.json" % (rep, f)))["selected_hyperparameters"]
        same_cfg += int((offp["C"], offp["penalty"], offp["class_weight"]) == best)
    ev["independent_nested_cv_%s_LR" % rep] = {"max_abs_probability_difference": mx, "folds_with_same_selected_config": same_cfg}
    print("(2) independent nested CV", rep, "LR: max |dp| =", mx, "| same selected config in", same_cfg, "of 5 folds")

# ---------- (3) all threshold metrics from the prediction files
cm = pd.read_csv(W + "/metrics/FINAL_V1/condition_metrics.csv"); reg = pd.read_csv(W + "/EXPERIMENT_REGISTRY.csv", keep_default_na=False)
bad = []
for c in cm.itertuples():
    d = pd.concat([pd.read_csv(W + "/predictions/FINAL_V1/%s-F%d.csv" % (c.condition_id, f)) for f in range(5)])
    g = d.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first"))
    pr = (g.s >= 0.5).astype(int); tp = int(((pr == 1) & (g.y == 1)).sum()); fn = int(((pr == 0) & (g.y == 1)).sum()); tn = int(((pr == 0) & (g.y == 0)).sum()); fp = int(((pr == 1) & (g.y == 0)).sum())
    se, spc = tp / (tp + fn), tn / (tn + fp); prc = tp / (tp + fp) if tp + fp else 0; f1 = 2 * prc * se / (prc + se) if prc + se else 0
    mine = dict(F1=f1, sensitivity=se, specificity=spc, balanced_accuracy=(se + spc) / 2, AUROC=roc_auc_score(g.y, g.s), AUPRC=average_precision_score(g.y, g.s))
    for k, v in mine.items():
        if abs(v - getattr(c, k)) > 1e-9: bad.append((c.condition_id, k, v, getattr(c, k)))
    rr = reg[(reg.row_type == "condition_pooled") & (reg.condition_id == c.condition_id)].iloc[0]
    for k in ["AUROC", "AUPRC", "F1", "sensitivity", "specificity", "balanced_accuracy"]:
        if abs(float(rr[k]) - mine[k]) > 5e-6: bad.append((c.condition_id, "registry_" + k, mine[k], rr[k]))
ev["threshold_metric_disagreements"] = bad
print("(3) metric disagreements (56 conditions x 6 metrics, table and registry):", len(bad))

# ---------- (4) bootstrap and paired differences
spk_sorted = sorted(sm.speaker_id.astype(str).unique()); r = np.random.RandomState(42); idx = r.randint(0, len(spk_sorted), size=(2000, len(spk_sorted)))
def boot(cid):
    d = pd.concat([pd.read_csv(W + "/predictions/FINAL_V1/%s-F%d.csv" % (cid, f)) for f in range(5)])
    g = d.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first")).reindex(spk_sorted)
    au = []; ap = []
    for ix in idx:
        yy = g.y.values[ix]; ss = g.s.values[ix]
        if yy.min() == yy.max(): au.append(np.nan); ap.append(np.nan); continue
        au.append(roc_auc_score(yy, ss)); ap.append(average_precision_score(yy, ss))
    return np.array(au), np.array(ap)
pc = pd.read_csv(W + "/metrics/FINAL_V1/paired_comparisons.csv")
chk = []
for a, b in [("FV1-H1-ALL-LR", "FV1-WLM-ALL-LR"), ("FV1-WLM-ALL-RSVM", "FV1-WLM-ALL-LR")]:
    A, B = boot(a), boot(b)
    for k, nm in [(0, "AUROC"), (1, "AUPRC")]:
        dl = A[k] - B[k]; dl = dl[np.isfinite(dl)]; lo, hi = np.percentile(dl, [2.5, 97.5])
        row = pc[(pc.condition_A == a) & (pc.condition_B == b) & (pc.metric == nm)].iloc[0]
        chk.append((a, b, nm, abs(lo - row.ci95_lo) < 1e-12 and abs(hi - row.ci95_hi) < 1e-12))
    for k, nm in [(0, "AUROC"), (1, "AUPRC")]:
        v = A[k][np.isfinite(A[k])]; lo, hi = np.percentile(v, [2.5, 97.5]); rr = cm[cm.condition_id == a].iloc[0]
        chk.append((a, "own CI", nm, abs(lo - getattr(rr, nm + "_ci_lo")) < 1e-12 and abs(hi - getattr(rr, nm + "_ci_hi")) < 1e-12))
ev["bootstrap_recomputation"] = chk
print("(4) bootstrap/paired recomputation:", chk)

# ---------- (5) feature selection identical across models
diff = 0
for fs in ["FORWARD", "BACKWARD"]:
    for f in range(5):
        sets = [tuple(json.load(open(W + "/metrics/FINAL_V1/FV1-AC-%s-%s-F%d.json" % (fs, m, f)))["selected_features"]) for m in ["LR", "LSVM", "RSVM", "XGB"]]
        diff += int(len(set(sets)) != 1)
ev["feature_selection_differs_across_models_count"] = diff
print("(5) folds where selected features differ across models:", diff, "of 10")

# ---------- (6) convergence of the final LR fits
import pipeline as pl
conv = {"fits": 0, "convergence_warnings": 0, "examples": []}
for c in cm[cm.model == "LR"].itertuples():
    rep = c.representation; X, names = pl.load_representation(rep, pl.load_manifest())
    for f in range(5):
        r = json.load(open(W + "/metrics/FINAL_V1/%s-F%d.json" % (c.condition_id, f)))
        cols = [names.index(n) for n in r["selected_features"]] if r["selected_features"] else list(range(X.shape[1]))
        tr = np.where(of != f)[0]; hp = r["selected_hyperparameters"]
        Xa = X[tr][:, cols]; mu, sd = Xa.mean(0), Xa.std(0); sd[sd < 1e-12] = 1
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            LogisticRegression(solver="liblinear", C=hp["C"], penalty=hp["penalty"], class_weight=hp["class_weight"], max_iter=3000, random_state=42).fit((Xa - mu) / sd, y[tr])
        conv["fits"] += 1
        if any(issubclass(w.category, ConvergenceWarning) for w in wl):
            conv["convergence_warnings"] += 1; conv["examples"].append((c.condition_id, f, hp))
ev["final_LR_fit_convergence"] = conv
print("(6) final LR fits:", conv["fits"], "| with ConvergenceWarning:", conv["convergence_warnings"], conv["examples"][:5])
json.dump(ev, open(W + "/audit_opus/evidence/a03_independent_recompute.json", "w"), indent=1, default=str)
