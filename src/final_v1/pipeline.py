"""Nested, speaker-disjoint training and out-of-fold prediction for FINAL_V1.

Everything that is learned (standardization, feature selection, hyperparameters, the model) is fitted only on the rows of the
applicable training partition. The outer-test fold is touched exactly once, after the configuration was chosen."""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import protocol_defs as P
from splits import assign_folds

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- data
def load_manifest():
    return pd.read_csv(WORKSPACE / "manifests/SAMPLE_MANIFEST.csv")


def load_component(key, sm):
    """Feature matrix of one component, rows aligned to the sample manifest by sample_id (never by position alone)."""
    if key == "AC":
        df = pd.read_csv(WORKSPACE / "features/acoustic_features.csv", index_col="sample_id").loc[sm.sample_id]
        return df.values.astype(np.float64), list(df.columns)
    order = pd.read_csv(WORKSPACE / "features/manifests" / (key + "_sample_order.csv"))["sample_id"].tolist()
    arr = np.load(WORKSPACE / "features" / (key + "_embeddings.npy"))
    pos = {s: i for i, s in enumerate(order)}
    arr = arr[[pos[s] for s in sm.sample_id]]
    return arr.astype(np.float64), ["%s_%04d" % (key.lower(), i) for i in range(arr.shape[1])]


def load_representation(rep, sm):
    mats, names = [], []
    for c in P.components(rep):
        x, n = load_component(c, sm)
        mats.append(x); names += n
    return np.hstack(mats), names


# ---------------------------------------------------------------- models
def _standardize(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_predict(model, params, Xtr, ytr, Xte):
    """Return pseudo/probabilities in [0, 1] for the rows of Xte. Standardization uses the training rows only."""
    if model == "XGB":
        from xgboost import XGBClassifier
        spw = params["scale_pos_weight"]
        if spw == "balanced":
            spw = float((ytr == 0).sum()) / max(1.0, float((ytr == 1).sum()))
        clf = XGBClassifier(tree_method="hist", n_jobs=2, random_state=P.SEED, eval_metric="logloss", verbosity=0, learning_rate=params["learning_rate"],
                            max_depth=params["max_depth"], n_estimators=params["n_estimators"], min_child_weight=params["min_child_weight"],
                            subsample=params["subsample"], colsample_bytree=params["colsample_bytree"], reg_lambda=params["reg_lambda"], scale_pos_weight=spw)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    A, B = _standardize(Xtr, Xte)
    if model == "LR":
        clf = LogisticRegression(solver="liblinear", C=params["C"], penalty=params["penalty"], class_weight=params["class_weight"], max_iter=3000, random_state=P.SEED)
        clf.fit(A, ytr)
        return clf.predict_proba(B)[:, 1]
    if model == "LSVM":
        clf = SVC(kernel="linear", C=params["C"], class_weight=params["class_weight"], cache_size=500)
    else:
        clf = SVC(kernel="rbf", C=params["C"], gamma=params["gamma_mult"] / A.shape[1], class_weight=params["class_weight"], cache_size=500)
    clf.fit(A, ytr)
    d = clf.decision_function(B)
    return 1.0 / (1.0 + np.exp(-np.clip(d, -30, 30)))


# ---------------------------------------------------------------- metrics
def speaker_scores(prob, speakers, y):
    df = pd.DataFrame({"p": prob, "s": speakers, "y": y})
    g = df.groupby("s").agg(p=("p", "mean"), y=("y", "first"))
    return g.p.values, g.y.values, g.index.values


def speaker_auprc(prob, speakers, y):
    p, t, _ = speaker_scores(prob, speakers, y)
    return float(average_precision_score(t, p))


def binary_metrics(score, y, thr=P.THRESHOLD):
    score, y = np.asarray(score), np.asarray(y)
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (tp + fp) and (tp + fn) and (prec + sens) > 0 else (0.0 if (tp + fp + fn) else float("nan"))
    out = {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
           "accuracy": (tp + tn) / len(y), "balanced_accuracy": (sens + spec) / 2}
    if len(set(y.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y, score)); out["auprc"] = float(average_precision_score(y, score))
    else:
        out["auroc"] = float("nan"); out["auprc"] = float("nan")
    return out


# ---------------------------------------------------------------- feature selection (acoustic only)
def _cv_auprc(cols, X, y, spk, folds):
    probs = np.zeros(len(y))
    for tr, va in folds:
        probs[va] = fit_predict("LR", {"C": 1.0, "penalty": "l2", "class_weight": "balanced"}, X[tr][:, cols], y[tr], X[va][:, cols])
    return speaker_auprc(probs, spk, y)


def select_features(X, y, spk, method, seed):
    """Sequential selection to exactly P.FS_K columns, scored by a speaker-disjoint 3-fold CV inside (X, y, spk) only."""
    labels = {s: int(y[spk == s][0]) for s in np.unique(spk)}
    fmap = assign_folds(list(labels), labels, P.N_INNER, seed)
    f = np.array([fmap[s] for s in spk])
    folds = [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(P.N_INNER)]
    d = X.shape[1]
    if method == "FORWARD":
        chosen = []
        while len(chosen) < P.FS_K:
            best, bs = None, -1.0
            for j in range(d):
                if j in chosen:
                    continue
                s = _cv_auprc(chosen + [j], X, y, spk, folds)
                if s > bs + 1e-12:
                    best, bs = j, s
            chosen.append(best)
        return sorted(chosen)
    keep = list(range(d))
    while len(keep) > P.FS_K:
        best, bs = None, -1.0
        for j in keep:
            s = _cv_auprc([c for c in keep if c != j], X, y, spk, folds)
            if s > bs + 1e-12:
                best, bs = j, s
        keep.remove(best)
    return sorted(keep)


# ---------------------------------------------------------------- one outer fold of one condition
def run_fold(cond, fold, X, names, sm):
    """Returns dict(prediction frame, selected params, inner table, selected features, runtime)."""
    t0 = time.time()
    model, fs = cond["model"], cond["feature_selection"]
    spk_all = sm.speaker_id.values
    y_all = sm.label.values.astype(int)
    split = pd.read_csv(WORKSPACE / "SPLIT_MANIFEST.csv").set_index("sample_id").loc[sm.sample_id]
    of = split.outer_fold.values
    tr_idx, te_idx = np.where(of != fold)[0], np.where(of == fold)[0]
    labels = {s: int(y_all[spk_all == s][0]) for s in np.unique(spk_all[tr_idx])}
    inner_map = assign_folds(list(labels), labels, P.N_INNER, 142 + fold)
    inner_f = np.array([inner_map[s] for s in spk_all[tr_idx]])
    inner = []
    for k in range(P.N_INNER):
        a, b = tr_idx[inner_f != k], tr_idx[inner_f == k]
        cols = list(range(X.shape[1]))
        if fs != "ALL":
            cols = select_features(X[a], y_all[a], spk_all[a], fs, 242 + fold)
        inner.append((a, b, cols))
    grid = P.grid(model)
    table = []
    for ci, params in enumerate(grid):
        probs = np.full(len(tr_idx), np.nan)
        pos = {i: n for n, i in enumerate(tr_idx)}
        for a, b, cols in inner:
            p = fit_predict(model, params, X[a][:, cols], y_all[a], X[b][:, cols])
            for i, v in zip(b, p):
                probs[pos[i]] = v
        score = speaker_auprc(probs, spk_all[tr_idx], y_all[tr_idx])
        table.append({"config_index": ci, "params": params, "inner_speaker_auprc": score})
    best = max(table, key=lambda r: (round(r["inner_speaker_auprc"], 12), -r["config_index"]))
    cols = list(range(X.shape[1]))
    if fs != "ALL":
        cols = select_features(X[tr_idx], y_all[tr_idx], spk_all[tr_idx], fs, 242 + fold)
    probs = fit_predict(model, best["params"], X[tr_idx][:, cols], y_all[tr_idx], X[te_idx][:, cols])
    pred = pd.DataFrame({"sample_id": sm.sample_id.values[te_idx], "speaker_id": spk_all[te_idx], "outer_fold": fold, "true_label": y_all[te_idx],
                         "predicted_probability": probs})
    return {"pred": pred, "best_params": best["params"], "inner_table": table, "selected_features": [names[c] for c in cols] if fs != "ALL" else [],
            "n_features_used": len(cols), "runtime_seconds": time.time() - t0, "n_train_speakers": len(labels), "n_test_speakers": int(len(set(spk_all[te_idx])))}
