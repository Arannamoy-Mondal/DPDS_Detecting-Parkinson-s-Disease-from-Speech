"""Single source of truth for the frozen FINAL_V1 design (grids, matrix, seeds). Frozen: do not edit after FREEZE_RECORD.json exists."""
import itertools
import numpy as np

SEED = 42
N_OUTER, N_INNER = 5, 3
THRESHOLD = 0.5
FS_K = 10
BOOTSTRAP = 2000
ENCODERS = ["WLM", "W2V", "XLSR"]
BASE_REPS = ["AC", "WLM", "W2V", "XLSR"]
HYBRIDS = {"H1": ["WLM", "AC"], "H2": ["W2V", "AC"], "H3": ["XLSR", "AC"], "H4": ["WLM", "W2V"], "H5": ["WLM", "XLSR"], "H6": ["W2V", "XLSR"],
           "H7": ["WLM", "W2V", "XLSR"], "H8": ["WLM", "W2V", "XLSR", "AC"]}
MODELS = ["LR", "LSVM", "RSVM", "XGB"]
DIMS = {"AC": 40, "WLM": 1024, "W2V": 1024, "XLSR": 1024}


def components(rep):
    return HYBRIDS[rep] if rep in HYBRIDS else [rep]


def grid(model):
    if model == "LR":
        return [{"C": c, "penalty": p, "class_weight": w} for c, p, w in itertools.product([0.01, 0.1, 1, 10, 100], ["l1", "l2"], [None, "balanced"])]
    if model == "LSVM":
        return [{"C": c, "class_weight": w} for c, w in itertools.product([0.01, 0.1, 1, 10, 100], [None, "balanced"])]
    if model == "RSVM":
        return [{"C": c, "gamma_mult": g, "class_weight": w} for c, g, w in itertools.product([0.1, 1, 10, 100], [0.1, 1, 10], [None, "balanced"])]
    if model == "XGB":
        full = [dict(learning_rate=a, max_depth=b, n_estimators=c, min_child_weight=d, subsample=e, colsample_bytree=f, reg_lambda=g, scale_pos_weight=h)
                for a, b, c, d, e, f, g, h in itertools.product([0.03, 0.1], [2, 4], [100, 300], [1, 5], [0.7, 1.0], [0.3, 1.0], [1, 10], [1, "balanced"])]
        keep = sorted(np.random.RandomState(SEED).choice(len(full), 24, replace=False).tolist())
        return [full[i] for i in keep]
    raise SystemExit("unknown model " + model)


def conditions():
    out = []
    for rep in BASE_REPS + list(HYBRIDS):
        fss = ["ALL", "FORWARD", "BACKWARD"] if rep == "AC" else ["ALL"]
        for fs in fss:
            for m in MODELS:
                out.append({"condition_id": "FV1-%s-%s-%s" % (rep, fs, m), "representation": rep, "feature_selection": fs, "model": m})
    return out


def protocol_dict():
    return {"protocol_version": "FINAL_V1", "seed": SEED, "n_outer": N_OUTER, "n_inner": N_INNER, "threshold": THRESHOLD, "feature_selection_k": FS_K,
            "bootstrap_resamples": BOOTSTRAP, "representations": {"base": BASE_REPS, "hybrids": HYBRIDS, "dims": DIMS},
            "models": MODELS, "grids": {m: grid(m) for m in MODELS}, "conditions": conditions(),
            "inner_objective": "speaker-level AUPRC pooled over the inner validation folds", "fwl": "BLOCKED_PROTOCOL_DECISION_REQUIRED"}
