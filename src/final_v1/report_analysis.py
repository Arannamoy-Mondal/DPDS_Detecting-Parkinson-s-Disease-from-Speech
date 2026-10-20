"""Extra verified analyses for the written reports (all from FINAL_V1 files): replay check, train/inner/test table, error analysis, parameter counts."""
import json, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import protocol_defs as P
import pipeline as pl
import runner

MET = WORKSPACE / "metrics/FINAL_V1"; PRD = WORKSPACE / "predictions/FINAL_V1"


def replay():
    """Re-execute three completed conditions into tests/replay and compare with the official predictions (determinism / reproduction check)."""
    out = WORKSPACE / "tests/replay"
    shutil.rmtree(out, ignore_errors=True)
    sm = pl.load_manifest()
    ph, sh = runner.frozen_hashes()
    res = []
    for cid in ["FV1-WLM-ALL-LR", "FV1-AC-ALL-LR", "FV1-XLSR-ALL-LSVM"]:
        rep = cid.split("-")[1]
        X, names = pl.load_representation(rep, sm)
        mx = 0.0
        for f in range(5):
            rid = "%s-F%d" % (cid, f)
            r = runner.run_one(rid, X, names, sm, out, ph, sh)
            a = pd.read_csv(PRD / (rid + ".csv")).set_index("sample_id").predicted_probability
            b = pd.read_csv(out / "predictions/FINAL_V1" / (rid + ".csv")).set_index("sample_id").predicted_probability.loc[a.index]
            mx = max(mx, float(np.abs(a - b).max()))
        res.append({"condition": cid, "max_abs_probability_difference": mx})
    return res


def train_val_test():
    rows = []
    for cid in ["FV1-AC-ALL-LR", "FV1-WLM-ALL-LR", "FV1-W2V-ALL-LR", "FV1-XLSR-ALL-LR", "FV1-H3-ALL-LR", "FV1-WLM-ALL-XGB"]:
        inner, test = [], []
        for f in range(5):
            r = json.loads((MET / ("%s-F%d.json" % (cid, f))).read_text())
            best = max(r["inner_table"], key=lambda t: (round(t["inner_speaker_auprc"], 12), -t["config_index"]))
            inner.append(best["inner_speaker_auprc"]); test.append(r["speaker_level"]["auprc"])
        rows.append({"condition": cid, "inner_val_auprc_mean": float(np.mean(inner)), "inner_val_auprc_sd": float(np.std(inner, ddof=1)),
                     "heldout_fold_auprc_mean": float(np.mean(test)), "heldout_fold_auprc_sd": float(np.std(test, ddof=1))})
    return rows


def errors():
    sm = pl.load_manifest()
    sp = sm.drop_duplicates("speaker_id").set_index("speaker_id")
    cm = pd.read_csv(MET / "condition_metrics.csv")
    wrong = {}
    for cid in cm.condition_id:
        d = pd.concat([pd.read_csv(PRD / ("%s-F%d.csv" % (cid, f))) for f in range(5)])
        g = d.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first"))
        for s, r in g.iterrows():
            if int(r.s >= 0.5) != int(r.y):
                wrong.setdefault(s, []).append(cid)
    out = {"speaker_error_counts": {s: len(v) for s, v in sorted(wrong.items(), key=lambda kv: -len(kv[1]))}, "n_conditions": len(cm)}
    detail = {}
    for s in list(out["speaker_error_counts"])[:3]:
        r = sp.loc[s]
        ss = sm[sm.speaker_id == s]
        info = {"label": int(r.label), "age_token": ss.age_in_name.iloc[0], "sex": ss.sex.iloc[0], "dates": sorted(set(ss.date)), "sample_rates": sorted(set(map(int, ss.sample_rate))), "n_recordings": int(len(ss))}
        for cid in ["FV1-WLM-ALL-LR", "FV1-AC-ALL-LR"]:
            d = pd.concat([pd.read_csv(PRD / ("%s-F%d.csv" % (cid, f))) for f in range(5)])
            d = d[d.speaker_id == s].merge(sm[["sample_id", "task_code"]], on="sample_id")
            info[cid] = {"speaker_score": float(d.predicted_probability.mean()), "by_task": {t: round(float(p), 3) for t, p in zip(d.task_code, d.predicted_probability)}}
        detail[s] = info
    out["detail"] = detail
    # recording-level error by task family for two reference conditions
    fam = lambda t: "vowel" if t.startswith("V") else ("syllable" if t.startswith("D") else ("passage" if t.startswith("B") else ("word list" if t == "PR1" else "phrase")))
    fam_err = {}
    for cid in ["FV1-WLM-ALL-LR", "FV1-AC-ALL-LR"]:
        d = pd.concat([pd.read_csv(PRD / ("%s-F%d.csv" % (cid, f))) for f in range(5)]).merge(sm[["sample_id", "task_code"]], on="sample_id")
        d["family"] = d.task_code.map(fam)
        d["wrong"] = (d.predicted_class != d.true_label)
        fam_err[cid] = {k: {"n": int(len(g)), "wrong": int(g.wrong.sum()), "rate": float(g.wrong.mean())} for k, g in d.groupby("family")}
        fam_err[cid]["_all"] = {"n": int(len(d)), "wrong": int(d.wrong.sum()), "rate": float(d.wrong.mean())}
    out["recording_error_by_task_family"] = fam_err
    return out


def main():
    res = {"replay": replay(), "train_val_test": train_val_test(), "errors": errors(),
           "parameter_counts_logistic_regression": {r: sum(P.DIMS[c] for c in P.components(r)) + 1 for r in P.BASE_REPS + list(P.HYBRIDS)}}
    write_json(MET / "report_analysis.json", res)
    print(json.dumps(res, indent=1)[:3500])


main()
