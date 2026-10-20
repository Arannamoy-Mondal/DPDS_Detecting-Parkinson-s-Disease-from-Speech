"""Audit A05: demographics from the corpus spreadsheets (TAB 5 for PD, Tab 3 for controls), consistency with file names, and the
spreadsheet asterisk notes on four controls; error share of the flagged controls across the 56 conditions."""
import json, re
import numpy as np, pandas as pd
W = "/root/Projects/Project4a_FINAL_V1_ISOLATED"
pdx = pd.read_excel(W + "/data_input/28 People with Parkinson's disease/TAB 5.xlsx", header=None).iloc[2:30, 1:5]
pdx.columns = ["name", "surname", "sex", "age"]; pdx = pdx.dropna(subset=["name"])
pdx["name"] = pdx.name.astype(str).str.strip(); pdx["surname"] = pdx.surname.astype(str).str.strip()
pd_unique = pdx.drop_duplicates(["name", "surname", "sex", "age"])
hc = pd.read_excel(W + "/data_input/22 Elderly Healthy Control/Tab 3.xlsx", header=None).iloc[2:24, 0:4]
hc.columns = ["name", "surname", "sex", "age"]; hc["starred"] = hc.name.astype(str).str.contains(r"\*") | hc.surname.astype(str).str.contains(r"\*")
notes = [str(x) for x in pd.read_excel(W + "/data_input/22 Elderly Healthy Control/Tab 3.xlsx", header=None).iloc[24:30, 0].dropna().tolist()]
def stats(a): a = np.asarray(a, float); return {"n": int(len(a)), "mean": round(float(a.mean()), 2), "sd": round(float(a.std(ddof=1)), 2), "min": int(a.min()), "max": int(a.max()), "median": float(np.median(a))}
res = {"PD_sheet_rows": int(len(pdx)), "PD_unique_people": int(len(pd_unique)), "PD_age": stats(pd_unique.age), "PD_sex": pd_unique.sex.value_counts().to_dict(),
       "HC_rows": int(len(hc)), "HC_age": stats(hc.age), "HC_sex": hc.sex.value_counts().to_dict(), "HC_starred": hc[hc.starred][["name", "surname"]].astype(str).apply(lambda r: (r["name"] + " " + r["surname"]).replace("*", ""), axis=1).tolist(),
       "sheet_notes_for_asterisks": notes}
sm = pd.read_csv(W + "/manifests/SAMPLE_MANIFEST.csv")
sex_file = sm.drop_duplicates("speaker_id").groupby("label").sex.value_counts().to_dict()
res["sex_from_file_names_by_label"] = {str(k): v for k, v in sex_file.items()}
# age in sheet vs two-digit number in file names (aggregate)
res["file_name_number_by_label"] = {str(l): stats(g.drop_duplicates("speaker_id").age_in_name.astype(int)) for l, g in sm.groupby("label")}
# map HC speakers to folder names (speaker ids are assigned in folder-name order)
spk = pd.read_csv(W + "/manifests/SPEAKER_MANIFEST.csv")
hc_spk = spk[spk.label == 0].copy(); hc_spk["folder"] = hc_spk.source_keys.str.split("/").str[1]
flag_folders = {"AGNESE P": "Agnese P", "GILDA C": "Gilda C", "GIOVANNA G": "Giovanna G", "VITANTONIO D": "Vitantonio D"}
hc_spk["flagged"] = hc_spk.folder.isin(flag_folders)
res["flagged_control_speaker_ids"] = hc_spk[hc_spk.flagged][["speaker_id", "folder", "n_recordings", "tasks"]].to_dict("records")
# false-positive events across the 56 conditions
cm = pd.read_csv(W + "/metrics/FINAL_V1/condition_metrics.csv")
fp_events, fp_flagged, fn_events = 0, 0, {}
flagged_ids = set(hc_spk[hc_spk.flagged].speaker_id)
per_spk = {}
for cid in cm.condition_id:
    d = pd.concat([pd.read_csv(W + "/predictions/FINAL_V1/%s-F%d.csv" % (cid, f)) for f in range(5)])
    g = d.groupby("speaker_id").agg(s=("predicted_probability", "mean"), y=("true_label", "first"))
    wrong = g[(g.s >= 0.5).astype(int) != g.y]
    for s, r in wrong.iterrows():
        per_spk[s] = per_spk.get(s, 0) + 1
        if r.y == 0:
            fp_events += 1; fp_flagged += int(s in flagged_ids)
        else:
            fn_events[s] = fn_events.get(s, 0) + 1
res["false_positive_events_total"] = fp_events; res["false_positive_events_from_flagged_controls"] = fp_flagged
res["false_negative_events_by_speaker"] = fn_events; res["errors_by_speaker"] = dict(sorted(per_spk.items(), key=lambda kv: -kv[1]))
json.dump(res, open(W + "/audit_opus/evidence/a05_demographics_and_flags.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
