"""Audit A02: independent re-derivation of labels and speakers from the raw tree; cross-speaker identity and near-duplicate audio checks."""
import os, re, sys, json
from collections import defaultdict
import numpy as np, pandas as pd
ROOT = "/root/Projects/PatchTST/Italian Parkinson's Voice and speech"
W = "/root/Projects/Project4a_FINAL_V1_ISOLATED"
rows = []
for dp, dn, fn in os.walk(ROOT):
    for f in fn:
        if not f.lower().endswith(".wav"): continue
        rel = os.path.relpath(os.path.join(dp, f), ROOT); parts = rel.split(os.sep)
        g = parts[0]
        if g.startswith("15 Young"): continue
        lab = 1 if g.startswith("28 People") else (0 if g.startswith("22 Elderly") else None)
        s = re.sub(r"[^A-Za-z0-9.]", "", f).upper()
        m = re.match(r"^(B1|B2|D1|D2|PR1|FB1|V[AEIOU][12])([A-Z]+)(\d\d)([MF])\d+\.+WAV$", s)
        rows.append({"rel": rel, "label": lab, "folder": parts[-2], "sub": parts[1] if len(parts) > 3 else "", "task": m.group(1), "code": m.group(2), "age": m.group(3), "sex": m.group(4)})
raw = pd.DataFrame(rows)
# independent identity: (label, folder name, sex, sorted letters of code)
raw["sig"] = raw.apply(lambda r: (r.label, r.folder.strip().upper(), r.sex, "".join(sorted(r.code))), axis=1)
ids = {s: i for i, s in enumerate(sorted(raw.sig.unique()))}
raw["indep_speaker"] = raw.sig.map(ids)
sm = pd.read_csv(W + "/manifests/SAMPLE_MANIFEST.csv")
inv = pd.read_csv(W + "/DATA_INVENTORY.csv", keep_default_na=False)
dupl = set(inv[(inv.included == 0) & inv.inclusion_reason.str.contains("duplicate")].relative_path)
raw = raw[~raw.rel.isin(dupl)]
mm = raw.merge(sm[["relative_path", "speaker_id", "label"]], left_on="rel", right_on="relative_path", how="outer", indicator=True)
res = {"raw_included_files": int(len(raw)), "manifest_files": int(len(sm)), "merge_status": mm._merge.value_counts().to_dict(),
       "label_mismatches": int((mm.label_x != mm.label_y).sum()),
       "indep_speakers": int(raw.indep_speaker.nunique()), "manifest_speakers": int(sm.speaker_id.nunique()),
       "speaker_partition_identical": bool(mm.groupby("indep_speaker").speaker_id.nunique().max() == 1 and mm.groupby("speaker_id").indep_speaker.nunique().max() == 1)}
# cross-speaker identity collisions ignoring folder names (same sex, same age, same letters) and across groups
sp = raw.groupby("indep_speaker").agg(label=("label", "first"), sex=("sex", "first"), ages=("age", lambda s: sorted(set(s))), letters=("code", lambda s: sorted(set("".join(sorted(c)) for c in s))))
coll = defaultdict(list)
for i, r in sp.iterrows():
    for L in r.letters: coll[(r.sex, L)].append(int(i))
res["identity_collisions_ignoring_folder"] = {str(k): v for k, v in coll.items() if len(set(v)) > 1}
# near-duplicate audio across speakers using WavLM embeddings and acoustic features
X = np.load(W + "/features/WLM_embeddings.npy").astype(np.float64)
order = pd.read_csv(W + "/features/manifests/WLM_sample_order.csv").sample_id.tolist()
spk = np.asarray(sm.set_index("sample_id").loc[order].speaker_id.astype(str).tolist())
Xn = X / np.linalg.norm(X, axis=1, keepdims=True); S = Xn @ Xn.T
np.fill_diagonal(S, -1)
same = spk[:, None] == spk[None, :]
res["max_cosine_cross_speaker_WLM"] = float(S[~same].max()); res["max_cosine_within_speaker_WLM"] = float(S[same & (S > -1)].max())
i, j = np.unravel_index(np.argmax(np.where(~same, S, -1)), S.shape)
res["most_similar_cross_speaker_pair"] = [order[i], order[j], str(spk[i]), str(spk[j])]
res["cross_speaker_pairs_cosine_ge_0.999"] = int((np.triu(np.where(~same, S, -1), 1) >= 0.999).sum())
A = pd.read_csv(W + "/features/acoustic_features.csv", index_col="sample_id").loc[order].values
Az = (A - A.mean(0)) / A.std(0); D = ((Az[:, None, :] - Az[None, :, :]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
res["min_acoustic_distance_cross_speaker"] = float(D[~same].min()); res["min_acoustic_distance_within_speaker"] = float(D[same & np.isfinite(D)].min())
json.dump(res, open(W + "/audit_opus/evidence/a02_data_identity.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
