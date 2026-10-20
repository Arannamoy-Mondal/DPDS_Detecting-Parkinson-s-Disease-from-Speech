"""S0: descriptive univariate analysis of each of the 40 acoustic features at speaker level (whole-data, descriptive: not used for any model decision)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
from scipy import stats
from sklearn.metrics import roc_auc_score

sm, X, names = load_ac()
df = pd.DataFrame(X, columns=names); df["speaker_id"] = sm.speaker_id.values; df["label"] = sm.label.values; df["date"] = sm.date.values; df["sr"] = sm.sample_rate.values
spk = df.groupby("speaker_id").agg({**{n: "mean" for n in names}, "label": "first", "sr": "first"})
datemode = df.groupby("speaker_id").date.agg(lambda s: s.mode().iloc[0]); spk["date"] = datemode
pdm, hcm = spk[spk.label == 1], spk[spk.label == 0]
# native-16-kHz subset: recordings at 16 kHz only (removes the sample-rate cue), speaker means
d16 = df[df.sr == 16000].groupby("speaker_id").agg({**{n: "mean" for n in names}, "label": "first"})
rows = []
def eta2(g, col):
    grand = g[col].mean(); ss_t = ((g[col] - grand) ** 2).sum()
    ss_b = sum(len(x) * (x[col].mean() - grand) ** 2 for _, x in g.groupby("date"))
    return float(ss_b / ss_t) if ss_t > 0 else float("nan")
for n in names:
    a, b = pdm[n].values, hcm[n].values
    auc = roc_auc_score(spk.label.values, spk[n].values)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp
    # within-class session effect: variance of the speaker means explained by recording date
    e_pd, e_hc = eta2(pdm, n), eta2(hcm, n)
    # within PD: 44.1 kHz (2016 sessions) versus 16 kHz (2017 sessions)
    sr_auc = roc_auc_score((pdm.sr == 44100).astype(int), pdm[n]) if pdm.sr.nunique() == 2 else float("nan")
    # total variance share of class vs date (two-way indicator: class alone)
    eta_class = float(1 - ((spk[n] - spk.groupby("label")[n].transform("mean")) ** 2).sum() / ((spk[n] - spk[n].mean()) ** 2).sum())
    fam = n[3:].rsplit("_", 1)[0]; stat = n.rsplit("_", 1)[1]
    rows.append({"feature": n, "family": fam, "statistic": stat, "mean_PD": a.mean(), "mean_control": b.mean(), "AUROC_class": auc, "AUROC_abs": max(auc, 1 - auc), "direction": "PD higher" if auc > 0.5 else "PD lower",
                 "AUROC_class_native16k_only": roc_auc_score(d16.label.values, d16[n].values), "cohens_d": d, "mann_whitney_p": float(stats.mannwhitneyu(a, b).pvalue), "eta2_class": eta_class, "eta2_date_within_PD": e_pd, "eta2_date_within_controls": e_hc,
                 "AUROC_within_PD_44k_vs_16k": sr_auc, "abs_AUROC_within_PD_sr": max(sr_auc, 1 - sr_auc)})
out = pd.DataFrame(rows).sort_values("AUROC_abs", ascending=False)
out.to_csv(RES / "S0_univariate_features.csv", index=False)
fam = out.groupby("family").agg(mean_abs_auroc=("AUROC_abs", "mean"), max_abs_auroc=("AUROC_abs", "max"), mean_eta2_date_PD=("eta2_date_within_PD", "mean"), mean_eta2_date_HC=("eta2_date_within_controls", "mean")).sort_values("mean_abs_auroc", ascending=False)
fam.to_csv(RES / "S0_family_summary.csv")
stt = out.groupby("statistic").agg(mean_abs_auroc=("AUROC_abs", "mean"), max_abs_auroc=("AUROC_abs", "max")).sort_values("mean_abs_auroc", ascending=False); stt.to_csv(RES / "S0_statistic_summary.csv")
# correlation among features (speaker level) to describe redundancy
corr = spk[names].corr(method="spearman"); corr.to_csv(RES / "S0_feature_spearman_corr.csv")
hi = [(a, b, corr.loc[a, b]) for i, a in enumerate(names) for b in names[i + 1:] if abs(corr.loc[a, b]) >= 0.9]
pd.DataFrame(hi, columns=["feature_a", "feature_b", "spearman"]).to_csv(RES / "S0_highly_correlated_pairs.csv", index=False)
print("native-16k speakers:", int(d16.label.sum()), "PD /", int((1 - d16.label).sum()), "controls"); print(out[["feature", "AUROC_class", "AUROC_class_native16k_only", "cohens_d", "eta2_class", "eta2_date_within_PD", "eta2_date_within_controls", "AUROC_within_PD_44k_vs_16k"]].round(3).head(15).to_string())
print(fam.round(3)); print(stt.round(3)); print("pairs |rho|>=0.9:", len(hi), "| features with AUROC>=0.8:", int((out.AUROC_abs >= .8).sum()), "| >=0.9:", int((out.AUROC_abs >= .9).sum()))
