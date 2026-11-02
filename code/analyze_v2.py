"""
Analysis for experiment_v2.py
=============================
Usage:  python analyze_v2.py results_v2        (Jupyter: %run analyze_v2.py results_v2)

Writes <out>/summary_v2.md, CSV tables and figures:
  A  RQ1  overlap of flagged sets on clean data (classical vs target-aware)
  B  RQ4a detection of injected corruption (AUROC, precision@k)
  C  RQ3  mechanism: minority-class enrichment and instance hardness (kDN)
  D  RQ2/RQ4b downstream gain of every treatment vs. no treatment,
          Wilcoxon signed-rank across datasets with Holm correction
  E  honest-oracle headroom per condition (is a recommender worthwhile?)
All accuracy numbers are balanced accuracy in POINTS (x100). Every dataset weighs
equally: seeds are averaged within a dataset before averaging across datasets.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CLASSICAL = ["IQR", "MAD", "RobustMahalanobis", "LOF", "IsolationForest"]
TARGET_AWARE = ["DataOOB", "KNNShapley", "SelfConfidence"]
METHODS = CLASSICAL + TARGET_AWARE

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results_v2")
parts = sorted((out / "parts").glob("*.pkl"))
if not parts:
    sys.exit(f"No finished jobs in {out / 'parts'}")
res = [pd.read_pickle(f) for f in parts]
ov = pd.DataFrame([r for x in res for r in x["overlap"]])
det = pd.DataFrame([r for x in res for r in x["detect"]])
mech = pd.DataFrame([r for x in res for r in x["mech"]])
dn = pd.DataFrame([r for x in res for r in x["downstream"]])
dn["score"] = 100 * dn["balanced_accuracy"]
dn["option"] = np.where(dn["treatment"] == "none", "none",
                        dn["treatment"] + ":" + dn["method"] +
                        np.where(dn["treatment"] == "remove",
                                 "@" + dn["frac"].map(lambda f: f"{f:g}"), ""))

COND_ORDER = sorted(dn["condition"].unique(),
                    key=lambda c: (c != "clean", c.split("@")[0], float(c.split("@")[1]) if "@" in c else 0))
fam = {m: ("classical" if m in CLASSICAL else "target-aware") for m in METHODS}
pd.set_option("display.width", 220)
md = ["# Experiment v2: summary\n",
      f"Datasets: {dn['dataset'].nunique()} | seeds: {dn['seed'].nunique()} | "
      f"conditions: {', '.join(COND_ORDER)}\n",
      "All accuracy figures: balanced accuracy in points (x100), datasets weighted equally.\n"]


def ds_mean(df, keys, cols):
    """average seeds within dataset, then datasets"""
    return df.groupby(["dataset"] + keys)[cols].mean().reset_index()


def holm(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    adj = np.full_like(p, np.nan)
    if ok.sum():
        idx = np.where(ok)[0][np.argsort(p[ok])]
        m = len(idx)
        run = 0.0
        for rank, i in enumerate(idx):
            run = max(run, min(1.0, (m - rank) * p[i]))
            adj[i] = run
    return adj


# ---------------------------------------------------------------- A  RQ1
o = ov[ov["condition"] == "clean"].copy()
o["pair_type"] = ["within-classical" if fam[a] == fam[b] == "classical"
                  else "within-target-aware" if fam[a] == fam[b] else "cross-family"
                  for a, b in zip(o["a"], o["b"])]
A = (ds_mean(o, ["pair_type"], ["jaccard", "spearman"])
     .groupby("pair_type")[["jaccard", "spearman"]].agg(["mean", "std"]).round(3))
md += ["## A. RQ1 - Overlap of top-5% flagged points (clean data)\n",
       f"Random expectation of Jaccard: {0.05 / 1.95:.3f}\n", A.to_markdown(), ""]

# ---------------------------------------------------------------- B  RQ4a
if len(det):
    B = (ds_mean(det, ["condition", "method"], ["auroc", "precision_at_k"])
         .groupby(["condition", "method"])[["auroc", "precision_at_k"]].mean()
         .unstack("condition").round(3))
    B = B.reindex(METHODS)
    md += ["## B. RQ4a - Detection of injected corruption\n",
           "AUROC (0.5 = chance) and precision@k (k = true number of corrupted rows).\n",
           B.to_markdown(), ""]
    B.to_csv(out / "v2_detection.csv")

# ---------------------------------------------------------------- C  RQ3
c = mech[mech["condition"] == "clean"].copy()
c["hardness_ratio"] = c["kdn_flagged"] / c["kdn_all"].replace(0, np.nan)
C = (ds_mean(c, ["method"], ["minority_enrichment", "hardness_ratio", "kdn_flagged"])
     .groupby("method")[["minority_enrichment", "hardness_ratio", "kdn_flagged"]]
     .agg(["mean", "median"]).round(2).reindex(METHODS))
md += ["## C. RQ3 - Who gets flagged? (clean data, top-5%)\n",
       "minority_enrichment = share of smallest class among flagged / its share in T "
       "(1 = no bias). hardness_ratio = mean kDN of flagged / mean kDN of all "
       "(>1 = flagged points are harder, i.e. near class boundaries).\n",
       C.to_markdown(), ""]
C.to_csv(out / "v2_mechanism.csv")

# ---------------------------------------------------------------- D  RQ2 / RQ4b
per = ds_mean(dn, ["condition", "model", "option"], ["score"])
piv = per.pivot_table(index=["dataset", "condition", "model"], columns="option", values="score")
rows = []
for (cond, model), g in piv.groupby(level=["condition", "model"]):
    for opt in [x for x in g.columns if x != "none"]:
        d = (g[opt] - g["none"]).dropna()
        if len(d) == 0:
            continue
        p = wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else np.nan
        rows.append(dict(condition=cond, model=model, option=opt, n=len(d),
                         mean_gain=d.mean(), median_gain=d.median(),
                         wins=int((d > 0).sum()), losses=int((d < 0).sum()), p=p))
D = pd.DataFrame(rows)
D["p_holm"] = holm(D["p"].values)
D = D.round(4)
D.to_csv(out / "v2_gains_all.csv", index=False)

best = (D.sort_values("mean_gain", ascending=False)
          .groupby(["condition", "model"]).head(3)
          .sort_values(["condition", "model", "mean_gain"], ascending=[True, True, False]))
best["condition"] = pd.Categorical(best["condition"], COND_ORDER, ordered=True)
best = best.sort_values(["condition", "model", "mean_gain"], ascending=[True, True, False])
md += ["## D. RQ2/RQ4b - Top-3 treatments per condition and model (gain vs. no treatment)\n",
       "p: Wilcoxon signed-rank across datasets; p_holm: Holm-corrected over ALL "
       f"{len(D)} tests (see v2_gains_all.csv for every option).\n",
       best.to_markdown(index=False), ""]

# family-level view: best option of each family per condition (mean over models)
def family_of(opt):
    if opt.startswith("winsorize"):
        return "winsorize"
    m = opt.split(":")[1].split("@")[0]
    if m == "random":
        return "random-removal"
    return f"{opt.split(':')[0]}-{fam[m]}"
D["family"] = D["option"].map(family_of)
F = (D.groupby(["condition", "family", "option"])["mean_gain"].mean().reset_index()
       .sort_values("mean_gain", ascending=False).groupby(["condition", "family"]).head(1))
F = F.pivot(index="family", columns="condition", values="mean_gain")[COND_ORDER].round(2)
md += ["## D2. Best option of each family per condition (mean gain over the 4 models)\n",
       "Note: picking the best option post hoc is optimistic; use it to see trends only.\n",
       F.to_markdown(), ""]
F.to_csv(out / "v2_family_trends.csv")

# ---------------------------------------------------------------- E  headroom
rows = []
dn_opt = dn[~dn["option"].str.contains("random")]
for (ds, cond, model), g in dn_opt.groupby(["dataset", "condition", "model"]):
    t = g.pivot_table(index="seed", columns="option", values="score").dropna(axis=1)
    if "none" not in t or len(t) < 2:
        continue
    honest = [t.loc[s, t.drop(index=s).mean().idxmax()] - t.loc[s, "none"] for s in t.index]
    rows.append(dict(dataset=ds, condition=cond, model=model, honest_gain=np.mean(honest),
                     naive_gain=t.mean().max() - t["none"].mean(),
                     noise_sd=t["none"].std(ddof=1)))
if rows:
    E0 = pd.DataFrame(rows)
    bf = (D.groupby(["condition", "model"])["mean_gain"].max().rename("best_fixed_gain"))
    E = (E0.groupby(["condition", "model"])
           .agg(naive=("naive_gain", "mean"), honest=("honest_gain", "mean"),
                pct_honest_gt_1pt=("honest_gain", lambda x: 100 * (x > 1).mean()),
                p_honest_gt_0=("honest_gain", lambda x: wilcoxon(x, alternative="greater").pvalue
                               if len(x) >= 6 and np.any(x != 0) else np.nan),
                noise_sd=("noise_sd", "mean"))
           .join(bf).round(3).reset_index())
    E["condition"] = pd.Categorical(E["condition"], COND_ORDER, ordered=True)
    E = E.sort_values(["condition", "model"])
    md += ["## E. Headroom for a per-dataset recommender (leave-one-seed-out)\n",
           "A recommender is justified only where `honest` is clearly > 0 (small p) "
           "AND clearly above `best_fixed_gain` (one rule for all datasets).\n",
           E.to_markdown(index=False), ""]
    E.to_csv(out / "v2_headroom.csv", index=False)

log = out / "failures.log"
if log.exists() and log.stat().st_size:
    lines = log.read_text(encoding="utf-8").splitlines()
    wiped = sum("wiped out a class" in l for l in lines)
    md += [f"## Notes\nfailures.log: {len(lines)} lines "
           f"({wiped} removals skipped because they deleted an entire class).\n"]

(out / "summary_v2.md").write_text("\n".join(md), encoding="utf-8")

# ---------------------------------------------------------------- figures
try:
    from matplotlib.figure import Figure

    # Fig 1: gain vs corruption level for representative options
    max_frac = dn.loc[dn["treatment"] == "remove", "frac"].max()
    reps = [f"remove:SelfConfidence@{max_frac:g}", f"remove:DataOOB@{max_frac:g}",
            f"remove:IsolationForest@{max_frac:g}", f"remove:IQR@{max_frac:g}",
            "reweight:SelfConfidence", "winsorize:winsor_iqr", f"remove:random@{max_frac:g}"]
    g_mean = D.groupby(["condition", "option"])["mean_gain"].mean()
    for kind in ["label", "feature"]:
        conds = ["clean"] + [c for c in COND_ORDER if c.startswith(kind)]
        if len(conds) < 2:
            continue
        xs = [0.0] + [float(c.split("@")[1]) for c in conds[1:]]
        fig = Figure(figsize=(7.5, 5))
        ax = fig.subplots()
        for opt in reps:
            ys = [g_mean.get((c, opt), np.nan) for c in conds]
            if not np.all(np.isnan(ys)):
                ax.plot(xs, ys, marker="o", label=opt)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel(f"Injected {kind} corruption rate")
        ax.set_ylabel("Mean gain vs. no treatment (BA points, avg over models)")
        ax.set_title(f"When does treatment start to help? ({kind} corruption)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"fig_v2_gain_vs_{kind}.png", dpi=200)

    # Fig 2: detection AUROC per method and condition
    if len(det):
        au = (ds_mean(det, ["condition", "method"], ["auroc"])
              .groupby(["method", "condition"])["auroc"].mean().unstack("condition")
              .reindex(METHODS))
        fig = Figure(figsize=(8, 5))
        ax = fig.subplots()
        au.plot.bar(ax=ax)
        ax.axhline(0.5, color="black", lw=0.8, ls="--")
        ax.set_ylabel("AUROC for injected corrupted rows")
        ax.set_title("Which methods detect which corruption?")
        ax.tick_params(axis="x", rotation=40)
        ax.set_ylim(0, 1.05)
        ax.legend(title="condition", fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        fig.savefig(out / "fig_v2_detection.png", dpi=200)
except Exception as e:
    print(f"figures skipped: {e}")

print((out / "summary_v2.md").read_text(encoding="utf-8")[:6000])
print(f"\nFull summary: {out / 'summary_v2.md'}")
