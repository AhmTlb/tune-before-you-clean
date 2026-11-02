"""
Oracle analysis: is there real headroom for a per-dataset recommender?
======================================================================
Uses results/downstream.csv from the motivating experiment (no recomputation).

For every (dataset, removal fraction, model) it compares:
  * naive oracle   : best option by mean over ALL seeds  (optimistic: winner's curse)
  * honest oracle  : best option chosen on the other seeds, scored on the held-out
                     seed (leave-one-seed-out). This is the realistic ceiling of
                     any recommender, because it removes selection-on-noise.
  * best fixed     : the single option that is best on average across datasets
                     (what a "one rule for all" practitioner would use).
All gains are in balanced-accuracy points (x100) relative to "none" (no removal).
Noise reference: the seed-to-seed standard deviation of "none".

Usage:  python oracle_analysis.py results
        (in Jupyter:  %run oracle_analysis.py results)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
dn = pd.read_csv(out / "downstream.csv")
dn = dn[dn["method"] != "random"]                     # random is a control, not an option
dn["score"] = dn["balanced_accuracy"] * 100

rows = []
for (ds, frac, model), g in dn.groupby(["dataset", "frac", "model"]):
    t = g.pivot_table(index="seed", columns="method", values="score")
    if "none" not in t or len(t) < 2:
        continue
    t = t.dropna(axis=1)                              # options available on every seed
    seeds = t.index.tolist()
    naive_best = t.mean().idxmax()
    naive_gain = t.mean().max() - t["none"].mean()
    honest = []
    for s in seeds:                                   # leave-one-seed-out selection
        pick = t.drop(index=s).mean().idxmax()
        honest.append(t.loc[s, pick] - t.loc[s, "none"])
    rows.append(dict(dataset=ds, frac=frac, model=model,
                     naive_best=naive_best, naive_gain=naive_gain,
                     honest_gain=np.mean(honest),
                     none_seed_sd=t["none"].std(ddof=1)))
per = pd.DataFrame(rows)

# best single fixed option per (frac, model) across datasets
fixed = (dn.groupby(["frac", "model", "method", "dataset"])["score"].mean()
           .unstack("method"))
fixed_gain = (fixed.sub(fixed["none"], axis=0).groupby(level=["frac", "model"]).mean())
best_fixed = fixed_gain.drop(columns="none").idxmax(axis=1).rename("best_fixed")
best_fixed_gain = fixed_gain.drop(columns="none").max(axis=1).rename("best_fixed_gain")


def p_greater(x):
    x = np.asarray(x)
    return wilcoxon(x, alternative="greater").pvalue if len(x) >= 6 and np.any(x != 0) else np.nan


summary = (per.groupby(["frac", "model"])
             .agg(datasets=("dataset", "nunique"),
                  naive_gain=("naive_gain", "mean"),
                  honest_gain=("honest_gain", "mean"),
                  honest_gain_median=("honest_gain", "median"),
                  pct_honest_positive=("honest_gain", lambda x: 100 * (x > 0).mean()),
                  pct_honest_gt_1pt=("honest_gain", lambda x: 100 * (x > 1).mean()),
                  p_honest_gt_0=("honest_gain", p_greater),
                  noise_sd_none=("none_seed_sd", "mean"))
             .join(best_fixed).join(best_fixed_gain)
             .round(3).reset_index())

pd.set_option("display.width", 200)
print("Gains in balanced-accuracy POINTS vs. no removal (mean over datasets)\n")
print(summary.to_string(index=False))

print("\nHow to read it")
print(" - naive_gain   : optimistic ceiling (inflated by picking the luckiest option).")
print(" - honest_gain  : realistic ceiling of a per-dataset recommender.")
print(" - noise_sd_none: typical seed-to-seed noise; honest gains well below it are not real.")
print(" - Recommender is worth building only if honest_gain > 0 with small p_honest_gt_0,")
print("   and clearly above best_fixed_gain.")

top = (per.sort_values("honest_gain", ascending=False)
          .head(10)[["dataset", "frac", "model", "naive_best", "honest_gain", "none_seed_sd"]]
          .round(2))
print("\nDatasets with the largest honest gain (where treatment really helps):")
print(top.to_string(index=False))

per.to_csv(out / "oracle_per_dataset.csv", index=False)
summary.to_csv(out / "oracle_summary.csv", index=False)
print(f"\nSaved: {out / 'oracle_summary.csv'} and {out / 'oracle_per_dataset.csv'}")
