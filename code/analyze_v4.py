"""
Analysis for experiment_v4.py
=============================
Usage:  python analyze_v4.py results_v4        (Jupyter: %run analyze_v4.py results_v4)

A  Gains per family: every (variant, treatment) vs. the family's DEFAULT model with
   NO treatment, per condition (points of balanced accuracy).
B  Pre-registered tests (PROTOCOL_v4.md), Holm-corrected over B only:
   H11a recovery: robust tuned LogReg (quantile+tuned) WITHOUT cleaning recovers >= 50%
        of the gain that default LogReg gets from matched cleaning (dirty conditions).
   H11b on top: matched cleaning still helps the robust tuned LogReg (per dirty condition).
   H11c trees: on clean data, removal still harms TUNED RandomForest and HistGB.
C  Class deletion: how often each treatment deletes a whole class and what it costs.
Matched cleaning: feature_* -> remove:IF ; label_* -> remove:OOB ; mixed -> both tested.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results_v4")
parts = sorted((out / "parts").glob("*.pkl"))
if not parts:
    sys.exit(f"No finished jobs in {out / 'parts'}")
df = pd.DataFrame([r for f in parts for r in pd.read_pickle(f)["rows"]])
METRIC = sys.argv[2] if len(sys.argv) > 2 else "balanced_accuracy"   # ba_prior | auc_macro (BA-threshold check)
SUF = "" if METRIC == "balanced_accuracy" else f"_{METRIC}"
df["score"] = 100 * df[METRIC]
CONDS = sorted(df["condition"].unique(), key=lambda c: (c != "clean", c))
DIRTY = [c for c in CONDS if c != "clean"]
key = ["dataset", "seed", "condition", "family"]
base_ = (df[(df["variant"] == "default") & (df["treatment"] == "none")]
         .set_index(key)["score"].rename("base"))
df = df.join(base_, on=key)
df["gain"] = df["score"] - df["base"]
df["option"] = df["variant"] + " | " + df["treatment"]


def per_ds(sub):
    """seed-average within dataset -> Series indexed by dataset"""
    return sub.groupby("dataset")["gain"].mean()


def wil(x, alternative="two-sided"):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return wilcoxon(x, alternative=alternative).pvalue if len(x) >= 6 and np.any(x != 0) else np.nan


def holm(p):
    p = np.asarray(p, float)
    adj = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    run = 0.0
    for r, i in enumerate(ok[np.argsort(p[ok])]):
        run = max(run, min(1.0, (len(ok) - r) * p[i]))
        adj[i] = run
    return adj


def matched(cond):
    if cond.startswith("feature"):
        return ["remove:IF"]
    if cond.startswith("label"):
        return ["remove:OOB"]
    return ["remove:IF", "remove:OOB"]


md = ["# Experiment v4: tuning / robust pipelines vs. cleaning\n",
      f"Datasets: {df['dataset'].nunique()} | seeds: {df['seed'].nunique()} | conditions: {', '.join(CONDS)}\n",
      "Gain = balanced-accuracy points vs. the family's DEFAULT model with NO treatment. "
      "Datasets weigh equally (seeds averaged first).\n"]

# ------------------------------------------------------------------ A
md += ["## A. Gains per family\n"]
for fam in ["LogReg", "KNN", "RandomForest", "HistGB"]:
    sub = df[df["family"] == fam]
    if sub.empty:
        continue
    T = (sub.groupby(["option", "condition", "dataset"])["gain"].mean()
         .groupby(["option", "condition"]).mean().unstack("condition")[CONDS])
    T["dirty mean"] = T[DIRTY].mean(axis=1)
    order = [f"{v} | {t}" for v in sub["variant"].unique()
             for t in ["none", "remove:IF", "remove:OOB", "winsorize"]]
    T = T.reindex([o for o in order if o in T.index]).round(2)
    md += [f"### {fam}\n", T.to_markdown(), ""]
    T.to_csv(out / f"v4_gains_{fam}{SUF}.csv")

# ------------------------------------------------------------------ B
tests, rec_rows = [], []
lr = df[df["family"] == "LogReg"]
for c in DIRTY:
    for m in matched(c):
        g_clean_default = per_ds(lr[(lr["condition"] == c) & (lr["option"] == f"default | {m}")])
        g_robust_none = per_ds(lr[(lr["condition"] == c) & (lr["option"] == "quantile+tuned | none")])
        g_robust_clean = per_ds(lr[(lr["condition"] == c) & (lr["option"] == f"quantile+tuned | {m}")])
        rec_rows.append(dict(condition=c, cleaning=m,
                             default_cleaning_gain=g_clean_default.mean(),
                             robust_tuned_no_cleaning_gain=g_robust_none.mean(),
                             recovery=(g_robust_none.mean() / g_clean_default.mean()
                                       if g_clean_default.mean() > 0 else np.nan)))
        d = (g_robust_clean - g_robust_none).dropna()
        tests.append(dict(hypothesis="H11b cleaning on top of robust tuned LogReg", condition=c,
                          comparison=f"quantile+tuned | {m}  vs  quantile+tuned | none",
                          n=len(d), mean_diff=d.mean(), p=wil(d, "greater")))
for fam in ["RandomForest", "HistGB"]:
    t = df[(df["family"] == fam) & (df["condition"] == "clean")]
    ref = per_ds(t[t["option"] == "tuned | none"])
    for m in ["remove:IF", "remove:OOB"]:
        d = (per_ds(t[t["option"] == f"tuned | {m}"]) - ref).dropna()
        tests.append(dict(hypothesis=f"H11c removal harms tuned {fam} on clean", condition="clean",
                          comparison=f"tuned | {m}  vs  tuned | none", n=len(d), mean_diff=d.mean(),
                          p=wil(d, "less")))
R = pd.DataFrame(rec_rows).round(3)
dirty_avg = R.groupby("condition")[["default_cleaning_gain", "robust_tuned_no_cleaning_gain"]].mean().mean()
overall_recovery = dirty_avg["robust_tuned_no_cleaning_gain"] / dirty_avg["default_cleaning_gain"]
B = pd.DataFrame(tests)
B["p_holm"] = holm(B["p"].values)
B = B.round(4)
h11b_ok = (B[B["hypothesis"].str.startswith("H11b")]["p_holm"] < 0.05).sum()
h11b_n = B["hypothesis"].str.startswith("H11b").sum()
h11c = B[B["hypothesis"].str.startswith("H11c")]
md += ["## B. Pre-registered tests\n",
       "### H11a: does a robust tuned LogReg (quantile + tuned C) without cleaning recover the cleaning gain?\n",
       R.to_markdown(index=False), "",
       f"Average over dirty conditions: default+cleaning = {dirty_avg['default_cleaning_gain']:.3f}, "
       f"robust tuned without cleaning = {dirty_avg['robust_tuned_no_cleaning_gain']:.3f} -> "
       f"recovery = **{overall_recovery:.0%}** -> H11a "
       f"{'SUPPORTED' if overall_recovery >= 0.5 else 'NOT supported'} (criterion >= 50%).\n",
       "### H11b and H11c (one-sided Wilcoxon over datasets, Holm over this table)\n",
       B.to_markdown(index=False), "",
       f"H11b: cleaning adds significantly on top of the robust tuned LogReg in **{h11b_ok} of {h11b_n}** "
       "dirty-condition tests.",
       f"H11c: removal significantly harms tuned trees on clean data in "
       f"**{(h11c['p_holm'] < 0.05).sum()} of {len(h11c)}** tests.\n"]
B.to_csv(out / f"v4_primary_tests{SUF}.csv", index=False)
R.to_csv(out / f"v4_recovery{SUF}.csv", index=False)

# ------------------------------------------------------------------ C
w = df[df["variant"] == "default"]
C1 = (w.groupby(["treatment"])["wiped_class"].mean() * 100).round(1).rename("% cases deleting a class")
cost = (w[w["treatment"].str.startswith("remove")]
        .groupby(["treatment", "wiped_class"])["gain"].mean().unstack("wiped_class").round(2))
cost.columns = [f"gain when wiped={c}" for c in cost.columns]
fb = (df[df["variant"].str.contains("tuned")].groupby("family")["tuning_fallback"].mean() * 100).round(1)
md += ["## C. Deleting whole classes (now evaluated, not skipped)\n",
       C1.to_frame().to_markdown(), "",
       "Mean gain (default models) when the removal did / did not delete a class:\n",
       cost.to_markdown(), "",
       "Tuned variants that fell back to defaults (smallest class too small for inner CV), %:\n",
       fb.to_frame("fallback %").to_markdown(), ""]

(out / f"summary_v4{SUF}.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print(f"\nSaved: {out / f'summary_v4{SUF}.md'}  (metric: {METRIC})")
