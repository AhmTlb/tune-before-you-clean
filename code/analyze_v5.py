"""
Analysis for experiment_v5.py
=============================
Usage:  python analyze_v5.py results_v5        (Jupyter: %run analyze_v5.py results_v5)

A  Detector precision / recall for right-sized (rs) and fixed-10% (fix) removal.
B  Gains vs. no treatment per family, condition and option (default variant;
   tuned variant in a second table when it was run).
C  Pre-registered hypotheses (PROTOCOL_v5.md):
   H12a  even ORACLE removal of feature-corrupted rows gains < 0.5 points for RF and HistGB
   H12b  for LogReg the detection gap (oracle - right-sized detector) is < 0.5 points under
         feature corruption and > 1 point under asymmetric label noise
   H12c  right-sized removal >= fixed-10% removal (matched detector; mean over the 4 families)
   H12d  oracle removal helps every family under 20% symmetric label noise
   Holm correction over the Wilcoxon tests of H12c and H12d.
Gains in balanced-accuracy points; seeds averaged within dataset, datasets weighted equally.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results_v5")
parts = sorted((out / "parts").glob("*.pkl"))
if not parts:
    sys.exit(f"No finished jobs in {out / 'parts'}")
res = [pd.read_pickle(f) for f in parts]
df = pd.DataFrame([r for x in res for r in x["rows"]])
pr = pd.DataFrame([r for x in res for r in x["precision"]])
METRIC = sys.argv[2] if len(sys.argv) > 2 else "balanced_accuracy"   # ba_prior | auc_macro (BA-threshold check)
SUF = "" if METRIC == "balanced_accuracy" else f"_{METRIC}"
df["score"] = 100 * df[METRIC]
key = ["dataset", "seed", "condition", "family", "variant"]
base_ = df[df["option"] == "none"].set_index(key)["score"].rename("base")
df = df.join(base_, on=key)
df["gain"] = df["score"] - df["base"]
CONDS = sorted(df["condition"].unique(), key=lambda c: (c != "clean", c))
OPTS = ["oracle", "oracle_label", "oracle_feature", "rs:IF", "rs:OOB", "rs:both",
        "fix:IF@0.1", "fix:OOB@0.1", "random"]
FAMS = ["LogReg", "KNN", "RandomForest", "HistGB"]


def ds_gain(sub):
    return sub.groupby("dataset")["gain"].mean()


def wil(x, alternative):
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


def g(fam, cond, opt, variant="default"):
    s = df[(df["family"] == fam) & (df["condition"] == cond) & (df["option"] == opt) & (df["variant"] == variant)]
    return ds_gain(s)


md = ["# Experiment v5: oracle vs. right-sized vs. fixed removal\n",
      f"Datasets: {df['dataset'].nunique()} | seeds: {df['seed'].nunique()} | "
      f"variants: {', '.join(sorted(df['variant'].unique()))}\n"]

# ------------------------------------------------------------------ A
if len(pr):
    A = (pr.groupby(["dataset", "condition", "option"])[["precision", "precision_own_type", "recall"]].mean()
           .groupby(["condition", "option"]).mean().round(3))
    md += ["## A. Detector precision and recall\n",
           "precision = share of removed rows that were corrupted (any type); precision_own_type = "
           "share corrupted in the detector's own type; recall = share of corrupted rows removed.\n",
           A.to_markdown(), ""]
    A.to_csv(out / f"v5_precision{SUF}.csv")

# ------------------------------------------------------------------ B
md += ["## B. Gains vs. no treatment\n"]
for variant in sorted(df["variant"].unique()):
    for fam in FAMS:
        s = df[(df["family"] == fam) & (df["variant"] == variant) & (df["option"] != "none")]
        if s.empty:
            continue
        T = (s.groupby(["option", "condition", "dataset"])["gain"].mean()
               .groupby(["option", "condition"]).mean().unstack("condition"))
        T = T.reindex([o for o in OPTS if o in T.index])[[c for c in CONDS if c in T.columns]].round(2)
        md += [f"### {fam} ({variant})\n", T.to_markdown(), ""]
        T.to_csv(out / f"v5_gains_{fam}_{variant}{SUF}.csv")

# ------------------------------------------------------------------ C
feat = [c for c in CONDS if c.startswith("feature")]
lab = [c for c in CONDS if c.startswith("label")]
md += ["## C. Pre-registered hypotheses\n"]

rows = []
for fam in ["RandomForest", "HistGB"]:
    for c in feat:
        x = g(fam, c, "oracle")
        rows.append(dict(family=fam, condition=c, oracle_gain=x.mean(), p_gt0=wil(x, "greater")))
H12a = pd.DataFrame(rows).round(4)
ok_a = (H12a["oracle_gain"] < 0.5).all()
md += ["### H12a: even oracle removal of feature errors barely helps tree ensembles\n",
       H12a.to_markdown(index=False), "",
       f"Criterion: every oracle_gain < 0.5 -> **{'SUPPORTED' if ok_a else 'NOT supported'}**.\n"]

rows = []
for c in feat + lab:
    det = "rs:IF" if c.startswith("feature") else "rs:OOB"
    o, r = g("LogReg", c, "oracle"), g("LogReg", c, det)
    d = (o - r).dropna()
    rows.append(dict(condition=c, detector=det, oracle=o.mean(), right_sized=r.mean(), gap=d.mean()))
H12b = pd.DataFrame(rows).round(3)
gap_feat = H12b[H12b["condition"].str.startswith("feature")]["gap"]
gap_asym = H12b[H12b["condition"].str.startswith("label_asym")]["gap"]
ok_b = (gap_feat < 0.5).all() and (gap_asym > 1).all()
md += ["### H12b: where does logistic regression lose, detection or removal?\n",
       H12b.to_markdown(index=False), "",
       f"Criterion: gap < 0.5 for feature conditions and > 1 for label_asym -> "
       f"**{'SUPPORTED' if ok_b else 'NOT supported'}**.\n"]

tests = []
for c in feat + lab:
    det = "IF" if c.startswith("feature") else "OOB"
    d = pd.concat([(g(f, c, f"rs:{det}") - g(f, c, f"fix:{det}@0.1")) for f in FAMS], axis=1).mean(axis=1)
    tests.append(dict(hypothesis="H12c right-sized >= fixed 10%", condition=c,
                      comparison=f"rs:{det} - fix:{det}@0.1 (mean of 4 families)",
                      n=d.notna().sum(), mean_diff=d.mean(), p=wil(d, "greater")))
for f in FAMS:
    c = "label_sym@0.2"
    if c in CONDS:
        x = g(f, c, "oracle")
        tests.append(dict(hypothesis="H12d oracle helps under 20% label noise", condition=c,
                          comparison=f"{f}: oracle - none", n=x.notna().sum(), mean_diff=x.mean(),
                          p=wil(x, "greater")))
T = pd.DataFrame(tests)
T["p_holm"] = holm(T["p"].values)
T = T.round(4)
c_ok = (T[T["hypothesis"].str.startswith("H12c")]["p_holm"] < 0.05).sum()
c_n = T["hypothesis"].str.startswith("H12c").sum()
d_ok = (T[T["hypothesis"].str.startswith("H12d")]["p_holm"] < 0.05).sum()
d_n = T["hypothesis"].str.startswith("H12d").sum()
md += ["### H12c and H12d (one-sided Wilcoxon, Holm over this table)\n",
       T.to_markdown(index=False), "",
       f"H12c: right-sized beats fixed removal significantly in **{c_ok} of {c_n}** conditions.",
       f"H12d: oracle removal helps significantly for **{d_ok} of {d_n}** families.\n"]
T.to_csv(out / f"v5_tests{SUF}.csv", index=False)

wip = df[df["option"] != "none"].groupby("option")["wiped_class"].mean().mul(100).round(2)
md += ["## D. Removals that deleted a whole class (%)\n", wip.to_frame("% wiped").to_markdown(), ""]

(out / f"summary_v5{SUF}.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print(f"\nSaved: {out / f'summary_v5{SUF}.md'}  (metric: {METRIC})")
