"""
Analysis for experiment_v6.py (PROTOCOL_v6.md)
==============================================
Usage:  python analyze_v6.py results_v6        (Jupyter: %run analyze_v6.py results_v6)
A  Detector precision / recall / share removed (asymmetric noise split binary vs multiclass)
B  Gains vs. no treatment per family, condition and option (balanced accuracy; plain accuracy)
C  Pre-specified hypotheses H13a-c, Holm over the 8 tests
Gains in balanced-accuracy points; seeds averaged within dataset, datasets weighted equally.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results_v6")
parts = sorted((out / "parts").glob("*.pkl"))
if not parts:
    sys.exit(f"No finished jobs in {out / 'parts'}")
res = [pd.read_pickle(f) for f in parts]
df = pd.DataFrame([r for x in res for r in x["rows"]])
pr = pd.DataFrame([r for x in res for r in x["precision"]])
key = ["dataset", "seed", "condition", "family"]
FAMS = ["LogReg", "KNN", "RandomForest", "HistGB"]
OPTS = ["oracle", "ef:majority", "ef:consensus", "rs:EF", "rs:OOB", "fix:EF@0.1", "fix:OOB@0.1"]
CONDS = sorted(df["condition"].unique(), key=lambda c: (c != "clean", c))


def with_gain(metric):
    d = df.copy()
    d["score"] = 100 * d[metric]
    d = d.join(d[d["option"] == "none"].set_index(key)["score"].rename("base"), on=key)
    d["gain"] = d["score"] - d["base"]
    return d


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


G = with_gain("balanced_accuracy")


def g(fam, cond, opt, d=G):
    s = d[(d["family"] == fam) & (d["condition"] == cond) & (d["option"] == opt)]
    return s.groupby("dataset")["gain"].mean()


md = ["# Experiment v6: classical ensemble filter vs. Data-OOB\n",
      f"Datasets: {df['dataset'].nunique()} | seeds: {df['seed'].nunique()}\n"]

# ------------------------------------------------------------------ A
pr["type"] = np.where(pr["n_classes"] == 2, "binary", "multiclass")
A = (pr.groupby(["dataset", "condition", "option"])[["share_removed", "precision", "recall"]].mean()
       .groupby(["condition", "option"]).mean().round(3))
md += ["## A. Detectors: share of rows removed, precision, recall\n", A.to_markdown(), ""]
a = pr[pr["condition"] == "label_asym@0.2"]
if len(a):
    A2 = (a.groupby(["type", "dataset", "option"])[["precision", "recall"]].mean()
            .groupby(["type", "option"]).mean().round(3))
    md += ["### Asymmetric noise, binary vs multiclass\n", A2.to_markdown(), ""]
A.to_csv(out / "v6_precision.csv")

# ------------------------------------------------------------------ B
for metric in ["balanced_accuracy", "accuracy"]:
    d = with_gain(metric)
    md += [f"## B. Gains vs. no treatment ({metric}, points)\n"]
    for fam in FAMS:
        s = d[(d["family"] == fam) & (d["option"] != "none")]
        T = (s.groupby(["option", "condition", "dataset"])["gain"].mean()
               .groupby(["option", "condition"]).mean().unstack("condition"))
        T = T.reindex([o for o in OPTS if o in T.index])[[c for c in CONDS if c in T.columns]].round(2)
        md += [f"### {fam}\n", T.to_markdown(), ""]
        T.to_csv(out / f"v6_gains_{fam}_{metric}.csv")

# ------------------------------------------------------------------ C
tests = []
for fam in ["RandomForest", "HistGB"]:
    x = g(fam, "clean", "ef:majority")
    tests.append(dict(hypothesis="H13a EF (native) harms trees on clean", family=fam, condition="clean",
                      n=x.notna().sum(), mean=x.mean(), p=wil(x, "less")))
gaps = {}
for fam in FAMS:
    x = (g(fam, "label_asym@0.2", "oracle") - g(fam, "label_asym@0.2", "rs:EF")).dropna()
    gaps[fam] = x.mean()
    tests.append(dict(hypothesis="H13b asym gap oracle - rs:EF > 0", family=fam, condition="label_asym@0.2",
                      n=len(x), mean=x.mean(), p=wil(x, "greater")))
for c in ["label_sym@0.2", "label_asym@0.2"]:
    x = pd.concat([g(f, c, "rs:EF") - g(f, c, "rs:OOB") for f in FAMS], axis=1).mean(axis=1).dropna()
    tests.append(dict(hypothesis="H13c rs:EF vs rs:OOB (two-sided)", family="mean of 4", condition=c,
                      n=len(x), mean=x.mean(), p=wil(x, "two-sided")))
T = pd.DataFrame(tests)
T["p_holm"] = holm(T["p"].values)
T = T.round(4)
ok = lambda h: T[T["hypothesis"].str.startswith(h)]
a_ok = (ok("H13a")["p_holm"] < 0.05).all()
b_ok = (ok("H13b")["p_holm"] < 0.05).all() and all(v > 1 for v in gaps.values())
md += ["## C. Pre-specified hypotheses (Wilcoxon over datasets, Holm over this table)\n",
       T.to_markdown(index=False), "",
       f"- H13a (EF harms RF and HistGB on clean data): **{'SUPPORTED' if a_ok else 'NOT supported'}**",
       f"- H13b (asymmetric-noise gap > 1 point and significant for every family with EF): "
       f"**{'SUPPORTED' if b_ok else 'NOT supported'}**",
       "- H13c (EF vs Data-OOB at the right size): " +
       "; ".join(f"{r.condition}: diff = {r.mean:+.2f}, p_holm = {r.p_holm:.3g}"
                 for r in ok("H13c").itertuples()) +
       " (non-significance is not evidence of equivalence).\n"]
T.to_csv(out / "v6_tests.csv", index=False)

wip = df[df["option"] != "none"].groupby("option")["wiped_class"].mean().mul(100).round(2)
md += ["## D. Removals that deleted a whole class (%)\n", wip.to_frame("% wiped").to_markdown(), ""]

(out / "summary_v6.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print(f"\nSaved: {out / 'summary_v6.md'}")
