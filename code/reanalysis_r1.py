"""
Revision reanalysis for "Tune Before You Clean" (no new experiments)
====================================================================
Answers the pre-submission review using the result files already in the repository
(results/<folder>/*.csv.gz written by export_results.py).

Usage (from code/):
    python reanalysis_r1.py ../results            (Jupyter: %run reanalysis_r1.py ../results)
Writes ../results/summary_r1.md and prints it.

A  Stage 3 (results_v3_full, 40 datasets x 5 seeds), clean data, remove top 10%:
   gain vs no treatment AND vs random removal of the same size, with 95% bootstrap CIs.
B  Model x treatment interaction: does the gain from removal differ between model families?
   Friedman test over the four families on per-dataset gains (clean data; and matched removal
   averaged over the corrupted conditions), plus LogReg-vs-family paired differences.
C  Equivalence (TOST, margin +-0.5 points) for "removal does not help trees/KNN under corruption".
D  Stage 4 (results_v4b): recovery of the cleaning gain by a quantile-transformed tuned LogReg,
   reported as a difference of mean gains with CI (instead of a ratio), under BA, BAP and AUC.
E  Stage 5 (results_v5b): asymmetric 20% label noise - oracle gain, right-sized Data-OOB gain and
   the gap, under BA, BAP and AUC, separately for binary and multiclass tasks, plus the realised
   flip share.
All of this is post hoc (written after the review) and is reported as such.
Datasets weigh equally: seeds are averaged within a dataset first.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

root = Path(sys.argv[1] if len(sys.argv) > 1 else "../results")
RNG = np.random.default_rng(2026)
B = 5000
FAMS = ["LogReg", "KNN", "RandomForest", "HistGB"]
MET = {"BA": "balanced_accuracy", "BAP": "ba_prior", "AUC": "auc_macro"}
md = ["# Revision reanalysis (post hoc, after the pre-submission review)\n",
      "Gains in points (x100). 95% CI = percentile bootstrap over datasets (5000 resamples). "
      "p = two-sided Wilcoxon signed-rank over datasets unless stated.\n"]


def load(folder, table):
    f = root / folder / f"{table}.csv.gz"
    if not f.exists():
        sys.exit(f"Missing {f}")
    return pd.read_csv(f)


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    bs = RNG.choice(x, (B, len(x)), replace=True).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


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


def fmt(m, lo, hi):
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def tost(x, margin=0.5):
    """equivalence within +-margin: both one-sided Wilcoxon tests must reject; returns max p"""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    p1 = wil(x + margin, "greater")
    p2 = wil(x - margin, "less")
    return max(p1, p2)


# ===================================================================== Stage 3
d3 = load("results_v3_full", "downstream")
d3["score"] = 100 * d3["balanced_accuracy"]
d3 = d3.rename(columns={"model": "family"})
key = ["dataset", "seed", "condition", "family"]
none3 = d3[d3["treatment"] == "none"].set_index(key)["score"].rename("none")


def removal(method, frac):
    s = d3[(d3["treatment"] == "remove") & (d3["method"] == method) & np.isclose(d3["frac"], frac)]
    return s.set_index(key)["score"]


# ---- A
md += ["## A. Stage 3, clean data, remove top 10%: against no treatment and against random removal\n",
       "Random removal of the same number of rows separates the cost of *which* rows are removed from the "
       "cost of having fewer training rows.\n"]
rand10 = removal("random", 0.10).rename("random")
rows, pv = [], []
for meth in ["IsolationForest", "DataOOB", "KNNShapley", "SelfConfidence"]:
    r = removal(meth, 0.10).rename("rem")
    J = pd.concat([r, none3, rand10], axis=1, join="inner").reset_index()
    J = J[J["condition"] == "clean"]
    for fam in FAMS:
        g = J[J["family"] == fam].groupby("dataset")[["rem", "none", "random"]].mean()
        a, b, c = g["rem"] - g["none"], g["rem"] - g["random"], g["random"] - g["none"]
        rows.append(dict(scorer=meth, family=fam, n=len(g), vs_none=fmt(*ci(a)), p_vs_none=wil(a),
                         vs_random=fmt(*ci(b)), p_vs_random=wil(b), random_vs_none=fmt(*ci(c))))
A = pd.DataFrame(rows)
A["pHolm_vs_none"] = holm(A["p_vs_none"])
A["pHolm_vs_random"] = holm(A["p_vs_random"])
md += [A.round(4).to_markdown(index=False), "",
       "Holm over the 16 rows of each column. Rows where a removal would have deleted a class were skipped "
       "in Stage 3, so pairs are inner-joined.\n"]

# ---- B
md += ["## B. Does the effect of removal depend on the model family? (interaction)\n",
       "Friedman test over the four families on per-dataset gains (H0: the gain is the same for every family). "
       "Paired differences: gain(LogReg) - gain(family).\n"]


def matched_gain(cond):
    k = cond.split("@")[0]
    meth = "IsolationForest" if k.startswith("feature") else ("DataOOB" if k.startswith("label") else None)
    return meth


rowsB = []
for label, conds, meth_fn in [("clean, Isolation Forest 10%", ["clean"], lambda c: "IsolationForest"),
                              ("clean, Data-OOB 10%", ["clean"], lambda c: "DataOOB"),
                              ("corrupted, matched scorer 10%, mean over conditions",
                               sorted(c for c in d3["condition"].unique() if c != "clean" and not c.startswith("mixed")),
                               matched_gain)]:
    parts = []
    for cond in conds:
        r = removal(meth_fn(cond), 0.10).rename("rem")
        J = pd.concat([r, none3], axis=1, join="inner").reset_index()
        J = J[J["condition"] == cond]
        J["gain"] = J["rem"] - J["none"]
        parts.append(J)
    J = pd.concat(parts)
    G = J.groupby(["dataset", "family"])["gain"].mean().unstack()[FAMS].dropna()
    stat, p = friedmanchisquare(*[G[f] for f in FAMS])
    r = dict(comparison=label, n_datasets=len(G), friedman_chi2=round(stat, 2), friedman_p=p)
    for f in FAMS[1:]:
        dlt = G["LogReg"] - G[f]
        r[f"LogReg-{f}"] = fmt(*ci(dlt))
        r[f"p(LogReg-{f})"] = wil(dlt)
    rowsB.append(r)
md += [pd.DataFrame(rowsB).round(4).to_markdown(index=False), ""]

# ---- C
md += ["## C. Equivalence: is matched removal under corruption within +-0.5 points of no treatment?\n",
       "TOST with two one-sided Wilcoxon tests (Stage 3, matched scorer at 10%, per condition). "
       "p_TOST < 0.05 means the gain is shown to lie within +-0.5 points; a large p_TOST means equivalence is "
       "NOT shown (the effect may be larger, or the data too noisy).\n"]
rowsC = []
for cond in sorted(c for c in d3["condition"].unique() if c != "clean" and not c.startswith("mixed")):
    r = removal(matched_gain(cond), 0.10).rename("rem")
    J = pd.concat([r, none3], axis=1, join="inner").reset_index()
    J = J[J["condition"] == cond]
    for fam in FAMS:
        g = J[J["family"] == fam].groupby("dataset")[["rem", "none"]].mean()
        x = g["rem"] - g["none"]
        rowsC.append(dict(condition=cond, family=fam, gain=fmt(*ci(x)), p_TOST=tost(x)))
md += [pd.DataFrame(rowsC).round(4).to_markdown(index=False), ""]

# ===================================================================== Stage 4
v4 = load("results_v4b", "rows")
md += ["## D. Stage 4: how much of the cleaning gain does a tuned, quantile-transformed LogReg recover?\n",
       "Per dirty condition: cleaning gain = default LogReg + matched removal - default LogReg (none); "
       "robust gain = quantile+tuned LogReg (none) - default LogReg (none). Reported: both gains and their "
       "difference (robust - cleaning) with CI, pooled over the corrupted conditions except mixed (whose matched scorer is ambiguous) (dataset x condition "
       "means, then averaged per dataset). A ratio of mean gains is shown only for continuity.\n"]
MATCH4 = lambda c: "remove:IF" if c.startswith("feature") else ("remove:OOB" if c.startswith("label") else "remove:IF")
rowsD = []
for lab, col in MET.items():
    if col not in v4.columns:
        continue
    L = v4[v4["family"] == "LogReg"].copy()
    L["s"] = 100 * L[col]
    k = ["dataset", "seed", "condition"]
    base = L[(L["variant"] == "default") & (L["treatment"] == "none")].set_index(k)["s"]
    rob = L[(L["variant"] == "quantile+tuned") & (L["treatment"] == "none")].set_index(k)["s"]
    dirty = [c for c in L["condition"].unique() if c != "clean" and not c.startswith("mixed")]
    L = L[L["condition"].isin(dirty)]
    base, rob = base[base.index.get_level_values("condition").isin(dirty)], rob[rob.index.get_level_values("condition").isin(dirty)]
    cl = pd.concat([L[(L["variant"] == "default") & (L["condition"] == c) & (L["treatment"] == MATCH4(c))]
                    for c in dirty]).set_index(k)["s"]
    J = pd.concat([base.rename("b"), rob.rename("r"), cl.rename("c")], axis=1, join="inner").reset_index()
    J = J.groupby("dataset")[["b", "r", "c"]].mean()
    gc, gr = J["c"] - J["b"], J["r"] - J["b"]
    rowsD.append(dict(metric=lab, n=len(J), cleaning_gain=fmt(*ci(gc)), robust_gain=fmt(*ci(gr)),
                      robust_minus_cleaning=fmt(*ci(gr - gc)), p=wil(gr - gc),
                      ratio_of_means=round(gr.mean() / gc.mean(), 2) if gc.mean() else np.nan))
md += [pd.DataFrame(rowsD).round(4).to_markdown(index=False), ""]

# ===================================================================== Stage 5
v5 = load("results_v5b", "rows")
md += ["## E. Stage 5: asymmetric 20% label noise - oracle, right-sized Data-OOB and the gap\n",
       "Gain over no treatment (default models). Gap = oracle - right-sized Data-OOB. Realised flip share = "
       "rows removed by the oracle / training rows.\n"]
A5 = v5[(v5["condition"] == "label_asym@0.2") & (v5["variant"] == "default")].copy()
A5["task"] = np.where(A5["n_classes"] == 2, "binary", "multiclass")
k = ["dataset", "seed", "family"]
flip = (A5[A5["option"] == "oracle"].assign(share=lambda d: d["removed"] / d["n_train"])
        .groupby(["task", "dataset"])["share"].mean())
rowsE = []
for lab, col in MET.items():
    A5["s"] = 100 * A5[col]
    P = A5.pivot_table(index=k + ["task"], columns="option", values="s").reset_index()
    for task in ["binary", "multiclass", "all"]:
        S = P if task == "all" else P[P["task"] == task]
        for fam in FAMS:
            g = S[S["family"] == fam].groupby("dataset")[["none", "oracle", "rs:OOB", "random"]].mean()
            o, d = g["oracle"] - g["none"], g["rs:OOB"] - g["none"]
            rowsE.append(dict(metric=lab, task=task, family=fam, n=len(g), oracle=fmt(*ci(o)),
                              rs_DataOOB=fmt(*ci(d)), gap=fmt(*ci(o - d)),
                              random=fmt(*ci(g["random"] - g["none"]))))
md += [pd.DataFrame(rowsE).to_markdown(index=False), "",
       "Realised flip share at nominal 20%: " + "; ".join(
           f"{t}: mean {flip[t].mean():.3f} (range {flip[t].min():.3f}-{flip[t].max():.3f}, {flip[t].size} datasets)"
           for t in ["binary", "multiclass"] if t in flip.index.get_level_values(0)), ""]

text = "\n".join(md)
(root / "summary_r1.md").write_text(text, encoding="utf-8")
print(text)
print(f"\nSaved: {root / 'summary_r1.md'}")
