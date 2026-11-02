"""
Analysis for experiment_v3.py
=============================
Usage:  python analyze_v3.py results_v3        (Jupyter: %run analyze_v3.py results_v3)

Sections of <out>/summary_v3.md
  A  RQ1   overlap of flagged sets on clean data
  B  RQ4a  detection of injected corruption, per corruption type
  C  RQ3   mechanism on clean data (minority enrichment, hardness)
  D  RQ6   accuracy of the label-free diagnostics
  E  PRIMARY pre-specified tests (matched family @10% vs none), Holm over these only
  F  RQ7   HEADLINE: diagnose-then-treat (thresholds learned leave-one-dataset-out)
           vs. never / always strategies, by corruption group
  G  family trends (exploratory)
Accuracy = balanced accuracy in points (x100). Datasets weigh equally.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

CLASSICAL = ["IQR", "MAD", "RobustMahalanobis", "LOF", "IsolationForest"]
TARGET_AWARE = ["DataOOB", "KNNShapley", "SelfConfidence"]
METHODS = CLASSICAL + TARGET_AWARE
HARM = -1.0                                     # a case "harmed" if gain < -1 point

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results_v3")
parts = sorted((out / "parts").glob("*.pkl"))
if not parts:
    sys.exit(f"No finished jobs in {out / 'parts'}")
res = [pd.read_pickle(f) for f in parts]
get = lambda k: pd.DataFrame([r for x in res for r in x[k]])
ov, det, mech, diag, dn = get("overlap"), get("detect"), get("mech"), get("diag"), get("downstream")
dn["score"] = 100 * dn["balanced_accuracy"]
dn["option"] = np.select(
    [dn["treatment"] == "none", dn["treatment"] == "remove", dn["treatment"] == "dtt"],
    ["none", "remove:" + dn["method"] + "@" + dn["frac"].map(lambda f: f"{f:g}"), dn["method"]],
    dn["treatment"] + ":" + dn["method"])


def group_of(cond):
    if cond == "clean":
        return "clean"
    k = cond.split("@")[0]
    return "mixed" if k == "mixed" else ("label" if k.startswith("label") else "feature")


def cond_key(c):
    return (c != "clean", c.split("@")[0], float(c.split("@")[1]) if "@" in c else 0)


COND_ORDER = sorted(dn["condition"].unique(), key=cond_key)
fam = {m: ("classical" if m in CLASSICAL else "target-aware") for m in METHODS}
md = ["# Experiment v3: summary\n",
      f"Datasets: {dn['dataset'].nunique()} | seeds: {dn['seed'].nunique()} | "
      f"conditions: {', '.join(COND_ORDER)}\n",
      "Accuracy = balanced accuracy in points (x100); datasets weigh equally.\n"]


def ds_mean(df, keys, cols):
    return df.groupby(["dataset"] + keys)[cols].mean().reset_index()


def holm(p):
    p = np.asarray(p, float)
    adj = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    order = ok[np.argsort(p[ok])]
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * p[i]))
        adj[i] = run
    return adj


def wil(x, alternative="two-sided"):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return wilcoxon(x, alternative=alternative).pvalue if len(x) >= 6 and np.any(x != 0) else np.nan


# ---------------------------------------------------------------- A
o = ov[ov["condition"] == "clean"].copy()
o["pair_type"] = ["within-classical" if fam[a] == fam[b] == "classical"
                  else "within-target-aware" if fam[a] == fam[b] else "cross-family"
                  for a, b in zip(o["a"], o["b"])]
A = (ds_mean(o, ["pair_type"], ["jaccard", "spearman"])
     .groupby("pair_type")[["jaccard", "spearman"]].mean().round(3))
md += ["## A. RQ1 - Overlap of top-5% flagged points (clean)\n",
       f"Random expectation of Jaccard: {0.05 / 1.95:.3f}\n", A.to_markdown(), ""]

# ---------------------------------------------------------------- B
if len(det):
    det["col"] = det["condition"] + " [" + det["target"] + "]"
    B = (ds_mean(det, ["col", "method"], ["auroc"]).groupby(["method", "col"])["auroc"].mean()
         .unstack("col").reindex(METHODS).round(3))
    B = B[sorted(B.columns, key=lambda c: cond_key(c.split(" ")[0]) + (c,))]
    md += ["## B. RQ4a - AUROC for detecting injected corrupted rows\n",
           "Column = condition [which corrupted rows are the target]. 0.5 = chance.\n",
           B.to_markdown(), ""]
    B.to_csv(out / "v3_detection.csv")

# ---------------------------------------------------------------- C
c = mech[mech["condition"] == "clean"].copy()
c["hardness_ratio"] = c["kdn_flagged"] / c["kdn_all"].replace(0, np.nan)
C = (ds_mean(c, ["method"], ["minority_enrichment", "hardness_ratio"])
     .groupby("method")[["minority_enrichment", "hardness_ratio"]].agg(["mean", "median"])
     .round(2).reindex(METHODS))
md += ["## C. RQ3 - Who gets flagged on clean data (top-5%)\n", C.to_markdown(), ""]

# ---------------------------------------------------------------- D  diagnostics
D_COLS = ["est_label_noise", "true_label_rate", "extreme_rate", "true_feature_rate"]
D = (ds_mean(diag, ["condition"], D_COLS).groupby("condition")[D_COLS]
     .agg(["mean", "std"]).round(3))
D = D.reindex(COND_ORDER)
rl = spearmanr(diag["est_label_noise"], diag["true_label_rate"]).statistic
rf = spearmanr(diag["extreme_rate"], diag["true_feature_rate"]).statistic
md += ["## D. RQ6 - Can corruption be diagnosed without knowing it?\n",
       f"Spearman(est_label_noise, true label rate) = **{rl:.2f}**; "
       f"Spearman(extreme_rate, true feature rate) = **{rf:.2f}** (over all dataset x seed x condition cases).\n",
       "On clean data both diagnostics should be low; their clean level is the 'false alarm' baseline "
       "(inherent class overlap inflates est_label_noise).\n",
       D.to_markdown(), ""]
D.to_csv(out / "v3_diagnostics.csv")

# ---------------------------------------------------------------- E  primary tests
per = ds_mean(dn, ["condition", "model", "option"], ["score"])
piv = per.pivot_table(index=["dataset", "condition", "model"], columns="option", values="score")
prim_opts = {"label": ["remove:DataOOB@0.1"], "feature": ["remove:IsolationForest@0.1"],
             "mixed": ["remove:DataOOB@0.1", "remove:IsolationForest@0.1"],
             "clean": ["remove:DataOOB@0.1", "remove:IsolationForest@0.1"]}
rows = []
for (cond, model), g in piv.groupby(level=["condition", "model"]):
    for opt in prim_opts[group_of(cond)]:
        if opt not in g:
            continue
        d = (g[opt] - g["none"]).dropna()
        rows.append(dict(condition=cond, model=model, option=opt, n=len(d), mean_gain=d.mean(),
                         wins=int((d > 0).sum()), losses=int((d < 0).sum()), p=wil(d)))
E = pd.DataFrame(rows)
E["p_holm"] = holm(E["p"].values)
E["condition"] = pd.Categorical(E["condition"], COND_ORDER, ordered=True)
E = E.sort_values(["condition", "option", "model"]).round(4)
md += ["## E. PRIMARY pre-specified tests (matched family, remove 10%, vs no treatment)\n",
       f"Holm correction over these {len(E)} tests only (fixed in PROTOCOL_v3.md before the run).\n",
       E.to_markdown(index=False), ""]
E.to_csv(out / "v3_primary_tests.csv", index=False)

# ---------------------------------------------------------------- F  strategies
case = (dn.pivot_table(index=["dataset", "seed", "condition", "model"], columns="option",
                       values="score").reset_index())
case = case.merge(diag[["dataset", "seed", "condition", "est_label_noise", "extreme_rate"]],
                  on=["dataset", "seed", "condition"])
case["group"] = case["condition"].map(group_of)
gain = lambda opt: (case[opt] - case["none"]).fillna(0.0).values if opt in case else np.zeros(len(case))
G = {"cls": gain("dtt_classical"), "tgt": gain("dtt_target"), "both": gain("dtt_both")}
est, ext = case["est_label_noise"].values, case["extreme_rate"].values


def rule_gain(tl, tf, idx):
    L, F = est[idx] > tl, ext[idx] > tf
    return np.where(L & F, G["both"][idx], np.where(L, G["tgt"][idx], np.where(F, G["cls"][idx], 0.0)))


grid_l = np.r_[-np.inf, np.unique(np.quantile(est, np.linspace(0, 1, 41))), np.inf]
grid_f = np.r_[-np.inf, np.unique(np.quantile(ext, np.linspace(0, 1, 41))), np.inf]
dtt = np.zeros(len(case))
chosen = []
datasets = case["dataset"].unique()
for d in datasets:                                    # leave-one-dataset-out tuning
    tr = np.where(case["dataset"].values != d)[0]
    te = np.where(case["dataset"].values == d)[0]
    best = max(((rule_gain(tl, tf, tr).mean(), tl, tf) for tl in grid_l for tf in grid_f))
    dtt[te] = rule_gain(best[1], best[2], te)
    chosen.append(dict(held_out=d, theta_label=best[1], theta_feature=best[2]))
chosen = pd.DataFrame(chosen)

truth = {"clean": np.zeros(len(case)), "label": G["tgt"], "feature": G["cls"], "mixed": G["both"]}
type_oracle = np.select([case["group"] == k for k in truth], [truth[k] for k in truth])
strategies = {
    "never treat": np.zeros(len(case)),
    "always classical (IF 5%)": gain("remove:IsolationForest@0.05"),
    "always target-aware (OOB 5%)": gain("remove:DataOOB@0.05"),
    "always both (DTT sizes)": G["both"],
    "always reweight (OOB)": gain("reweight:DataOOB"),
    "DIAGNOSE-THEN-TREAT (LODO)": dtt,
    "type oracle (knows type)": type_oracle,
    "action oracle (upper bound)": np.maximum.reduce([np.zeros(len(case)), G["cls"], G["tgt"], G["both"]]),
}
S = case[["dataset", "group"]].copy()
for k, v in strategies.items():
    S[k] = v
rows = []
best_always = None
for grp in ["clean", "label", "feature", "mixed", "ALL"]:
    sub = S if grp == "ALL" else S[S["group"] == grp]
    per_ds = sub.groupby("dataset")[list(strategies)].mean()
    for k in strategies:
        rows.append(dict(group=grp, strategy=k, mean_gain=per_ds[k].mean(),
                         harm_rate=100 * (sub[k] < HARM).mean(),
                         p_vs_never=wil(per_ds[k]) if k != "never treat" else np.nan))
F = pd.DataFrame(rows).round(3)
allrows = F[F["group"] == "ALL"].set_index("strategy")
always = [k for k in strategies if k.startswith("always")]
best_always = allrows.loc[always, "mean_gain"].idxmax()
per_all = S.groupby("dataset")[list(strategies)].mean()
p_dtt_vs_best = wil(per_all["DIAGNOSE-THEN-TREAT (LODO)"] - per_all[best_always])
Fp = F.pivot(index="strategy", columns="group", values="mean_gain")[["clean", "label", "feature", "mixed", "ALL"]]
Fh = F.pivot(index="strategy", columns="group", values="harm_rate")[["clean", "label", "feature", "mixed", "ALL"]]
Fp, Fh = Fp.reindex(list(strategies)), Fh.reindex(list(strategies))
md += ["## F. HEADLINE - Diagnose-then-treat vs. fixed strategies\n",
       "Mean gain vs. no treatment (points), averaged per dataset. DTT thresholds are learned on the "
       "OTHER datasets (leave-one-dataset-out), so the DTT row is an honest estimate. "
       "The two oracle rows are upper bounds, not methods.\n",
       Fp.round(3).to_markdown(), "",
       f"Harm rate (% of dataset x seed x condition x model cases with gain < {HARM:g} point):\n",
       Fh.round(1).to_markdown(), "",
       f"Wilcoxon over datasets, ALL conditions: DTT vs never p = {allrows.loc['DIAGNOSE-THEN-TREAT (LODO)', 'p_vs_never']:.4g}; "
       f"DTT vs best fixed strategy ('{best_always}') p = {p_dtt_vs_best:.4g}.\n",
       "Thresholds chosen per fold (median): theta_label = "
       f"{chosen['theta_label'].replace([np.inf, -np.inf], np.nan).median():.3f}, "
       f"theta_feature = {chosen['theta_feature'].replace([np.inf, -np.inf], np.nan).median():.3f}.\n"]
F.to_csv(out / "v3_strategies.csv", index=False)
chosen.to_csv(out / "v3_dtt_thresholds.csv", index=False)

# ---------------------------------------------------------------- G  family trends
def family_of(opt):
    if opt.startswith("winsor"):
        return "winsorize"
    if opt.startswith("dtt"):
        return opt
    m = opt.split(":")[1].split("@")[0]
    return "random-removal" if m == "random" else f"{opt.split(':')[0]}-{fam[m]}"


rows = []
for (cond, model), g in piv.groupby(level=["condition", "model"]):
    for opt in [x for x in g.columns if x != "none"]:
        rows.append(dict(condition=cond, option=opt, gain=(g[opt] - g["none"]).mean()))
T = pd.DataFrame(rows).groupby(["condition", "option"])["gain"].mean().reset_index()
T["family"] = T["option"].map(family_of)
T = (T.sort_values("gain", ascending=False).groupby(["condition", "family"]).head(1)
       .pivot(index="family", columns="condition", values="gain")[COND_ORDER].round(2))
md += ["## G. Best option of each family per condition (exploratory, post hoc)\n", T.to_markdown(), ""]

log = out / "failures.log"
if log.exists() and log.stat().st_size:
    lines = log.read_text(encoding="utf-8").splitlines()
    md += [f"## Notes\nfailures.log: {len(lines)} lines "
           f"({sum('wiped out a class' in l for l in lines)} removals skipped: they deleted a whole class).\n"]
(out / "summary_v3.md").write_text("\n".join(md), encoding="utf-8")

# ---------------------------------------------------------------- figures
try:
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    show = ["never treat", "always classical (IF 5%)", "always target-aware (OOB 5%)",
            "always reweight (OOB)", "DIAGNOSE-THEN-TREAT (LODO)", "type oracle (knows type)"]
    Fp.loc[show].T.plot.bar(ax=ax, width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean gain vs. no treatment (BA points)")
    ax.set_title("Diagnose-then-treat vs. fixed cleaning strategies")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(out / "fig_v3_strategies.png", dpi=200)

    fig = Figure(figsize=(10, 4))
    a1, a2 = fig.subplots(1, 2)
    for ax_, col, title in [(a1, "est_label_noise", "Estimated label noise"),
                            (a2, "extreme_rate", "Extreme-row rate (|robust z| > 6)")]:
        data = [diag.loc[diag["condition"] == c, col].values for c in COND_ORDER]
        ax_.boxplot(data, showfliers=False)
        ax_.set_xticks(range(1, len(COND_ORDER) + 1), COND_ORDER, rotation=60, ha="right", fontsize=7)
        ax_.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_v3_diagnostics.png", dpi=200)
except Exception as e:
    print(f"figures skipped: {e}")

print((out / "summary_v3.md").read_text(encoding="utf-8")[:7000])
print(f"\nFull summary: {out / 'summary_v3.md'}")
