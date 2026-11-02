"""
Per-model and confirmatory analysis for experiment_v3.py
========================================================
Usage
  python analyze_v3_confirm.py results_v3                         # per-model view only
  python analyze_v3_confirm.py results_v3_full --discovery-from results_v3
                                                                  # + confirmatory test
(Jupyter: %run analyze_v3_confirm.py results_v3_full --discovery-from results_v3)

Part 1 (exploratory): strategy gains and harm rates PER MODEL.
  Note: KNN has no sample weights, so "always reweight" is undefined (NaN) for KNN.

Part 2 (confirmatory, pre-registered in PROTOCOL_v3_confirm.md):
  Discovery datasets  = datasets found in --discovery-from (the 20 of the preliminary run).
  Confirmation datasets = all other datasets in the full run (never seen while designing).
  Using DISCOVERY datasets only:
    - learn the diagnose-then-treat (DTT) thresholds,
    - pick, for each model, the strategy with the highest mean gain
      -> "model-aware policy",
    - pick the single strategy with the highest mean gain over all models
      -> "best global strategy".
  Then on CONFIRMATION datasets: compare model-aware policy, best global strategy,
  DTT and never-treat (Wilcoxon over datasets, harm rate).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HARM = -1.0
ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", default="results_v3")
ap.add_argument("--discovery-from", default=None)
args = ap.parse_args()
out = Path(args.out)

parts = sorted((out / "parts").glob("*.pkl"))
if not parts:
    raise SystemExit(f"No finished jobs in {out / 'parts'}")
res = [pd.read_pickle(f) for f in parts]
dn = pd.DataFrame([r for x in res for r in x["downstream"]])
diag = pd.DataFrame([r for x in res for r in x["diag"]])
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


case = (dn.pivot_table(index=["dataset", "seed", "condition", "model"], columns="option",
                       values="score").reset_index())
case = case.merge(diag[["dataset", "seed", "condition", "est_label_noise", "extreme_rate"]],
                  on=["dataset", "seed", "condition"])
case["group"] = case["condition"].map(group_of)


def gain(opt, fill=True):
    if opt not in case:
        return np.full(len(case), 0.0 if fill else np.nan)
    g = (case[opt] - case["none"]).to_numpy(dtype=float, copy=True)
    return np.nan_to_num(g, nan=0.0) if fill else g


G = {"cls": gain("dtt_classical"), "tgt": gain("dtt_target"), "both": gain("dtt_both")}
EST, EXT = case["est_label_noise"].values, case["extreme_rate"].values


def rule_gain(tl, tf, idx):
    L, F = EST[idx] > tl, EXT[idx] > tf
    return np.where(L & F, G["both"][idx], np.where(L, G["tgt"][idx], np.where(F, G["cls"][idx], 0.0)))


def learn_thresholds(idx):
    gl = np.r_[-np.inf, np.unique(np.quantile(EST[idx], np.linspace(0, 1, 41))), np.inf]
    gf = np.r_[-np.inf, np.unique(np.quantile(EXT[idx], np.linspace(0, 1, 41))), np.inf]
    best = max(((rule_gain(tl, tf, idx).mean(), tl, tf) for tl in gl for tf in gf))
    return best[1], best[2]


def harm_rate(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]                    # undefined cases (e.g. reweight for KNN) excluded
    return 100 * np.mean(v < HARM) if len(v) else np.nan


def wil(x, alternative="two-sided"):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return wilcoxon(x, alternative=alternative).pvalue if len(x) >= 6 and np.any(x != 0) else np.nan


# strategies; reweight is undefined for KNN (no sample weights) -> NaN, not 0
rew = gain("reweight:DataOOB", fill=False)
rew[case["model"].values == "KNN"] = np.nan
STRAT = {
    "never treat": np.zeros(len(case)),
    "always classical (IF 5%)": gain("remove:IsolationForest@0.05"),
    "always target-aware (OOB 5%)": gain("remove:DataOOB@0.05"),
    "always both (DTT sizes)": G["both"],
    "always reweight (OOB)": rew,
}
md = ["# Experiment v3: per-model and confirmatory analysis\n",
      f"Datasets: {case['dataset'].nunique()} | seeds: {case['seed'].nunique()} | "
      f"models: {', '.join(sorted(case['model'].unique()))}\n",
      "Gains = balanced-accuracy points vs. no treatment; averaged per dataset, then across datasets. "
      f"Harm rate = % of cases with gain < {HARM:g} point.\n"]

# ---------------------------------------------------------------- Part 1: per model (LODO DTT)
dtt = np.zeros(len(case))
for d in case["dataset"].unique():
    tr = np.where(case["dataset"].values != d)[0]
    te = np.where(case["dataset"].values == d)[0]
    tl, tf = learn_thresholds(tr)
    dtt[te] = rule_gain(tl, tf, te)
S1 = dict(STRAT, **{"diagnose-then-treat (LODO)": dtt})

rows = []
for model in sorted(case["model"].unique()):
    m = case["model"].values == model
    for grp in ["clean", "label", "feature", "mixed", "ALL"]:
        g = m if grp == "ALL" else m & (case["group"].values == grp)
        for k, v in S1.items():
            vv = pd.Series(v[g], index=case.index[g])
            per_ds = vv.groupby(case.loc[g, "dataset"]).mean()
            rows.append(dict(model=model, group=grp, strategy=k,
                             mean_gain=per_ds.mean(),
                             harm=harm_rate(v[g])))
P = pd.DataFrame(rows)
md += ["## Part 1. Strategies per model (exploratory; DTT thresholds leave-one-dataset-out)\n"]
for model in sorted(case["model"].unique()):
    t = P[P["model"] == model].pivot(index="strategy", columns="group", values="mean_gain")
    h = P[(P["model"] == model) & (P["group"] == "ALL")].set_index("strategy")["harm"]
    t = t[["clean", "label", "feature", "mixed", "ALL"]].reindex(list(S1)).round(3)
    t["harm % (ALL)"] = h.reindex(list(S1)).round(1)
    md += [f"### {model}\n", t.to_markdown(), ""]
P.to_csv(out / "v3_per_model.csv", index=False)

# ---------------------------------------------------------------- Part 2: confirmatory
if args.discovery_from:
    disc_names = {pd.read_pickle(f)["downstream"][0]["dataset"]
                  for f in (Path(args.discovery_from) / "parts").glob("*.pkl")}
    is_disc = case["dataset"].isin(disc_names).values
    n_disc = case.loc[is_disc, "dataset"].nunique()
    n_conf = case.loc[~is_disc, "dataset"].nunique()
    md += ["## Part 2. CONFIRMATORY test on held-out datasets (pre-registered)\n",
           f"Discovery datasets (used to choose everything): **{n_disc}**. "
           f"Confirmation datasets (never used for any choice): **{n_conf}**.\n"]
    if n_disc == 0 or n_conf == 0:
        md += ["Not enough datasets on one side; confirmatory test skipped.\n"]
    else:
        idx_d, idx_c = np.where(is_disc)[0], np.where(~is_disc)[0]
        tl, tf = learn_thresholds(idx_d)
        dtt_c = np.full(len(case), np.nan)
        dtt_c[idx_c] = rule_gain(tl, tf, idx_c)
        S2 = dict(STRAT, **{"diagnose-then-treat": np.where(is_disc, np.nan, dtt_c)})
        S2_disc = dict(STRAT, **{"diagnose-then-treat": rule_gain(tl, tf, np.arange(len(case)))})

        def disc_mean(v, mask):
            vv = pd.Series(v[mask], index=case.index[mask])
            return vv.groupby(case.loc[mask, "dataset"]).mean().mean()

        policy = {}
        for model in sorted(case["model"].unique()):
            mk = is_disc & (case["model"].values == model)
            scores = {k: disc_mean(v, mk) for k, v in S2_disc.items()}
            scores = {k: s for k, s in scores.items() if np.isfinite(s)}
            policy[model] = max(scores, key=scores.get)
        glob_scores = {k: disc_mean(v, is_disc) for k, v in S2_disc.items()
                       if k != "always reweight (OOB)"}          # must exist for every model
        best_global = max(glob_scores, key=glob_scores.get)

        ma = np.full(len(case), np.nan)
        for model, k in policy.items():
            mk = case["model"].values == model
            ma[mk] = S2[k][mk] if k != "diagnose-then-treat" else dtt_c[mk]
        S2["MODEL-AWARE POLICY"] = ma
        S2["best global strategy"] = S2[best_global] if best_global != "diagnose-then-treat" else dtt_c

        conf = case.loc[~is_disc]
        rows = []
        per_ds = {}
        for k, v in S2.items():
            vc = pd.Series(v[~is_disc], index=conf.index)
            per_ds[k] = vc.groupby(conf["dataset"]).mean()
            rows.append(dict(strategy=k, mean_gain=per_ds[k].mean(),
                             harm=harm_rate(v[~is_disc]),
                             p_vs_never=wil(per_ds[k]) if k != "never treat" else np.nan))
        R = pd.DataFrame(rows).set_index("strategy").round(4)
        p_ma_glob = wil(per_ds["MODEL-AWARE POLICY"] - per_ds["best global strategy"])
        p_ma_never_gt = wil(per_ds["MODEL-AWARE POLICY"], "greater")
        harm_ma = R.loc["MODEL-AWARE POLICY", "harm"]
        harm_best_fixed = R.loc[[k for k in STRAT if k != "never treat"], "harm"].min()
        clean_mask = (~is_disc) & (case["group"].values == "clean")
        ma_clean = pd.Series(ma[clean_mask]).groupby(case.loc[clean_mask, "dataset"].values).mean().mean()

        md += [f"DTT thresholds learned on discovery: theta_label = {tl:.3f}, theta_feature = {tf:.3f}.\n",
               "Model-aware policy chosen on discovery:\n",
               pd.Series(policy, name="strategy").to_frame().to_markdown(), "",
               f"Best global strategy chosen on discovery: **{best_global}**.\n",
               "### Results on CONFIRMATION datasets\n",
               R.to_markdown(), "",
               "### Pre-registered decisions\n",
               f"- H9a  model-aware > never (one-sided Wilcoxon): p = {p_ma_never_gt:.4g} "
               f"-> {'SUPPORTED' if p_ma_never_gt < 0.05 else 'NOT supported'}",
               f"- H9b  model-aware vs best global (two-sided): diff = "
               f"{(per_ds['MODEL-AWARE POLICY'] - per_ds['best global strategy']).mean():.3f}, p = {p_ma_glob:.4g} "
               f"-> {'SUPPORTED' if (p_ma_glob < 0.05 and (per_ds['MODEL-AWARE POLICY'] - per_ds['best global strategy']).mean() > 0) else 'NOT supported'}",
               f"- H9c  harm rate of model-aware ({harm_ma:.1f}%) <= lowest harm of any fixed treating strategy "
               f"({harm_best_fixed:.1f}%) -> {'SUPPORTED' if harm_ma <= harm_best_fixed else 'NOT supported'}",
               f"- H9d  model-aware on clean confirmation data >= -0.2: {ma_clean:.3f} "
               f"-> {'SUPPORTED' if ma_clean >= -0.2 else 'NOT supported'}", ""]
        R.to_csv(out / "v3_confirmatory.csv")

(out / "summary_v3_confirm.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print(f"\nSaved: {out / 'summary_v3_confirm.md'}")
