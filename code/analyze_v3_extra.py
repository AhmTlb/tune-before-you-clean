"""
Extra analyses on saved v3 results (no recomputation)
=====================================================
Usage:  python analyze_v3_extra.py results_v3_full
        (Jupyter: %run analyze_v3_extra.py results_v3_full)

1. Class wipe-outs: how often each removal would have deleted an entire class
   (those fits were skipped in v3), by method and fraction.
2. Sensitivity of the headline strategy table to how skipped cases are counted:
     optimistic   skipped case = no change (gain 0)   <- what analyze_v3 did
     complete     skipped cases dropped
     pessimistic  skipped case = worst gain observed among all options in that case
3. Noise floor: how often random removal of the same size "harms" (< -1 point),
   and the seed-to-seed SD of the untreated model -> what a harm rate of ~X% means.
4. Break-even contamination probability per model and strategy (exploratory):
   expected gain = pi * gain_dirty + (1 - pi) * gain_clean  ->  pi* = -g_clean / (g_dirty - g_clean)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HARM = -1.0
out = Path(sys.argv[1] if len(sys.argv) > 1 else "results_v3_full")
parts = sorted((out / "parts").glob("*.pkl"))
if not parts:
    sys.exit(f"No finished jobs in {out / 'parts'}")

dn_rows, wipe_rows = [], []
pat = re.compile(r"^(?P<cond>\S+) (?P<method>[A-Za-z_]+)@(?P<frac>[0-9.]+): removal wiped out a class")
for f in parts:
    r = pd.read_pickle(f)
    dn_rows += r["downstream"]
    if not r["downstream"]:
        continue
    ds, seed = r["downstream"][0]["dataset"], r["downstream"][0]["seed"]
    for line in r["log"]:
        m = pat.match(line)
        if m:
            wipe_rows.append(dict(dataset=ds, seed=seed, condition=m["cond"],
                                  method=m["method"], frac=round(float(m["frac"]), 3)))
dn = pd.DataFrame(dn_rows)
wipe = pd.DataFrame(wipe_rows, columns=["dataset", "seed", "condition", "method", "frac"])
dn["score"] = 100 * dn["balanced_accuracy"]
dn["option"] = np.select(
    [dn["treatment"] == "none", dn["treatment"] == "remove", dn["treatment"] == "dtt"],
    ["none", "remove:" + dn["method"] + "@" + dn["frac"].map(lambda f: f"{f:g}"), dn["method"]],
    dn["treatment"] + ":" + dn["method"])
n_ds = dn["dataset"].nunique()
md = ["# Extra analyses on v3 results\n",
      f"Datasets: {n_ds} | seeds: {dn['seed'].nunique()}\n"]

# ------------------------------------------------------------------ 1. wipe-outs
CLASSICAL = ["IQR", "MAD", "RobustMahalanobis", "LOF", "IsolationForest"]
if len(wipe):
    wipe["family"] = np.where(wipe["method"].isin(CLASSICAL), "classical",
                              np.where(wipe["method"] == "random", "random",
                                       np.where(wipe["method"].str.startswith("dtt"), "dtt", "target-aware")))
    W = (wipe.groupby(["family", "method"]).agg(skipped_fits=("dataset", "size"),
                                                datasets=("dataset", "nunique"))
         .sort_values("skipped_fits", ascending=False))
    Wf = wipe.pivot_table(index="method", columns="frac", values="dataset", aggfunc="size", fill_value=0)
    Wc = wipe.groupby("condition").size().rename("skipped").to_frame()
    total_jobs = dn.groupby(["dataset", "seed"]).ngroups
    md += ["## 1. Removals that would delete an entire class (skipped in v3)\n",
           f"Total skipped removal settings: **{len(wipe)}** across "
           f"**{wipe['dataset'].nunique()}** of {n_ds} datasets "
           f"({total_jobs} dataset x seed jobs).\n",
           "By method (a removal setting = one condition x fraction; each was skipped for all 4 models):\n",
           W.to_markdown(), "", "By method and removal fraction:\n", Wf.to_markdown(), "",
           "By condition:\n", Wc.to_markdown(), ""]
    W.to_csv(out / "extra_wipeouts_by_method.csv")
else:
    md += ["## 1. Class wipe-outs\nNone recorded.\n"]

# ------------------------------------------------------------------ 2. sensitivity
case = dn.pivot_table(index=["dataset", "seed", "condition", "model"], columns="option",
                      values="score").reset_index()


def group_of(c):
    if c == "clean":
        return "clean"
    k = c.split("@")[0]
    return "mixed" if k == "mixed" else ("label" if k.startswith("label") else "feature")


case["group"] = case["condition"].map(group_of)
opt_cols = [c for c in case.columns if c not in ("dataset", "seed", "condition", "model", "group", "none")]
gains = case[opt_cols].sub(case["none"], axis=0)
worst = gains.min(axis=1)
STRATS = {"always classical (IF 5%)": "remove:IsolationForest@0.05",
          "always target-aware (OOB 5%)": "remove:DataOOB@0.05",
          "always both (DTT sizes)": "dtt_both",
          "remove IF 10%": "remove:IsolationForest@0.1",
          "remove OOB 10%": "remove:DataOOB@0.1",
          "remove SelfConfidence 10%": "remove:SelfConfidence@0.1"}
rows = []
for name, opt in STRATS.items():
    if opt not in gains:
        continue
    g = gains[opt]
    missing = g.isna()
    for policy, v in [("optimistic (skip=0)", g.fillna(0.0)),
                      ("complete cases", g),
                      ("pessimistic (skip=worst)", g.fillna(worst))]:
        sub = case.assign(v=v)
        for grp in ["clean", "label", "feature", "mixed", "ALL"]:
            s = sub if grp == "ALL" else sub[sub["group"] == grp]
            per_ds = s.groupby("dataset")["v"].mean()
            rows.append(dict(strategy=name, policy=policy, group=grp, mean_gain=per_ds.mean(),
                             skipped_pct=100 * missing[s.index].mean()))
S = pd.DataFrame(rows)
St = S.pivot_table(index=["strategy", "policy"], columns="group", values="mean_gain")[
    ["clean", "label", "feature", "mixed", "ALL"]].round(3)
Sk = S[S["policy"] == "complete cases"].pivot_table(index="strategy", columns="group",
                                                    values="skipped_pct")[["clean", "label", "feature", "mixed", "ALL"]].round(1)
md += ["## 2. Sensitivity of strategy gains to skipped (class-deleting) removals\n",
       "If the three policies agree, the skipped cases do not change the conclusion.\n",
       St.to_markdown(), "", "Share of cases skipped (%):\n", Sk.to_markdown(), ""]
St.to_csv(out / "extra_sensitivity.csv")

# ------------------------------------------------------------------ 3. noise floor
rand_opts = [c for c in gains.columns if c.startswith("remove:random@")]
nf = []
for c in rand_opts:
    g = gains[c].dropna()
    nf.append(dict(option=c, harm_rate=100 * (g < HARM).mean(), mean_gain=g.mean(),
                   sd_gain=g.std()))
NF = pd.DataFrame(nf).round(3)
seed_sd = (case.groupby(["dataset", "condition", "model"])["none"].std().groupby("model").mean().round(3))
md += ["## 3. Noise floor\n",
       f"'Harm' = gain < {HARM:g} point. Random removal of the same size is a pure-noise reference: "
       "its harm rate is the share of cases that look harmed by chance alone.\n",
       NF.to_markdown(index=False), "",
       "Seed-to-seed SD of the untreated model (points), mean over datasets x conditions:\n",
       seed_sd.to_frame("sd_none").to_markdown(), ""]

# ------------------------------------------------------------------ 4. break-even
BE_OPTS = {"remove IF 5%": "remove:IsolationForest@0.05", "remove OOB 5%": "remove:DataOOB@0.05",
           "remove both (DTT sizes)": "dtt_both", "reweight OOB": "reweight:DataOOB",
           "winsorize p1-99": "winsorize:winsor_p1_99"}
rows = []
for model in sorted(case["model"].unique()):
    cm = case["model"] == model
    for name, opt in BE_OPTS.items():
        if opt not in gains or gains.loc[cm, opt].isna().all():
            continue
        g = case[cm].assign(v=gains.loc[cm, opt])
        clean = g[g["group"] == "clean"].groupby("dataset")["v"].mean().mean()
        dirty = (g[g["group"] != "clean"].groupby(["dataset", "group"])["v"].mean()
                 .groupby("dataset").mean().mean())       # label/feature/mixed weighted equally
        if clean >= 0 and dirty > 0:
            pi = "always worth it"
        elif dirty <= 0:
            pi = "never worth it"
        else:
            pi = f"{-clean / (dirty - clean):.2f}"
        rows.append(dict(model=model, strategy=name, gain_clean=round(clean, 3),
                         gain_dirty=round(dirty, 3), break_even_pi=pi))
BE = pd.DataFrame(rows)
md += ["## 4. Break-even probability of contamination (exploratory, post hoc)\n",
       "Treating is worth it on average when the probability that the data are contaminated "
       "exceeds pi*. gain_dirty averages the label, feature and mixed groups equally; "
       "it depends on the corruption levels simulated here.\n",
       BE.to_markdown(index=False), ""]
BE.to_csv(out / "extra_break_even.csv", index=False)

(out / "summary_v3_extra.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print(f"\nSaved: {out / 'summary_v3_extra.md'}")
