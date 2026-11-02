"""
BA-threshold check for paper 1 (PROTOCOL_ba_check.md): do the main gains survive threshold-free metrics?
=====================================================================================================
Usage:  %run compare_ba_check.py results_v4b results_v5b

For the claims in the abstract of paper 1, mean gain over no treatment (points; seeds averaged within dataset,
datasets weighted equally) under three metrics side by side:
  BA05  balanced accuracy of predict() (as in the paper)
  BAP   balanced accuracy of the prior-corrected decision argmax p_k / pi_k
  AUC   one-vs-rest macro ROC AUC (x100)
plus the share of datasets with a positive gain, and a verdict per row:
  survives            same sign under BAP and |BAP gain| >= 50% of |BA05 gain|
  threshold-dependent sign flips under BAP, or |BAP gain| < 50% of |BA05 gain|
(rows with |BA05 gain| < 0.3 points are marked 'small' and not judged)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

v4_dir, v5_dir = Path(sys.argv[1]), Path(sys.argv[2])
METRICS = {"BA05": "balanced_accuracy", "BAP": "ba_prior", "AUC": "auc_macro"}
FAMS = ["LogReg", "KNN", "RandomForest", "HistGB"]


def load(d):
    return pd.DataFrame([r for f in sorted((d / "parts").glob("*.pkl")) for r in pd.read_pickle(f)["rows"]])


def gains(df, optcol, key):
    out = {}
    for lab, col in METRICS.items():
        d = df.copy()
        d["s"] = 100 * d[col]
        b = d[d[optcol] == "none"].set_index(key)["s"].rename("b")
        d = d.join(b, on=key)
        d["g"] = d["s"] - d["b"]
        out[lab] = d
    return out


def row(G, optcol, claim, fam, cond, opt, variant="default"):
    r = dict(claim=claim, family=fam, condition=cond, treatment=opt, variant=variant)
    for lab, d in G.items():
        s = d[(d["family"] == fam) & (d["condition"] == cond) & (d[optcol] == opt) & (d["variant"] == variant)]
        x = s.groupby("dataset")["g"].mean()
        r[f"{lab}_gain"] = x.mean()
        r[f"{lab}_share_pos"] = (x > 0).mean()
    a, b = r["BA05_gain"], r["BAP_gain"]
    if not np.isfinite(a):
        r["verdict"] = "n/a"
    elif abs(a) < 0.3:
        r["verdict"] = "small"
    elif np.sign(a) == np.sign(b) and abs(b) >= 0.5 * abs(a):
        r["verdict"] = "survives"
    else:
        r["verdict"] = "threshold-dependent"
    return r


rows = []
v4 = load(v4_dir)
G4 = gains(v4, "treatment", ["dataset", "seed", "condition", "family", "variant"])
for fam in FAMS:                                         # C1: removal on clean data (default and tuned)
    for var in ["default", "tuned"]:
        for opt in ["remove:OOB", "remove:IF"]:
            rows.append(row(G4, "treatment", "C1 removal on clean data", fam, "clean", opt, var))
for cond in sorted(set(v4["condition"]) - {"clean"}):    # C3: robust tuned LogReg without cleaning
    rows.append(row(G4, "treatment", "C3 quantile+tuned LogReg, no removal (vs default, none)", "LogReg", cond,
                    "none", "quantile+tuned"))
v5 = load(v5_dir)
G5 = gains(v5, "option", ["dataset", "seed", "condition", "family", "variant"])
for cond, opt in [("feature_gross@0.1", "rs:IF"), ("feature_gross@0.05", "rs:IF"), ("label_sym@0.1", "rs:OOB"),
                  ("label_sym@0.2", "rs:OOB")]:          # C2: LogReg matched removal
    rows.append(row(G5, "option", "C2 matched removal, LogReg", "LogReg", cond, opt))
for fam in FAMS:                                         # C4: oracle under 20% label noise; C1 again (v5)
    for cond in ["label_sym@0.2", "label_asym@0.2", "feature_gross@0.1"]:
        rows.append(row(G5, "option", "C4 oracle removal", fam, cond, "oracle"))
    rows.append(row(G5, "option", "C1 fixed 10% OOB removal on clean (v5)", fam, "clean", "fix:OOB@0.1"))
T = pd.DataFrame(rows)
# C3 note: the gain of 'none' with variant quantile+tuned must be computed against default 'none'
c3 = T["claim"].str.startswith("C3")
for i in T[c3].index:
    cond = T.at[i, "condition"]
    for lab, d in G4.items():
        a = d[(d["family"] == "LogReg") & (d["condition"] == cond) & (d["treatment"] == "none")]
        x = (a[a["variant"] == "quantile+tuned"].groupby("dataset")["s"].mean()
             - a[a["variant"] == "default"].groupby("dataset")["s"].mean())
        T.at[i, f"{lab}_gain"], T.at[i, f"{lab}_share_pos"] = x.mean(), (x > 0).mean()
    a, b = T.at[i, "BA05_gain"], T.at[i, "BAP_gain"]
    T.at[i, "verdict"] = ("small" if abs(a) < 0.3 else "survives" if np.sign(a) == np.sign(b) and abs(b) >= 0.5 * abs(a)
                          else "threshold-dependent")
md = ["# Paper 1: do the main gains survive threshold-free metrics? (PROTOCOL_ba_check.md)\n",
      f"v4: {v4['dataset'].nunique()} datasets | v5: {v5['dataset'].nunique()} datasets\n",
      T.round(2).to_markdown(index=False), "",
      "Verdict counts: " + ", ".join(f"{k}: {v}" for k, v in T["verdict"].value_counts().items()), ""]
(v5_dir / "summary_ba_check.md").write_text("\n".join(md), encoding="utf-8")
T.to_csv(v5_dir / "ba_check.csv", index=False)
print("\n".join(md))
print(f"\nSaved: {v5_dir / 'summary_ba_check.md'}")
