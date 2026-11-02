"""
Follow-up analyses requested by the internal peer review (no new experiments except M3-c)
========================================================================================
Usage (Jupyter):
    %run analyze_review.py --v3 results_v3 --v4 results_v4 --v5 results_v5 --v5asym results_v5_asym
Any folder that does not exist is skipped. Writes summary_review.md in the current folder.

M3  Asymmetric label noise (Stage 5): is the 7-10 point oracle-vs-detector gap a binary-only,
    balanced-accuracy-only effect?
      a) realised share of flipped training rows, binary vs multiclass
      b) oracle / right-sized Data-OOB / random gains and detector precision, split by binary vs multiclass
      c) the same gains in plain accuracy (needs the short rerun:
         python experiment_v5.py --preset asym --out results_v5_asym)
M4  The Stage 4 "recovery" figure: every component, how 83% is computed, per-dataset medians, and a
    difference-in-differences that removes the part of the tuning gain that also appears on clean data.
M5  Mechanism (Stage 3, clean data): do datasets/methods whose flagged rows are harder (kDN) or more
    minority-heavy also lose more accuracy when those rows are removed?
      a) within each dataset, Spearman over the 8 methods between hardness ratio and removal gain;
         Wilcoxon of the per-dataset rho against 0 (dataset effects cancel)
      b) across datasets, per target-aware method and model
Gains in balanced-accuracy points (x100); seeds averaged within dataset; datasets weighted equally.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--v3", default="results_v3")
ap.add_argument("--v4", default="results_v4")
ap.add_argument("--v5", default="results_v5")
ap.add_argument("--v5asym", default="results_v5_asym")
ap.add_argument("--out", default="summary_review.md")
ap.add_argument("--m5-exclude-from", default=None,
                help="M5 only: drop datasets found in this folder (e.g. results_v3) to test on held-out datasets")
ap.add_argument("--only", default=None, help="comma list of sections to run, e.g. M5")
args = ap.parse_args()

FAMS = ["LogReg", "KNN", "RandomForest", "HistGB"]
md = ["# Follow-up analyses for the peer review\n"]


def load(folder, key="rows"):
    parts = sorted((Path(folder) / "parts").glob("*.pkl"))
    if not parts:
        return None
    return pd.DataFrame([r for f in parts for r in pd.read_pickle(f)[key]])


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


def ds(sub, col="gain"):
    return sub.groupby("dataset")[col].mean()


# =================================================================== M3
ONLY = set(args.only.split(",")) if args.only else {"M3", "M4", "M5"}
v5 = load(args.v5) if "M3" in ONLY else None
if v5 is not None and (v5["condition"] == "label_asym@0.2").any():
    pr = load(args.v5, "precision")
    a = v5[(v5["condition"] == "label_asym@0.2") & (v5["variant"] == "default")].copy()
    a["type"] = np.where(a["n_classes"] == 2, "binary", "multiclass")
    a["score"] = 100 * a["balanced_accuracy"]
    key = ["dataset", "seed", "family"]
    a = a.join(a[a["option"] == "none"].set_index(key)["score"].rename("base"), on=key)
    a["gain"] = a["score"] - a["base"]

    # realised flip share: oracle removes exactly the flipped rows
    if "n_train" in a.columns:
        n_tr = a.groupby(["dataset", "seed"])["n_train"].first()
    else:   # older runs: n from the fixed-10% removal, round(0.1 n)
        fx = a[a["option"] == "fix:IF@0.1"].groupby(["dataset", "seed"])["removed"].first()
        n_tr = fx / 0.10
    k = a[a["option"] == "oracle"].groupby(["dataset", "seed"])["removed"].first()
    flip = (k / n_tr).rename("flip_share").reset_index()
    flip = flip.merge(a[["dataset", "type"]].drop_duplicates(), on="dataset")
    F = flip.groupby(["type", "dataset"])["flip_share"].mean().groupby("type").agg(
        datasets="count", mean="mean", min="min", max="max").round(3)
    md += ["## M3. Asymmetric label noise (nominal 20%), binary vs multiclass\n",
           "### M3-a. Realised share of training rows whose label was flipped\n",
           "Binary data flips minority -> majority, capped at half of the minority class.\n",
           F.to_markdown(), ""]

    rows = []
    for t in ["binary", "multiclass", "all"]:
        s = a if t == "all" else a[a["type"] == t]
        for fam in FAMS:
            r = dict(type=t, family=fam, datasets=s["dataset"].nunique())
            for opt in ["oracle", "rs:OOB", "random"]:
                r[opt] = ds(s[(s["family"] == fam) & (s["option"] == opt)]).mean()
            r["gap (oracle - rs:OOB)"] = r["oracle"] - r["rs:OOB"]
            rows.append(r)
    G = pd.DataFrame(rows).round(2)
    md += ["### M3-b. Gains in balanced accuracy (points) vs. no removal\n", G.to_markdown(index=False), ""]

    if pr is not None and len(pr):
        p = pr[(pr["condition"] == "label_asym@0.2") & (pr["option"] == "rs:OOB")].copy()
        p["type"] = np.where(p["n_classes"] == 2, "binary", "multiclass")
        P = (p.groupby(["type", "dataset"])[["precision", "recall"]].mean()
              .groupby("type").mean().round(3))
        md += ["Right-sized Data-OOB precision/recall on asymmetric noise:\n", P.to_markdown(), ""]
    G.to_csv("review_M3_gains.csv", index=False)
else:
    md += ["## M3\nSkipped: no label_asym@0.2 rows in " + args.v5 + "\n"]

va = load(args.v5asym) if "M3" in ONLY else None
if va is not None and "accuracy" in va.columns:
    va = va[va["variant"] == "default"].copy()
    va["type"] = np.where(va["n_classes"] == 2, "binary", "multiclass")
    key = ["dataset", "seed", "family"]
    rows = []
    for metric in ["balanced_accuracy", "accuracy"]:
        va["score"] = 100 * va[metric]
        b = va[va["option"] == "none"].set_index(key)["score"].rename("base")
        v = va.drop(columns="base", errors="ignore").join(b, on=key)
        v["gain"] = v["score"] - v["base"]
        for t in ["binary", "multiclass", "all"]:
            s = v if t == "all" else v[v["type"] == t]
            for fam in FAMS:
                o = ds(s[(s["family"] == fam) & (s["option"] == "oracle")])
                r_ = ds(s[(s["family"] == fam) & (s["option"] == "rs:OOB")])
                rows.append(dict(metric=metric, type=t, family=fam, oracle=o.mean(),
                                 rs_OOB=r_.mean(), gap=(o - r_).mean(),
                                 p_gap_gt0=wil((o - r_).dropna(), "greater")))
    A = pd.DataFrame(rows).round(3)
    md += ["### M3-c. Same comparison in plain accuracy (rerun results_v5_asym)\n",
           "If the gap stays large in plain accuracy and in multiclass data, it is not an artefact of "
           "the binary construction or of balanced accuracy.\n", A.to_markdown(index=False), ""]
    A.to_csv("review_M3_accuracy.csv", index=False)
else:
    md += ["### M3-c\nSkipped: run `python experiment_v5.py --preset asym --out results_v5_asym` first.\n"]

# =================================================================== M4
v4 = load(args.v4) if "M4" in ONLY else None
if v4 is not None:
    v4["score"] = 100 * v4["balanced_accuracy"]
    key = ["dataset", "seed", "condition", "family"]
    b = v4[(v4["variant"] == "default") & (v4["treatment"] == "none")].set_index(key)["score"].rename("base")
    v4 = v4.join(b, on=key)
    v4["gain"] = v4["score"] - v4["base"]
    lr = v4[v4["family"] == "LogReg"]
    conds = sorted(c for c in lr["condition"].unique() if c != "clean")

    def matched(c):
        k = c.split("@")[0]
        return ["remove:IF"] if k.startswith("feature") else (
            ["remove:OOB"] if k.startswith("label") else ["remove:IF", "remove:OOB"])

    def opt(c, variant, treat):
        return ds(lr[(lr["condition"] == c) & (lr["variant"] == variant) & (lr["treatment"] == treat)])

    robust_clean = opt("clean", "quantile+tuned", "none")      # tuning/transform gain with no contamination
    rows, per_ds_rows = [], []
    for c in conds:
        for m in matched(c):
            gc = opt(c, "default", m)                 # default LogReg + matched removal
            gr = opt(c, "quantile+tuned", "none")     # robust tuned LogReg, no removal
            did = (gr - robust_clean)                 # contamination-specific part of the robust gain
            rows.append(dict(condition=c, cleaning=m, default_cleaning_gain=gc.mean(),
                             robust_no_cleaning_gain=gr.mean(), robust_gain_on_clean=robust_clean.mean(),
                             robust_gain_minus_clean=did.mean(),
                             ratio=gr.mean() / gc.mean() if gc.mean() > 0 else np.nan,
                             ratio_did=did.mean() / gc.mean() if gc.mean() > 0 else np.nan,
                             median_diff_robust_minus_cleaning=(gr - gc).median(),
                             share_ds_robust_ge_cleaning=((gr - gc).dropna() >= 0).mean()))
            per_ds_rows.append(pd.DataFrame(dict(gc=gc, gr=gr, did=did)).assign(condition=c, cleaning=m))
    R = pd.DataFrame(rows)
    avg = R.groupby("condition")[["default_cleaning_gain", "robust_no_cleaning_gain",
                                  "robust_gain_minus_clean"]].mean().mean()
    md += ["## M4. Stage 4 recovery: components and definitions\n",
           "Gains are relative to default LogReg without removal (balanced-accuracy points). "
           "`robust` = quantile transform + tuned C, no removal. `robust_gain_on_clean` is the same "
           "option on clean data; subtracting it isolates the part of the gain that is specific to "
           "contamination (difference-in-differences).\n",
           R.round(3).to_markdown(index=False), "",
           f"- Ratio of means as reported in the draft (conditions averaged first, mixed averaged over its two "
           f"cleanings): **{avg['robust_no_cleaning_gain'] / avg['default_cleaning_gain']:.0%}**",
           f"- Mean of the per-condition ratios: **{R.groupby('condition')['ratio'].mean().mean():.0%}**",
           f"- Same ratio after removing the clean-data gain (difference-in-differences): "
           f"**{avg['robust_gain_minus_clean'] / avg['default_cleaning_gain']:.0%}**",
           f"- Share of dataset x condition cases where robust-no-cleaning >= default+cleaning: "
           f"**{(pd.concat(per_ds_rows).eval('gr >= gc')).mean():.0%}**\n"]
    R.to_csv("review_M4_recovery.csv", index=False)
else:
    md += ["## M4\nSkipped: no results in " + args.v4 + "\n"]

# =================================================================== M5
mech, dn = (load(args.v3, "mech"), load(args.v3, "downstream")) if "M5" in ONLY else (None, None)
if mech is not None and args.m5_exclude_from:
    seen = set(load(args.m5_exclude_from, "mech")["dataset"])
    mech, dn = mech[~mech["dataset"].isin(seen)], dn[~dn["dataset"].isin(seen)]
    md += [f"M5 restricted to {mech['dataset'].nunique()} datasets not in {args.m5_exclude_from} (held out).\n"]
    if mech.empty:
        mech = None
if mech is not None and dn is not None:
    c = mech[mech["condition"] == "clean"].copy()
    c["hardness_ratio"] = c["kdn_flagged"] / c["kdn_all"].replace(0, np.nan)
    H = c.groupby(["dataset", "method"])[["hardness_ratio", "minority_enrichment"]].mean()
    d = dn[dn["condition"] == "clean"].copy()
    d["score"] = 100 * d["balanced_accuracy"]
    k = ["dataset", "seed", "model"]
    d = d.join(d[d["treatment"] == "none"].set_index(k)["score"].rename("base"), on=k)
    d["gain"] = d["score"] - d["base"]
    frac = 0.05                                     # mechanism was measured on the top 5%
    g = (d[(d["treatment"] == "remove") & np.isclose(d["frac"], frac)]
         .groupby(["dataset", "method", "model"])["gain"].mean().reset_index())
    g = g.join(H, on=["dataset", "method"])
    if "model" in g:
        g = g.rename(columns={"model": "family"})

    rows = []
    for fam in g["family"].unique():
        for feat in ["hardness_ratio", "minority_enrichment"]:
            rhos = []
            for _, s in g[g["family"] == fam].groupby("dataset"):
                s = s.dropna(subset=[feat, "gain"])
                if len(s) >= 5 and s[feat].nunique() > 1 and s["gain"].nunique() > 1:
                    rhos.append(spearmanr(s[feat], s["gain"]).statistic)
            rhos = np.array(rhos)
            rows.append(dict(family=fam, predictor=feat, datasets=len(rhos),
                             median_rho=np.median(rhos) if len(rhos) else np.nan,
                             share_negative=(rhos < 0).mean() if len(rhos) else np.nan,
                             p_rho_lt0=wil(rhos, "less")))
    W = pd.DataFrame(rows)
    W["p_holm"] = holm(W["p_rho_lt0"].values)
    W = W.round(3)
    md += ["## M5. Do harder / more minority-heavy removals cost more? (Stage 3, clean data, top 5%)\n",
           "### M5-a. Within each dataset, across the 8 methods\n",
           "Negative rho = methods that flag harder (or more minority) rows lose more accuracy. "
           "One-sided Wilcoxon of the per-dataset rho against 0.\n",
           W.to_markdown(index=False), ""]

    rows = []
    for meth in g["method"].unique():
        for fam in g["family"].unique():
            s = g[(g["method"] == meth) & (g["family"] == fam)].dropna(subset=["hardness_ratio", "gain"])
            if len(s) >= 8:
                r = spearmanr(s["hardness_ratio"], s["gain"])
                rows.append(dict(method=meth, family=fam, datasets=len(s), rho=r.statistic, p=r.pvalue))
    X = pd.DataFrame(rows).round(3)
    md += ["### M5-b. Across datasets, per method and model (hardness ratio vs. gain)\n",
           "Descriptive only (many correlations, no correction).\n", X.to_markdown(index=False), ""]
    W.to_csv("review_M5_within.csv", index=False)
    X.to_csv("review_M5_across.csv", index=False)
else:
    md += ["## M5\nSkipped: no results in " + args.v3 + "\n"]

Path(args.out).write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print(f"\nSaved: {args.out}")
