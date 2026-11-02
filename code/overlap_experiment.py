"""
Motivating experiment (optimized: parallel + resumable + vectorized KNN-Shapley)
================================================================================
Do classical outlier detectors flag the same training points that
data-valuation / label-quality methods consider harmful?
And which removal actually helps downstream supervised models?

Harm scores (higher = more suspicious)
  Classical (target-agnostic): IQR, MAD (robust z), Robust Mahalanobis (MCD),
                               LOF, Isolation Forest
  Target-aware:                Data-OOB, KNN-Shapley, Self-confidence
                               (cleanlab's default label-quality score)
  Controls:                    none (no removal), random removal

Each (dataset, seed) is an independent job:
  * jobs run in parallel on CPU cores (joblib / loky),
  * each finished job is saved to  <out>/parts/<dataset>__seed<k>.pkl,
  * re-running the same command skips finished jobs (resume after a crash
    or a closed laptop lid).

Outputs (in --out):
  overlap.csv, downstream.csv, downstream_tests.csv, summary.md,
  fig_jaccard_k*.png, fig_delta.png, failures.log

Usage
  python overlap_experiment.py --source openml --n-jobs 10 --seeds 5
  python overlap_experiment.py --source sklearn --n-jobs 4 --seeds 2   # smoke test
  python overlap_experiment.py --summarize-only --out results           # rebuild tables
"""

from __future__ import annotations

import argparse
import re
import time
import traceback
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import spearmanr, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.covariance import MinCovDet
from sklearn.ensemble import (HistGradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

CLASSICAL = ["IQR", "MAD", "RobustMahalanobis", "LOF", "IsolationForest"]
TARGET_AWARE = ["DataOOB", "KNNShapley", "SelfConfidence"]
METHODS = CLASSICAL + TARGET_AWARE


# --------------------------------------------------------------------------
# Data loading (runs in the main process; OpenML caches to disk)
# --------------------------------------------------------------------------
def load_sklearn_datasets():
    """Offline datasets bundled with scikit-learn (smoke test only)."""
    from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
    for name, loader in [("breast_cancer", load_breast_cancer), ("wine", load_wine),
                         ("digits", load_digits), ("iris", load_iris)]:
        d = loader(as_frame=True)
        yield name, d.data, d.target


SUITE_CACHE = Path(__file__).with_name("cc18_task_ids.json")


def _with_retry(fn, what, tries=5, wait=20):
    """Retry OpenML calls: the server sometimes answers 503 (temporarily unavailable)."""
    for k in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            if k == tries:
                raise
            print(f"  OpenML {what} failed ({type(e).__name__}); retry {k}/{tries - 1} in {wait * k}s")
            time.sleep(wait * k)


def cc18_task_ids():
    """Task ids of OpenML-CC18 in suite order. The list is saved locally the first time it is
    fetched, so later runs (and runs while the OpenML server is down) do not need the suite API
    and always use the same order."""
    import json
    import openml
    if SUITE_CACHE.exists():
        return json.loads(SUITE_CACHE.read_text())
    ids = [int(t) for t in _with_retry(lambda: openml.study.get_suite(99), "suite 99").tasks]
    SUITE_CACHE.write_text(json.dumps(ids))
    return ids


def load_openml_cc18(max_datasets: int, max_rows: int, max_features: int):
    """OpenML-CC18 benchmark suite (id 99). Requires `pip install openml`.
    Tasks and datasets already downloaded are read from the OpenML disk cache."""
    import openml
    count = 0
    for task_id in cc18_task_ids():
        if count >= max_datasets:
            break
        try:
            task = _with_retry(lambda: openml.tasks.get_task(task_id, download_splits=False),
                               f"task {task_id}")
            ds = _with_retry(task.get_dataset, f"dataset of task {task_id}")
            X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        except Exception as e:
            print(f"  skip task {task_id}: {e}")
            continue
        if X.shape[1] > max_features:
            print(f"  skip {ds.name}: {X.shape[1]} features > {max_features}")
            continue
        if len(X) > max_rows:  # fixed stratified subsample (same for every seed)
            X, _, y, _ = train_test_split(X, y, train_size=max_rows,
                                          stratify=y, random_state=0)
        count += 1
        yield f"{ds.name}_{ds.dataset_id}", X.reset_index(drop=True), \
            pd.Series(y).reset_index(drop=True)


def to_numeric(X: pd.DataFrame, y: pd.Series):
    """Ordinal-encode categoricals, median-impute, drop constant columns.
    Label-free and row-wise independent of the split, so no target leakage;
    the scaler is fitted later on the training part only."""
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    ct = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OrdinalEncoder(handle_unknown="use_encoded_value",
                                             unknown_value=-1)), cat_cols),
    ])
    Xn = np.asarray(ct.fit_transform(X), dtype=float)
    Xn = Xn[:, Xn.std(axis=0) > 0]
    yn = LabelEncoder().fit_transform(y.astype(str))
    keep = np.bincount(yn)[yn] >= 10           # classes too small to stratify
    return Xn[keep], LabelEncoder().fit_transform(yn[keep])


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


# --------------------------------------------------------------------------
# Harm scores (higher = more suspicious). All fitted on the training part only.
# --------------------------------------------------------------------------
def score_iqr(X):
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    iqr = np.where(q3 - q1 == 0, 1e-12, q3 - q1)
    dist = np.maximum(q1 - 1.5 * iqr - X, X - (q3 + 1.5 * iqr)) / iqr
    # many points score 0 -> tie-break by overall magnitude (deterministic)
    return np.clip(dist, 0, None).max(axis=1) + 1e-9 * np.abs(X).sum(axis=1)


def score_mad(X):
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    mad = np.where(mad == 0, 1e-12, mad)
    return np.abs((X - med) / mad).max(axis=1)


def score_robust_mahalanobis(X, seed):
    n, p = X.shape
    if p >= n / 3:
        raise ValueError("too many features for MCD")
    return MinCovDet(random_state=seed).fit(X).mahalanobis(X)


def score_lof(X):
    lof = LocalOutlierFactor(n_neighbors=min(20, len(X) - 1))
    lof.fit(X)
    return -lof.negative_outlier_factor_


def score_iforest(X, seed):
    return -IsolationForest(n_estimators=200, random_state=seed).fit(X).score_samples(X)


def score_data_oob(X, y, seed, n_trees=200):
    """Data-OOB (Kwon & Zou, ICML 2023): value_i = mean out-of-bag correctness
    of bagged trees that did not see i. Harm = 1 - value."""
    rng = np.random.default_rng(seed)
    n = len(y)
    correct, seen = np.zeros(n), np.zeros(n)
    all_idx = np.arange(n)
    for _ in range(n_trees):
        idx = rng.integers(0, n, n)
        oob_mask = np.ones(n, bool)
        oob_mask[idx] = False
        oob = all_idx[oob_mask]
        if len(oob) == 0 or len(np.unique(y[idx])) < 2:
            continue
        tree = DecisionTreeClassifier(random_state=int(rng.integers(1 << 30)))
        tree.fit(X[idx], y[idx])
        correct[oob] += tree.predict(X[oob]) == y[oob]
        seen[oob] += 1
    value = np.divide(correct, seen, out=np.full(n, 0.5), where=seen > 0)
    return 1.0 - value


def score_knn_shapley(X, y, Xv, yv, k=10, chunk=256):
    """Exact KNN-Shapley (Jia et al., VLDB 2019), fully vectorized.
    For each validation point, with training points sorted by distance
    (rank 1 = nearest) and m_i = 1[y_(i) == y_val]:
        s_N = m_N / N
        s_i = s_{i+1} + (m_i - m_{i+1}) / K * min(K, i) / i
    -> s_i = s_N + reverse_cumsum(d)_i ; values are averaged over validation.
    Harm = -value."""
    n = len(y)
    ranks = np.arange(1, n)                              # i = 1..N-1
    coef = np.minimum(k, ranks) / ranks / k              # shape (N-1,)
    sq_tr = (X ** 2).sum(axis=1)
    values = np.zeros(n)
    for start in range(0, len(yv), chunk):               # bounds memory
        xv, lv = Xv[start:start + chunk], yv[start:start + chunk]
        d2 = (xv ** 2).sum(axis=1)[:, None] + sq_tr[None, :] - 2.0 * xv @ X.T
        order = np.argsort(d2, axis=1, kind="stable")    # (b, N)
        m = (y[order] == lv[:, None]).astype(float)       # (b, N)
        d = (m[:, :-1] - m[:, 1:]) * coef                # (b, N-1)
        s = np.empty_like(m)
        s[:, -1] = m[:, -1] / n
        s[:, :-1] = s[:, -1:] + np.cumsum(d[:, ::-1], axis=1)[:, ::-1]
        np.add.at(values, order.ravel(), s.ravel())
    return -(values / len(yv))


def score_self_confidence(X, y, seed):
    """cleanlab's default label-quality score: out-of-fold p(given label).
    Uses cleanlab if installed, otherwise the identical formula."""
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    try:
        from cleanlab.rank import get_label_quality_scores
        q = get_label_quality_scores(y, proba, method="self_confidence")
    except ImportError:
        q = proba[np.arange(len(y)), y]
    return 1.0 - q


def compute_scores(Xs, y, Xv_s, yv, seed, log):
    fns = {
        "IQR": lambda: score_iqr(Xs),
        "MAD": lambda: score_mad(Xs),
        "RobustMahalanobis": lambda: score_robust_mahalanobis(Xs, seed),
        "LOF": lambda: score_lof(Xs),
        "IsolationForest": lambda: score_iforest(Xs, seed),
        "DataOOB": lambda: score_data_oob(Xs, y, seed),
        "KNNShapley": lambda: score_knn_shapley(Xs, y, Xv_s, yv),
        "SelfConfidence": lambda: score_self_confidence(Xs, y, seed),
    }
    out, timing = {}, {}
    for name, fn in fns.items():
        t = time.time()
        try:
            s = np.asarray(fn(), dtype=float)
            if not np.all(np.isfinite(s)):
                raise ValueError("non-finite scores")
            out[name] = s
        except Exception as e:
            log.append(f"{name}: {e}")
        timing[name] = time.time() - t
    return out, timing


# --------------------------------------------------------------------------
# Downstream models (single-threaded: parallelism is across jobs)
# --------------------------------------------------------------------------
def make_models(seed):
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        "HistGB": HistGradientBoostingClassifier(random_state=seed),
    }


def top_k(scores, frac):
    k = max(1, int(round(frac * len(scores))))
    return set(np.argsort(-scores, kind="stable")[:k].tolist())


# --------------------------------------------------------------------------
# One job = one (dataset, seed)
# --------------------------------------------------------------------------
def run_job(name, X, y, seed, fracs, part_path):
    warnings.filterwarnings("ignore")   # worker processes don't inherit filters
    t0 = time.time()
    log = []
    try:
        # Split: T (scored & pruned) / V (valuation only) / test
        X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.25,
                                                  stratify=y, random_state=seed)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.2,
                                                  stratify=y_tv, random_state=seed)
        scaler = StandardScaler().fit(X_tr)              # fitted on T only
        Xs_tr, Xs_va = scaler.transform(X_tr), scaler.transform(X_va)

        scores, timing = compute_scores(Xs_tr, y_tr, Xs_va, y_va, seed, log)

        overlap_rows, down_rows = [], []
        for frac in fracs:
            sets = {m: top_k(s, frac) for m, s in scores.items()}
            for a, b in combinations(sets, 2):
                inter, union = len(sets[a] & sets[b]), len(sets[a] | sets[b])
                overlap_rows.append(dict(
                    dataset=name, seed=seed, frac=frac, a=a, b=b,
                    jaccard=inter / union,
                    jaccard_random=frac / (2 - frac),   # E[J] for two random sets
                    spearman=spearmanr(scores[a], scores[b]).statistic))

        rng = np.random.default_rng(seed)
        n_classes = len(np.unique(y_tr))
        for frac in fracs:
            k = max(1, int(round(frac * len(y_tr))))
            removals = {"none": set(),
                        "random": set(rng.choice(len(y_tr), k, replace=False).tolist())}
            removals.update({m: top_k(s, frac) for m, s in scores.items()})
            for method, drop in removals.items():
                keep = np.setdiff1d(np.arange(len(y_tr)), np.fromiter(drop, int, len(drop)))
                if len(np.unique(y_tr[keep])) < n_classes:
                    log.append(f"{method}@{frac}: removal wiped out a class")
                    continue
                for mname, model in make_models(seed).items():
                    model.fit(X_tr[keep], y_tr[keep])
                    down_rows.append(dict(
                        dataset=name, seed=seed, frac=frac, method=method, model=mname,
                        balanced_accuracy=balanced_accuracy_score(y_te, model.predict(X_te))))

        result = dict(overlap=overlap_rows, downstream=down_rows, log=log,
                      timing=timing, seconds=time.time() - t0)
        tmp = part_path.with_suffix(".tmp")
        pd.to_pickle(result, tmp)
        tmp.replace(part_path)                     # atomic: no half-written parts
        return name, seed, time.time() - t0, log, None
    except Exception:
        return name, seed, time.time() - t0, log, traceback.format_exc()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(args):
    out = Path(args.out)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        datasets = (load_sklearn_datasets() if args.source == "sklearn"
                    else load_openml_cc18(args.max_datasets, args.max_rows, args.max_features))
        jobs, skipped = [], 0
        for name, X_raw, y_raw in datasets:
            X, y = to_numeric(X_raw, y_raw)
            if len(np.unique(y)) < 2 or len(y) < 100:
                print(f"skip {name}: too small after cleaning")
                continue
            for seed in range(args.seeds):
                part = parts / f"{safe_name(name)}__seed{seed}.pkl"
                if part.exists():
                    skipped += 1
                    continue
                jobs.append((name, X, y, seed, part))
            print(f"queued {name}: n={len(y)} p={X.shape[1]} classes={len(np.unique(y))}")

        print(f"\n{len(jobs)} jobs to run, {skipped} already done (resumed). "
              f"Workers: {args.n_jobs}\n")
        t0 = time.time()
        # one thread per worker -> no oversubscription of the 24 logical CPUs
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
                delayed(run_job)(n, X, y, s, args.fracs, p) for n, X, y, s, p in jobs)
            with open(out / "failures.log", "a", encoding="utf-8") as flog:
                for i, (name, seed, secs, log, err) in enumerate(results, 1):
                    status = "FAILED" if err else "ok"
                    print(f"[{i}/{len(jobs)}] {name} seed={seed} {status} "
                          f"({secs:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min)")
                    for line in log:
                        flog.write(f"{name} seed={seed}: {line}\n")
                    if err:
                        flog.write(f"{name} seed={seed}: JOB FAILED\n{err}\n")
                    flog.flush()

    # ---- collect all finished parts --------------------------------------
    ov, dn = [], []
    for f in sorted(parts.glob("*.pkl")):
        r = pd.read_pickle(f)
        ov += r["overlap"]
        dn += r["downstream"]
    if not ov:
        print("No finished jobs yet.")
        return
    ov, dn = pd.DataFrame(ov), pd.DataFrame(dn)
    ov.to_csv(out / "overlap.csv", index=False)
    dn.to_csv(out / "downstream.csv", index=False)
    summarize(ov, dn, out)


# --------------------------------------------------------------------------
# Summary & figures
# --------------------------------------------------------------------------
def summarize(ov: pd.DataFrame, dn: pd.DataFrame, out: Path):
    lines = ["# Motivating experiment: summary\n",
             f"Datasets: {ov['dataset'].nunique()}. "
             f"Seeds per dataset: {ov['seed'].nunique()}.\n"]

    ov = ov.copy()
    fam = {m: ("classical" if m in CLASSICAL else "target-aware") for m in METHODS}
    ov["pair_type"] = [
        "within-classical" if fam[a] == fam[b] == "classical"
        else "within-target-aware" if fam[a] == fam[b]
        else "cross-family" for a, b in zip(ov["a"], ov["b"])]
    # average over seeds first, then over datasets (each dataset weighs equally)
    per_ds = ov.groupby(["dataset", "frac", "pair_type"])[
        ["jaccard", "jaccard_random", "spearman"]].mean().reset_index()
    agg = (per_ds.groupby(["frac", "pair_type"])
           .agg(jaccard=("jaccard", "mean"), jaccard_sd=("jaccard", "std"),
                jaccard_random=("jaccard_random", "mean"),
                spearman=("spearman", "mean")).round(3).reset_index())
    lines += ["## Overlap by pair type (mean over datasets)\n",
              agg.to_markdown(index=False), ""]

    pair = (ov.groupby(["dataset", "frac", "a", "b"])[["jaccard", "spearman"]].mean()
              .groupby(["frac", "a", "b"]).mean().round(3).reset_index())
    lines += ["## Overlap per method pair\n", pair.to_markdown(index=False), ""]

    # ---- downstream: delta vs none / random, Wilcoxon across datasets ----
    per_ds = (dn.groupby(["dataset", "frac", "model", "method"])["balanced_accuracy"]
                .mean().reset_index())
    piv = per_ds.pivot_table(index=["dataset", "frac", "model"], columns="method",
                             values="balanced_accuracy")

    def pval(d):
        d = d.dropna()
        return wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else np.nan

    rows = []
    for (frac, model), g in piv.groupby(level=["frac", "model"]):
        for m in [c for c in g.columns if c != "none"]:
            d_none = (g[m] - g["none"]).dropna()
            d_rand = (g[m] - g["random"]).dropna() if m != "random" else pd.Series(dtype=float)
            rows.append(dict(
                frac=frac, model=model, method=m,
                mean_delta_vs_none=d_none.mean(),
                wins=int((d_none > 0).sum()), losses=int((d_none < 0).sum()),
                p_vs_none=pval(d_none),
                mean_delta_vs_random=d_rand.mean() if len(d_rand) else np.nan,
                p_vs_random=pval(d_rand) if len(d_rand) else np.nan))
    res = pd.DataFrame(rows).round(4)
    lines += ["## Downstream effect of removing top-k (balanced accuracy)\n",
              "Wilcoxon signed-rank across datasets (needs >= 6 datasets). "
              "With many tests, apply Holm correction before claiming significance.\n",
              res.to_markdown(index=False), ""]

    best = (per_ds[per_ds["method"] != "random"]
            .sort_values("balanced_accuracy", ascending=False)
            .groupby(["dataset", "frac", "model"]).head(1))
    share = (best.groupby(["frac", "model"])["method"]
                 .value_counts(normalize=True).round(2).rename("share").reset_index())
    lines += ["## How often each option is the best (per dataset)\n",
              "If no single option dominates, a recommender is justified.\n",
              share.to_markdown(index=False), ""]

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    res.to_csv(out / "downstream_tests.csv", index=False)

    try:
        # Object-oriented API: no pyplot, so the notebook's own plotting
        # backend is never changed and no windows pop up.
        from matplotlib.figure import Figure
        for frac in sorted(ov["frac"].unique()):
            sub = pair[pair["frac"] == frac]
            present = [m for m in METHODS if m in set(sub.a) | set(sub.b)]
            mat = pd.DataFrame(np.eye(len(present)), index=present, columns=present)
            for _, r in sub.iterrows():
                mat.loc[r.a, r.b] = mat.loc[r.b, r.a] = r.jaccard
            fig = Figure(figsize=(7, 6))
            ax = fig.subplots()
            im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(mat)), mat.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(mat)), mat.index)
            for i in range(len(mat)):
                for j in range(len(mat)):
                    ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center",
                            color="white" if mat.values[i, j] < 0.5 else "black", fontsize=8)
            ax.set_title(f"Mean Jaccard of top-{int(round(frac * 100))}% flagged points\n"
                         f"(random expectation ≈ {frac / (2 - frac):.2f})")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(out / f"fig_jaccard_k{int(round(frac * 100))}.png", dpi=200)

        f0 = sorted(res["frac"].unique())[-1]
        piv2 = res[res["frac"] == f0].pivot(index="method", columns="model",
                                             values="mean_delta_vs_none")
        piv2 = piv2.loc[[m for m in ["random"] + METHODS if m in piv2.index]]
        fig = Figure(figsize=(8, 6))
        ax = fig.subplots()
        piv2.plot.barh(ax=ax)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Mean Δ balanced accuracy vs. no removal")
        ax.set_title(f"Removing top-{int(round(f0 * 100))}% flagged training points")
        fig.tight_layout()
        fig.savefig(out / "fig_delta.png", dpi=200)
    except Exception as e:
        print(f"figure generation skipped: {e}")

    print(f"\nSummary written to {out / 'summary.md'}")


if __name__ == "__main__":       # required on Windows for parallel workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["openml", "sklearn"], default="openml")
    ap.add_argument("--max-datasets", type=int, default=72)
    ap.add_argument("--max-rows", type=int, default=3000)
    ap.add_argument("--max-features", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--n-jobs", type=int, default=10,
                    help="parallel workers (10 suits a 12-core / 24-thread laptop)")
    ap.add_argument("--summarize-only", action="store_true",
                    help="skip computation; rebuild tables/figures from saved parts")
    ap.add_argument("--out", default="results")
    run(ap.parse_args())
