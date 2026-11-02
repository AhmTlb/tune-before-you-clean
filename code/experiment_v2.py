"""
Experiment v2: When does outlier/noise cleaning help supervised learning, and why?
==================================================================================

Research questions
  RQ1  Do classical (feature-space) detectors and target-aware (data-valuation /
       label-quality) methods flag the same training points?
  RQ2  On real benchmark data, does treating flagged points help downstream models?
       Treatments: removal (1/2/5/10 %), soft down-weighting, winsorizing.
  RQ3  Mechanism: what kind of points are flagged? (minority-class enrichment,
       instance hardness kDN)
  RQ4  Controlled corruption: with KNOWN injected label noise or feature outliers,
       (a) which family detects them (AUROC, precision@k)?
       (b) from which corruption level does each treatment start to help?

Design (leakage-safe)
  * Split per seed: T (train, may be corrupted, scored and treated) /
    V (validation, clean, used only by KNN-Shapley) / test (clean).
  * Corruption is injected into T only, before any preprocessing is fitted.
  * Every detector / scaler / clip bound is fitted on T only.
  * Controls: no treatment, and random removal of the same size.

One job = one (dataset, seed); it runs all corruption conditions.
Finished jobs are saved atomically to <out>/parts/ -> re-run to resume.

Requires overlap_experiment.py in the same folder (data loading and scorers).

Usage
  python experiment_v2.py --source openml --max-datasets 20 --seeds 3 --n-jobs 10 --out results_v2
  python experiment_v2.py --source sklearn --max-datasets 2 --seeds 1 --n-jobs 2 --out v2_smoke
Then:   python analyze_v2.py results_v2
"""

from __future__ import annotations

import argparse
import time
import traceback
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import overlap_experiment as base

warnings.filterwarnings("ignore")

CLASSICAL = ["IQR", "MAD", "RobustMahalanobis", "LOF", "IsolationForest"]
TARGET_AWARE = ["DataOOB", "KNNShapley", "SelfConfidence"]
METHODS = CLASSICAL + TARGET_AWARE
REWEIGHT_TOP, REWEIGHT_W = 0.10, 0.5      # soft treatment: top-10% get weight 0.5


# --------------------------------------------------------------------------
# Fixed robust Mahalanobis: PCA (bounded dims) + MCD, Ledoit-Wolf fallback
# --------------------------------------------------------------------------
def score_robust_mahalanobis(X, seed):
    n, p = X.shape
    q = int(min(p, max(2, n // 10), 30))
    Z = PCA(n_components=q, random_state=seed).fit_transform(X) if q < p else X
    try:
        d = MinCovDet(support_fraction=0.75, random_state=seed).fit(Z).mahalanobis(Z)
        if not np.all(np.isfinite(d)):
            raise ValueError("non-finite")
        return d
    except Exception:
        return LedoitWolf().fit(Z).mahalanobis(Z)


def compute_scores(Xs, y, Xv_s, yv, seed, log):
    fns = {
        "IQR": lambda: base.score_iqr(Xs),
        "MAD": lambda: base.score_mad(Xs),
        "RobustMahalanobis": lambda: score_robust_mahalanobis(Xs, seed),
        "LOF": lambda: base.score_lof(Xs),
        "IsolationForest": lambda: base.score_iforest(Xs, seed),
        "DataOOB": lambda: base.score_data_oob(Xs, y, seed),
        "KNNShapley": lambda: base.score_knn_shapley(Xs, y, Xv_s, yv),
        "SelfConfidence": lambda: base.score_self_confidence(Xs, y, seed),
    }
    out = {}
    for name, fn in fns.items():
        try:
            s = np.asarray(fn(), dtype=float)
            if not np.all(np.isfinite(s)):
                raise ValueError("non-finite scores")
            out[name] = s
        except Exception as e:
            log.append(f"{name}: {e}")
    return out


# --------------------------------------------------------------------------
# Controlled corruption (applied to the training part T only)
# --------------------------------------------------------------------------
def corrupt(X, y, kind, rate, rng):
    """kind='label': symmetric label flips on `rate` of rows.
    kind='feature': on `rate` of rows, ~20% of features get gross errors
    (median +/- 5..10 standard deviations). Returns X, y, corrupted-row mask."""
    X, y = X.copy(), y.copy()
    n, p = X.shape
    mask = np.zeros(n, bool)
    m = int(round(rate * n))
    if kind == "clean" or m == 0:
        return X, y, mask
    idx = rng.choice(n, m, replace=False)
    mask[idx] = True
    if kind == "label":
        classes = np.unique(y)
        for i in idx:
            y[i] = rng.choice(classes[classes != y[i]])
    elif kind == "feature":
        med = np.median(X, axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        n_f = max(1, int(np.ceil(0.2 * p)))
        for i in idx:
            f = rng.choice(p, n_f, replace=False)
            X[i, f] = med[f] + rng.choice([-1, 1], n_f) * rng.uniform(5, 10, n_f) * sd[f]
    return X, y, mask


# --------------------------------------------------------------------------
# Models and treatments
# --------------------------------------------------------------------------
def make_models(seed):
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1),
        "HistGB": HistGradientBoostingClassifier(random_state=seed),
    }


WEIGHT_PARAM = {"LogReg": "logisticregression__sample_weight",
                "RandomForest": "sample_weight", "HistGB": "sample_weight"}  # KNN: none


def top_idx(scores, frac):
    k = max(1, int(round(frac * len(scores))))
    return np.argsort(-scores, kind="stable")[:k]


def evaluate(model, X_fit, y_fit, X_te, y_te, fit_params=None):
    model.fit(X_fit, y_fit, **(fit_params or {}))
    pred = model.predict(X_te)
    return balanced_accuracy_score(y_te, pred), accuracy_score(y_te, pred)


def kdn(Xs, y, k=5):
    """Instance hardness: fraction of the k nearest neighbours with another label."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(y))).fit(Xs)
    idx = nn.kneighbors(Xs, return_distance=False)[:, 1:]
    return (y[idx] != y[:, None]).mean(axis=1)


# --------------------------------------------------------------------------
# One job = one (dataset, seed), all corruption conditions
# --------------------------------------------------------------------------
def run_job(name, X, y, seed, conditions, fracs, part_path):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    log, overlap, detect, mech, down = [], [], [], [], []
    try:
        X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.25,
                                                  stratify=y, random_state=seed)
        X_tr0, X_va, y_tr0, y_va = train_test_split(X_tv, y_tv, test_size=0.2,
                                                    stratify=y_tv, random_state=seed)
        n_classes = len(np.unique(y_tr0))

        for kind, rate in conditions:
            cond = f"{kind}@{rate:g}" if kind != "clean" else "clean"
            rng = np.random.default_rng(10_000 * seed + int(rate * 1000) + len(kind))
            X_tr, y_tr, cmask = corrupt(X_tr0, y_tr0, kind, rate, rng)

            scaler = StandardScaler().fit(X_tr)
            Xs_tr, Xs_va = scaler.transform(X_tr), scaler.transform(X_va)
            scores = compute_scores(Xs_tr, y_tr, Xs_va, y_va, seed, log)
            common = dict(dataset=name, seed=seed, condition=cond, kind=kind, rate=rate)

            # ---- RQ1 overlap (top-5%) -----------------------------------
            sets = {m: set(top_idx(s, 0.05).tolist()) for m, s in scores.items()}
            for a, b in combinations(sets, 2):
                overlap.append(dict(**common, a=a, b=b,
                                    jaccard=len(sets[a] & sets[b]) / len(sets[a] | sets[b]),
                                    spearman=spearmanr(scores[a], scores[b]).statistic))

            # ---- RQ4a detection of injected corruption -------------------
            if cmask.any():
                k_true = int(cmask.sum())
                for m, s in scores.items():
                    top = np.argsort(-s, kind="stable")[:k_true]
                    detect.append(dict(**common, method=m,
                                       auroc=roc_auc_score(cmask, s),
                                       precision_at_k=cmask[top].mean()))

            # ---- RQ3 mechanism: who gets flagged? ------------------------
            counts = np.bincount(y_tr, minlength=n_classes)
            minority = int(np.argmin(np.where(counts > 0, counts, np.inf)))
            base_share = (y_tr == minority).mean()
            hard = kdn(Xs_tr, y_tr)
            for m, s in scores.items():
                top = top_idx(s, 0.05)
                mech.append(dict(**common, method=m,
                                 minority_enrichment=(y_tr[top] == minority).mean() / base_share,
                                 kdn_flagged=hard[top].mean(), kdn_all=hard.mean(),
                                 corrupted_share=cmask[top].mean() if cmask.any() else np.nan))

            # ---- RQ2/RQ4b downstream treatments --------------------------
            def record(treatment, method, frac, bacc, acc, model):
                down.append(dict(**common, treatment=treatment, method=method, frac=frac,
                                 model=model, balanced_accuracy=bacc, accuracy=acc))

            all_rows = np.arange(len(y_tr))
            for mname, model in make_models(seed).items():
                b, a = evaluate(model, X_tr, y_tr, X_te, y_te)
                record("none", "none", 0.0, b, a, mname)

            jobs = []                                         # (treatment, method, frac, keep)
            for frac in fracs:
                k = max(1, int(round(frac * len(y_tr))))
                jobs.append(("remove", "random", frac, np.setdiff1d(all_rows, rng.choice(len(y_tr), k, replace=False))))
                for m, s in scores.items():
                    jobs.append(("remove", m, frac, np.setdiff1d(all_rows, top_idx(s, frac))))
            for treat, m, frac, keep in jobs:
                if len(np.unique(y_tr[keep])) < n_classes:
                    log.append(f"{cond} {m}@{frac}: removal wiped out a class -> skipped")
                    continue
                for mname, model in make_models(seed).items():
                    b, a = evaluate(model, X_tr[keep], y_tr[keep], X_te, y_te)
                    record(treat, m, frac, b, a, mname)

            # soft down-weighting (KNN has no sample weights)
            for m, s in scores.items():
                w = np.ones(len(y_tr))
                w[top_idx(s, REWEIGHT_TOP)] = REWEIGHT_W
                for mname, model in make_models(seed).items():
                    if mname in WEIGHT_PARAM:
                        b, a = evaluate(model, X_tr, y_tr, X_te, y_te,
                                        {WEIGHT_PARAM[mname]: w})
                        record("reweight", m, REWEIGHT_TOP, b, a, mname)

            # winsorizing (feature-level, detector-independent), bounds from T
            q1, q3 = np.percentile(X_tr, [25, 75], axis=0)
            iqr = q3 - q1
            bounds = {"winsor_p1_99": tuple(np.percentile(X_tr, [1, 99], axis=0)),
                      "winsor_iqr": (q1 - 1.5 * iqr, q3 + 1.5 * iqr)}
            for wname, (lo, hi) in bounds.items():
                Xc_tr, Xc_te = np.clip(X_tr, lo, hi), np.clip(X_te, lo, hi)
                for mname, model in make_models(seed).items():
                    b, a = evaluate(model, Xc_tr, y_tr, Xc_te, y_te)
                    record("winsorize", wname, 0.0, b, a, mname)

        result = dict(overlap=overlap, detect=detect, mech=mech, downstream=down,
                      log=log, seconds=time.time() - t0)
        tmp = part_path.with_suffix(".tmp")
        pd.to_pickle(result, tmp)
        tmp.replace(part_path)
        return name, seed, time.time() - t0, log, None
    except Exception:
        return name, seed, time.time() - t0, log, traceback.format_exc()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build_conditions(label_rates, feature_rates):
    return ([("clean", 0.0)] + [("label", r) for r in label_rates]
            + [("feature", r) for r in feature_rates])


def run(args):
    out = Path(args.out)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    conditions = build_conditions(args.label_noise, args.feature_outliers)

    if args.source == "sklearn":
        datasets = list(base.load_sklearn_datasets())[: args.max_datasets]
    else:
        datasets = base.load_openml_cc18(args.max_datasets, args.max_rows, args.max_features)

    jobs, skipped = [], 0
    for name, X_raw, y_raw in datasets:
        X, y = base.to_numeric(X_raw, y_raw)
        if len(np.unique(y)) < 2 or len(y) < 100:
            print(f"skip {name}: too small after cleaning")
            continue
        for seed in range(args.seeds):
            part = parts / f"{base.safe_name(name)}__seed{seed}.pkl"
            if part.exists():
                skipped += 1
                continue
            jobs.append((name, X, y, seed, part))
        print(f"queued {name}: n={len(y)} p={X.shape[1]} classes={len(np.unique(y))}")

    print(f"\nConditions: {[c for c in conditions]}")
    print(f"{len(jobs)} jobs to run, {skipped} already done (resumed). Workers: {args.n_jobs}\n")
    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
            delayed(run_job)(n, X, y, s, conditions, args.fracs, p) for n, X, y, s, p in jobs)
        with open(out / "failures.log", "a", encoding="utf-8") as flog:
            for i, (name, seed, secs, log, err) in enumerate(results, 1):
                print(f"[{i}/{len(jobs)}] {name} seed={seed} {'FAILED' if err else 'ok'} "
                      f"({secs:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min)")
                for line in log:
                    flog.write(f"{name} seed={seed}: {line}\n")
                if err:
                    flog.write(f"{name} seed={seed}: JOB FAILED\n{err}\n")
                flog.flush()
    print(f"\nDone. Now run:  python analyze_v2.py {out}")


if __name__ == "__main__":       # required on Windows for parallel workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["openml", "sklearn"], default="openml")
    ap.add_argument("--max-datasets", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=3000)
    ap.add_argument("--max-features", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10])
    ap.add_argument("--label-noise", type=float, nargs="*", default=[0.05, 0.10, 0.20])
    ap.add_argument("--feature-outliers", type=float, nargs="*", default=[0.05, 0.10])
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--out", default="results_v2")
    run(ap.parse_args())
