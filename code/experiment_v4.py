"""
Experiment v4: does tuning or a robust pipeline replace data cleaning?
=====================================================================
Question a reviewer will ask: v3 used default hyper-parameters. Perhaps a tuned
model (e.g. regularised logistic regression) or a rank-based (quantile) input
transform gets the same benefit as cleaning, without deleting anything.

For each family we compare model VARIANTS x TREATMENTS:
  LogReg : default | tuned (C by inner CV) | quantile (QuantileTransformer->normal) | quantile+tuned
  KNN    : default | tuned (k, weights)    | quantile
  RF     : default | tuned (min_samples_leaf, max_features)
  HistGB : default | tuned (random search, 8 configs)
  Treatments: none | remove IsolationForest 10% | remove Data-OOB 10% | winsorize p1-p99
Tuning uses an inner stratified CV on the (possibly corrupted, possibly cleaned)
training part only; scoring = balanced accuracy.

Change from v3: removals that delete an entire class are NOT skipped. The model is
trained on the remaining classes and evaluated normally (the missing class gets
recall 0), and the case is flagged wiped_class=True. This counts the real cost.

Same splits, corruption code and data loading as v3 (imports experiment_v3).

Usage
  python experiment_v4.py --source openml --max-datasets 40 --seeds 3 --n-jobs 10 --out results_v4
  python experiment_v4.py --source sklearn --max-datasets 2 --seeds 1 --n-jobs 2 --preset smoke --out v4_smoke
Then:  python analyze_v4.py results_v4
"""
from __future__ import annotations

import argparse
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

import experiment_v3 as v3
import overlap_experiment as base

warnings.filterwarnings("ignore")

PRESETS = {
    "full": [("clean", 0.0), ("feature_gross", 0.10), ("feature_mild", 0.10),
             ("label_sym", 0.20), ("label_asym", 0.20), ("mixed", 0.10)],
    "smoke": [("clean", 0.0), ("feature_gross", 0.10), ("label_sym", 0.20)],
}
REMOVE_FRAC = 0.10
C_GRID = np.logspace(-3, 2, 8)


def inner_cv(y, k_max, seed):
    """Stratified inner CV with as many folds as the smallest class allows (None if < 2)."""
    k = min(k_max, int(np.bincount(y)[np.bincount(y) > 0].min()))
    return StratifiedKFold(k, shuffle=True, random_state=seed) if k >= 2 else None


def quantile(n):
    return QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, n), random_state=0)


def build(family, variant, y, seed):
    """Return an unfitted estimator. Tuned variants fall back to default if the
    smallest class is too small for an inner CV (flag returned)."""
    n = len(y)
    if family == "LogReg":
        cv = inner_cv(y, 5, seed)
        lr_cv = (LogisticRegressionCV(Cs=C_GRID, cv=cv, scoring="balanced_accuracy", max_iter=3000)
                 if cv is not None else LogisticRegression(max_iter=3000))
        return {"default": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                "tuned": make_pipeline(StandardScaler(), lr_cv),
                "quantile": make_pipeline(quantile(n), LogisticRegression(max_iter=3000)),
                "quantile+tuned": make_pipeline(quantile(n), lr_cv)}[variant], cv is None
    if family == "KNN":
        cv = inner_cv(y, 3, seed)
        knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
        tuned = (GridSearchCV(knn, {"kneighborsclassifier__n_neighbors": [5, 15, 31],
                                    "kneighborsclassifier__weights": ["uniform", "distance"]},
                              cv=cv, scoring="balanced_accuracy") if cv is not None else knn)
        return {"default": knn, "tuned": tuned,
                "quantile": make_pipeline(quantile(n), KNeighborsClassifier())}[variant], cv is None
    if family == "RandomForest":
        cv = inner_cv(y, 3, seed)
        rf = RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1)
        tuned = (GridSearchCV(rf, {"min_samples_leaf": [1, 3, 10], "max_features": ["sqrt", 0.5]},
                              cv=cv, scoring="balanced_accuracy") if cv is not None else rf)
        return {"default": rf, "tuned": tuned}[variant], cv is None
    if family == "HistGB":
        cv = inner_cv(y, 3, seed)
        hgb = HistGradientBoostingClassifier(random_state=seed)
        tuned = (RandomizedSearchCV(hgb, {"learning_rate": [0.03, 0.1, 0.3],
                                          "max_leaf_nodes": [15, 31, 63],
                                          "l2_regularization": [0.0, 1.0, 10.0],
                                          "min_samples_leaf": [5, 20, 50]},
                                    n_iter=8, cv=cv, scoring="balanced_accuracy", random_state=seed)
                 if cv is not None else hgb)
        return {"default": hgb, "tuned": tuned}[variant], cv is None
    raise ValueError(family)


def threshold_free_metrics(est, X_test, y_test, y_fit):
    """Added for the BA-threshold check (PROTOCOL_ba_check.md). Returns
    ba_prior  balanced accuracy of the prior-corrected decision argmax_k p_k(x) / pi_k, pi_k = class shares of the
              labels the model was fitted on (the balanced-accuracy-optimal rule for calibrated probabilities;
              for two classes it is the threshold p_1 >= pi_1)
    auc_macro one-vs-rest macro ROC AUC on the test set (threshold-free)
    Classes missing from the fitted model get probability 0. Errors give NaN, never stop the run."""
    try:
        classes = np.unique(y_test)
        P = np.zeros((len(X_test), len(classes)))
        proba = est.predict_proba(X_test)
        for j, c in enumerate(est.classes_):
            if c in classes:
                P[:, np.searchsorted(classes, c)] = proba[:, j]
        share = np.array([(y_fit == c).mean() for c in classes])
        adj = np.where(share > 0, P / np.maximum(share, 1e-12), -np.inf)
        ba_prior = balanced_accuracy_score(y_test, classes[adj.argmax(1)])
        if len(classes) == 2:
            auc = roc_auc_score(y_test, P[:, 1])
        else:
            Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-12)
            auc = roc_auc_score(y_test, Pn, multi_class="ovr", average="macro", labels=classes)
        return dict(ba_prior=ba_prior, auc_macro=auc)
    except Exception:
        return dict(ba_prior=np.nan, auc_macro=np.nan)


VARIANTS = {"LogReg": ["default", "tuned", "quantile", "quantile+tuned"],
            "KNN": ["default", "tuned", "quantile"],
            "RandomForest": ["default", "tuned"],
            "HistGB": ["default", "tuned"]}


def run_job(name, X, y, seed, conditions, part_path):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    rows, log = [], []
    try:
        X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        X_tr0, _, y_tr0, _ = train_test_split(X_tv, y_tv, test_size=0.2, stratify=y_tv,
                                              random_state=seed)   # same T as v3 (V unused here)
        K = len(np.unique(y_tr0))
        for kind, rate in conditions:
            cond = "clean" if kind == "clean" else f"{kind}@{rate:g}"
            rng = np.random.default_rng(10_000 * seed + int(rate * 1000) + 97 * len(kind))
            X_tr, y_tr, lmask, fmask = v3.corrupt(X_tr0, y_tr0, kind, rate, rng)
            Xs = StandardScaler().fit(X_tr).transform(X_tr)
            s_if = base.score_iforest(Xs, seed)
            s_oob = base.score_data_oob(Xs, y_tr, seed)
            all_rows = np.arange(len(y_tr))
            lo, hi = np.percentile(X_tr, [1, 99], axis=0)
            treatments = {
                "none": (X_tr, y_tr, X_te),
                "remove:IF": (np.delete(X_tr, v3.v2.top_idx(s_if, REMOVE_FRAC), 0),
                              np.delete(y_tr, v3.v2.top_idx(s_if, REMOVE_FRAC)), X_te),
                "remove:OOB": (np.delete(X_tr, v3.v2.top_idx(s_oob, REMOVE_FRAC), 0),
                               np.delete(y_tr, v3.v2.top_idx(s_oob, REMOVE_FRAC)), X_te),
                "winsorize": (np.clip(X_tr, lo, hi), y_tr, np.clip(X_te, lo, hi)),
            }
            for treat, (Xf, yf, Xt) in treatments.items():
                wiped = len(np.unique(yf)) < K
                for family, variants in VARIANTS.items():
                    for variant in variants:
                        try:
                            est, fallback = build(family, variant, yf, seed)
                            est.fit(Xf, yf)
                            pred = est.predict(Xt)
                            ba = balanced_accuracy_score(y_te, pred)
                            extra = threshold_free_metrics(est, Xt, y_te, yf)
                        except Exception as e:
                            log.append(f"{cond} {treat} {family}/{variant}: {e}")
                            continue
                        rows.append(dict(dataset=name, seed=seed, condition=cond, kind=kind, rate=rate,
                                         n_classes=K, treatment=treat, family=family, variant=variant,
                                         balanced_accuracy=ba, wiped_class=wiped, **extra,
                                         tuning_fallback=fallback))
        result = dict(rows=rows, log=log, seconds=time.time() - t0)
        tmp = part_path.with_suffix(".tmp")
        pd.to_pickle(result, tmp)
        tmp.replace(part_path)
        return name, seed, time.time() - t0, log, None
    except Exception:
        return name, seed, time.time() - t0, log, traceback.format_exc()


def run(args):
    out = Path(args.out)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    conditions = PRESETS[args.preset]
    datasets = (list(base.load_sklearn_datasets())[: args.max_datasets] if args.source == "sklearn"
                else base.load_openml_cc18(args.max_datasets, args.max_rows, args.max_features))
    jobs, skipped = [], 0
    for name, X_raw, y_raw in datasets:
        X, y = base.to_numeric(X_raw, y_raw)
        if len(np.unique(y)) < 2 or len(y) < 100:
            continue
        for seed in range(args.seeds):
            part = parts / f"{base.safe_name(name)}__seed{seed}.pkl"
            if part.exists():
                skipped += 1
                continue
            jobs.append((name, X, y, seed, part))
        print(f"queued {name}: n={len(y)} p={X.shape[1]} classes={len(np.unique(y))}")
    jobs.sort(key=lambda j: -len(j[2]) * max(1, len(np.unique(j[2]))) * j[1].shape[1] ** 0.5)
    print(f"\nConditions: {conditions}\n{len(jobs)} jobs to run, {skipped} already done. "
          f"Workers: {args.n_jobs}\n")
    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
            delayed(run_job)(n, X, y, s, conditions, p) for n, X, y, s, p in jobs)
        with open(out / "failures.log", "a", encoding="utf-8") as flog:
            for i, (name, seed, secs, log, err) in enumerate(results, 1):
                print(f"[{i}/{len(jobs)}] {name} seed={seed} {'FAILED' if err else 'ok'} "
                      f"({secs:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min)")
                for line in log:
                    flog.write(f"{name} seed={seed}: {line}\n")
                if err:
                    flog.write(f"{name} seed={seed}: JOB FAILED\n{err}\n")
                flog.flush()
    print(f"\nDone. Now run:  python analyze_v4.py {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["openml", "sklearn"], default="openml")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--max-datasets", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--max-features", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--out", default="results_v4")
    run(ap.parse_args())
