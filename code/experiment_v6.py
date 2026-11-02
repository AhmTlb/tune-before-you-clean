"""
Experiment v6: does a classical ensemble filter do better than Data-OOB?
=======================================================================
Reviewer question: Smith & Martinez (2018) found an ensemble of classifiers to be the best
filter for mislabelled instances. Is our conclusion (removal hurts tree ensembles; asymmetric
label noise is a detection gap) specific to Data-OOB, or does it hold for the classical
ensemble filter (Brodley & Friedl, 1999) as well?

Ensemble filter (EF): 5-fold out-of-fold predictions of four diverse learners
(logistic regression, 5-NN, decision tree, Gaussian naive Bayes) on the standardized training set.
  votes_i = number of learners that misclassify row i (0..4)
  EF score = votes_i + 0.99 * (1 - mean predicted probability of the given label)   (ties broken)
Options (same splits, corruption and seeds as v5, so v5 and v6 rows are directly comparable):
  none            no treatment
  oracle          remove exactly the corrupted rows
  ef:majority     remove rows misclassified by >= 3 of 4 learners (native size)
  ef:consensus    remove rows misclassified by all 4 learners (native size)
  rs:EF / rs:OOB  EF / Data-OOB, removing as many rows as are label-corrupted (right-sized)
  fix:EF@0.1 / fix:OOB@0.1   top 10%
Feature-error conditions use the total corrupted count for right-sizing.
Precision/recall stored for every detector-based removal. Class-deleting removals are evaluated.

Usage
  python experiment_v6.py --source openml --max-datasets 40 --seeds 3 --n-jobs 10 --out results_v6
  python experiment_v6.py --source sklearn --max-datasets 2 --seeds 1 --n-jobs 2 --preset smoke --out v6_smoke
Then:  python analyze_v6.py results_v6
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import experiment_v3 as v3
import experiment_v4 as v4
import overlap_experiment as base

PRESETS = {
    "full": [("clean", 0.0), ("feature_gross", 0.10), ("label_sym", 0.10), ("label_sym", 0.20),
             ("label_asym", 0.20), ("mixed", 0.10)],
    "smoke": [("clean", 0.0), ("label_sym", 0.20), ("label_asym", 0.20)],
}
FIX = 0.10


def top_k(scores, k):
    return np.argsort(-scores, kind="stable")[:k] if k > 0 else np.array([], dtype=int)


def ensemble_filter(Xs, y, seed):
    """Returns (votes, score). votes = # of 4 learners misclassifying each row out-of-fold."""
    _, counts = np.unique(y, return_counts=True)
    k = int(min(5, counts.min()))
    cv = (StratifiedKFold(k, shuffle=True, random_state=seed) if k >= 2
          else KFold(5, shuffle=True, random_state=seed))
    learners = [LogisticRegression(max_iter=3000), KNeighborsClassifier(5),
                DecisionTreeClassifier(random_state=seed), GaussianNB()]
    classes = np.unique(y)
    col = np.searchsorted(classes, y)
    votes = np.zeros(len(y))
    p_label = np.zeros(len(y))
    for est in learners:
        proba = cross_val_predict(est, Xs, y, cv=cv, method="predict_proba")
        if proba.shape[1] != len(classes):          # a fold missed a class: fall back to labels
            pred = cross_val_predict(est, Xs, y, cv=cv)
            votes += pred != y
            p_label += pred == y
            continue
        votes += classes[proba.argmax(1)] != y
        p_label += proba[np.arange(len(y)), col]
    p_label /= len(learners)
    return votes, votes + 0.99 * (1.0 - p_label)


def run_job(name, X, y, seed, conditions, part_path):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    rows, prec, log = [], [], []
    try:
        # identical to v5
        X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        X_tr0, _, y_tr0, _ = train_test_split(X_tv, y_tv, test_size=0.2, stratify=y_tv, random_state=seed)
        K = len(np.unique(y_tr0))
        for kind, rate in conditions:
            cond = "clean" if kind == "clean" else f"{kind}@{rate:g}"
            rng = np.random.default_rng(10_000 * seed + int(rate * 1000) + 97 * len(kind))
            X_tr, y_tr, lmask, fmask = v3.corrupt(X_tr0, y_tr0, kind, rate, rng)
            cmask = lmask | fmask
            n = len(y_tr)
            Xs = StandardScaler().fit(X_tr).transform(X_tr)
            s_oob = base.score_data_oob(Xs, y_tr, seed)
            votes, s_ef = ensemble_filter(Xs, y_tr, seed)
            k_l, k_all = int(lmask.sum()), int(cmask.sum())
            k_rs = k_l if k_l else k_all
            k_fix = max(1, int(round(FIX * n)))

            drops = {"none": np.array([], dtype=int),
                     "ef:majority": np.where(votes >= 3)[0],
                     "ef:consensus": np.where(votes >= 4)[0],
                     "fix:EF@0.1": top_k(s_ef, k_fix), "fix:OOB@0.1": top_k(s_oob, k_fix)}
            if k_all:
                drops["oracle"] = np.where(cmask)[0]
                drops["rs:EF"] = top_k(s_ef, k_rs)
                drops["rs:OOB"] = top_k(s_oob, k_rs)

            common = dict(dataset=name, seed=seed, condition=cond, kind=kind, rate=rate,
                          n_classes=K, n_train=n)
            for opt, d in drops.items():
                if opt not in ("none", "oracle"):
                    prec.append(dict(**common, option=opt, removed=len(d),
                                     share_removed=len(d) / n,
                                     precision=float(cmask[d].mean()) if len(d) and k_all else np.nan,
                                     recall=float(cmask[d].sum() / k_all) if k_all else np.nan))
                keep = np.setdiff1d(np.arange(n), d)
                Xf, yf = X_tr[keep], y_tr[keep]
                wiped = len(np.unique(yf)) < K
                for family in v4.VARIANTS:
                    try:
                        est, _ = v4.build(family, "default", yf, seed)
                        est.fit(Xf, yf)
                        pred = est.predict(X_te)
                    except Exception as e:
                        log.append(f"{cond} {opt} {family}: {e}")
                        continue
                    rows.append(dict(**common, option=opt, removed=len(d), family=family,
                                     balanced_accuracy=balanced_accuracy_score(y_te, pred),
                                     accuracy=accuracy_score(y_te, pred), wiped_class=wiped))
        result = dict(rows=rows, precision=prec, log=log, seconds=time.time() - t0)
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
    print(f"\nDone. Now run:  python analyze_v6.py {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["openml", "sklearn"], default="openml")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--max-datasets", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--max-features", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--out", default="results_v6")
    run(ap.parse_args())
