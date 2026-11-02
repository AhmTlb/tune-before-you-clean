"""
Experiment v3: realistic corruption + "diagnose, then treat"
============================================================
Builds on v2. New in v3:
  1. Realistic corruption types (injected into the training part T only):
       label_sym     symmetric label flips (as in v2)
       label_asym    class-conditional flips: multiclass -> nearest class (centroids);
                     binary -> minority flipped to majority (under-reporting)
       feature_gross gross feature errors, 5-10 SD (as in v2)
       feature_mild  mild feature errors, 2.5-4 SD (overlap with natural tails)
       mixed         label_sym + feature_gross on disjoint rows
  2. Label-free diagnostics computed on T (no knowledge of the corruption):
       est_label_noise  confident-learning estimate of the label-noise rate
                        (Northcutt et al., 2021) from out-of-fold probabilities
       extreme_rate     share of rows whose max robust |z| (median/MAD) exceeds 6
  3. Diagnose-then-treat (DTT) actions, with removal sizes set by the diagnostics:
       dtt_classical  remove top IsolationForest rows, fraction = clip(extreme_rate)
       dtt_target     remove top Data-OOB rows,       fraction = clip(est_label_noise)
       dtt_both       union of the two
     The decision RULE (which action, given the two diagnostics) is learned
     afterwards in analyze_v3.py with leave-one-dataset-out thresholds.

Everything else (8 scorers, fixed-fraction removal, re-weighting, winsorizing,
4 models, leakage-safe splits, resumable parallel jobs) is as in v2.

Requires overlap_experiment.py and experiment_v2.py in the same folder.

Usage
  python experiment_v3.py --source openml --max-datasets 20 --seeds 3 --n-jobs 10 --out results_v3
  python experiment_v3.py --source sklearn --max-datasets 2 --seeds 1 --n-jobs 2 --preset smoke --out v3_smoke
Then:   python analyze_v3.py results_v3
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

import experiment_v2 as v2
import overlap_experiment as base

warnings.filterwarnings("ignore")

CLASSICAL, TARGET_AWARE = v2.CLASSICAL, v2.TARGET_AWARE
METHODS = CLASSICAL + TARGET_AWARE
DTT_CLASSICAL, DTT_TARGET = "IsolationForest", "DataOOB"
FRAC_MIN, FRAC_MAX = 0.01, 0.15
Z_EXTREME = 6.0

PRESETS = {
    "full": [("clean", 0.0),
             ("label_sym", 0.10), ("label_sym", 0.20),
             ("label_asym", 0.10), ("label_asym", 0.20),
             ("feature_gross", 0.05), ("feature_gross", 0.10),
             ("feature_mild", 0.05), ("feature_mild", 0.10),
             ("mixed", 0.05), ("mixed", 0.10)],
    "smoke": [("clean", 0.0), ("label_asym", 0.10), ("feature_mild", 0.10), ("mixed", 0.10)],
}


# --------------------------------------------------------------------------
# Corruption
# --------------------------------------------------------------------------
def _flip_sym(y, idx, rng):
    classes = np.unique(y)
    for i in idx:
        y[i] = rng.choice(classes[classes != y[i]])


def _feature_errors(X, idx, rng, lo, hi):
    med = np.median(X, axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    p = X.shape[1]
    n_f = max(1, int(np.ceil(0.2 * p)))
    for i in idx:
        f = rng.choice(p, n_f, replace=False)
        X[i, f] = med[f] + rng.choice([-1, 1], n_f) * rng.uniform(lo, hi, n_f) * sd[f]


def corrupt(X, y, kind, rate, rng):
    """Returns X, y, label_mask, feature_mask (rows actually corrupted)."""
    X, y = X.copy(), y.copy()
    n = len(y)
    lmask, fmask = np.zeros(n, bool), np.zeros(n, bool)
    m = int(round(rate * n))
    if kind == "clean" or m == 0:
        return X, y, lmask, fmask

    if kind == "label_sym":
        idx = rng.choice(n, m, replace=False)
        _flip_sym(y, idx, rng)
        lmask[idx] = True
    elif kind == "label_asym":
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) == 2:                      # under-reporting: minority -> majority
            src, dst = classes[np.argmin(counts)], classes[np.argmax(counts)]
            pool = np.where(y == src)[0]
            idx = rng.choice(pool, min(m, len(pool) // 2), replace=False)
            y[idx] = dst
        else:                                      # confusion with the nearest class
            Z = StandardScaler().fit_transform(X)
            cent = np.array([Z[y == c].mean(axis=0) for c in classes])
            d = ((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d, np.inf)
            nearest = dict(zip(classes, classes[d.argmin(axis=1)]))
            idx = rng.choice(n, m, replace=False)
            y[idx] = [nearest[c] for c in y[idx]]
        lmask[idx] = True
    elif kind in ("feature_gross", "feature_mild"):
        lo, hi = (5, 10) if kind == "feature_gross" else (2.5, 4)
        idx = rng.choice(n, m, replace=False)
        _feature_errors(X, idx, rng, lo, hi)
        fmask[idx] = True
    elif kind == "mixed":
        idx = rng.choice(n, 2 * m, replace=False)
        li, fi = idx[:m], idx[m:]
        _flip_sym(y, li, rng)
        _feature_errors(X, fi, rng, 5, 10)
        lmask[li], fmask[fi] = True, True
    else:
        raise ValueError(kind)
    return X, y, lmask, fmask


# --------------------------------------------------------------------------
# Diagnostics (label-free w.r.t. the corruption)
# --------------------------------------------------------------------------
def oof_proba(Xs, y, seed):
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(clf, Xs, y, cv=cv, method="predict_proba")


def estimate_label_noise(y, proba):
    """Confident-learning estimate of the fraction of mislabeled rows.
    Per-class threshold t_j = mean p(j) over rows labeled j; confident joint C[i, j]
    counts rows labeled i whose most likely above-threshold class is j; rows of C
    are calibrated to the observed label counts; noise = off-diagonal mass / n."""
    n, K = proba.shape
    t = np.array([proba[y == j, j].mean() if np.any(y == j) else 1.0 for j in range(K)])
    above = proba >= t[None, :]
    has = above.any(axis=1)
    j_star = np.where(above, proba, -1.0).argmax(axis=1)
    C = np.zeros((K, K))
    np.add.at(C, (y[has], j_star[has]), 1)
    counts = np.bincount(y, minlength=K).astype(float)
    rows = C.sum(axis=1, keepdims=True)
    Q = np.divide(C, rows, out=np.zeros_like(C), where=rows > 0) * counts[:, None]
    return float((Q.sum() - np.trace(Q)) / n)


def extreme_rate(X, z=Z_EXTREME):
    """Share of rows whose largest robust z-score (median / MAD, std fallback) exceeds z."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    sd = X.std(axis=0)
    scale = np.where(mad > 0, mad, sd)
    keep = scale > 0
    if not keep.any():
        return 0.0
    rz = np.abs((X[:, keep] - med[keep]) / scale[keep])
    return float((rz.max(axis=1) > z).mean())


def compute_scores(Xs, y, Xv_s, yv, proba, seed, log):
    fns = {
        "IQR": lambda: base.score_iqr(Xs),
        "MAD": lambda: base.score_mad(Xs),
        "RobustMahalanobis": lambda: v2.score_robust_mahalanobis(Xs, seed),
        "LOF": lambda: base.score_lof(Xs),
        "IsolationForest": lambda: base.score_iforest(Xs, seed),
        "DataOOB": lambda: base.score_data_oob(Xs, y, seed),
        "KNNShapley": lambda: base.score_knn_shapley(Xs, y, Xv_s, yv),
        "SelfConfidence": lambda: 1.0 - proba[np.arange(len(y)), y],   # cleanlab default
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
# One job = one (dataset, seed), all conditions
# --------------------------------------------------------------------------
def run_job(name, X, y, seed, conditions, fracs, part_path):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    log, overlap, detect, mech, diag, down = [], [], [], [], [], []
    try:
        X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.25,
                                                  stratify=y, random_state=seed)
        X_tr0, X_va, y_tr0, y_va = train_test_split(X_tv, y_tv, test_size=0.2,
                                                    stratify=y_tv, random_state=seed)
        n_classes = len(np.unique(y_tr0))

        for kind, rate in conditions:
            cond = "clean" if kind == "clean" else f"{kind}@{rate:g}"
            rng = np.random.default_rng(10_000 * seed + int(rate * 1000) + 97 * len(kind))
            X_tr, y_tr, lmask, fmask = corrupt(X_tr0, y_tr0, kind, rate, rng)
            cmask = lmask | fmask
            common = dict(dataset=name, seed=seed, condition=cond, kind=kind, rate=rate)

            scaler = StandardScaler().fit(X_tr)
            Xs_tr, Xs_va = scaler.transform(X_tr), scaler.transform(X_va)
            proba = oof_proba(Xs_tr, y_tr, seed)
            scores = compute_scores(Xs_tr, y_tr, Xs_va, y_va, proba, seed, log)

            # ---- diagnostics -------------------------------------------
            est_l, ext = estimate_label_noise(y_tr, proba), extreme_rate(X_tr)
            diag.append(dict(**common, est_label_noise=est_l, extreme_rate=ext,
                             true_label_rate=lmask.mean(), true_feature_rate=fmask.mean()))

            # ---- overlap (top-5%) ----------------------------------------
            sets = {m: set(v2.top_idx(s, 0.05).tolist()) for m, s in scores.items()}
            for a, b in combinations(sets, 2):
                overlap.append(dict(**common, a=a, b=b,
                                    jaccard=len(sets[a] & sets[b]) / len(sets[a] | sets[b]),
                                    spearman=spearmanr(scores[a], scores[b]).statistic))

            # ---- detection of injected corruption (by corruption type) ----
            for mtype, mask in [("label", lmask), ("feature", fmask)]:
                if mask.any():
                    k_true = int(mask.sum())
                    for m, s in scores.items():
                        top = np.argsort(-s, kind="stable")[:k_true]
                        detect.append(dict(**common, target=mtype, method=m,
                                           auroc=roc_auc_score(mask, s),
                                           precision_at_k=mask[top].mean()))

            # ---- mechanism -----------------------------------------------
            counts = np.bincount(y_tr, minlength=n_classes)
            minority = int(np.argmin(np.where(counts > 0, counts, np.inf)))
            base_share = (y_tr == minority).mean()
            hard = v2.kdn(Xs_tr, y_tr)
            for m, s in scores.items():
                top = v2.top_idx(s, 0.05)
                mech.append(dict(**common, method=m,
                                 minority_enrichment=(y_tr[top] == minority).mean() / base_share,
                                 kdn_flagged=hard[top].mean(), kdn_all=hard.mean(),
                                 corrupted_share=cmask[top].mean() if cmask.any() else np.nan))

            # ---- downstream ----------------------------------------------
            def record(treatment, method, frac, b, a, model):
                down.append(dict(**common, treatment=treatment, method=method, frac=frac,
                                 model=model, balanced_accuracy=b, accuracy=a))

            all_rows = np.arange(len(y_tr))
            for mname, model in v2.make_models(seed).items():
                b, a = v2.evaluate(model, X_tr, y_tr, X_te, y_te)
                record("none", "none", 0.0, b, a, mname)

            jobs = []
            for frac in fracs:
                k = max(1, int(round(frac * len(y_tr))))
                jobs.append(("remove", "random", frac,
                             np.setdiff1d(all_rows, rng.choice(len(y_tr), k, replace=False))))
                for m, s in scores.items():
                    jobs.append(("remove", m, frac, np.setdiff1d(all_rows, v2.top_idx(s, frac))))
            # diagnose-then-treat actions (sizes from diagnostics)
            f_cls = float(np.clip(ext, FRAC_MIN, FRAC_MAX))
            f_tgt = float(np.clip(est_l, FRAC_MIN, FRAC_MAX))
            if DTT_CLASSICAL in scores and DTT_TARGET in scores:
                drop_c = v2.top_idx(scores[DTT_CLASSICAL], f_cls)
                drop_t = v2.top_idx(scores[DTT_TARGET], f_tgt)
                jobs += [("dtt", "dtt_classical", f_cls, np.setdiff1d(all_rows, drop_c)),
                         ("dtt", "dtt_target", f_tgt, np.setdiff1d(all_rows, drop_t)),
                         ("dtt", "dtt_both", f_cls + f_tgt,
                          np.setdiff1d(all_rows, np.union1d(drop_c, drop_t)))]
            for treat, m, frac, keep in jobs:
                if len(np.unique(y_tr[keep])) < n_classes:
                    log.append(f"{cond} {m}@{frac:.3f}: removal wiped out a class -> skipped")
                    continue
                for mname, model in v2.make_models(seed).items():
                    b, a = v2.evaluate(model, X_tr[keep], y_tr[keep], X_te, y_te)
                    record(treat, m, frac, b, a, mname)

            for m, s in scores.items():                        # soft down-weighting
                w = np.ones(len(y_tr))
                w[v2.top_idx(s, v2.REWEIGHT_TOP)] = v2.REWEIGHT_W
                for mname, model in v2.make_models(seed).items():
                    if mname in v2.WEIGHT_PARAM:
                        b, a = v2.evaluate(model, X_tr, y_tr, X_te, y_te,
                                           {v2.WEIGHT_PARAM[mname]: w})
                        record("reweight", m, v2.REWEIGHT_TOP, b, a, mname)

            lo, hi = np.percentile(X_tr, [1, 99], axis=0)      # winsorizing
            Xc_tr, Xc_te = np.clip(X_tr, lo, hi), np.clip(X_te, lo, hi)
            for mname, model in v2.make_models(seed).items():
                b, a = v2.evaluate(model, Xc_tr, y_tr, Xc_te, y_te)
                record("winsorize", "winsor_p1_99", 0.0, b, a, mname)

        result = dict(overlap=overlap, detect=detect, mech=mech, diag=diag, downstream=down,
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
def run(args):
    out = Path(args.out)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    conditions = PRESETS[args.preset]

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

    # largest jobs first: long jobs start early instead of dragging out the end of the run
    jobs.sort(key=lambda j: -len(j[2]) * max(1, len(np.unique(j[2]))) * j[1].shape[1] ** 0.5)
    print(f"\nConditions ({len(conditions)}): {conditions}")
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
    print(f"\nDone. Now run:  python analyze_v3.py {out}")


if __name__ == "__main__":       # required on Windows for parallel workers
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["openml", "sklearn"], default="openml")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--max-datasets", type=int, default=20)
    ap.add_argument("--max-rows", type=int, default=3000)
    ap.add_argument("--max-features", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--out", default="results_v3")
    run(ap.parse_args())
