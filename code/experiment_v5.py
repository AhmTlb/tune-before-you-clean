"""
Experiment v5: is the problem WHO we remove, or HOW MANY, or removal itself?
===========================================================================
Because the corrupted rows are known, we can compare real detectors with an
oracle that removes exactly the corrupted rows:

  none            no treatment
  oracle          remove exactly the corrupted rows (label and/or feature)
  oracle_label    mixed only: remove only the label-corrupted rows
  oracle_feature  mixed only: remove only the feature-corrupted rows
  rs:IF           Isolation Forest, removing as many rows as are feature-corrupted (right-sized)
  rs:OOB          Data-OOB, removing as many rows as are label-corrupted (right-sized)
  rs:both         mixed only: union of rs:IF and rs:OOB
  fix:IF@0.1      Isolation Forest, top 10% (as in v3/v4)
  fix:OOB@0.1     Data-OOB, top 10%
  random          random removal of as many rows as are corrupted (control)

Precision (share of removed rows that were truly corrupted) is stored for every
detector-based removal. Removals that delete a whole class are evaluated, not
skipped (flag wiped_class). Splits, corruption and scorers are those of v3/v4.

Usage
  python experiment_v5.py --source openml --max-datasets 40 --seeds 3 --n-jobs 10 --out results_v5
  python experiment_v5.py --source sklearn --max-datasets 2 --seeds 1 --n-jobs 2 --preset smoke --out v5_smoke
  (add --tuned to also run the tuned variants of v4; roughly triples the run time)
Then:  python analyze_v5.py results_v5
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import experiment_v3 as v3
import experiment_v4 as v4
import overlap_experiment as base

warnings.filterwarnings("ignore")

PRESETS = {
    "full": [("clean", 0.0), ("feature_gross", 0.05), ("feature_gross", 0.10),
             ("feature_mild", 0.10), ("label_sym", 0.10), ("label_sym", 0.20),
             ("label_asym", 0.20), ("mixed", 0.10)],
    "smoke": [("clean", 0.0), ("feature_gross", 0.10), ("label_sym", 0.20), ("mixed", 0.10)],
    # review follow-up (M3): rerun only asymmetric noise to record plain accuracy as well
    "asym": [("label_asym", 0.20)],
}
FIX = 0.10


def top_k(scores, k):
    return np.argsort(-scores, kind="stable")[:k] if k > 0 else np.array([], dtype=int)


def run_job(name, X, y, seed, conditions, tuned, part_path):
    warnings.filterwarnings("ignore")
    t0 = time.time()
    rows, prec, log = [], [], []
    variants = {f: (["default", "tuned"] if tuned else ["default"]) for f in v4.VARIANTS}
    try:
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
            s_if = base.score_iforest(Xs, seed)
            s_oob = base.score_data_oob(Xs, y_tr, seed)
            k_l, k_f, k_all = int(lmask.sum()), int(fmask.sum()), int(cmask.sum())
            k_fix = max(1, int(round(FIX * n)))

            drops = {"none": np.array([], dtype=int),
                     "fix:IF@0.1": top_k(s_if, k_fix), "fix:OOB@0.1": top_k(s_oob, k_fix)}
            if k_all:
                drops["oracle"] = np.where(cmask)[0]
                drops["random"] = rng.choice(n, k_all, replace=False)
            if k_f:
                drops["rs:IF"] = top_k(s_if, k_f)
            if k_l:
                drops["rs:OOB"] = top_k(s_oob, k_l)
            if k_l and k_f:
                drops["oracle_label"] = np.where(lmask)[0]
                drops["oracle_feature"] = np.where(fmask)[0]
                drops["rs:both"] = np.union1d(drops["rs:IF"], drops["rs:OOB"])

            common = dict(dataset=name, seed=seed, condition=cond, kind=kind, rate=rate, n_classes=K, n_train=n)
            for opt, d in drops.items():
                if opt.startswith(("fix", "rs")) and k_all:
                    target = fmask if "IF" in opt else lmask
                    if opt == "rs:both":
                        target = cmask
                    prec.append(dict(**common, option=opt, removed=len(d),
                                     precision=float(cmask[d].mean()) if len(d) else np.nan,
                                     precision_own_type=float(target[d].mean()) if len(d) else np.nan,
                                     recall=float(cmask[d].sum() / k_all)))
                keep = np.setdiff1d(np.arange(n), d)
                Xf, yf = X_tr[keep], y_tr[keep]
                wiped = len(np.unique(yf)) < K
                for family, vs in variants.items():
                    for variant in vs:
                        try:
                            est, fallback = v4.build(family, variant, yf, seed)
                            est.fit(Xf, yf)
                            pred = est.predict(X_te)
                            ba = balanced_accuracy_score(y_te, pred)
                            acc = accuracy_score(y_te, pred)
                            extra = v4.threshold_free_metrics(est, X_te, y_te, yf)
                        except Exception as e:
                            log.append(f"{cond} {opt} {family}/{variant}: {e}")
                            continue
                        rows.append(dict(**common, option=opt, removed=len(d), family=family,
                                         variant=variant, balanced_accuracy=ba, accuracy=acc, **extra,
                                         wiped_class=wiped))
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
    print(f"\nConditions: {conditions}\nTuned variants: {args.tuned}\n"
          f"{len(jobs)} jobs to run, {skipped} already done. Workers: {args.n_jobs}\n")
    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
            delayed(run_job)(n, X, y, s, conditions, args.tuned, p) for n, X, y, s, p in jobs)
        with open(out / "failures.log", "a", encoding="utf-8") as flog:
            for i, (name, seed, secs, log, err) in enumerate(results, 1):
                print(f"[{i}/{len(jobs)}] {name} seed={seed} {'FAILED' if err else 'ok'} "
                      f"({secs:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min)")
                for line in log:
                    flog.write(f"{name} seed={seed}: {line}\n")
                if err:
                    flog.write(f"{name} seed={seed}: JOB FAILED\n{err}\n")
                flog.flush()
    print(f"\nDone. Now run:  python analyze_v5.py {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["openml", "sklearn"], default="openml")
    ap.add_argument("--preset", choices=list(PRESETS), default="full")
    ap.add_argument("--max-datasets", type=int, default=40)
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--max-features", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tuned", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--out", default="results_v5")
    run(ap.parse_args())
