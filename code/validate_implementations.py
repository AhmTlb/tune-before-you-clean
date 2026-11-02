"""
Validate our own implementations of the target-aware scorers
============================================================
Usage:  python validate_implementations.py            (Jupyter: %run validate_implementations.py)
Writes validation_report.md in the current folder.

Checks (offline datasets bundled with scikit-learn, 10% symmetric label noise injected):
  1. KNN-Shapley (closed form, Jia et al. 2019)  vs  Monte-Carlo permutation Shapley computed
     directly from the definition of the KNN utility. Independent of the recursion we coded.
  2. Data-OOB (our bagging loop)  vs  out-of-bag correctness computed from scikit-learn's
     BaggingClassifier (an independent bagging implementation).
  3. Self-confidence  vs  cleanlab.rank.get_label_quality_scores(method="self_confidence")
     on the same out-of-fold probabilities (needs `pip install cleanlab`; skipped otherwise).
  4. Confident-learning noise-rate estimate  vs  cleanlab.count.num_label_issues
     (calibrated off-diagonal method) / n  (needs cleanlab).
  5. Optional: KNN-Shapley and Data-OOB from the OpenDataVal library, if it is installed and
     its API matches the one tried here; any mismatch is reported, not hidden.

Pass criteria (fixed before running):
  1, 2  Spearman rank correlation >= 0.90 on every dataset
  3     max absolute difference < 1e-9
  4     absolute difference of the noise-rate estimates <= 0.02 on every dataset
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import experiment_v3 as v3
import overlap_experiment as base

RNG = np.random.default_rng(0)
lines = ["# Validation of scorer implementations\n"]


def datasets():
    for name, loader in [("breast_cancer", load_breast_cancer), ("wine", load_wine),
                         ("iris", load_iris), ("digits", load_digits)]:
        d = loader()
        X, y = d.data.astype(float), d.target
        if len(y) > 600:
            X, _, y, _ = train_test_split(X, y, train_size=600, stratify=y, random_state=0)
        X = StandardScaler().fit_transform(X)
        Xc, yc, lmask, _ = v3.corrupt(X, y, "label_sym", 0.10, np.random.default_rng(1))
        yield name, Xc, yc, lmask


# ------------------------------------------------------------------ 1. KNN-Shapley vs Monte Carlo
def mc_knn_shapley(X, y, Xv, yv, k, perms, rng):
    """Permutation-sampling Shapley values of the KNN utility
    u(S) = mean_v (1/K) * #{matching labels among the min(K,|S|) nearest points of S to v}."""
    n, V = len(y), len(yv)
    d = ((Xv[:, None, :] - X[None, :, :]) ** 2).sum(-1)          # V x n
    m = (yv[:, None] == y[None, :]).astype(float)                # V x n
    phi = np.zeros(n)
    for _ in range(perms):
        topd = np.full((V, k), np.inf)
        topm = np.zeros((V, k))
        u_old = 0.0
        for j in rng.permutation(n):
            cd = np.concatenate([topd, d[:, [j]]], axis=1)
            cm = np.concatenate([topm, m[:, [j]]], axis=1)
            idx = np.argsort(cd, axis=1, kind="stable")[:, :k]
            topd = np.take_along_axis(cd, idx, 1)
            topm = np.take_along_axis(cm, idx, 1)
            u_new = (topm * np.isfinite(topd)).sum(1).mean() / k
            phi[j] += u_new - u_old
            u_old = u_new
    return phi / perms


rows = []
for name, X, y, lmask in datasets():
    tr, va = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=0)
    sub = tr[:60]                                              # small n so Monte Carlo converges
    ours = -base.score_knn_shapley(X[sub], y[sub], X[va[:30]], y[va[:30]], k=5)
    mc = mc_knn_shapley(X[sub], y[sub], X[va[:30]], y[va[:30]], k=5, perms=3000, rng=RNG)
    rho = spearmanr(ours, mc).statistic
    rows.append(dict(check="1 KNN-Shapley vs Monte-Carlo Shapley", dataset=name, n=len(sub),
                     spearman=rho, max_abs_diff=np.abs(ours - mc).max(),
                     passed=rho >= 0.90))
    print(f"[1] {name}: spearman={rho:.3f}")

# ------------------------------------------------------------------ 2. Data-OOB vs sklearn Bagging
def bagging_oob_value(X, y, n_trees, seed):
    bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=n_trees, bootstrap=True,
                            random_state=seed).fit(X, y)
    n = len(y)
    correct, seen = np.zeros(n), np.zeros(n)
    for est, idx in zip(bag.estimators_, bag.estimators_samples_):
        mask = np.ones(n, bool)
        mask[idx] = False
        if mask.any():
            # sub-estimators of BaggingClassifier are trained on encoded labels 0..K-1
            pred = bag.classes_[est.predict(X[mask]).astype(int)]
            correct[mask] += pred == y[mask]
            seen[mask] += 1
    return np.divide(correct, seen, out=np.full(n, 0.5), where=seen > 0)


for name, X, y, lmask in datasets():
    ours = 1.0 - base.score_data_oob(X, y, seed=0, n_trees=500)
    ref = bagging_oob_value(X, y, 500, seed=1)
    rho = spearmanr(ours, ref).statistic
    rows.append(dict(check="2 Data-OOB vs sklearn BaggingClassifier OOB", dataset=name, n=len(y),
                     spearman=rho, auroc_ours=roc_auc_score(lmask, -ours),
                     auroc_ref=roc_auc_score(lmask, -ref), passed=rho >= 0.90))
    print(f"[2] {name}: spearman={rho:.3f}")

# ------------------------------------------------------------------ 3-4. cleanlab
try:
    import cleanlab
    from cleanlab.count import num_label_issues
    from cleanlab.rank import get_label_quality_scores
    have_cleanlab = True
except ImportError:
    have_cleanlab = False
    lines.append("cleanlab is not installed: checks 3 and 4 skipped (`pip install cleanlab`).\n")

if have_cleanlab:
    for name, X, y, lmask in datasets():
        proba = v3.oof_proba(X, y, seed=0)
        ours = 1.0 - proba[np.arange(len(y)), y]
        ref = 1.0 - get_label_quality_scores(y, proba, method="self_confidence")
        diff = np.abs(ours - ref).max()
        rows.append(dict(check="3 Self-confidence vs cleanlab", dataset=name, n=len(y),
                         max_abs_diff=diff, passed=diff < 1e-9))
        est_ours = v3.estimate_label_noise(y, proba)
        est_ref = num_label_issues(y, proba, estimation_method="off_diagonal_calibrated") / len(y)
        rows.append(dict(check="4 Noise-rate estimate vs cleanlab", dataset=name, n=len(y),
                         ours=est_ours, reference=est_ref, true_rate=lmask.mean(),
                         abs_diff=abs(est_ours - est_ref), passed=abs(est_ours - est_ref) <= 0.02))
        print(f"[3/4] {name}: selfconf diff={diff:.2e}, noise ours={est_ours:.3f} ref={est_ref:.3f}")
    lines.append(f"cleanlab version: {cleanlab.__version__}\n")

# ------------------------------------------------------------------ 5. OpenDataVal (optional)
try:
    from opendataval.dataloader import DataFetcher
    from opendataval.dataval import DataOob, KNNShapley
    from opendataval.model import ModelFactory
    for name, X, y, lmask in datasets():
        tr, va = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=0)
        fetcher = DataFetcher.from_data_splits(X[tr], y[tr], X[va], y[va], X[va], y[va], one_hot=True)
        model = ModelFactory("sklogreg", fetcher)
        knn_ref = np.asarray(KNNShapley(k_neighbors=10).train(fetcher=fetcher, pred_model=model).evaluate_data_values())
        knn_ours = -base.score_knn_shapley(X[tr], y[tr], X[va], y[va], k=10)
        rows.append(dict(check="5a KNN-Shapley vs OpenDataVal", dataset=name, n=len(tr),
                         spearman=spearmanr(knn_ours, knn_ref).statistic,
                         passed=spearmanr(knn_ours, knn_ref).statistic >= 0.90))
        oob_ref = np.asarray(DataOob(num_models=500).train(fetcher=fetcher, pred_model=ModelFactory("skdecisiontree", fetcher)).evaluate_data_values())
        oob_ours = 1.0 - base.score_data_oob(X[tr], y[tr], seed=0, n_trees=500)
        rows.append(dict(check="5b Data-OOB vs OpenDataVal", dataset=name, n=len(tr),
                         spearman=spearmanr(oob_ours, oob_ref).statistic,
                         passed=spearmanr(oob_ours, oob_ref).statistic >= 0.90))
        print(f"[5] {name}: done")
except ImportError:
    lines.append("OpenDataVal is not installed: optional check 5 skipped (`pip install opendataval`).\n")
except Exception as e:
    lines.append(f"OpenDataVal is installed but its API differs from the one tried here, so check 5 "
                 f"was not run: `{type(e).__name__}: {e}`. Checks 1-2 already validate both scorers "
                 "against independent implementations.\n")

R = pd.DataFrame(rows)
summary = R.groupby("check")["passed"].agg(["sum", "count"])
summary["result"] = np.where(summary["sum"] == summary["count"], "PASS", "FAIL")
lines += ["## Summary\n", summary.rename(columns={"sum": "passed", "count": "datasets"}).to_markdown(), "",
          "## Details\n", R.round(4).to_markdown(index=False), ""]
open("validation_report.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
print("\nSaved: validation_report.md")
