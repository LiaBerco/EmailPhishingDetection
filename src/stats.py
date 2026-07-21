"""Evaluation metrics + the statistical machinery that makes comparisons honest.

  * metrics()      - the usual battery (accuracy, precision, recall, F1, MCC, AUC).
  * bootstrap_ci() - 95% confidence interval for any metric, by resampling the
                     test set. Stops us reporting a single point estimate as if
                     it were exact.
  * mcnemar()      - paired test for "is model A really better than model B on the
                     same test set, or is it noise?".
"""
from __future__ import annotations
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, matthews_corrcoef,
)


def metrics(y_true, prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    y_hat = (prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_hat),
        "precision": precision_score(y_true, y_hat, zero_division=0),
        "recall": recall_score(y_true, y_hat, zero_division=0),
        "f1": f1_score(y_true, y_hat, zero_division=0),
        # F2 weights recall above precision (beta=2) -- the right emphasis for
        # phishing, where a miss (FN) is costlier than a false alarm (FP).
        "f2": fbeta_score(y_true, y_hat, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_hat),
    }
    # AUC needs both classes present and probabilistic scores.
    try:
        out["auc"] = roc_auc_score(y_true, prob)
    except ValueError:
        out["auc"] = float("nan")
    return out


def bootstrap_ci(y_true, prob, metric="f1", threshold=0.5, n=1000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for one metric. Returns (point, lo, hi)."""
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y_true))
    point = metrics(y_true, prob, threshold)[metric]
    vals = []
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        # skip degenerate resamples with a single class (undefined metrics)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metrics(y_true[idx], prob[idx], threshold)[metric])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test comparing two classifiers' errors on the same set.

    Returns (b, c, p_value) where b = A wrong / B right, c = A right / B wrong.
    A small p-value means the two models disagree in a way unlikely to be chance.
    """
    y_true = np.asarray(y_true)
    a_wrong = pred_a != y_true
    b_wrong = pred_b != y_true
    b = int(np.sum(a_wrong & ~b_wrong))   # A wrong, B right
    c = int(np.sum(~a_wrong & b_wrong))   # A right, B wrong
    # exact binomial test on the discordant pairs
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return b, c, p


def cost_optimal_threshold(y_true, prob, cost_fn=5.0, cost_fp=1.0, grid=None):
    """Pick the probability threshold that minimises expected cost on val data.

    FN (missed phishing) is weighted more heavily than FP (false alarm).
    """
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_cost = 0.5, float("inf")
    for t in grid:
        y_hat = (prob >= t).astype(int)
        fn = int(np.sum((y_true == 1) & (y_hat == 0)))
        fp = int(np.sum((y_true == 0) & (y_hat == 1)))
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t
