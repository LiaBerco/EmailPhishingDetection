"""Explainability for the linear model.

For a linear classifier over TF-IDF features the signed coefficient of a token IS
its contribution, so we can read the model directly:

  * top_features()  - the words that push hardest towards phishing / legitimate.
  * explain_email() - for a single email, the tokens that drove *this* decision.

A caveat we make explicit in the report (RQ4): the character n-gram features that
give robustness are not human-readable, so explanations are drawn from the *word*
part of the feature space. Robustness and interpretability pull in opposite
directions.
"""
from __future__ import annotations
import numpy as np


def linear_weights(clf):
    """Signed per-feature weights for a linear model (LogReg / linear SVM)."""
    return clf.coef_[0]


def top_features(feature_names, weights, k=15):
    """Return (top_phishing, top_legit) as lists of (token, weight)."""
    order = np.argsort(weights)
    top_legit = [(feature_names[i], float(weights[i])) for i in order[:k]]
    top_phish = [(feature_names[i], float(weights[i])) for i in order[-k:][::-1]]
    return top_phish, top_legit


def explain_email(text, analyzer, weight_map, k=5):
    """Human-readable reason: the k strongest phishing tokens present in `text`."""
    toks = {t for t in analyzer(text) if weight_map.get(t, 0) > 0}
    toks = sorted(toks, key=lambda t: weight_map[t], reverse=True)[:k]
    if not toks:
        return "(weak text signal)"
    return "flagged for: " + ", ".join(toks)


def _strip_prefix(name):
    """Drop the FeatureUnion 'word__' prefix so tokens read cleanly."""
    return name.split("__", 1)[-1]


def instance_contributions(text, word_vectorizer, clf, k=6):
    """Exact per-token contributions for ONE email (a faithful local explanation).

    For a linear model the log-odds is a sum over features:
        logit(x) = bias + sum_i  coef_i * tfidf_i(x)
    so ``coef_i * tfidf_i(x)`` is the *exact* signed contribution of token i to
    THIS prediction (positive -> pushes towards phishing, negative -> towards
    legit). Unlike a keyword list this is not a heuristic: it is the model's own
    arithmetic, and it is exact rather than an approximation like LIME/SHAP would
    give for a non-linear model.

    Returns (top_phishing, top_legit), each a list of (token, contribution).
    """
    X = word_vectorizer.transform([str(text)])
    coef = clf.coef_[0]
    names = np.asarray(word_vectorizer.get_feature_names_out())
    cc = X.multiply(coef).tocoo()
    pairs = sorted(zip(names[cc.col], cc.data), key=lambda p: p[1])
    top_phish = [(_strip_prefix(t), float(v)) for t, v in pairs[-k:][::-1] if v > 0]
    top_legit = [(_strip_prefix(t), float(v)) for t, v in pairs[:k] if v < 0]
    return top_phish, top_legit
