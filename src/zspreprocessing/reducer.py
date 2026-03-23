"""
Feature selection for zero-shot preprocessing.

Chooses between no reduction and correlation-based feature filtering based on
dataset characteristics (n/p ratio). Uses unsupervised selection only: removes
features that are highly correlated (|r| > 0.90) with another feature.

All returned transformers are skl2onnx-compatible (pure feature masks,
no matrix decomposition).
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection._base import SelectorMixin
from sklearn.pipeline import Pipeline

from .inspector import PreprocessingProfile


class CorrelationFilter(BaseEstimator, SelectorMixin):
    """
    Unsupervised feature selector that removes highly correlated features.

    For each pair of features with |Pearson r| > threshold, the one with
    lower variance is dropped.
    """

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        corr = np.corrcoef(X.T)  # p × p
        variances = X.var(axis=0)
        mask = np.ones(p, dtype=bool)
        for i in range(p):
            if not mask[i]:
                continue
            for j in range(i + 1, p):
                if not mask[j]:
                    continue
                if abs(corr[i, j]) > self.threshold:
                    if variances[i] >= variances[j]:
                        mask[j] = False
                    else:
                        mask[i] = False
                        break
        self.mask_ = mask
        return self

    def _get_support_mask(self):
        return self.mask_


def select_reducer(profile: PreprocessingProfile) -> str:
    """
    Choose a reducer name based on dataset characteristics.

    Decision rule:
      p ≤ 50 or n/p ≥ 20  →  "variance_threshold"  (no reduction needed)
      everything else      →  "correlation_filter"

    Parameters
    ----------
    profile : PreprocessingProfile

    Returns
    -------
    str
    """
    p = profile.n_features
    ratio = profile.n_p_ratio

    if p <= 50 or ratio >= 20:
        return "variance_threshold"

    return "correlation_filter"


def build_reducer(reducer_name: str, profile: PreprocessingProfile):
    """
    Instantiate and return the sklearn transformer for the given reducer name.

    Parameters
    ----------
    reducer_name : str
    profile : PreprocessingProfile

    Returns
    -------
    sklearn transformer (Pipeline or single step)
    """
    vt = VarianceThreshold(threshold=1e-6)

    if reducer_name == "variance_threshold":
        return vt

    if reducer_name == "correlation_filter":
        return Pipeline([
            ("vt", VarianceThreshold(threshold=1e-6)),
            ("select", CorrelationFilter(threshold=0.90)),
        ])

    raise ValueError(
        f"Unknown reducer {reducer_name!r}. "
        f"Valid options: variance_threshold, correlation_filter"
    )
