"""
Feature selection for zero-shot preprocessing.

Chooses between no reduction, variance thresholding, and supervised feature
selection based on dataset characteristics (n/p ratio, sparsity, EPV).
Rules are calibrated for XGBoost and tree-based downstream models.

All returned transformers are skl2onnx-compatible (pure feature masks,
no matrix decomposition).
"""

import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.pipeline import Pipeline

from .inspector import PreprocessingProfile


def select_reducer(profile: PreprocessingProfile) -> str:
    """
    Choose a reducer name based on dataset characteristics.

    Decision tree (calibrated for XGBoost; Grinsztajn et al. 2022):

    p ≤ 50
        → "variance_threshold"  (too few features to meaningfully reduce)

    is_sparse_counts and p > 200
        → "select_k_mutual_info"
          MI is theoretically optimal for binary/count features.
          k = max(p//2, 2√n_eff), capped at 10×n_eff, floored at min_features.
          Rogers & Hahn (2010); Guyon & Elisseeff (2003).

    n/p ≥ 20
        → "variance_threshold"  (XGBoost internals sufficient; Harrell 2001)

    n/p ≥ 1
        → "select_80"  (light 20% drop; XGBoost handles this regime internally)

    n/p < 1  (p > n: overfitting risk even with tree regularization)
        sparsity > 0.5  → "select_k_mutual_info"
        else            → "select_60"  (keep 60%; conservative for p > n dense data)

    All selectors enforce min_features = max(50, n_eff // 2, p // 3) as a floor.

    Parameters
    ----------
    profile : PreprocessingProfile

    Returns
    -------
    str
    """
    p = profile.n_features
    ratio = profile.n_p_ratio

    if p <= 50:
        return "variance_threshold"

    if profile.is_sparse_counts and p > 200:
        return "select_k_mutual_info"

    if ratio >= 20:
        return "variance_threshold"

    if ratio >= 1:
        return "select_80"

    # Severely underdetermined (p > n): XGBoost overfits to noise features.
    if profile.sparsity > 0.5:
        return "select_k_mutual_info"
    return "select_60"


def _score_func(task: str, mutual_info: bool = False):
    """Return the appropriate sklearn score function."""
    if mutual_info:
        return (
            mutual_info_classif
            if task == "binary_classification"
            else mutual_info_regression
        )
    return f_classif if task == "binary_classification" else f_regression


def build_reducer(reducer_name: str, profile: PreprocessingProfile):
    """
    Instantiate and return the sklearn transformer for the given reducer name.

    All reducers start with a VarianceThreshold(1e-6) sub-step to remove
    any remaining constant features before the main reduction step.

    Parameters
    ----------
    reducer_name : str
    profile : PreprocessingProfile
        Used to compute k / n_components safely.

    Returns
    -------
    sklearn transformer (Pipeline or single step)
    """
    p = profile.n_features
    n = profile.n_samples
    task = profile.task

    # Effective sample size: for classification use minority class count (EPV rule,
    # Peduzzi et al. 1996); for regression use total n.
    n_eff = (
        profile.n_minority_class
        if task == "binary_classification" and profile.n_minority_class > 0
        else n
    )

    # Minimum features floor: max(50, n_eff // 2, p // 3).
    # - 50: absolute floor (EPV literature).
    # - n_eff // 2: need at least half the effective samples worth of features.
    # - p // 3: never reduce original feature set by more than 67%.
    min_features = max(50, n_eff // 2, p // 3)
    min_features = min(min_features, p)  # cannot exceed original feature count

    vt = VarianceThreshold(threshold=1e-6)

    if reducer_name == "variance_threshold":
        return vt

    if reducer_name == "select_k_mutual_info":
        # Target: max(p // 2, 2√n_eff); ceiling at 10×n_eff (10-EPV cap).
        # Rogers & Hahn (2010); Guyon & Elisseeff (2003).
        k = max(p // 2, int(np.sqrt(n_eff) * 2))
        k = min(k, 10 * n_eff, p)
        k = max(k, min_features)
        score_fn = _score_func(task, mutual_info=True)
        return Pipeline([
            ("vt", VarianceThreshold(threshold=1e-6)),
            ("select", SelectKBest(score_fn, k=k)),
        ])

    if reducer_name == "select_80":
        k = max(min_features, int(0.80 * p))
        k = min(k, p)
        return Pipeline([
            ("vt", VarianceThreshold(threshold=1e-6)),
            ("select", SelectKBest(_score_func(task), k=k)),
        ])

    if reducer_name == "select_60":
        k = max(min_features, int(0.60 * p))
        k = min(k, p)
        return Pipeline([
            ("vt", VarianceThreshold(threshold=1e-6)),
            ("select", SelectKBest(_score_func(task), k=k)),
        ])

    raise ValueError(
        f"Unknown reducer {reducer_name!r}. "
        f"Valid options: variance_threshold, select_k_mutual_info, select_80, select_60"
    )
