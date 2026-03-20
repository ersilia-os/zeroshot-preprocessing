# Zero-shot preprocessing for Ersilia's ML tasks

Zero-shot preprocessing pipeline selection for scikit-learn. Automatically picks a scaler and dimensionality reducer from a single pass over your dataset — no cross-validation, no grid search.

## Installation

```bash
pip install -e .
# Optional: ONNX export support
pip install -e ".[onnx]"
```

## Quick start

```python
from zspreprocessing import ZeroShotClassifierPreprocessor, ZeroShotRegressorPreprocessor

# Classification
pre = ZeroShotClassifierPreprocessor(verbose=True)
X_train_t = pre.fit_transform(X_train, y_train)
X_test_t  = pre.transform(X_test)

# Regression
pre = ZeroShotRegressorPreprocessor(verbose=True)
X_train_t = pre.fit_transform(X_train, y_train)

print(pre.scaler_name_)   # e.g. "standard"
print(pre.reducer_name_)  # e.g. "variance_threshold"
```

The base class `ZeroShotPreprocessor(task="auto")` is also available when the task should be inferred from `y`.

## How it works

`fit()` runs in three stages:

```
1. Inspect  →  compute a lightweight dataset profile
2. Select   →  choose scaler and reducer from the profile via rule-based heuristics
3. Build    →  assemble and fit a fixed sklearn Pipeline
```

The fitted pipeline is always:

```
SimpleImputer(strategy="median")
  → VarianceThreshold(1e-6)
  → <selected scaler>
  → <selected reducer>
```

### Stage 1 — Dataset profiling (`inspector.py`)

A `PreprocessingProfile` is computed from a single pass (subsampled where noted):

| Field | Description |
|---|---|
| `n_samples`, `n_features`, `n_p_ratio` | Basic shape and sample-to-feature ratio |
| `sparsity` | Fraction of zero entries in X |
| `is_sparse_counts` | True if X looks like fingerprint/count data (sparse + integer-valued + small max) |
| `binary_feature_fraction` | Fraction of features with only {0, 1} values |
| `median_feature_skewness` | Median absolute skewness across features (subsampled) |
| `outlier_fraction` | Fraction of features where >5% of values fall outside 1.5×IQR |
| `near_zero_variance_fraction` | Fraction of features with variance < 1e-6 |
| `median_abs_correlation` | Median absolute Pearson correlation across random feature pairs |
| `feature_signal_p90` | 90th-percentile absolute correlation of features with the target |
| `n_minority_class` | Minority class count for binary classification (0 for regression) |
| `y_skewness`, `y_all_positive` | Target distribution stats (regression only) |

### Stage 2 — Scaler selection (`scaler.py`)

Rules are evaluated in order; first match wins:

| Condition | Scaler | Rationale |
|---|---|---|
| `is_sparse_counts` | `MaxAbsScaler` | Preserves sparsity; bits/counts are already on a bounded scale |
| `binary_feature_fraction ≥ 0.8` | `MaxAbsScaler` | Effectively a no-op on {0,1} data; avoids densifying sparse matrices |
| `sparsity > 0.5` | `MaxAbsScaler` | Checked before outlier/power rules to avoid dense operations on sparse data |
| `outlier_fraction > 0.3` | `RobustScaler` | Median+IQR centering is resistant to extreme values |
| `median_feature_skewness > 1.5` | `PowerTransformer` (Yeo-Johnson) | Reduces heavy skew; handles negative values |
| *(default)* | `StandardScaler` | Zero mean, unit variance; appropriate for approximately normal features |

### Stage 3 — Reducer selection (`reducer.py`)

Rules are calibrated for XGBoost and tree-based models (Grinsztajn et al. 2022, TabZilla 2023). All reducers are pure feature selectors — no matrix decomposition — keeping the ONNX graph compact.

```
p ≤ 50
    → variance_threshold           (too few features to meaningfully reduce)

is_sparse_counts and p > 200
    → select_k_mutual_info         (MI optimal for binary features; Rogers & Hahn 2010,
                                    Guyon & Elisseeff 2003)

n/p ≥ 20
    → variance_threshold           (XGBoost internals sufficient; Harrell 2001)

n/p ≥ 1
    → select_80                    (light 20% drop; XGBoost handles this regime internally)

n/p < 1  (p > n: overfitting risk even with tree regularization)
    sparsity > 0.5  → select_k_mutual_info
    else            → select_60    (keep 60%; conservative reduction for p > n dense data)
```

**EPV adjustment**: for binary classification, `n_eff = min(n_positives, n_negatives)` (minority class count) is used instead of `n_samples` when computing `k`. This reflects the EPV (Events Per Variable) rule — the number of features you can reliably learn from is bounded by the minority class, not total samples (Peduzzi et al. 1996). For regression, `n_eff = n_samples`.

**Feature floor**: to avoid over-aggressive reduction, all selectors enforce:

```
min_features = max(50, n_eff // 2, p // 3)
```

This guarantees at least 50 features, at least half the effective sample count, and never more than a 67% reduction from the original feature count.

**`select_k_mutual_info` k formula**:

```
k = max(p // 2, 2 * sqrt(n_eff))   # keep at least half; 2√n_eff is FDR-motivated
k = min(k, 10 * n_eff)              # 10-EPV ceiling for very imbalanced data
k = max(k, min_features)            # floor
```

Every selector prepends a `VarianceThreshold(1e-6)` sub-step to remove any remaining constant features before selection.

### Casistic summary

| Dataset type | Typical scaler | Typical reducer |
|---|---|---|
| Dense, low-dimensional (p ≤ 50) | `standard` | `variance_threshold` |
| Dense, well-determined (n/p ≥ 20) | `standard` or `robust` | `variance_threshold` |
| Dense, n/p ≥ 1 | `standard` | `select_80` |
| Dense, p > n | `standard` or `power` | `select_60` |
| Sparse / fingerprint (ECFP, Morgan) | `max_abs` | `select_k_mutual_info` |
| Sparse, p > n | `max_abs` | `select_k_mutual_info` |
| Heavy outliers | `robust` | *(ratio-dependent)* |
| Heavy skew, dense | `power` | *(ratio-dependent)* |

## Low-level API

```python
from zspreprocessing import inspect

profile = inspect(X, y, task="binary_classification")
print(profile)
```

`inspect()` returns a `PreprocessingProfile` without building any pipeline. Useful for exploring dataset characteristics or building custom logic on top.

## ONNX export

```python
pre.to_onnx("preprocessor.onnx")

import onnxruntime as rt
sess = rt.InferenceSession("preprocessor.onnx")
X_out = sess.run(None, {"float_input": X_test.astype("float32")})[0]
```

Requires `pip install zspreprocessing[onnx]`. All reducers are skl2onnx-compatible.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
