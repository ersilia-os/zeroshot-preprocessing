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
print(pre.n_features_in_, "→", pre.n_features_out_)
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

Profile computation subsamples to at most 5 000 rows and 500 features to scale to large datasets.

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

> **Fallback**: if `PowerTransformer` raises a scipy bracketing error during `fit`, the scaler is automatically downgraded to `RobustScaler`.

### Stage 3 — Reducer selection (`reducer.py`)

All reducers are pure feature selectors — no matrix decomposition — keeping the pipeline interpretable and the ONNX graph compact. Rules are calibrated for XGBoost and tree-based models.

| Condition | Reducer |
|---|---|
| `p ≤ 50` or `n/p ≥ 20` | `variance_threshold` — dataset is small enough that no further reduction is needed |
| *(everything else)* | `correlation_filter` — removes redundant features by pairwise Pearson correlation |

**`variance_threshold`**: `VarianceThreshold(1e-6)` removes any remaining constant features (those with near-zero variance after imputation and scaling).

**`correlation_filter`**: a sequential `[VarianceThreshold(1e-6) → CorrelationFilter(threshold=0.90)]` pipeline. `CorrelationFilter` iterates over features and removes any feature whose absolute Pearson correlation with a previously kept feature exceeds 0.90; when two features are correlated, the one with lower variance is dropped.

### Casistic summary

| Dataset type | Typical scaler | Typical reducer |
|---|---|---|
| Dense, low-dimensional (p ≤ 50) | `standard` | `variance_threshold` |
| Dense, well-determined (n/p ≥ 20) | `standard` or `robust` | `variance_threshold` |
| Dense, underdetermined (p > n) | `standard` or `power` | `correlation_filter` |
| Sparse / fingerprint (ECFP, Morgan) | `max_abs` | `variance_threshold` or `correlation_filter` |
| Heavy outliers | `robust` | *(ratio-dependent)* |
| Heavy skew, dense | `power` | *(ratio-dependent)* |

## Low-level API

```python
from zspreprocessing import inspect, select_scaler, select_reducer, build_scaler, build_reducer

profile = inspect(X, y, task="binary_classification")
print(profile)

scaler_name  = select_scaler(profile)   # e.g. "standard"
reducer_name = select_reducer(profile)  # e.g. "correlation_filter"

scaler  = build_scaler(scaler_name)
reducer = build_reducer(reducer_name)
```

`inspect()` returns a `PreprocessingProfile` without building any pipeline. Useful for exploring dataset characteristics or building custom selection logic on top.

## Command-line interface

```bash
python -m zspreprocessing data.csv --target <column> [--task auto|binary_classification|regression]
```

Reads a CSV file, profiles the dataset, and prints the `PreprocessingProfile` together with the recommended scaler and reducer name. No pipeline is built or fitted.

## ONNX export

```python
pre.to_onnx("preprocessor.onnx")

import onnxruntime as rt
sess = rt.InferenceSession("preprocessor.onnx")
X_out = sess.run(None, {"float_input": X_test.astype("float32")})[0]
```

Requires `pip install zspreprocessing[onnx]`. The entire pipeline — including `CorrelationFilter` — is skl2onnx-compatible (target opset 15).

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
