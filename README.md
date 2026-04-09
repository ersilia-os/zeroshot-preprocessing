**THIS REPOSITORY HAS BEEN ARCHIVED AND INCORPORATED INTO [LAZY-QSAR](https://github.com/ersilia-os/lazy-qsar)**

# Zero-shot preprocessing for Ersilia's ML tasks

Automatic preprocessing pipeline selection for scikit-learn. Picks a scaler and feature reducer from a single pass over your dataset — no cross-validation, no grid search.

## Installation

```bash
pip install -e .
```

## Usage

### Classification

```python
from zspreprocessing import ZeroShotClassifierPreprocessor

pre = ZeroShotClassifierPreprocessor()
X_train_t = pre.fit_transform(X_train, y_train)
X_test_t  = pre.transform(X_test)

print(pre.scaler_name_)   # e.g. "standard"
print(pre.reducer_name_)  # e.g. "correlation_filter"
print(pre.n_features_in_, "→", pre.n_features_out_)
```

### Regression

```python
from zspreprocessing import ZeroShotRegressorPreprocessor

pre = ZeroShotRegressorPreprocessor()
X_train_t = pre.fit_transform(X_train, y_train)
X_test_t  = pre.transform(X_test)
```

### Save to disk

```python
pre.save("my_model/")
# writes:
#   my_model/preprocessor.onnx
#   my_model/preprocessor.json
```

`preprocessor.json` contains the selected operations and the indices of the original columns that survive the pipeline.

### Load and run inference

```python
from zspreprocessing import PreprocessorArtifact

artifact = PreprocessorArtifact.load("my_model/")
X_out = artifact.run(X_test)
```

`PreprocessorArtifact` has no fit method — it is inference-only and requires only `onnxruntime` at serving time.

## How it works

`fit()` profiles the dataset in a single pass, then applies deterministic rules to select a scaler and a feature reducer. The fitted pipeline is always:

```
SimpleImputer(strategy="median")
  → VarianceThreshold(1e-6)
  → <selected scaler>
  → <selected reducer>
```

**Scaler rules** (first match wins):

| Condition | Scaler |
|---|---|
| Sparse fingerprint/count data | `MaxAbsScaler` |
| ≥80% binary features | `MaxAbsScaler` |
| Sparsity > 50% | `MaxAbsScaler` |
| Outlier fraction > 30% | `RobustScaler` |
| Median skewness > 1.5 | `PowerTransformer` (Yeo-Johnson) |
| Default | `StandardScaler` |

**Reducer rules:**

| Condition | Reducer |
|---|---|
| p ≤ 50 or n/p ≥ 20 | `VarianceThreshold` — no further reduction needed |
| Otherwise | `CorrelationFilter` — drops features with \|r\| > 0.90 with another feature |

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
