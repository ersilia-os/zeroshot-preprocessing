"""
zspreprocessing — Zero-shot scikit-learn preprocessing pipeline selection.

Automatically selects imputation, scaling, and dimensionality reduction
strategies for binary classification and regression tasks based on
lightweight dataset profiling — no hyperparameter search required.

Public API
----------
inspect(X, y, task=None) -> PreprocessingProfile
    Profile a dataset and return its preprocessing characteristics.

ZeroShotPreprocessor(task="auto", verbose=False)
    Sklearn-compatible transformer with automatically selected preprocessing.
    Exports to ONNX via ``to_onnx(path)``.

ZeroShotClassifierPreprocessor(verbose=False)
    Convenience subclass fixed to binary_classification task.

ZeroShotRegressorPreprocessor(verbose=False)
    Convenience subclass fixed to regression task.

Typical usage
-------------
    from zspreprocessing import ZeroShotClassifierPreprocessor

    pre = ZeroShotClassifierPreprocessor(verbose=True)
    X_train_t = pre.fit_transform(X_train, y_train)
    X_test_t  = pre.transform(X_test)
    pre.to_onnx("preprocessor.onnx")

Low-level usage
---------------
    from zspreprocessing import inspect
    from zspreprocessing import select_scaler, select_reducer

    profile = inspect(X, y)
    print(select_scaler(profile), select_reducer(profile))
"""

from .inspector import PreprocessingProfile, inspect
from .pipeline import (
    ZeroShotPreprocessor,
    ZeroShotClassifierPreprocessor,
    ZeroShotRegressorPreprocessor,
)
from .reducer import build_reducer, select_reducer
from .scaler import build_scaler, select_scaler

__all__ = [
    "inspect",
    "PreprocessingProfile",
    "ZeroShotPreprocessor",
    "ZeroShotClassifierPreprocessor",
    "ZeroShotRegressorPreprocessor",
    "select_scaler",
    "build_scaler",
    "select_reducer",
    "build_reducer",
]
