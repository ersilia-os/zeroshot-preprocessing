"""
ZeroShotPreprocessor — sklearn-compatible transformer that automatically
selects imputation, scaling, and dimensionality reduction strategies based
on dataset characteristics.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .inspector import PreprocessingProfile, inspect
from .reducer import build_reducer, select_reducer
from .scaler import build_scaler, select_scaler
from .utils.logging import logger


class ZeroShotPreprocessor(BaseEstimator, TransformerMixin):
    """
    Automatically selects and fits a preprocessing pipeline for binary
    classification and regression tasks.

    The pipeline always consists of:

        SimpleImputer(strategy="median")   — handles missing values (no-op if none)
        VarianceThreshold(1e-6)            — removes constant features before scaling
        <selected scaler>                  — chosen from dataset profile
        <selected reducer>                 — chosen from dataset profile

    Scaler and reducer are selected using lightweight, rule-based heuristics
    derived from dataset shape, sparsity, distribution, and redundancy.

    The fitted pipeline is fully exportable to ONNX via ``to_onnx()``.

    Parameters
    ----------
    task : str, default "auto"
        "binary_classification", "regression", or "auto" (detected from y).
    verbose : bool, default False
        Whether to emit Rich/loguru logging during fit.

    Attributes
    ----------
    profile_ : PreprocessingProfile
        Dataset profile computed during ``fit``.
    scaler_name_ : str
        Name of the selected scaler.
    reducer_name_ : str
        Name of the selected reducer.
    pipeline_ : sklearn.pipeline.Pipeline
        The fitted preprocessing pipeline.
    n_features_in_ : int
        Number of input features seen during ``fit``.
    n_features_out_ : int
        Number of output features after transformation.
    """

    def __init__(self, task: str = "auto", verbose: bool = False):
        self.task = task
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "ZeroShotPreprocessor":
        """
        Profile the dataset, select and fit the preprocessing pipeline.

        Parameters
        ----------
        X : array-like or scipy sparse, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        logger.set_verbosity(self.verbose)
        logger.rule("ZeroShotPreprocessor")

        y = np.asarray(y).ravel()
        resolved_task = None if self.task == "auto" else self.task

        # -- Profile --
        self.profile_: PreprocessingProfile = inspect(X, y, task=resolved_task)
        logger.profile_summary(self.profile_)

        # -- Select strategies --
        self.scaler_name_: str = select_scaler(self.profile_)
        self.reducer_name_: str = select_reducer(self.profile_)
        logger.info(
            f"scaler={self.scaler_name_} | reducer={self.reducer_name_}"
        )

        # -- Build pipeline --
        scaler = build_scaler(self.scaler_name_)
        reducer = build_reducer(self.reducer_name_, self.profile_)

        self.pipeline_ = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("vt0",     VarianceThreshold(threshold=1e-6)),
            ("scaler",  scaler),
            ("reducer", reducer),
        ])

        self.n_features_in_: int = X.shape[1]

        # -- Fit (pass y so supervised selectors receive labels) --
        try:
            self.pipeline_.fit(X, y)
        except Exception as exc:
            # PowerTransformer (Yeo-Johnson) can fail with a BracketError when
            # scipy cannot bracket a minimum for some feature distributions.
            # Fall back to RobustScaler, which handles skewed data without
            # requiring numerical optimization per column.
            if self.scaler_name_ == "power":
                logger.warning(
                    f"PowerTransformer fit failed ({exc!r}); "
                    "falling back to RobustScaler."
                )
                self.scaler_name_ = "robust"
                scaler = build_scaler("robust")
                self.pipeline_ = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("vt0",     VarianceThreshold(threshold=1e-6)),
                    ("scaler",  scaler),
                    ("reducer", reducer),
                ])
                self.pipeline_.fit(X, y)
            else:
                raise

        # Cache output shape using a single cheap transform on a zero row
        self._n_features_out: int = self.pipeline_.transform(
            np.zeros((1, self.n_features_in_))
        ).shape[1]

        logger.success(
            f"scaler={self.scaler_name_} | reducer={self.reducer_name_} | "
            f"{self.n_features_in_} → {self._n_features_out} features"
        )
        return self

    def transform(self, X) -> np.ndarray:
        """
        Apply the fitted preprocessing pipeline.

        Parameters
        ----------
        X : array-like or scipy sparse, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features_out_)
        """
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.transform(X)

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step, passing y to supervised selectors."""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_features_out_(self) -> int:
        check_is_fitted(self, "_n_features_out")
        return self._n_features_out

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def to_onnx(self, path: str) -> None:
        """
        Export the fitted pipeline to an ONNX file.

        Requires the optional ``onnx`` dependencies::

            pip install zspreprocessing[onnx]

        Parameters
        ----------
        path : str
            Destination file path (e.g. "preprocessor.onnx").
        """
        check_is_fitted(self, "pipeline_")
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as exc:
            raise ImportError(
                "ONNX export requires optional dependencies. "
                "Install with:  pip install zspreprocessing[onnx]"
            ) from exc

        initial_type = [
            ("float_input", FloatTensorType([None, self.n_features_in_]))
        ]
        onnx_model = convert_sklearn(self.pipeline_, initial_types=initial_type, target_opset=15)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())


class ZeroShotClassifierPreprocessor(ZeroShotPreprocessor):
    """ZeroShotPreprocessor fixed to binary_classification task."""

    def __init__(self, verbose: bool = False):
        super().__init__(task="binary_classification", verbose=verbose)


class ZeroShotRegressorPreprocessor(ZeroShotPreprocessor):
    """ZeroShotPreprocessor fixed to regression task."""

    def __init__(self, verbose: bool = False):
        super().__init__(task="regression", verbose=verbose)
