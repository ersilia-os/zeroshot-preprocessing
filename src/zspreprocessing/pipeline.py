"""
ZeroShotPreprocessor — sklearn-compatible transformer that automatically
selects imputation, scaling, and dimensionality reduction strategies based
on dataset characteristics.
"""

import json
import os
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
    Automatically selects and fits a preprocessing pipeline for
    classification and regression tasks.

    The pipeline always consists of:

        SimpleImputer(strategy="median", keep_empty_features=True)   — handles missing values (no-op if none)
        VarianceThreshold(1e-6)            — removes constant features before scaling
        <selected scaler>                  — chosen from dataset profile
        <selected reducer>                 — chosen from dataset profile

    Scaler and reducer are selected using lightweight, rule-based heuristics
    derived from dataset shape, sparsity, distribution, and redundancy.

    The fitted pipeline is fully exportable to ONNX via ``to_onnx()``.

    Parameters
    ----------
    task : str, default "classification"
        "classification" or "regression".
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

    def __init__(self, task: str = "classification", verbose: bool = False):
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

        # -- Profile --
        self.profile_: PreprocessingProfile = inspect(X, y, task=self.task)
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
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
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
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
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

        # Compute which original column indices survive the full pipeline
        self.kept_feature_indices_: list = self._compute_kept_indices()

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_kept_indices(self) -> list:
        """
        Return the indices (in the original feature space) of columns that
        survive the full pipeline.

        The pipeline has four steps:
          imputer  → no column change
          vt0      → VarianceThreshold, may drop columns
          scaler   → no column change
          reducer  → VarianceThreshold or CorrelationFilter (inner pipeline),
                     may drop further columns
        """
        # Columns kept after vt0
        vt0_mask = self.pipeline_.named_steps["vt0"].get_support()
        vt0_indices = np.where(vt0_mask)[0]

        reducer = self.pipeline_.named_steps["reducer"]

        if self.reducer_name_ == "variance_threshold":
            # Single VarianceThreshold
            reducer_mask = reducer.get_support()
        else:
            # correlation_filter is a Pipeline: vt → CorrelationFilter
            vt_mask = reducer.named_steps["vt"].get_support()
            cf_mask = reducer.named_steps["select"].mask_
            # Compose: first vt removes some, then cf removes from the remainder
            vt_indices = np.where(vt_mask)[0]
            cf_indices = np.where(cf_mask)[0]
            reducer_mask = np.zeros(len(vt0_indices), dtype=bool)
            reducer_mask[vt_indices[cf_indices]] = True

        kept = vt0_indices[reducer_mask]
        return kept.tolist()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_features_out_(self) -> int:
        check_is_fitted(self, "_n_features_out")
        return self._n_features_out

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _metadata_dict(self) -> dict:
        """Return a JSON-serialisable dict describing the fitted pipeline."""
        check_is_fitted(self, "pipeline_")
        return {
            "task": self.task,
            "scaler": self.scaler_name_,
            "reducer": self.reducer_name_,
            "n_features_in": self.n_features_in_,
            "n_features_out": self.n_features_out_,
            "kept_feature_indices": self.kept_feature_indices_,
        }

    def save(self, directory: str, onnx: bool = True) -> None:
        """
        Save the fitted pipeline to *directory* as two files:

            preprocessor.onnx   (or preprocessor.joblib)
            preprocessor.json

        Parameters
        ----------
        directory : str
            Destination directory (created if it does not exist).
        onnx : bool, default True
            If True, serialize the pipeline as ``preprocessor.onnx``.
            If False, serialize with joblib as ``preprocessor.joblib``.
        """
        check_is_fitted(self, "pipeline_")
        os.makedirs(directory, exist_ok=True)
        base = os.path.join(directory, "preprocessor")

        if onnx:
            self.to_onnx(base + ".onnx")
        else:
            import joblib
            joblib.dump(self.pipeline_, base + ".joblib")

        with open(base + ".json", "w") as f:
            json.dump(self._metadata_dict(), f, indent=2)

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
                "ONNX export requires skl2onnx. Install with:  pip install skl2onnx"
            ) from exc

        from .reducer import _register_correlation_filter_onnx_converter
        _register_correlation_filter_onnx_converter()

        initial_type = [
            ("float_input", FloatTensorType([None, self.n_features_in_]))
        ]
        onnx_model = convert_sklearn(self.pipeline_, initial_types=initial_type, target_opset=15)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())


class PreprocessorArtifact:
    """
    A loaded, inference-only preprocessor.

    Usage
    -----
    artifact = PreprocessorArtifact.load("path/to/directory")
    X_out = artifact.run(X)

    Attributes
    ----------
    task : str
    scaler : str
    reducer : str
    n_features_in : int
    n_features_out : int
    kept_feature_indices : list[int]
    """

    @classmethod
    def load(cls, directory: str) -> "PreprocessorArtifact":
        """
        Load a preprocessor saved by ``ZeroShotPreprocessor.save()``.

        Parameters
        ----------
        directory : str
            Folder containing ``preprocessor.json`` and either
            ``preprocessor.onnx`` or ``preprocessor.joblib``.
            ONNX is preferred when both are present.
        """
        self = cls.__new__(cls)

        json_path = os.path.join(directory, "preprocessor.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"No preprocessor.json found in {directory!r}")
        with open(json_path) as f:
            meta = json.load(f)

        self.task: str                  = meta["task"]
        self.scaler: str                = meta["scaler"]
        self.reducer: str               = meta["reducer"]
        self.n_features_in: int         = meta["n_features_in"]
        self.n_features_out: int        = meta["n_features_out"]
        self.kept_feature_indices: list = meta["kept_feature_indices"]

        onnx_path   = os.path.join(directory, "preprocessor.onnx")
        joblib_path = os.path.join(directory, "preprocessor.joblib")

        if os.path.exists(onnx_path):
            import onnxruntime as rt
            self._session    = rt.InferenceSession(onnx_path)
            self._input_name = self._session.get_inputs()[0].name
            self._backend    = "onnx"
        elif os.path.exists(joblib_path):
            import joblib
            self._pipeline = joblib.load(joblib_path)
            self._backend  = "joblib"
        else:
            raise FileNotFoundError(
                f"No preprocessor.onnx or preprocessor.joblib found in {directory!r}"
            )

        return self

    def run(self, X) -> np.ndarray:
        """
        Transform X using the loaded pipeline.

        Parameters
        ----------
        X : array-like or scipy sparse, shape (n_samples, n_features_in)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features_out)
        """
        if self._backend == "onnx":
            if hasattr(X, "toarray"):
                X = X.toarray()
            return self._session.run(None, {self._input_name: np.asarray(X, dtype=np.float32)})[0]
        else:
            return self._pipeline.transform(X)

    def __repr__(self) -> str:
        return (
            f"PreprocessorArtifact("
            f"task={self.task!r}, "
            f"scaler={self.scaler!r}, "
            f"reducer={self.reducer!r}, "
            f"features={self.n_features_in}→{self.n_features_out}, "
            f"backend={self._backend!r})"
        )


class ZeroShotClassifierPreprocessor(ZeroShotPreprocessor):
    """ZeroShotPreprocessor fixed to classification task."""

    def __init__(self, verbose: bool = False):
        super().__init__(task="classification", verbose=verbose)


class ZeroShotRegressorPreprocessor(ZeroShotPreprocessor):
    """ZeroShotPreprocessor fixed to regression task."""

    def __init__(self, verbose: bool = False):
        super().__init__(task="regression", verbose=verbose)
