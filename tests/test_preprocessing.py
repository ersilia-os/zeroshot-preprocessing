"""
Tests for zspreprocessing.

Structure mirrors test_zsxgboost.py:
  TestInspector          — dataset profiling
  TestScalerReducer      — selection rule unit tests
  TestFitTransform       — end-to-end pipeline tests
"""

import importlib

import numpy as np
import pytest
import scipy.sparse as sp

from zspreprocessing import (
    PreprocessingProfile,
    ZeroShotPreprocessor,
    inspect,
    select_reducer,
    select_scaler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_clf_data(n=300, p=20, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = (rng.randn(n) > 0).astype(int)
    return X, y


def make_reg_data(n=300, p=20, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = rng.randn(n)
    return X, y


def make_fingerprint_data(n=200, p=512, seed=0):
    rng = np.random.RandomState(seed)
    X = (rng.rand(n, p) < 0.05).astype(float)
    y = (rng.randn(n) > 0).astype(int)
    return X, y


def make_skewed_data(n=300, p=20, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.exponential(scale=2.0, size=(n, p))
    y = rng.randn(n)
    return X, y


def make_outlier_data(n=300, p=20, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    # Inject outliers into all features
    X[:10] *= 100
    y = (rng.randn(n) > 0).astype(int)
    return X, y


def make_profile(**kwargs) -> PreprocessingProfile:
    """Build a minimal PreprocessingProfile with sensible defaults."""
    defaults = dict(
        n_samples=500,
        n_features=100,
        n_p_ratio=5.0,
        sparsity=0.0,
        is_sparse_counts=False,
        binary_feature_fraction=0.0,
        median_feature_skewness=0.3,
        outlier_fraction=0.1,
        near_zero_variance_fraction=0.0,
        median_abs_correlation=0.2,
        feature_signal_p90=0.2,
        task="binary_classification",
        y_skewness=0.0,
        y_all_positive=False,
    )
    defaults.update(kwargs)
    return PreprocessingProfile(**defaults)


# ---------------------------------------------------------------------------
# TestInspector
# ---------------------------------------------------------------------------

class TestInspector:
    def test_profile_fields_present(self):
        X, y = make_clf_data()
        p = inspect(X, y)
        assert hasattr(p, "n_samples")
        assert hasattr(p, "n_features")
        assert hasattr(p, "n_p_ratio")
        assert hasattr(p, "sparsity")
        assert hasattr(p, "is_sparse_counts")
        assert hasattr(p, "binary_feature_fraction")
        assert hasattr(p, "median_feature_skewness")
        assert hasattr(p, "outlier_fraction")
        assert hasattr(p, "near_zero_variance_fraction")
        assert hasattr(p, "median_abs_correlation")
        assert hasattr(p, "feature_signal_p90")
        assert hasattr(p, "task")
        assert hasattr(p, "y_skewness")
        assert hasattr(p, "y_all_positive")

    def test_profile_fractions_in_range(self):
        X, y = make_clf_data()
        p = inspect(X, y)
        assert 0.0 <= p.sparsity <= 1.0
        assert 0.0 <= p.binary_feature_fraction <= 1.0
        assert 0.0 <= p.outlier_fraction <= 1.0
        assert 0.0 <= p.near_zero_variance_fraction <= 1.0
        assert 0.0 <= p.median_abs_correlation <= 1.0
        assert 0.0 <= p.feature_signal_p90 <= 1.0
        assert p.n_p_ratio > 0

    def test_task_auto_classification(self):
        X, y = make_clf_data()
        p = inspect(X, y)
        assert p.task == "binary_classification"

    def test_task_auto_regression(self):
        X, y = make_reg_data()
        p = inspect(X, y)
        assert p.task == "regression"

    def test_task_explicit(self):
        X, y = make_clf_data()
        p = inspect(X, y, task="binary_classification")
        assert p.task == "binary_classification"

    def test_task_invalid(self):
        X, y = make_clf_data()
        with pytest.raises(ValueError):
            inspect(X, y, task="multiclass")

    def test_sparse_counts_detected(self):
        X, y = make_fingerprint_data()
        p = inspect(X, y)
        assert p.is_sparse_counts is True

    def test_sparse_counts_not_detected_for_dense(self):
        X, y = make_clf_data()
        p = inspect(X, y)
        assert p.is_sparse_counts is False

    def test_near_zero_variance_detected(self):
        rng = np.random.RandomState(0)
        X = rng.randn(200, 20)
        X[:, :5] = 0.0  # 25% constant features
        y = (rng.randn(200) > 0).astype(int)
        p = inspect(X, y)
        assert p.near_zero_variance_fraction > 0.1

    def test_sparse_scipy_input(self):
        X_dense, y = make_fingerprint_data()
        X_sparse = sp.csr_matrix(X_dense)
        p = inspect(X_sparse, y)
        assert p.n_samples == X_dense.shape[0]
        assert p.n_features == X_dense.shape[1]
        assert p.sparsity > 0.5

    def test_regression_y_stats(self):
        X, y = make_reg_data()
        y = np.abs(y) + 1  # all positive
        p = inspect(X, y, task="regression")
        assert p.y_all_positive is True
        assert np.isfinite(p.y_skewness)

    def test_repr(self):
        X, y = make_clf_data()
        p = inspect(X, y)
        r = repr(p)
        assert "PreprocessingProfile" in r
        assert "n_samples" in r


# ---------------------------------------------------------------------------
# TestScalerReducer
# ---------------------------------------------------------------------------

class TestScalerReducer:

    # --- Scaler rules ---

    def test_scaler_sparse_counts(self):
        p = make_profile(is_sparse_counts=True, sparsity=0.95)
        assert select_scaler(p) == "max_abs"

    def test_scaler_binary_features(self):
        p = make_profile(binary_feature_fraction=0.9)
        assert select_scaler(p) == "max_abs"

    def test_scaler_sparse_data(self):
        p = make_profile(sparsity=0.6)
        assert select_scaler(p) == "max_abs"

    def test_scaler_outliers(self):
        p = make_profile(outlier_fraction=0.5)
        assert select_scaler(p) == "robust"

    def test_scaler_power(self):
        p = make_profile(median_feature_skewness=2.0, outlier_fraction=0.1, sparsity=0.0)
        assert select_scaler(p) == "power"

    def test_scaler_default(self):
        p = make_profile(median_feature_skewness=0.3, outlier_fraction=0.05, sparsity=0.0)
        assert select_scaler(p) == "standard"

    def test_scaler_sparse_beats_outlier(self):
        # Sparse data should return max_abs even if outlier_fraction is high
        p = make_profile(sparsity=0.7, outlier_fraction=0.5)
        assert select_scaler(p) == "max_abs"

    # --- Reducer rules ---

    def test_reducer_few_features(self):
        p = make_profile(n_features=30, n_p_ratio=10.0)
        assert select_reducer(p) == "variance_threshold"

    def test_reducer_fingerprints(self):
        p = make_profile(n_features=512, is_sparse_counts=True, n_p_ratio=3.0)
        assert select_reducer(p) == "select_k_mutual_info"

    def test_reducer_well_determined(self):
        p = make_profile(n_features=100, n_p_ratio=25.0)
        assert select_reducer(p) == "variance_threshold"

    def test_reducer_moderate_correlated(self):
        p = make_profile(n_features=100, n_p_ratio=10.0, median_abs_correlation=0.7)
        assert select_reducer(p) == "pca_99"

    def test_reducer_moderate_independent(self):
        p = make_profile(n_features=100, n_p_ratio=10.0, median_abs_correlation=0.2)
        assert select_reducer(p) == "select_80"

    def test_reducer_underdetermined_sparse(self):
        p = make_profile(n_features=500, n_p_ratio=1.5, sparsity=0.7)
        assert select_reducer(p) == "truncated_svd"

    def test_reducer_underdetermined_correlated(self):
        p = make_profile(n_features=200, n_p_ratio=2.0, sparsity=0.0, median_abs_correlation=0.8)
        assert select_reducer(p) == "pca_95"

    def test_reducer_underdetermined_independent(self):
        p = make_profile(n_features=200, n_p_ratio=2.0, sparsity=0.0, median_abs_correlation=0.1)
        assert select_reducer(p) == "select_50"


# ---------------------------------------------------------------------------
# TestFitTransform
# ---------------------------------------------------------------------------

class TestFitTransform:

    def test_basic_classification(self):
        X, y = make_clf_data()
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X, y)
        assert X_t.ndim == 2
        assert X_t.shape[0] == X.shape[0]
        assert X_t.shape[1] <= X.shape[1]

    def test_basic_regression(self):
        X, y = make_reg_data()
        pre = ZeroShotPreprocessor(task="regression")
        X_t = pre.fit_transform(X, y)
        assert X_t.shape[0] == X.shape[0]

    def test_attributes_set_after_fit(self):
        X, y = make_clf_data()
        pre = ZeroShotPreprocessor()
        pre.fit(X, y)
        assert isinstance(pre.profile_, PreprocessingProfile)
        assert isinstance(pre.scaler_name_, str)
        assert isinstance(pre.reducer_name_, str)
        assert pre.n_features_in_ == X.shape[1]
        assert pre.n_features_out_ >= 1

    def test_transform_consistency(self):
        X, y = make_clf_data(n=200, p=20)
        pre = ZeroShotPreprocessor()
        pre.fit(X, y)
        X_t1 = pre.fit_transform(X, y)
        X_t2 = pre.transform(X)
        np.testing.assert_allclose(X_t1, X_t2)

    def test_transform_before_fit_raises(self):
        from sklearn.exceptions import NotFittedError
        X, y = make_clf_data()
        pre = ZeroShotPreprocessor()
        with pytest.raises(NotFittedError):
            pre.transform(X)

    def test_sparse_fingerprint_input(self):
        X_dense, y = make_fingerprint_data()
        X_sparse = sp.csr_matrix(X_dense)
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X_sparse, y)
        assert X_t.shape[0] == X_dense.shape[0]

    def test_missing_values_handled(self):
        X, y = make_clf_data(n=200, p=20)
        X[0, 0] = np.nan
        X[5, 3] = np.nan
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X, y)
        assert not np.isnan(X_t).any()

    def test_constant_column_removed(self):
        rng = np.random.RandomState(0)
        X = rng.randn(200, 20)
        X[:, 0] = 5.0  # constant column
        y = (rng.randn(200) > 0).astype(int)
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X, y)
        assert X_t.shape[1] < X.shape[1]

    def test_verbose_does_not_crash(self):
        X, y = make_clf_data()
        pre = ZeroShotPreprocessor(verbose=True)
        pre.fit_transform(X, y)

    def test_skewed_data_uses_power_or_robust(self):
        X, y = make_skewed_data()
        pre = ZeroShotPreprocessor(task="regression")
        pre.fit(X, y)
        assert pre.scaler_name_ in ("power", "robust", "standard")

    def test_outlier_data_uses_robust(self):
        X, y = make_outlier_data()
        pre = ZeroShotPreprocessor()
        pre.fit(X, y)
        assert pre.scaler_name_ == "robust"

    def test_high_dimensional_underdetermined(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 300)  # n/p = 0.17
        y = (rng.randn(50) > 0).astype(int)
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X, y)
        assert X_t.shape[0] == 50
        assert X_t.shape[1] < 300

    def test_auto_task_detection(self):
        X, y = make_clf_data()
        pre = ZeroShotPreprocessor(task="auto")
        pre.fit(X, y)
        assert pre.profile_.task == "binary_classification"


# ---------------------------------------------------------------------------
# TestONNX
# ---------------------------------------------------------------------------

onnx_available = importlib.util.find_spec("skl2onnx") is not None


@pytest.mark.skipif(not onnx_available, reason="skl2onnx not installed")
class TestONNX:

    def test_onnx_export_and_load(self, tmp_path):
        import onnxruntime as rt

        X, y = make_clf_data(n=200, p=20)
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X, y)

        onnx_path = str(tmp_path / "preprocessor.onnx")
        pre.to_onnx(onnx_path)

        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        X_onnx = sess.run(None, {input_name: X.astype(np.float32)})[0]

        np.testing.assert_allclose(
            X_t.astype(np.float32), X_onnx, rtol=1e-4, atol=1e-5
        )

    def test_onnx_output_shape(self, tmp_path):
        import onnxruntime as rt

        X, y = make_reg_data(n=300, p=30)
        pre = ZeroShotPreprocessor(task="regression")
        pre.fit(X, y)

        onnx_path = str(tmp_path / "reg_preprocessor.onnx")
        pre.to_onnx(onnx_path)

        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        X_onnx = sess.run(None, {input_name: X.astype(np.float32)})[0]

        assert X_onnx.shape[0] == X.shape[0]
        assert X_onnx.shape[1] == pre.n_features_out_
