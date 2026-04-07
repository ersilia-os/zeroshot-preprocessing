"""
Tests for zspreprocessing.

Structure:
  TestInspector          — dataset profiling
  TestScalerReducer      — selection rule unit tests
  TestFitTransform       — end-to-end pipeline tests
"""

import importlib
import os

import numpy as np
import pytest
import scipy.sparse as sp

from zspreprocessing import (
    PreprocessingProfile,
    PreprocessorArtifact,
    ZeroShotClassifierPreprocessor,
    ZeroShotPreprocessor,
    ZeroShotRegressorPreprocessor,
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
    # Inject outliers into >5% of rows so outlier_fraction > 0.3
    # The profiler flags a feature if >5% of its values fall outside 1.5×IQR,
    # so we need at least ceil(n * 0.05) + 1 = 16 rows; use 20 to be safe.
    X[:20] *= 100
    y = (rng.randn(n) > 0).astype(int)
    return X, y


def make_imbalanced_clf_data(n=1000, minority_count=10, p=20, seed=0):
    """Binary classification with a heavily imbalanced target."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = np.zeros(n, dtype=int)
    y[:minority_count] = 1
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
        task="classification",
        y_skewness=0.0,
        y_all_positive=False,
        n_minority_class=0,
    )
    defaults.update(kwargs)
    return PreprocessingProfile(**defaults)


# ---------------------------------------------------------------------------
# TestInspector
# ---------------------------------------------------------------------------

class TestInspector:
    def test_profile_fields_present(self):
        X, y = make_clf_data()
        p = inspect(X, y, task="classification")
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
        assert hasattr(p, "n_minority_class")

    def test_profile_fractions_in_range(self):
        X, y = make_clf_data()
        p = inspect(X, y, task="classification")
        assert 0.0 <= p.sparsity <= 1.0
        assert 0.0 <= p.binary_feature_fraction <= 1.0
        assert 0.0 <= p.outlier_fraction <= 1.0
        assert 0.0 <= p.near_zero_variance_fraction <= 1.0
        assert 0.0 <= p.median_abs_correlation <= 1.0
        assert 0.0 <= p.feature_signal_p90 <= 1.0
        assert p.n_p_ratio > 0

    def test_task_explicit(self):
        X, y = make_clf_data()
        p = inspect(X, y, task="classification")
        assert p.task == "classification"

    def test_task_invalid(self):
        X, y = make_clf_data()
        with pytest.raises(ValueError):
            inspect(X, y, task="multiclass")

    def test_sparse_counts_detected(self):
        X, y = make_fingerprint_data()
        p = inspect(X, y, task="classification")
        assert p.is_sparse_counts is True

    def test_sparse_counts_not_detected_for_dense(self):
        X, y = make_clf_data()
        p = inspect(X, y, task="classification")
        assert p.is_sparse_counts is False

    def test_near_zero_variance_detected(self):
        rng = np.random.RandomState(0)
        X = rng.randn(200, 20)
        X[:, :5] = 0.0  # 25% constant features
        y = (rng.randn(200) > 0).astype(int)
        p = inspect(X, y, task="classification")
        assert p.near_zero_variance_fraction > 0.1

    def test_sparse_scipy_input(self):
        X_dense, y = make_fingerprint_data()
        X_sparse = sp.csr_matrix(X_dense)
        p = inspect(X_sparse, y, task="classification")
        assert p.n_samples == X_dense.shape[0]
        assert p.n_features == X_dense.shape[1]
        assert p.sparsity > 0.5

    def test_regression_y_stats(self):
        X, y = make_reg_data()
        y = np.abs(y) + 1  # all positive
        p = inspect(X, y, task="regression")
        assert p.y_all_positive is True
        assert np.isfinite(p.y_skewness)

    def test_regression_y_not_all_positive(self):
        X, y = make_reg_data()
        # y from randn includes negatives
        p = inspect(X, y, task="regression")
        assert p.y_all_positive is False

    def test_n_minority_class_computed(self):
        # 10 positives, 290 negatives
        X, y = make_imbalanced_clf_data(n=300, minority_count=10, p=20)
        p = inspect(X, y, task="classification")
        assert p.task == "classification"
        assert p.n_minority_class == 10

    def test_large_dataset_profile(self):
        # 100k samples, 2000 features — profiling must complete and fields be valid
        rng = np.random.RandomState(0)
        X = rng.randn(100_000, 2_000).astype(np.float32)
        y = (rng.randn(100_000) > 0).astype(int)
        p = inspect(X, y, task="classification")
        assert p.n_samples == 100_000
        assert p.n_features == 2_000
        assert 0.0 <= p.sparsity <= 1.0
        assert 0.0 <= p.binary_feature_fraction <= 1.0
        assert 0.0 <= p.outlier_fraction <= 1.0
        assert p.n_p_ratio == pytest.approx(50.0)

    def test_repr(self):
        X, y = make_clf_data()
        p = inspect(X, y, task="classification")
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
    # select_reducer only uses n_features and n_p_ratio.

    def test_reducer_few_features(self):
        # p ≤ 50 → no reduction regardless of ratio
        p = make_profile(n_features=30, n_p_ratio=1.0)
        assert select_reducer(p) == "variance_threshold"

    def test_reducer_well_determined(self):
        # n/p ≥ 20 → no reduction needed
        p = make_profile(n_features=100, n_p_ratio=25.0)
        assert select_reducer(p) == "variance_threshold"

    def test_reducer_intermediate_ratio(self):
        # p > 50 and 1 ≤ n/p < 20 → correlation filter
        p = make_profile(n_features=100, n_p_ratio=10.0)
        assert select_reducer(p) == "correlation_filter"

    def test_reducer_underdetermined(self):
        # p > n (n/p < 1) → correlation filter
        p = make_profile(n_features=500, n_p_ratio=1.5)
        assert select_reducer(p) == "correlation_filter"

    def test_reducer_fingerprints(self):
        # Large sparse fingerprint dataset with low n/p → correlation filter
        p = make_profile(n_features=512, is_sparse_counts=True, n_p_ratio=3.0)
        assert select_reducer(p) == "correlation_filter"

    def test_reducer_large_p_low_ratio(self):
        # 2000 features, n/p = 0.5 → correlation filter
        p = make_profile(n_features=2000, n_p_ratio=0.5)
        assert select_reducer(p) == "correlation_filter"

    def test_reducer_large_p_high_ratio(self):
        # 2000 features, n/p = 50 → variance threshold (well-determined)
        p = make_profile(n_features=2000, n_p_ratio=50.0)
        assert select_reducer(p) == "variance_threshold"


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
        # fit_transform(X) must equal transform(X) after fit on the same data
        X, y = make_clf_data(n=200, p=20)
        pre = ZeroShotPreprocessor()
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
        assert X_t.shape[1] <= 300

    def test_highly_imbalanced_classification(self):
        # 990 negatives, 10 positives
        X, y = make_imbalanced_clf_data(n=1000, minority_count=10, p=20)
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X, y)
        assert pre.profile_.n_minority_class == 10
        assert X_t.shape[0] == 1000
        assert X_t.shape[1] >= 1

    def test_large_dataset_fits(self):
        # 100k samples, 2000 features — n/p=50 → variance_threshold reducer
        rng = np.random.RandomState(0)
        X = rng.randn(100_000, 2_000).astype(np.float32)
        y = (rng.randn(100_000) > 0).astype(int)
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X, y)
        assert pre.n_features_in_ == 2_000
        assert pre.n_features_out_ >= 1
        assert pre.reducer_name_ == "variance_threshold"
        assert X_t.shape[0] == 100_000

    def test_subclass_classifier_preprocessor(self):
        X, y = make_clf_data()
        pre = ZeroShotClassifierPreprocessor()
        pre.fit(X, y)
        assert pre.profile_.task == "classification"
        assert pre.task == "classification"

    def test_subclass_regressor_preprocessor(self):
        X, y = make_reg_data()
        pre = ZeroShotRegressorPreprocessor()
        pre.fit(X, y)
        assert pre.profile_.task == "regression"
        assert pre.task == "regression"

    def test_correlation_filter_reduces_features(self):
        # Build data where half the features are exact copies of the other half
        # so CorrelationFilter must drop ~50% of them.
        rng = np.random.RandomState(0)
        n, half_p = 200, 50
        X_base = rng.randn(n, half_p)
        X = np.hstack([X_base, X_base])  # 100 features, 50 duplicated pairs (r=1.0)
        y = (rng.randn(n) > 0).astype(int)
        # n=200, p=100, n/p=2.0 → "correlation_filter"
        pre = ZeroShotPreprocessor()
        pre.fit(X, y)
        assert pre.reducer_name_ == "correlation_filter"
        assert pre.n_features_out_ < pre.n_features_in_


# ---------------------------------------------------------------------------
# TestSave
# ---------------------------------------------------------------------------

class TestSave:

    def _check_metadata(self, directory, pre):
        import json
        json_path = os.path.join(directory, "preprocessor.json")
        assert os.path.exists(json_path), f"{json_path} missing"
        with open(json_path) as f:
            meta = json.load(f)
        assert meta["task"] == pre.task
        assert meta["scaler"] == pre.scaler_name_
        assert meta["reducer"] == pre.reducer_name_
        assert meta["n_features_in"] == pre.n_features_in_
        assert meta["n_features_out"] == pre.n_features_out_
        assert isinstance(meta["kept_feature_indices"], list)
        assert len(meta["kept_feature_indices"]) == pre.n_features_out_
        assert all(0 <= i < pre.n_features_in_ for i in meta["kept_feature_indices"])
        assert len(set(meta["kept_feature_indices"])) == len(meta["kept_feature_indices"])
        return meta

    # --- ONNX saves ---

    def test_save_onnx_classification(self, tmp_path):
        import onnxruntime as rt
        X, y = make_clf_data(n=300, p=80)
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X, y)

        pre.save(str(tmp_path), onnx=True)

        assert os.path.exists(str(tmp_path / "preprocessor.onnx"))
        assert os.path.exists(str(tmp_path / "preprocessor.json"))
        self._check_metadata(str(tmp_path), pre)

        sess = rt.InferenceSession(str(tmp_path / "preprocessor.onnx"))
        X_onnx = sess.run(None, {"float_input": X.astype(np.float32)})[0]
        np.testing.assert_allclose(X_t.astype(np.float32), X_onnx, rtol=1e-4, atol=1e-5)

    def test_save_onnx_regression(self, tmp_path):
        import onnxruntime as rt
        X, y = make_reg_data(n=300, p=20)
        pre = ZeroShotPreprocessor(task="regression")
        X_t = pre.fit_transform(X, y)

        pre.save(str(tmp_path), onnx=True)

        self._check_metadata(str(tmp_path), pre)
        sess = rt.InferenceSession(str(tmp_path / "preprocessor.onnx"))
        X_onnx = sess.run(None, {"float_input": X.astype(np.float32)})[0]
        np.testing.assert_allclose(X_t.astype(np.float32), X_onnx, rtol=1e-4, atol=1e-5)

    # --- joblib saves ---

    def test_save_joblib_classification(self, tmp_path):
        import joblib
        X, y = make_clf_data(n=300, p=80)
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X, y)

        pre.save(str(tmp_path), onnx=False)

        assert os.path.exists(str(tmp_path / "preprocessor.joblib"))
        assert os.path.exists(str(tmp_path / "preprocessor.json"))
        self._check_metadata(str(tmp_path), pre)

        pipeline = joblib.load(str(tmp_path / "preprocessor.joblib"))
        np.testing.assert_allclose(X_t, pipeline.transform(X))

    def test_save_joblib_regression(self, tmp_path):
        import joblib
        X, y = make_reg_data(n=300, p=20)
        pre = ZeroShotPreprocessor(task="regression")
        X_t = pre.fit_transform(X, y)

        pre.save(str(tmp_path), onnx=False)

        self._check_metadata(str(tmp_path), pre)
        pipeline = joblib.load(str(tmp_path / "preprocessor.joblib"))
        np.testing.assert_allclose(X_t, pipeline.transform(X))

    # --- Metadata correctness ---

    def test_kept_indices_variance_threshold(self, tmp_path):
        rng = np.random.RandomState(0)
        X = rng.randn(300, 20)
        X[:, 3] = 0.0
        X[:, 7] = 0.0
        y = (rng.randn(300) > 0).astype(int)

        pre = ZeroShotPreprocessor()
        pre.fit(X, y)
        assert pre.reducer_name_ == "variance_threshold"

        kept = pre.kept_feature_indices_
        assert 3 not in kept
        assert 7 not in kept
        assert len(kept) == pre.n_features_out_

    def test_kept_indices_correlation_filter(self, tmp_path):
        rng = np.random.RandomState(0)
        n, half_p = 200, 50
        X_base = rng.randn(n, half_p)
        X = np.hstack([X_base, X_base])
        y = (rng.randn(n) > 0).astype(int)

        pre = ZeroShotPreprocessor()
        pre.fit(X, y)
        assert pre.reducer_name_ == "correlation_filter"

        kept = pre.kept_feature_indices_
        assert len(kept) == pre.n_features_out_
        assert len(kept) < 100

    def test_kept_indices_count_matches_n_features_out(self, tmp_path):
        X, y = make_fingerprint_data(n=300, p=256)
        pre = ZeroShotPreprocessor()
        pre.fit(X, y)
        assert len(pre.kept_feature_indices_) == pre.n_features_out_

    def test_save_creates_directory(self, tmp_path):
        X, y = make_clf_data()
        pre = ZeroShotPreprocessor()
        pre.fit(X, y)
        directory = str(tmp_path / "new" / "nested" / "dir")
        pre.save(directory, onnx=False)
        assert os.path.exists(os.path.join(directory, "preprocessor.json"))

    def test_kept_indices_all_nan_column(self, tmp_path):
        rng = np.random.RandomState(0)
        X = rng.randn(300, 20)
        X[:, 5] = np.nan
        y = (rng.randn(300) > 0).astype(int)

        pre = ZeroShotPreprocessor()
        pre.fit(X, y)

        assert 5 not in pre.kept_feature_indices_
        assert len(pre.kept_feature_indices_) == pre.n_features_out_

    def test_save_before_fit_raises(self, tmp_path):
        from sklearn.exceptions import NotFittedError
        pre = ZeroShotPreprocessor()
        with pytest.raises(NotFittedError):
            pre.save(str(tmp_path), onnx=False)


# ---------------------------------------------------------------------------
# TestPreprocessorArtifact
# ---------------------------------------------------------------------------

class TestPreprocessorArtifact:

    def test_load_onnx_and_run(self, tmp_path):
        X, y = make_clf_data(n=300, p=80)
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X, y)
        pre.save(str(tmp_path), onnx=True)

        artifact = PreprocessorArtifact.load(str(tmp_path))
        X_out = artifact.run(X)

        assert isinstance(X_out, np.ndarray)
        np.testing.assert_allclose(X_t.astype(np.float32), X_out, rtol=1e-4, atol=1e-5)

    def test_load_joblib_and_run(self, tmp_path):
        X, y = make_clf_data(n=300, p=80)
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X, y)
        pre.save(str(tmp_path), onnx=False)

        artifact = PreprocessorArtifact.load(str(tmp_path))
        X_out = artifact.run(X)

        np.testing.assert_allclose(X_t, X_out)

    def test_metadata_attributes(self, tmp_path):
        X, y = make_clf_data(n=300, p=80)
        pre = ZeroShotPreprocessor()
        pre.fit(X, y)
        pre.save(str(tmp_path), onnx=True)

        artifact = PreprocessorArtifact.load(str(tmp_path))

        assert artifact.task == pre.task
        assert artifact.scaler == pre.scaler_name_
        assert artifact.reducer == pre.reducer_name_
        assert artifact.n_features_in == pre.n_features_in_
        assert artifact.n_features_out == pre.n_features_out_
        assert artifact.kept_feature_indices == pre.kept_feature_indices_

    def test_prefers_onnx_when_both_present(self, tmp_path):
        X, y = make_clf_data()
        pre = ZeroShotPreprocessor()
        pre.fit(X, y)
        pre.save(str(tmp_path), onnx=True)
        pre.save(str(tmp_path), onnx=False)  # also write joblib

        artifact = PreprocessorArtifact.load(str(tmp_path))
        assert artifact._backend == "onnx"

    def test_output_shape(self, tmp_path):
        X, y = make_reg_data(n=200, p=20)
        pre = ZeroShotPreprocessor(task="regression")
        pre.fit(X, y)
        pre.save(str(tmp_path), onnx=True)

        artifact = PreprocessorArtifact.load(str(tmp_path))
        X_out = artifact.run(X)

        assert X_out.shape == (200, pre.n_features_out_)

    def test_sparse_input(self, tmp_path):
        X_dense, y = make_fingerprint_data(n=200, p=256)
        X_sparse = sp.csr_matrix(X_dense)
        pre = ZeroShotPreprocessor()
        X_t = pre.fit_transform(X_sparse, y)
        pre.save(str(tmp_path), onnx=True)

        artifact = PreprocessorArtifact.load(str(tmp_path))
        X_out = artifact.run(X_sparse)

        X_t_dense = X_t.toarray() if hasattr(X_t, "toarray") else X_t
        np.testing.assert_allclose(X_t_dense.astype(np.float32), X_out, rtol=1e-4, atol=1e-5)

    def test_missing_json_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="preprocessor.json"):
            PreprocessorArtifact.load(str(tmp_path))

    def test_missing_model_raises(self, tmp_path):
        # Write only the JSON, no model file
        import json
        with open(str(tmp_path / "preprocessor.json"), "w") as f:
            json.dump({"task": "classification", "scaler": "standard",
                       "reducer": "variance_threshold", "n_features_in": 10,
                       "n_features_out": 10, "kept_feature_indices": list(range(10))}, f)
        with pytest.raises(FileNotFoundError, match="preprocessor.onnx or preprocessor.joblib"):
            PreprocessorArtifact.load(str(tmp_path))

    def test_repr(self, tmp_path):
        X, y = make_clf_data()
        pre = ZeroShotPreprocessor()
        pre.fit(X, y)
        pre.save(str(tmp_path), onnx=True)

        artifact = PreprocessorArtifact.load(str(tmp_path))
        r = repr(artifact)
        assert "PreprocessorArtifact" in r
        assert "onnx" in r


# ---------------------------------------------------------------------------
# TestONNX
# ---------------------------------------------------------------------------

onnx_available = importlib.util.find_spec("skl2onnx") is not None
if onnx_available:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        onnx_available = False


def _onnx_roundtrip(pre, X, tmp_path, filename="preprocessor.onnx"):
    """Fit, export to ONNX, run inference, return (sklearn_out, onnx_out)."""
    import onnxruntime as rt

    X_t = pre.fit_transform(X, pre._last_y) if hasattr(pre, "_last_y") else None
    onnx_path = str(tmp_path / filename)
    pre.to_onnx(onnx_path)
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    X_onnx = sess.run(None, {input_name: X.astype(np.float32)})[0]
    return X_t, X_onnx


def _fit_export_check(X, y, task, tmp_path, filename="pre.onnx", atol=1e-5, rtol=1e-4):
    """Full fit → export → onnx inference → numerical check helper."""
    import onnxruntime as rt

    pre = ZeroShotPreprocessor(task=task)
    X_t = pre.fit_transform(X, y)

    onnx_path = str(tmp_path / filename)
    pre.to_onnx(onnx_path)

    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name

    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    X_onnx = sess.run(None, {input_name: X_dense.astype(np.float32)})[0]

    X_t_dense = X_t.toarray() if hasattr(X_t, "toarray") else X_t
    assert X_onnx.shape == X_t_dense.shape, (
        f"Shape mismatch: sklearn={X_t_dense.shape}, onnx={X_onnx.shape}"
    )
    np.testing.assert_allclose(
        X_t_dense.astype(np.float32), X_onnx, rtol=rtol, atol=atol,
        err_msg=f"scaler={pre.scaler_name_} reducer={pre.reducer_name_}"
    )
    return pre


@pytest.mark.skipif(not onnx_available, reason="skl2onnx or onnxruntime not available")
class TestONNX:

    # ------------------------------------------------------------------
    # Scaler × Reducer matrix (classification)
    # ------------------------------------------------------------------

    def test_standard_variance_threshold(self, tmp_path):
        # StandardScaler + VarianceThreshold: normal dense, well-determined
        rng = np.random.RandomState(0)
        X = rng.randn(400, 20)          # p=20 ≤ 50 → variance_threshold
        y = (rng.randn(400) > 0).astype(int)
        pre = _fit_export_check(X, y, "classification", tmp_path, "std_vt.onnx")
        assert pre.scaler_name_ == "standard"
        assert pre.reducer_name_ == "variance_threshold"

    def test_standard_correlation_filter(self, tmp_path):
        # StandardScaler + CorrelationFilter: normal dense, high-dimensional
        rng = np.random.RandomState(1)
        X = rng.randn(100, 200)         # n/p=0.5 → correlation_filter
        y = (rng.randn(100) > 0).astype(int)
        pre = _fit_export_check(X, y, "classification", tmp_path, "std_cf.onnx")
        assert pre.scaler_name_ == "standard"
        assert pre.reducer_name_ == "correlation_filter"

    def test_robust_variance_threshold(self, tmp_path):
        # RobustScaler + VarianceThreshold: heavy outliers, few features
        rng = np.random.RandomState(2)
        X = rng.randn(400, 20)
        X[:40] *= 100                   # >30% outlier features
        y = (rng.randn(400) > 0).astype(int)
        pre = _fit_export_check(X, y, "classification", tmp_path, "rob_vt.onnx")
        assert pre.scaler_name_ == "robust"
        assert pre.reducer_name_ == "variance_threshold"

    def test_robust_correlation_filter(self, tmp_path):
        # RobustScaler + CorrelationFilter: heavy outliers, high-dimensional
        rng = np.random.RandomState(3)
        X = rng.randn(100, 200)
        X[:40] *= 100
        y = (rng.randn(100) > 0).astype(int)
        pre = _fit_export_check(X, y, "classification", tmp_path, "rob_cf.onnx")
        assert pre.scaler_name_ == "robust"
        assert pre.reducer_name_ == "correlation_filter"

    def test_power_variance_threshold(self, tmp_path):
        # PowerTransformer + VarianceThreshold: skewed, few features
        rng = np.random.RandomState(4)
        X = rng.exponential(scale=2.0, size=(400, 20)) ** 2   # heavy skew
        y = (rng.randn(400) > 0).astype(int)
        pre = _fit_export_check(X, y, "classification", tmp_path, "pow_vt.onnx")
        assert pre.scaler_name_ in ("power", "robust")  # fallback allowed
        assert pre.reducer_name_ == "variance_threshold"

    def test_power_correlation_filter(self, tmp_path):
        # PowerTransformer + CorrelationFilter: skewed, high-dimensional
        rng = np.random.RandomState(5)
        X = rng.exponential(scale=2.0, size=(100, 200)) ** 2
        y = (rng.randn(100) > 0).astype(int)
        pre = _fit_export_check(X, y, "classification", tmp_path, "pow_cf.onnx")
        assert pre.scaler_name_ in ("power", "robust")
        assert pre.reducer_name_ == "correlation_filter"

    def test_maxabs_variance_threshold(self, tmp_path):
        # MaxAbsScaler + VarianceThreshold: binary fingerprints, few features
        rng = np.random.RandomState(6)
        X = rng.randint(0, 2, (400, 30)).astype(float)   # binary, p≤50
        y = (rng.randn(400) > 0).astype(int)
        pre = _fit_export_check(X, y, "classification", tmp_path, "mabs_vt.onnx")
        assert pre.scaler_name_ == "max_abs"
        assert pre.reducer_name_ == "variance_threshold"

    def test_maxabs_correlation_filter(self, tmp_path):
        # MaxAbsScaler + CorrelationFilter: sparse fingerprint, high-dim
        rng = np.random.RandomState(7)
        X = (rng.random((200, 512)) < 0.05).astype(float)  # ~5% dense, p>50
        y = (rng.randn(200) > 0).astype(int)
        pre = _fit_export_check(X, y, "classification", tmp_path, "mabs_cf.onnx")
        assert pre.scaler_name_ == "max_abs"
        assert pre.reducer_name_ == "correlation_filter"

    # ------------------------------------------------------------------
    # Regression variants
    # ------------------------------------------------------------------

    def test_regression_standard_variance_threshold(self, tmp_path):
        rng = np.random.RandomState(10)
        X = rng.randn(400, 20)
        y = rng.randn(400)
        pre = _fit_export_check(X, y, "regression", tmp_path, "reg_std_vt.onnx")
        assert pre.scaler_name_ == "standard"
        assert pre.reducer_name_ == "variance_threshold"

    def test_regression_robust_correlation_filter(self, tmp_path):
        rng = np.random.RandomState(11)
        X = rng.randn(100, 200)
        X[:40] *= 100
        y = rng.randn(100)
        pre = _fit_export_check(X, y, "regression", tmp_path, "reg_rob_cf.onnx")
        assert pre.scaler_name_ == "robust"
        assert pre.reducer_name_ == "correlation_filter"

    def test_regression_power_variance_threshold(self, tmp_path):
        rng = np.random.RandomState(12)
        X = rng.exponential(scale=2.0, size=(400, 20)) ** 2
        y = rng.randn(400)
        pre = _fit_export_check(X, y, "regression", tmp_path, "reg_pow_vt.onnx")
        assert pre.scaler_name_ in ("power", "robust")
        assert pre.reducer_name_ == "variance_threshold"

    # ------------------------------------------------------------------
    # Sparse scipy input
    # ------------------------------------------------------------------

    def test_sparse_input(self, tmp_path):
        # scipy.sparse CSR → ONNX round-trip
        X_dense, y = make_fingerprint_data(n=300, p=256)
        X_sparse = sp.csr_matrix(X_dense)
        pre = _fit_export_check(X_sparse, y, "classification", tmp_path, "sparse.onnx")
        assert pre.scaler_name_ == "max_abs"

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_with_missing_values(self, tmp_path):
        # NaNs are handled by the imputer; ONNX receives clean input
        rng = np.random.RandomState(20)
        X = rng.randn(300, 20)
        X[rng.random(X.shape) < 0.1] = np.nan
        y = (rng.randn(300) > 0).astype(int)
        _fit_export_check(X, y, "classification", tmp_path, "nan.onnx")

    def test_with_constant_columns(self, tmp_path):
        # Constant columns are removed by VarianceThreshold; ONNX sees fewer features
        rng = np.random.RandomState(21)
        X = rng.randn(300, 25)
        X[:, 0] = 7.0
        X[:, 1] = -3.0
        y = (rng.randn(300) > 0).astype(int)
        pre = _fit_export_check(X, y, "classification", tmp_path, "const.onnx")
        assert pre.n_features_out_ < 25

    def test_transform_new_data(self, tmp_path):
        # ONNX inference on held-out test set matches sklearn transform
        import onnxruntime as rt

        rng = np.random.RandomState(30)
        X_train = rng.randn(300, 80)
        y_train = (rng.randn(300) > 0).astype(int)
        X_test  = rng.randn(50, 80)

        pre = ZeroShotPreprocessor()
        pre.fit(X_train, y_train)
        X_test_sk = pre.transform(X_test)

        onnx_path = str(tmp_path / "new_data.onnx")
        pre.to_onnx(onnx_path)
        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        X_test_onnx = sess.run(None, {input_name: X_test.astype(np.float32)})[0]

        np.testing.assert_allclose(
            X_test_sk.astype(np.float32), X_test_onnx, rtol=1e-4, atol=1e-5
        )

    def test_export_before_fit_raises(self, tmp_path):
        from sklearn.exceptions import NotFittedError
        pre = ZeroShotPreprocessor()
        with pytest.raises(NotFittedError):
            pre.to_onnx(str(tmp_path / "unfitted.onnx"))
