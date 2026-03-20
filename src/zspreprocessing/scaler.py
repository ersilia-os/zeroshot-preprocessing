"""
Scaler selection and construction for zero-shot preprocessing.

Applies a single sklearn scaler to all features, chosen based on the
dataset profile. Using one scaler across all features keeps the pipeline
simple and fully ONNX-exportable.
"""

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from .inspector import PreprocessingProfile


def select_scaler(profile: PreprocessingProfile) -> str:
    """
    Choose a scaler name based on dataset characteristics.

    Decision rules (first match wins):

    1. Fingerprint / sparse-count data  →  MaxAbsScaler
       Preserves sparsity; bits are already in {0, 1} or small counts.

    2. Mostly binary features (≥ 80%)  →  MaxAbsScaler
       Values are in {0, 1}; MaxAbsScaler is effectively a no-op.

    3. Sparse data (sparsity > 0.5)  →  MaxAbsScaler
       Checked *before* the outlier rule to avoid applying RobustScaler
       (which densifies sparse matrices) to genuinely sparse data.

    4. Heavy outliers (outlier_fraction > 0.3)  →  RobustScaler
       Uses median and IQR; immune to extreme values.

    5. Heavy skew (median_feature_skewness > 1.5, dense data)  →  PowerTransformer
       Yeo-Johnson handles both positive and negative values and reduces skew.

    6. Default  →  StandardScaler
       Zero mean, unit variance; appropriate for approximately normal features.

    Parameters
    ----------
    profile : PreprocessingProfile

    Returns
    -------
    str
        One of: "max_abs", "robust", "power", "standard", "minmax".
    """
    # 1. Fingerprints
    if profile.is_sparse_counts:
        return "max_abs"

    # 2. Mostly binary
    if profile.binary_feature_fraction >= 0.8:
        return "max_abs"

    # 3. Sparse (guard before outlier/power rules that require dense ops)
    if profile.sparsity > 0.5:
        return "max_abs"

    # 4. Heavy outliers
    if profile.outlier_fraction > 0.3:
        return "robust"

    # 5. Heavy skew on dense data
    if profile.median_feature_skewness > 1.5:
        return "power"

    # 6. Default
    return "standard"


_SCALER_FACTORIES = {
    "standard": StandardScaler,
    "robust":   RobustScaler,
    "power":    lambda: PowerTransformer(method="yeo-johnson"),
    "max_abs":  MaxAbsScaler,
    "minmax":   MinMaxScaler,
}


def build_scaler(scaler_name: str):
    """
    Instantiate and return the sklearn scaler for the given name.

    Parameters
    ----------
    scaler_name : str
        Must be one of: "standard", "robust", "power", "max_abs", "minmax".

    Returns
    -------
    sklearn transformer
    """
    if scaler_name not in _SCALER_FACTORIES:
        raise ValueError(
            f"Unknown scaler {scaler_name!r}. "
            f"Valid options: {sorted(_SCALER_FACTORIES)}"
        )
    return _SCALER_FACTORIES[scaler_name]()
