import numpy as np
import pandas as pd
import scipy.stats
from pandas.api.types import is_numeric_dtype


def extract_meta_features(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Simplified meta-feature extractor for tabular regression.
    Returns a dict mapping feature names to numeric values.
    """
    # Basic shape features
    n_instances, n_features = X.shape
    log_n_instances = np.log(n_instances)
    log_n_features = np.log(n_features)

    # Missing‐value stats
    na_mask = X.isna()
    total_missing = na_mask.values.sum()
    pct_missing = total_missing / (n_instances * n_features)
    inst_missing = na_mask.any(axis=1).sum()
    feat_missing = na_mask.any(axis=0).sum()
    pct_inst_missing = inst_missing / n_instances
    pct_feat_missing = feat_missing / n_features

    # Numeric columns detection
    is_num = X.dtypes.map(is_numeric_dtype).values
    n_numeric = int(is_num.sum())
    n_categorical = n_features - n_numeric
    ratio_num_cat = n_numeric / n_categorical if n_categorical else 0.0
    ratio_cat_num = n_categorical / n_numeric if n_numeric else 0.0

    # Numeric data slice and moments
    if n_numeric:
        num_vals = X.values[:, is_num].astype(float)
        col_skew = scipy.stats.skew(num_vals, axis=0, nan_policy="omit")
        col_kurt = scipy.stats.kurtosis(num_vals, axis=0, nan_policy="omit")
        skew_mean = float(np.nanmean(col_skew))
        skew_std = float(np.nanstd(col_skew))
        kurt_mean = float(np.nanmean(col_kurt))
        kurt_std = float(np.nanstd(col_kurt))
    else:
        skew_mean = skew_std = kurt_mean = kurt_std = 0.0

    # Target stats
    y_arr = np.asarray(y, dtype=float)
    t_mean = float(y_arr.mean())
    t_std = float(y_arr.std())
    t_skew = float(scipy.stats.skew(y_arr, nan_policy="omit"))
    t_kurt = float(scipy.stats.kurtosis(y_arr, nan_policy="omit"))

    meta = {
        "n_instances": n_instances,
        "n_features": n_features,
        "log_n_instances": log_n_instances,
        "log_n_features": log_n_features,
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "ratio_num_cat": ratio_num_cat,
        "ratio_cat_num": ratio_cat_num,
        "total_missing": total_missing,
        "pct_missing": pct_missing,
        "inst_missing": inst_missing,
        "pct_inst_missing": pct_inst_missing,
        "feat_missing": feat_missing,
        "pct_feat_missing": pct_feat_missing,
        "skew_mean": skew_mean,
        "skew_std": skew_std,
        "kurt_mean": kurt_mean,
        "kurt_std": kurt_std,
        "target_mean": t_mean,
        "target_std": t_std,
        "target_skew": t_skew,
        "target_kurt": t_kurt,
    }
    return meta


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, np.nan],
            "b": [4.0, np.nan, 6.0, 7.0],
            "c": ["x", "y", "x", "z"],
        }
    )
    y = pd.Series([0.5, 0.6, 0.7, 0.8])
    features = extract_meta_features(df, y)
    print(features)
