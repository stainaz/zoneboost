"""Shared input handling used by both ZoneBoostRegressor and
ZoneBoostClassifier -- kept in one place so the two estimators can't drift
apart on how they interpret X or auto-detect categorical columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array

__all__ = ["ensure_dataframe", "resolve_categorical_features"]


def ensure_dataframe(X, feature_names=None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    X = check_array(X, dtype=None, ensure_all_finite=False)
    columns = feature_names if feature_names is not None and len(feature_names) == X.shape[1] else None
    if columns is None:
        columns = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns)


def resolve_categorical_features(X: pd.DataFrame, declared) -> set:
    # is_numeric_dtype (rather than listing dtype names) also catches
    # pandas' newer arrow-backed / nullable string dtypes, not just legacy
    # numpy object dtype.
    auto_detected = {
        c for c in X.columns if pd.api.types.is_bool_dtype(X[c]) or not pd.api.types.is_numeric_dtype(X[c])
    }
    declared_set = set()
    if declared:
        for f in declared:
            declared_set.add(X.columns[f] if isinstance(f, (int, np.integer)) else f)
    return auto_detected | declared_set
