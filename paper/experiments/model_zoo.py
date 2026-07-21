"""Benchmark model factories, one per library, each returning a fresh
unfitted estimator plus a data-adapter function that reshapes X into the
form that library expects for categorical columns.

Philosophy (matching benchmarks/compare_lightgbm.py): every model at its
library's own out-of-the-box defaults, only random_state fixed -- no tuning
that favors any one model over another.
"""
import numpy as np
import pandas as pd

RANDOM_STATE = 0


# ---- data adapters -------------------------------------------------------

def adapt_native(X, cat_cols):
    """LightGBM / XGBoost / ZoneBoost: pandas 'category' dtype columns are
    auto-detected -- no reshaping needed."""
    return X


def adapt_catboost(X, cat_cols):
    """CatBoost wants categorical columns as string/object, not pandas
    'category' dtype, and NaN as a literal string sentinel since its
    categorical path does not accept float NaN."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), "__missing__").astype(str)
    return X


def adapt_ebm(X, cat_cols):
    """EBM (interpret): dense numeric array works most reliably across
    versions; categorical columns are ordinal-encoded (EBM bins whatever
    it's given -- a nominal column ordinal-encoded still gets one bin per
    category, since EBM's own binning is not a distance-based split search
    the way a generic cut-point regressor would be)."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
    return X.astype(float)


# ---- regression models ---------------------------------------------------

def regression_models(cat_cols):
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from interpret.glassbox import ExplainableBoostingRegressor
    from zoneboost import ZoneBoostRegressor

    # Factories (not fitted instances): sklearn's clone() fails on
    # CatBoostRegressor when cat_features is set (RuntimeError -- CatBoost's
    # get_params/set_params round-trip doesn't reproduce the constructor
    # call sklearn's clone() relies on), so every model here is built via a
    # fresh call per CV fold instead of clone(), matching this project's own
    # benchmarks/compare_lightgbm.py factory-function style.
    return {
        "ZoneBoost": (
            lambda: ZoneBoostRegressor(categorical_features=cat_cols, random_state=RANDOM_STATE),
            adapt_native,
        ),
        "XGBoost": (
            lambda: XGBRegressor(random_state=RANDOM_STATE, enable_categorical=True),
            adapt_native,
        ),
        "LightGBM": (
            lambda: LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
            adapt_native,
        ),
        "CatBoost": (
            lambda: CatBoostRegressor(
                random_state=RANDOM_STATE, verbose=False,
                cat_features=cat_cols if cat_cols else None,
            ),
            adapt_catboost,
        ),
        "EBM": (
            # n_jobs=1 pinned: EBM's default (n_jobs=-2) spawns a joblib/loky
            # worker process per outer bag, and on Windows the fixed
            # process-spawn overhead dominates wall-clock time across many
            # small CV folds -- an environment/parallelism-backend artifact,
            # not a model change (outer_bags stays at its own default), see
            # benchmarks/README.md for the same issue in this project's
            # existing LightGBM comparison.
            lambda: ExplainableBoostingRegressor(random_state=RANDOM_STATE, n_jobs=1),
            adapt_ebm,
        ),
    }


# ---- classification models ------------------------------------------------

def classification_models(cat_cols, n_classes):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from interpret.glassbox import ExplainableBoostingClassifier
    from zoneboost import ZoneBoostClassifier

    return {
        "ZoneBoost": (
            lambda: ZoneBoostClassifier(categorical_features=cat_cols, random_state=RANDOM_STATE),
            adapt_native,
        ),
        "XGBoost": (
            lambda: XGBClassifier(random_state=RANDOM_STATE, enable_categorical=True),
            adapt_native,
        ),
        "LightGBM": (
            lambda: LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            adapt_native,
        ),
        "CatBoost": (
            lambda: CatBoostClassifier(
                random_state=RANDOM_STATE, verbose=False,
                cat_features=cat_cols if cat_cols else None,
            ),
            adapt_catboost,
        ),
        "EBM": (
            lambda: ExplainableBoostingClassifier(random_state=RANDOM_STATE, n_jobs=1),
            adapt_ebm,
        ),
    }
