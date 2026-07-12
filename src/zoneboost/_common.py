"""Shared input handling used by both ZoneBoostRegressor and
ZoneBoostClassifier -- kept in one place so the two estimators can't drift
apart on how they interpret X or auto-detect categorical columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array

__all__ = [
    "ensure_dataframe",
    "resolve_categorical_features",
    "resolve_monotonic_constraints",
    "resolve_bounded_effects",
    "resolve_forbidden_interactions",
]


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


def resolve_monotonic_constraints(X: pd.DataFrame, declared, categorical_features: set) -> dict:
    """Normalize a user-declared ``{column_name_or_index: +1 or -1}`` dict
    to ``{column_name: direction}``, the same name/index convention
    ``resolve_categorical_features`` uses.

    A constraint declared on a categorical column is silently dropped
    rather than raising: there's no meaningful order to constrain for a
    nominal category, and the rest of the library prefers graceful
    degradation over crashing on this kind of ambiguous input (the same
    treatment an unseen category or a missing value gets elsewhere).
    An invalid direction (anything other than -1 or 1) does raise --
    unlike a stray categorical key, that's simply a usage mistake with no
    sensible silent interpretation.

    Also reused as-is for ``convexity_constraints`` (``{column: +1 convex,
    -1 concave}``) -- identical shape and validation, this function doesn't
    care what the direction semantically means.
    """
    if not declared:
        return {}
    resolved = {}
    for f, direction in declared.items():
        if direction not in (-1, 1):
            raise ValueError(f"monotonic_constraints values must be -1 or 1, got {direction!r} for {f!r}")
        name = X.columns[f] if isinstance(f, (int, np.integer)) else f
        if name in categorical_features:
            continue
        resolved[name] = direction
    return resolved


def resolve_bounded_effects(X: pd.DataFrame, declared, categorical_features: set) -> dict:
    """Normalize a user-declared ``{column_name_or_index: (lower, upper)}``
    dict to ``{column_name: (lower, upper)}``, the same name/index
    convention ``resolve_categorical_features`` uses.

    A bound declared on a categorical column is silently dropped, same
    precedent as ``resolve_monotonic_constraints``. ``lower > upper``
    raises -- simply invalid, no sensible silent interpretation.
    """
    if not declared:
        return {}
    resolved = {}
    for f, bounds in declared.items():
        lower, upper = bounds
        if lower > upper:
            raise ValueError(f"bounded_effects lower bound must be <= upper bound, got {bounds!r} for {f!r}")
        name = X.columns[f] if isinstance(f, (int, np.integer)) else f
        if name in categorical_features:
            continue
        resolved[name] = (float(lower), float(upper))
    return resolved


def resolve_forbidden_interactions(X: pd.DataFrame, declared) -> set:
    """Normalize a user-declared list of 2-column name/index pairs to a
    ``set`` of 2-element ``frozenset``s of column names -- the same
    name/index convention ``resolve_categorical_features`` uses.

    An entry that doesn't name exactly 2 distinct columns raises: unlike a
    stray categorical key on ``monotonic_constraints``, this is simply a
    usage mistake with no sensible silent interpretation.
    """
    if not declared:
        return set()
    resolved = set()
    for pair in declared:
        names = {X.columns[f] if isinstance(f, (int, np.integer)) else f for f in pair}
        if len(names) != 2:
            raise ValueError(f"forbidden_interactions entries must name exactly 2 distinct columns, got {pair!r}")
        resolved.add(frozenset(names))
    return resolved


def resolve_group_col(X: pd.DataFrame, declared):
    """Normalize a user-declared column name/index (or ``None``) to a
    column name, the same name/index convention ``resolve_categorical_
    features`` uses.

    Unlike a stray categorical key on ``monotonic_constraints``, a
    ``group_col`` that doesn't name a real column is simply a usage
    mistake -- there's no sensible silent interpretation, so this raises.
    """
    if declared is None:
        return None
    name = X.columns[declared] if isinstance(declared, (int, np.integer)) else declared
    if name not in X.columns:
        raise ValueError(f"group_col={declared!r} is not a column of X.")
    return name
