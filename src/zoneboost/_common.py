"""Shared input handling used by both ZoneBoostRegressor and
ZoneBoostClassifier -- kept in one place so the two estimators can't drift
apart on how they interpret X or auto-detect categorical columns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance, mean_tweedie_deviance
from sklearn.utils.validation import check_array

__all__ = [
    "ensure_dataframe",
    "resolve_categorical_features",
    "resolve_monotonic_constraints",
    "resolve_bounded_effects",
    "resolve_forbidden_interactions",
    "resolve_group_col",
    "_glm_residual",
    "_glm_inverse_link",
    "_glm_baseline",
    "_glm_deviance_score",
]

_GLM_POWERS = {"poisson": 1.0, "gamma": 2.0}


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


def _glm_power(loss: str, tweedie_power: float) -> float:
    """The Tweedie variance power unifying the three GLM losses --
    Poisson (``p=1``) and Gamma (``p=2``) are fixed special cases of the
    same family, ``loss="tweedie"`` exposes ``p`` directly (default
    ``1.5``, the usual insurance pure-premium setting)."""
    return _GLM_POWERS.get(loss, tweedie_power)


def _glm_inverse_link(eta: np.ndarray) -> np.ndarray:
    """Log link's inverse: ``mu = exp(eta)``. Clipped before
    exponentiating (a plain numerical-stability guard, not a modeling
    choice) so a large intermediate boosting score can't overflow
    ``np.exp`` into ``inf``."""
    return np.exp(np.clip(eta, -30.0, 30.0))


def _glm_residual(y: np.ndarray, mu: np.ndarray, power: float) -> np.ndarray:
    """Negative deviance gradient w.r.t. the link-scale linear predictor
    ``eta = log(mu)``, for the Tweedie family at variance power ``power``
    -- what gets boosted, exactly the role ``y - current_pred`` plays for
    ``loss="squared_error"`` and ``y - sigmoid(current_score)`` plays for
    :class:`zoneboost.ZoneBoostClassifier`'s log-odds booster.

    ``residual = mu**(1 - power) * (y - mu)`` -- reduces to ``y - mu`` at
    ``power=1`` (Poisson) and ``(y - mu) / mu`` at ``power=2`` (Gamma),
    the standard boosting residuals for each family; verified directly
    against both in tests.
    """
    return mu ** (1.0 - power) * (y - mu)


def _glm_baseline(y: np.ndarray, offset: np.ndarray, power: float, sample_weight: np.ndarray = None) -> float:
    """Best constant link-scale predictor given a (possibly nonzero,
    per-row) fixed ``offset``: ``log(sum(y) / sum(exp(offset)))`` -- the
    exact intercept-only MLE for Poisson with a fixed offset (the score
    equation ``sum(y) = sum(exp(beta0 + offset))`` solved in closed
    form). Reused as-is for Gamma/Tweedie: not their exact MLE, but the
    same mean-matching, disclosed approximation quantile-mode's
    machinery already relies on elsewhere in this codebase.

    ``sample_weight`` (default ``None``, bit-identical to every prior
    release) generalizes the same closed form to
    ``log(sum(w*y) / sum(w*exp(offset)))`` -- the weighted score
    equation ``sum(w*y) = sum(w*exp(beta0 + offset))``."""
    if sample_weight is None:
        return float(np.log(np.sum(y) / np.sum(np.exp(offset))))
    return float(np.log(np.sum(sample_weight * y) / np.sum(sample_weight * np.exp(offset))))


def _glm_deviance_score(
    y: np.ndarray, mu: np.ndarray, loss: str, tweedie_power: float, sample_weight: np.ndarray = None
) -> float:
    """Mean deviance for whichever GLM loss is active -- the ``_score``
    role RMSE/pinball loss play for ``squared_error``/``quantile``.
    Dispatches directly to scikit-learn's own, already-tested
    ``mean_poisson_deviance``/``mean_gamma_deviance``/
    ``mean_tweedie_deviance``, not a hand-derived formula. ``sample_
    weight`` (default ``None``) is passed straight through -- all three
    accept it natively."""
    if loss == "poisson":
        return float(mean_poisson_deviance(y, mu, sample_weight=sample_weight))
    if loss == "gamma":
        return float(mean_gamma_deviance(y, mu, sample_weight=sample_weight))
    return float(mean_tweedie_deviance(y, mu, power=tweedie_power, sample_weight=sample_weight))
