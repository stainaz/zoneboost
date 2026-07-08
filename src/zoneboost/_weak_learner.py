"""One boosting round's weak learner: per-column zone info, main effects
(single-variable zone -> average residual), interactions (variable-pair
zone grid -> average residual), and density-based confidence weighting.

Every "weak learner" in ZoneBoostRegressor is built from this module alone
-- no decision tree, no gradient computation beyond a plain residual, no
external model of any kind. What changes round to round is only the target
these functions are pointed at (the current residual) and which rows/
columns were sampled for that round.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ._zones import adaptive_zone_boundaries, categorical_zone_index, categorical_zone_map, zone_index

__all__ = ["weak_learner_fit", "weak_learner_score"]


def _zone_deviation_confidence(zone_values: np.ndarray, target_values: np.ndarray, overall_mean: float, n_zones: int):
    """For each zone: the actual average target among fit-rows in that
    zone (minus the overall mean), and a density-confidence weight
    relative to the best-supported zone. O(n) via bincount."""
    counts = np.bincount(zone_values, minlength=n_zones).astype(float)
    sums = np.bincount(zone_values, weights=target_values, minlength=n_zones)
    cell_mean = np.where(counts > 0, np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0), overall_mean)
    deviation = cell_mean - overall_mean
    base = counts.max()
    confidence = counts / base if base > 0 else counts
    return deviation, confidence


def _pair_deviation_confidence(
    za: np.ndarray, zb: np.ndarray, target_values: np.ndarray, overall_mean: float, n_zones_a: int, n_zones_b: int
):
    """Same idea, gridded over two variables' zones jointly. Combines both
    zone indices into one flat index and does a single bincount pass."""
    combined = za * n_zones_b + zb
    size = n_zones_a * n_zones_b
    counts = np.bincount(combined, minlength=size).astype(float).reshape(n_zones_a, n_zones_b)
    sums = np.bincount(combined, weights=target_values, minlength=size).reshape(n_zones_a, n_zones_b)
    cell_mean = np.where(counts > 0, np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0), overall_mean)
    deviation = cell_mean - overall_mean
    base = counts.max()
    confidence = counts / base if base > 0 else counts
    return deviation, confidence


def _column_zone_info(x_col: pd.Series, residual: np.ndarray, is_categorical: bool, max_zones: int, min_zone_frac: float):
    """Returns a ``("continuous", boundaries)`` or ``("categorical",
    category_map)`` tagged tuple -- the one place that decides which zone
    mechanism a column uses."""
    if is_categorical:
        return ("categorical", categorical_zone_map(x_col))
    col_cap = min(max_zones, x_col.nunique())
    bounds = adaptive_zone_boundaries(x_col, residual, max_zones=col_cap, min_zone_frac=min_zone_frac)
    return ("continuous", bounds)


def _column_zone_index(x_col: pd.Series, info: tuple) -> np.ndarray:
    kind, payload = info
    if kind == "categorical":
        return categorical_zone_index(x_col, payload)
    return zone_index(x_col, payload)


def _column_n_zones(info: tuple) -> int:
    kind, payload = info
    if kind == "categorical":
        return len(payload) + 2  # +2: dedicated "missing" zone, dedicated "unseen category" zone
    return len(payload) + 2  # +2: last cut point, dedicated "missing" zone


def weak_learner_fit(
    X: pd.DataFrame,
    residual: np.ndarray,
    predictor_subset: list,
    categorical_features: set,
    max_zones: int = 7,
    min_zone_frac: float = 0.02,
):
    """Fit one boosting round's weak learner: zone info (adaptive-continuous
    or exact-categorical per column), main effects, and interactions --
    ALL derived fresh from this round's (already row/column-subsampled)
    residual.

    Returns
    -------
    zone_info : dict
        column -> ``("continuous", boundaries)`` or ``("categorical", map)``.
    main_effects : dict
        column -> ``(deviation, confidence)`` arrays.
    interactions : dict
        ``(col_a, col_b)`` -> ``(deviation, confidence)`` 2D arrays.
    overall_mean : float
        The residual's mean this round -- the weak learner's own baseline.
    """
    zone_info = {
        c: _column_zone_info(X[c], residual, c in categorical_features, max_zones, min_zone_frac)
        for c in predictor_subset
    }
    n_zones = {c: _column_n_zones(zone_info[c]) for c in predictor_subset}
    zones = {c: _column_zone_index(X[c], zone_info[c]) for c in predictor_subset}
    overall_mean = float(residual.mean())

    main_effects = {
        col: _zone_deviation_confidence(zones[col], residual, overall_mean, n_zones[col]) for col in predictor_subset
    }
    interactions = {
        (a, b): _pair_deviation_confidence(zones[a], zones[b], residual, overall_mean, n_zones[a], n_zones[b])
        for a, b in itertools.combinations(predictor_subset, 2)
    }
    return zone_info, main_effects, interactions, overall_mean


def weak_learner_score(X: pd.DataFrame, zone_info: dict, main_effects: dict, interactions: dict) -> np.ndarray:
    """Score rows with an already-fit weak learner. Self-sufficient from
    main_effects/interactions' own keys, so it works whether that round
    used every predictor or only a random subset of them."""
    needed_cols = set(main_effects.keys())
    for a, b in interactions.keys():
        needed_cols.add(a)
        needed_cols.add(b)
    zones = {c: _column_zone_index(X[c], zone_info[c]) for c in needed_cols}

    contributions = []
    for col, (deviation, confidence) in main_effects.items():
        z = zones[col]
        contributions.append(deviation[z] * confidence[z])
    for (a, b), (deviation, confidence) in interactions.items():
        za, zb = zones[a], zones[b]
        contributions.append(deviation[za, zb] * confidence[za, zb])
    return np.column_stack(contributions).mean(axis=1)
