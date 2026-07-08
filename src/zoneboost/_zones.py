"""Zone construction: the transparent, non-parametric binning mechanism
ZoneBoostRegressor uses in place of decision-tree splits.

Two independent mechanisms, selected per column:

- Continuous columns use :func:`adaptive_zone_boundaries`, which recursively
  finds the cut point that most reduces the target's within-segment
  variance -- the same criterion a regression tree split search uses --
  rather than fixed percentile/quantile bins.
- Categorical (nominal) columns use :func:`categorical_zone_map` /
  :func:`categorical_zone_index`, which give each distinct value its own
  zone directly, with no ordering assumption at all. A cut-point search is
  the wrong tool for a nominal variable: there is no reason two
  label-encoded values that happen to be numerically adjacent behave alike.

Every zone boundary or category mapping produced here is plain, inspectable
data -- a sorted array of numbers, or a dict from category to integer index.
Nothing is hidden inside a fitted model object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "zone_index",
    "adaptive_zone_boundaries",
    "categorical_zone_map",
    "categorical_zone_index",
]


def zone_index(values, boundaries: np.ndarray) -> np.ndarray:
    """Map each value to a zone index via ``searchsorted`` against sorted
    cut points.

    Parameters
    ----------
    values : array-like of shape (n_samples,)
        The raw values of one continuous column.
    boundaries : ndarray of shape (n_cuts,)
        Sorted cut points, as returned by :func:`adaptive_zone_boundaries`.

    Returns
    -------
    ndarray of shape (n_samples,)
        Integer zone index in ``[0, n_cuts]`` for each value.
    """
    return np.searchsorted(boundaries, np.asarray(values), side="right")


def _best_split(y_sorted_seg: np.ndarray, x_sorted_seg: np.ndarray, min_size: int):
    """Vectorized search for the single best variance-reducing split point
    within one x-sorted segment. Returns ``(split_index, gain, cut_value)``,
    or ``(None, 0.0, None)`` if nothing in this segment beats leaving it
    whole (too small, or every candidate split's gain is non-positive)."""
    n = len(y_sorted_seg)
    if n < 2 * min_size:
        return None, 0.0, None

    cum_y = np.cumsum(y_sorted_seg)
    cum_y2 = np.cumsum(y_sorted_seg**2)
    total_sum, total_sq = cum_y[-1], cum_y2[-1]
    total_ss = total_sq - (total_sum**2) / n

    i = np.arange(1, n)  # split after position i-1: left gets i points, right gets n-i
    left_n, right_n = i, n - i
    left_sum = cum_y[:-1]
    right_sum = total_sum - left_sum
    left_sq = cum_y2[:-1]
    right_sq = total_sq - left_sq
    left_ss = left_sq - (left_sum**2) / left_n
    right_ss = right_sq - (right_sum**2) / right_n
    gains = total_ss - (left_ss + right_ss)

    no_tie = x_sorted_seg[:-1] != x_sorted_seg[1:]
    valid = (left_n >= min_size) & (right_n >= min_size) & no_tie
    if not valid.any():
        return None, 0.0, None

    gains_valid = np.where(valid, gains, -np.inf)
    best_i = int(np.argmax(gains_valid))
    best_gain = gains_valid[best_i]
    if best_gain <= 0:
        return None, 0.0, None

    cut_value = (x_sorted_seg[best_i] + x_sorted_seg[best_i + 1]) / 2.0
    return best_i + 1, float(best_gain), float(cut_value)


def adaptive_zone_boundaries(
    x, y, max_zones: int = 7, min_zone_frac: float = 0.02, min_zone_abs: int = 20
) -> np.ndarray:
    """Variable-width zone boundaries for one continuous variable.

    Recursively cuts wherever a split most reduces the target's
    within-zone variance -- the same criterion a regression tree uses to
    pick a split -- instead of fixed quantile bins. A flat/unrelated
    variable may end up with very few zones; a variable with rich
    structure uses the full budget of up to ``max_zones``.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        The predictor column.
    y : array-like of shape (n_samples,)
        The target (or residual) to reduce variance of.
    max_zones : int, default=7
        Upper bound on the number of zones. Kept conservative by default:
        a higher ceiling gives more per-round fitting flexibility, which
        helps when a variable genuinely has many distinct meaningful
        regimes but otherwise mostly adds capacity to fit noise. See
        ``categorical_features`` on :class:`~zoneboost.ZoneBoostRegressor`
        for variables that need many distinct groups (use categorical
        handling instead of raising this).
    min_zone_frac : float, default=0.02
        Minimum fraction of rows required on each side of a candidate
        split.
    min_zone_abs : int, default=20
        Minimum absolute row count required on each side of a candidate
        split (the binding constraint on small datasets).

    Returns
    -------
    ndarray
        Sorted cut points; ``len(result) + 1`` zones result from them.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    order = np.argsort(x_arr)
    x_sorted, y_sorted = x_arr[order], y_arr[order]
    n = len(x_sorted)
    min_size = max(min_zone_abs, int(n * min_zone_frac))

    segments = [(0, n)]  # (start, end) index ranges into the sorted arrays
    cut_values = []

    while len(segments) < max_zones:
        best_gain, best_seg_i, best_split, best_cut = 0.0, None, None, None
        for seg_i, (start, end) in enumerate(segments):
            split, gain, cut = _best_split(y_sorted[start:end], x_sorted[start:end], min_size)
            if split is not None and gain > best_gain:
                best_gain, best_seg_i, best_split, best_cut = gain, seg_i, split, cut

        if best_seg_i is None:
            break  # no segment has a beneficial split left

        start, end = segments[best_seg_i]
        segments[best_seg_i : best_seg_i + 1] = [(start, start + best_split), (start + best_split, end)]
        cut_values.append(best_cut)

    return np.sort(np.array(cut_values, dtype=float))


def categorical_zone_map(series) -> dict:
    """Map each distinct value of a nominal column to its own zone index,
    in first-seen order. No cut-point search: there is no meaningful
    "adjacent" for categories with no true order, so a value simply gets
    its own zone.

    Parameters
    ----------
    series : array-like of shape (n_samples,)

    Returns
    -------
    dict
        Maps each distinct value to an integer zone index
        ``0, 1, ..., n_categories - 1``.
    """
    categories = pd.unique(np.asarray(series))
    return {cat: i for i, cat in enumerate(categories)}


def categorical_zone_index(series, category_map: dict) -> np.ndarray:
    """Look up each value's zone index via a stored category map.

    A value absent from ``category_map`` (an unseen category, typically
    encountered at predict time on new data) maps to a dedicated "unknown"
    zone one past the end. That zone has zero fit-time support, so it
    naturally carries zero confidence and contributes nothing to a
    prediction, rather than raising an error.

    Parameters
    ----------
    series : array-like of shape (n_samples,)
    category_map : dict
        As returned by :func:`categorical_zone_map`.

    Returns
    -------
    ndarray of shape (n_samples,)
    """
    unknown_idx = len(category_map)
    mapped = pd.Series(np.asarray(series)).map(category_map)
    return mapped.fillna(unknown_idx).astype(int).to_numpy()
