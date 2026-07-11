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
    "zone_centers",
    "adaptive_zone_boundaries",
    "categorical_zone_map",
    "categorical_zone_index",
]


def zone_index(values, boundaries: np.ndarray) -> np.ndarray:
    """Map each value to a zone index via ``searchsorted`` against sorted
    cut points. A missing (NaN) value maps to a dedicated "missing" zone
    one past the last regular zone -- ``searchsorted`` against NaN is
    undefined, so this is handled explicitly rather than left to whatever
    numpy happens to do with it.

    Parameters
    ----------
    values : array-like of shape (n_samples,)
        The raw values of one continuous column.
    boundaries : ndarray of shape (n_cuts,)
        Sorted cut points, as returned by :func:`adaptive_zone_boundaries`.

    Returns
    -------
    ndarray of shape (n_samples,)
        Integer zone index in ``[0, n_cuts]`` for each present value, or
        ``n_cuts + 1`` (the missing zone) for NaN.
    """
    arr = np.asarray(values, dtype=float)
    is_missing = np.isnan(arr)
    idx = np.searchsorted(boundaries, arr, side="right")
    missing_idx = len(boundaries) + 1
    return np.where(is_missing, missing_idx, idx)


def zone_centers(x, boundaries: np.ndarray) -> np.ndarray:
    """Each *real* zone's centroid: the empirical mean training-x-value of
    the rows that landed in it (via :func:`zone_index`) -- one entry per
    real zone (``len(boundaries) + 1`` of them; the missing zone has no
    centroid, since it's never interpolated into or out of).

    Used to interpolate a lookup between two zones' fitted values rather
    than hard-assigning a value to exactly one zone (see
    ``_weak_learner._column_soft_zone_index``) -- a value exactly at its
    own zone's centroid is fully that zone; moving toward a neighboring
    zone's centroid blends linearly toward it.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        The same training column ``boundaries`` was fit from.
    boundaries : ndarray of shape (n_cuts,)

    Returns
    -------
    ndarray of shape (n_cuts + 1,)
        A pathologically empty zone (shouldn't normally happen given
        :func:`adaptive_zone_boundaries`'s own min-size guards) falls back
        to its own boundary midpoint rather than producing NaN.
    """
    x_arr = np.asarray(x, dtype=float)
    x_present = x_arr[~np.isnan(x_arr)]
    n_real = len(boundaries) + 1
    z = np.searchsorted(boundaries, x_present, side="right")

    sums = np.bincount(z, weights=x_present, minlength=n_real)
    counts = np.bincount(z, minlength=n_real)
    centers = np.divide(sums, counts, out=np.full(n_real, np.nan), where=counts > 0)

    empty = counts == 0
    if np.any(empty):
        if len(boundaries) > 0:
            edges = np.concatenate(([boundaries[0] - 1.0], boundaries, [boundaries[-1] + 1.0]))
        else:
            edges = np.array([-1.0, 1.0])
        midpoints = (edges[:-1] + edges[1:]) / 2.0
        centers = np.where(empty, midpoints, centers)
    return centers


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
        Sorted cut points; ``len(result) + 1`` regular zones result from
        them, plus one further "missing" zone reserved for NaN (see
        :func:`zone_index`) -- missing rows never enter this search, since
        a NaN sorts to an arbitrary position and would otherwise corrupt
        whichever segment's cumulative sums it lands in, including the cut
        *value* itself.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    present = ~np.isnan(x_arr)
    x_arr, y_arr = x_arr[present], y_arr[present]
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
    """Map each distinct, present value of a nominal column to its own
    zone index, in first-seen order. No cut-point search: there is no
    meaningful "adjacent" for categories with no true order, so a value
    simply gets its own zone.

    Missing values (NaN/None) are deliberately excluded from this map --
    they get their own dedicated zone in :func:`categorical_zone_index`,
    kept separate from "an unseen but real category", rather than being
    folded in as just another dict key (which is what a raw NaN key would
    otherwise become, relying on a fragile identity-comparison quirk in
    Python dict lookups rather than an explicit rule).

    Parameters
    ----------
    series : array-like of shape (n_samples,)

    Returns
    -------
    dict
        Maps each distinct present value to an integer zone index
        ``0, 1, ..., n_categories - 1``.
    """
    arr = np.asarray(series, dtype=object)
    present = arr[~pd.isna(arr)]
    categories = pd.unique(present)
    return {cat: i for i, cat in enumerate(categories)}


def categorical_zone_index(series, category_map: dict) -> np.ndarray:
    """Look up each value's zone index via a stored category map.

    Two distinct fallback zones, both one-past-the-regular-categories:
    a missing (NaN/None) value gets its own dedicated zone, kept separate
    from an unseen-but-real category (a value that exists but wasn't
    present at fit time, typically encountered at predict time on new
    data), which gets the zone after that. Both start with zero fit-time
    support, so they naturally carry zero confidence and contribute
    nothing to a prediction rather than raising an error or silently
    colliding with a real category's zone.

    Parameters
    ----------
    series : array-like of shape (n_samples,)
    category_map : dict
        As returned by :func:`categorical_zone_map`.

    Returns
    -------
    ndarray of shape (n_samples,)
    """
    arr = np.asarray(series, dtype=object)
    is_missing = pd.isna(arr)
    n_categories = len(category_map)
    missing_idx = n_categories
    unknown_idx = n_categories + 1

    result = np.full(len(arr), unknown_idx, dtype=int)
    result[is_missing] = missing_idx
    present_mask = ~is_missing
    mapped = pd.Series(arr[present_mask]).map(category_map)
    result[present_mask] = mapped.fillna(unknown_idx).astype(int).to_numpy()
    return result
