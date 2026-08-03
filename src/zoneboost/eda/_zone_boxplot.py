"""zone_boxplot: the visual, business-friendly layer the notebook pages
asked for -- "compare outcome & variables boxplots" -- built on top of the
identical zone construction :class:`zoneboost.ZoneProfileEncoder` already
uses, not a new binning mechanism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._common import resolve_categorical_features
from .._zones import (
    adaptive_zone_boundaries,
    categorical_zone_index,
    categorical_zone_map,
    zone_index,
)

__all__ = ["zone_boxplot"]

# Validated per the dataviz skill's reference palette (references/palette.md):
# categorical slot 1 (blue) for an ordinary zone, status "critical" (red) for
# a zone flagged by valid_range -- distinct hue families so a flagged zone
# never gets confused for "just another series."
_ZONE_COLOR = "#2a78d6"
_INVALID_COLOR = "#d03b3b"
_MISSING_COLOR = "#898781"


def _zone_labels(kind: str, boundaries, category_map: dict, n_real: int) -> list:
    if kind == "categorical":
        by_index = {i: cat for cat, i in category_map.items()}
        labels = [str(by_index[i]) for i in range(n_real)]
        return labels + ["missing", "unseen"]
    bounds = boundaries.tolist()
    if n_real == 1:
        labels = ["all"]
    else:
        labels = [f"< {bounds[0]:.3g}"]
        labels += [f"[{bounds[i]:.3g}, {bounds[i + 1]:.3g})" for i in range(len(bounds) - 1)]
        labels.append(f">= {bounds[-1]:.3g}")
    return labels + ["missing"]


def zone_boxplot(
    X,
    y,
    column,
    n_zones: int = 7,
    min_zone_frac: float = 0.02,
    min_zone_abs: int = 20,
    split_criterion: str = "variance",
    valid_range: tuple = None,
    plot: bool = True,
    ax=None,
):
    """Per-zone outcome distribution for one column -- the notebook's own
    "compare outcome & variables boxplots," reusing the exact zone
    construction :class:`zoneboost.ZoneProfileEncoder` already uses for
    ``column`` (adaptive variance/correlation-reducing boundaries for a
    continuous column, one zone per distinct value for a categorical one)
    rather than a fixed quantile-bin boxplot-by-decile.

    Always computes and returns the per-zone statistics table, whether or
    not a plot is drawn -- ``plot=False`` gives the identical numbers with
    no ``matplotlib`` dependency at all.

    Parameters
    ----------
    X : DataFrame or array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        The outcome whose per-zone distribution is being compared.
    column : str or int
        The column (name, or index if ``X`` isn't a DataFrame) to zone.
    n_zones : int, default=7
        Forwarded as ``max_zones`` to :func:`zoneboost._zones.
        adaptive_zone_boundaries`. Ignored for a categorical column (one
        zone per distinct value, no cap).
    min_zone_frac : float, default=0.02
        Forwarded to :func:`zoneboost._zones.adaptive_zone_boundaries`.
        Ignored for a categorical column.
    min_zone_abs : int, default=20
        Forwarded to :func:`zoneboost._zones.adaptive_zone_boundaries`.
        Ignored for a categorical column.
    split_criterion : {"variance", "correlation"}, default="variance"
        Forwarded to :func:`zoneboost._zones.adaptive_zone_boundaries` --
        see "Correlation-aware zone boundaries" in the docs. Ignored for a
        categorical column.
    valid_range : tuple of (lower, upper), default=None
        Caller-declared domain knowledge about ``column``'s own valid raw
        values (e.g. ``(18, 100)`` for an age column) -- the same
        "declare it, don't infer it" precedent
        ``monotonic_constraints``/``bounded_effects`` already set on
        :class:`zoneboost.ZoneBoostRegressor`. When given, each zone's
        ``pct_business_invalid`` column reports the fraction of that
        zone's own rows whose *raw* ``column`` value falls outside this
        range (a zone straddling the boundary gets a real fraction, not
        just 0 or 1). ``None`` (default) omits that column entirely --
        there is no statistical way to infer domain validity on its own.
        Raises ``ValueError`` if ``column`` is categorical (a numeric
        bound has no meaning there).
    plot : bool, default=True
        If ``True`` (default), additionally draws a boxplot with one box
        per zone via ``matplotlib`` (a new import, only reached when
        ``plot=True`` -- requires ``pip install zoneboost[eda]``), coloring
        any zone with ``pct_business_invalid > 0`` distinctly from an
        ordinary zone. If ``False``, returns only ``stats`` with no
        ``matplotlib`` import at all.
    ax : matplotlib.axes.Axes, default=None
        Draw onto this existing ``Axes`` instead of creating a new
        figure. Ignored if ``plot=False``.

    Returns
    -------
    stats : DataFrame
        Indexed by zone label (a value range like ``"[2, 5)"`` for a
        continuous column, or the category name for a categorical one,
        plus a trailing ``"missing"``/``"unseen"`` row only if any row
        actually landed there), columns ``count``, ``mean``, ``median``,
        ``q1``, ``q3``, ``iqr``, ``outlier_count`` (rows outside
        ``[q1 - 1.5*iqr, q3 + 1.5*iqr]``, the standard boxplot rule,
        applied to ``y`` within that zone), ``outlier_frac``, and (only
        when ``valid_range`` is given) ``pct_business_invalid``.
    ax : matplotlib.axes.Axes
        Only returned when ``plot=True`` (in which case the return value
        is the tuple ``(stats, ax)``, not just ``stats``).
    """
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if len(X) != len(y_arr):
        raise ValueError(f"X and y have inconsistent lengths: {len(X)} vs {len(y_arr)}")
    col = X.columns[column] if isinstance(column, (int, np.integer)) else column
    if col not in X.columns:
        raise ValueError(f"column {column!r} not found in X.")
    series = X[col]

    categorical = resolve_categorical_features(X[[col]], None)
    is_categorical = col in categorical
    if is_categorical and valid_range is not None:
        raise ValueError(
            f"valid_range has no meaning for categorical column {col!r} -- there is no "
            "numeric ordering to bound."
        )

    if is_categorical:
        category_map = categorical_zone_map(series)
        zone_idx = categorical_zone_index(series, category_map)
        n_real = len(category_map)
        labels = _zone_labels("categorical", None, category_map, n_real)
        raw_for_validity = None
    else:
        x_arr = series.to_numpy(dtype=float)
        boundaries = adaptive_zone_boundaries(
            x_arr, y_arr, max_zones=n_zones, min_zone_frac=min_zone_frac,
            min_zone_abs=min_zone_abs, split_criterion=split_criterion,
        )
        zone_idx = zone_index(x_arr, boundaries)
        n_real = len(boundaries) + 1
        labels = _zone_labels("continuous", boundaries, None, n_real)
        raw_for_validity = x_arr

    rows = []
    for z in sorted(np.unique(zone_idx)):
        mask = zone_idx == z
        y_zone = y_arr[mask]
        count = int(mask.sum())
        q1, median, q3 = np.percentile(y_zone, [25, 50, 75])
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(np.sum((y_zone < lo_fence) | (y_zone > hi_fence)))
        row = {
            "zone": labels[z],
            "count": count,
            "mean": float(y_zone.mean()),
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "outlier_count": outlier_count,
            "outlier_frac": outlier_count / count if count else 0.0,
        }
        if valid_range is not None:
            lo, hi = valid_range
            raw_zone = raw_for_validity[mask]
            present = ~np.isnan(raw_zone)
            invalid = present & ((raw_zone < lo) | (raw_zone > hi))
            row["pct_business_invalid"] = float(invalid.sum() / present.sum()) if present.any() else 0.0
        rows.append((z, row))

    rows.sort(key=lambda item: item[0])
    stats = pd.DataFrame([r for _, r in rows]).set_index("zone")

    if not plot:
        return stats

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.2 * len(stats)), 5))

    box_data = [y_arr[zone_idx == z] for z, _ in rows]
    positions = np.arange(1, len(box_data) + 1)
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
    invalid_flags = (
        stats["pct_business_invalid"].to_numpy() > 0
        if "pct_business_invalid" in stats.columns
        else np.zeros(len(stats), dtype=bool)
    )
    missing_flags = stats.index.to_series().isin(["missing", "unseen"]).to_numpy()
    for patch, invalid, missing in zip(bp["boxes"], invalid_flags, missing_flags):
        if missing:
            patch.set_facecolor(_MISSING_COLOR)
        elif invalid:
            patch.set_facecolor(_INVALID_COLOR)
        else:
            patch.set_facecolor(_ZONE_COLOR)
        patch.set_alpha(0.75)

    ax.set_xticks(positions)
    ax.set_xticklabels(stats.index, rotation=30, ha="right")
    ax.set_xlabel(col)
    ax.set_ylabel("y")
    ax.set_title(f"Outcome distribution by {col} zone")
    return stats, ax
