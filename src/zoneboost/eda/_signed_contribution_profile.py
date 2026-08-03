"""signed_contribution_profile: the "direction included" ask from the
notebook pages, at the level of a single feature -- how does this
feature's own main-effect contribution rise or fall as its value changes,
zone by zone, sign-colored. This is what the sketches' "new profile
showing the range of important" was reaching for -- a profile, not just
a table of numbers.

Reuses the exact same zone construction :class:`zoneboost.
ZoneProfileEncoder`/:func:`zoneboost.eda.zone_boxplot` already use, this
time binning against the feature's own already-computed contribution
column (from :meth:`zoneboost.ZoneBoostRegressor.explain`) rather than
the real target -- no new modeling math, a different "y" fed through the
identical machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._common import ensure_dataframe, resolve_categorical_features
from .._zones import (
    adaptive_zone_boundaries,
    categorical_zone_index,
    categorical_zone_map,
    zone_index,
)
from ._prediction_waterfall import _explain
from ._zone_boxplot import _zone_labels

__all__ = ["signed_contribution_profile"]

# Same validated palette as prediction_waterfall/zone_boxplot (dataviz
# skill reference palette): categorical slot 1 (blue) for a net-positive
# zone, status "critical" (red) for a net-negative one.
_POSITIVE_COLOR = "#2a78d6"
_NEGATIVE_COLOR = "#d03b3b"


def signed_contribution_profile(
    model,
    X,
    feature,
    n_zones: int = 7,
    min_zone_frac: float = 0.02,
    min_zone_abs: int = 20,
    split_criterion: str = "variance",
    purify: bool = False,
    plot: bool = True,
    ax=None,
):
    """How ``feature``'s own main-effect contribution rises or falls
    across its range, zone by zone -- "income adds +$120 once it clears
    $90k, but subtracts below $20k," read directly off the fitted model
    rather than eyeballed from a table.

    Bins ``feature`` into zones using the *exact same* construction
    :class:`zoneboost.ZoneProfileEncoder` already uses (adaptive
    variance/correlation-reducing boundaries for a continuous column --
    pass ``split_criterion="correlation"`` to reuse "Correlation-aware
    zone boundaries" -- one zone per distinct value for a categorical
    one), except the quantity being binned against is ``feature``'s own
    already-computed contribution column from ``explain(X)``, not the
    real target ``y``: each zone's own **mean contribution** is this
    profile's value for that zone.

    Always computes and returns this per-zone table, whether or not a
    plot is drawn -- ``plot=False`` gives the identical numbers with no
    ``matplotlib`` dependency at all.

    Parameters
    ----------
    model : ZoneBoostRegressor or binary ZoneBoostClassifier
        Already fitted.
    X : DataFrame or array-like of shape (n_samples, n_features)
    feature : str or int
        The column (name, or index if ``X`` isn't a DataFrame) to
        profile. Must have appeared as its own **main effect** term in
        ``explain(X)`` -- raises ``ValueError`` otherwise (a column only
        ever fit inside an interaction has no own contribution column to
        profile; this function doesn't attempt to fold an interaction's
        contribution into its constituent column's own profile).
    n_zones : int, default=7
        Forwarded as ``max_zones`` to ``adaptive_zone_boundaries``;
        ignored for a categorical column.
    min_zone_frac : float, default=0.02
        Forwarded to ``adaptive_zone_boundaries``; ignored for a
        categorical column.
    min_zone_abs : int, default=20
        Forwarded to ``adaptive_zone_boundaries``; ignored for a
        categorical column.
    split_criterion : {"variance", "correlation"}, default="variance"
        Forwarded to ``adaptive_zone_boundaries`` -- see
        "Correlation-aware zone boundaries" in the docs; ignored for a
        categorical column.
    purify : bool, default=False
        Forwarded to ``model.explain(X, purify=purify)`` before
        extracting ``feature``'s own contribution column -- see
        "Functional-ANOVA Purification" in the docs.
    plot : bool, default=True
        If ``True`` (default), additionally draws a bar chart via
        ``matplotlib`` (a new import, only reached when ``plot=True`` --
        requires ``pip install zoneboost[eda]``), one bar per zone in
        natural (ascending-value) order, colored by the sign of that
        zone's own mean contribution. If ``False``, returns only the
        table with no ``matplotlib`` import at all.
    ax : matplotlib.axes.Axes, default=None
        Draw onto this existing ``Axes`` instead of creating a new
        figure. Ignored if ``plot=False``.

    Returns
    -------
    table : DataFrame
        Indexed by zone label (a value range like ``"[2, 5)"`` for a
        continuous column, or the category name for a categorical one,
        in natural — not magnitude — order), columns ``count`` and
        ``mean_contribution``.
    ax : matplotlib.axes.Axes
        Only returned when ``plot=True`` (in which case the return value
        is the tuple ``(table, ax)``, not just ``table``).
    """
    contrib = _explain(model, X, purify)
    if isinstance(contrib, dict):
        raise ValueError(
            "signed_contribution_profile does not support multiclass classifiers -- "
            "explain(X) returns a {class_label: DataFrame} dict there, not the single "
            "flat table this function needs."
        )

    X = ensure_dataframe(X, None)
    col = X.columns[feature] if isinstance(feature, (int, np.integer)) else feature
    if col not in X.columns:
        raise ValueError(f"feature {feature!r} not found in X.")
    if col not in contrib.columns:
        raise ValueError(
            f"feature {col!r} never appeared as its own main-effect term in explain(X) -- "
            "only a column fit as a main effect at least once has a contribution column of "
            "its own to profile (a column that only ever entered an interaction has no such "
            "column)."
        )

    contribution = contrib[col].to_numpy(dtype=float)
    series = X[col]

    categorical = resolve_categorical_features(X[[col]], None)
    is_categorical = col in categorical

    if is_categorical:
        category_map = categorical_zone_map(series)
        zone_idx = categorical_zone_index(series, category_map)
        n_real = len(category_map)
        labels = _zone_labels("categorical", None, category_map, n_real)
    else:
        x_arr = series.to_numpy(dtype=float)
        boundaries = adaptive_zone_boundaries(
            x_arr, contribution, max_zones=n_zones, min_zone_frac=min_zone_frac,
            min_zone_abs=min_zone_abs, split_criterion=split_criterion,
        )
        zone_idx = zone_index(x_arr, boundaries)
        n_real = len(boundaries) + 1
        labels = _zone_labels("continuous", boundaries, None, n_real)

    rows = []
    for z in sorted(np.unique(zone_idx)):
        mask = zone_idx == z
        rows.append((z, labels[z], int(mask.sum()), float(contribution[mask].mean())))

    table = pd.DataFrame(
        [{"count": c, "mean_contribution": m} for _, _, c, m in rows],
        index=[label for _, label, _, _ in rows],
    )

    if not plot:
        return table

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.2 * len(table)), 5))

    positions = np.arange(len(table))
    colors = [_POSITIVE_COLOR if v >= 0 else _NEGATIVE_COLOR for v in table["mean_contribution"]]
    ax.bar(positions, table["mean_contribution"], color=colors, alpha=0.85)
    ax.axhline(0, color="#898781", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(table.index, rotation=30, ha="right")
    ax.set_xlabel(col)
    ax.set_ylabel("Mean contribution")
    ax.set_title(f"Signed contribution profile: {col}")
    return table, ax
