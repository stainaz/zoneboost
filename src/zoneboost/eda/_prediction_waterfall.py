"""prediction_waterfall: the "direction included" ask from the notebook
pages, at the level of a single prediction -- every term's own signed
contribution, stacked from baseline to the final predicted value, so a
prediction reads like "income adds +$120, but age x region subtracts
$45" instead of a table of numbers.

Invents no new modeling math: every value plotted is read directly off
:meth:`zoneboost.ZoneBoostRegressor.explain`'s own already-exact,
already-summed-to-the-prediction row -- this module is purely a
rendering/ordering layer on top of it.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

__all__ = ["prediction_waterfall"]


def _explain(model, X, purify: bool):
    """Calls ``model.explain(X)``, forwarding ``purify`` only when the
    model's own ``explain`` actually accepts it -- ``ZoneBoostClassifier``
    has no ``purify`` parameter at all (purification is regressor-only),
    so passing it unconditionally would break every classifier, not just
    multiclass ones."""
    if "purify" in inspect.signature(model.explain).parameters:
        return model.explain(X, purify=purify)
    if purify:
        raise ValueError("purify=True is not supported by this model's own explain() (regressor-only).")
    return model.explain(X)

# Same validated palette as zone_boxplot/drift_dashboard (dataviz skill
# reference palette): categorical slot 1 (blue) for a positive
# contribution, status "critical" (red) for a negative one.
_POSITIVE_COLOR = "#2a78d6"
_NEGATIVE_COLOR = "#d03b3b"
_BASELINE_COLOR = "#898781"


def prediction_waterfall(model, X, index: int = 0, max_terms: int = None, purify: bool = False, plot: bool = True, ax=None):
    """Per-prediction, per-term waterfall: every term's own contribution
    to **one** row's prediction, ordered by ``|contribution|`` descending,
    stacked from ``baseline`` to the final predicted value.

    Always computes and returns the underlying waterfall table, whether
    or not a plot is drawn -- ``plot=False`` gives the identical numbers
    with no ``matplotlib`` dependency at all.

    Parameters
    ----------
    model : ZoneBoostRegressor or binary ZoneBoostClassifier
        Already fitted. Multiclass (3+ classes) is not supported --
        ``explain(X)`` returns a ``{class_label: DataFrame}`` dict there,
        not the single flat table this function needs; raises
        ``ValueError`` if given one.
    X : DataFrame or array-like of shape (n_samples, n_features)
        May hold more than one row -- only row ``index`` is explained.
    index : int, default=0
        Positional row of ``X`` to explain.
    max_terms : int, default=None
        Cap on how many individual terms are shown. ``None`` (default)
        shows every term. When set and there are more terms than this,
        the smallest-magnitude excess terms are folded into a single
        ``"other"`` row (their contributions summed) rather than dropped
        -- the waterfall still sums exactly to the prediction either way.
    purify : bool, default=False
        Forwarded to ``model.explain(X, purify=purify)`` -- see
        "Functional-ANOVA Purification" in the docs. Moves any signal in
        a pairwise-interaction term that's really just a function of one
        constituent column alone into that column's own main effect,
        before the waterfall is built.
    plot : bool, default=True
        If ``True`` (default), additionally draws the waterfall via
        ``matplotlib`` (a new import, only reached when ``plot=True`` --
        requires ``pip install zoneboost[eda]``). If ``False``, returns
        only the table with no ``matplotlib`` import at all.
    ax : matplotlib.axes.Axes, default=None
        Draw onto this existing ``Axes`` instead of creating a new
        figure. Ignored if ``plot=False``.

    Returns
    -------
    table : DataFrame
        One row per bar in waterfall order: ``"baseline"`` first, then
        every term sorted by ``|contribution|`` descending (an
        ``"other"`` row folding in the smallest-magnitude excess terms
        when ``max_terms`` truncates), and ``"total"`` last. Columns
        ``contribution`` (0 for the ``"baseline"``/``"total"`` rows --
        those are reference points, not contributions of their own),
        ``cumulative_start``, ``cumulative_end``. ``cumulative_end`` of
        the final term row equals ``cumulative_end``/``cumulative_start``
        of the ``"total"`` row, which equals ``model.predict(X.iloc[[index]])``.
    ax : matplotlib.axes.Axes
        Only returned when ``plot=True`` (in which case the return value
        is the tuple ``(table, ax)``, not just ``table``).
    """
    if not hasattr(model, "explain"):
        raise ValueError("model must expose an explain(X) method (e.g. ZoneBoostRegressor).")
    contrib = _explain(model, X, purify)
    if isinstance(contrib, dict):
        raise ValueError(
            "prediction_waterfall does not support multiclass classifiers -- explain(X) "
            "returns a {class_label: DataFrame} dict there, not the single flat table this "
            "function needs."
        )

    row = contrib.iloc[index]
    baseline = float(row["baseline"])
    terms = row.drop("baseline")
    total = float(baseline + terms.sum())

    ordered = terms.reindex(terms.abs().sort_values(ascending=False).index)
    if max_terms is not None and len(ordered) > max_terms:
        shown, excess = ordered.iloc[:max_terms], ordered.iloc[max_terms:]
        ordered = pd.concat([shown, pd.Series({"other": float(excess.sum())})])

    names = ["baseline", *ordered.index.tolist(), "total"]
    term_values = ordered.to_numpy(dtype=float).tolist()
    contributions = [0.0, *term_values, 0.0]

    # baseline/total are reference points, not deltas -- NaN start (there is
    # nothing "before" baseline in this chart, and total's own start is
    # meaningless once every term has already been applied); every term row's
    # start is the running total before it, end is the running total after.
    cumulative_start = [np.nan]
    cumulative_end = [baseline]
    running = baseline
    for value in term_values:
        cumulative_start.append(running)
        running += value
        cumulative_end.append(running)
    cumulative_start.append(np.nan)
    cumulative_end.append(running)

    table = pd.DataFrame(
        {"contribution": contributions, "cumulative_start": cumulative_start, "cumulative_end": cumulative_end},
        index=names,
    )

    if not plot:
        return table

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(table))))

    y_positions = np.arange(len(table))[::-1]
    for y, (name, r) in zip(y_positions, table.iterrows()):
        if name in ("baseline", "total"):
            ax.barh(y, r["cumulative_end"], color=_BASELINE_COLOR, alpha=0.6, height=0.6)
        else:
            left = min(r["cumulative_start"], r["cumulative_end"])
            width = abs(r["cumulative_end"] - r["cumulative_start"])
            color = _POSITIVE_COLOR if r["contribution"] >= 0 else _NEGATIVE_COLOR
            ax.barh(y, width, left=left, color=color, alpha=0.85, height=0.6)

    ax.axvline(baseline, color=_BASELINE_COLOR, linewidth=1, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(table.index)
    ax.set_xlabel("Contribution")
    ax.set_title(f"Prediction waterfall (row {index}) -> {total:.4g}")
    return table, ax
