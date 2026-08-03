"""drift_dashboard: the visual layer over :func:`zoneboost.compare_models`
-- the notebook pages' own "the model will keep changing so using older
data might affect its performance" concern, made a picture instead of a
dict a human has to read field by field.

Computes zero new statistics of its own: every number plotted comes
directly from :func:`zoneboost.compare_models`'s already-shipped,
already-exact comparison (feature-importance change, boundary shift,
population migration, prediction shift) -- this module is purely a
rendering layer on top of it.
"""

from __future__ import annotations

import numpy as np

from .._drift import compare_models

__all__ = ["drift_dashboard"]

# Same validated palette as zone_boxplot (dataviz skill reference palette):
# categorical slots 1 (blue) / 2 (orange) for "old" vs "new", and the
# blue<->red diverging pair for a signed increase/decrease.
_OLD_COLOR = "#2a78d6"
_NEW_COLOR = "#eb6834"
_INCREASE_COLOR = "#2a78d6"
_DECREASE_COLOR = "#d03b3b"
_MIGRATION_COLOR = "#2a78d6"


def drift_dashboard(model_old, model_new, X_eval, y_eval=None, plot: bool = True, top_n: int = 15):
    """Visual comparison of two already-fitted
    :class:`zoneboost.ZoneBoostRegressor` snapshots -- a multi-panel
    figure over :func:`zoneboost.compare_models`'s own comparison dict,
    not a new statistic.

    Parameters
    ----------
    model_old, model_new : ZoneBoostRegressor
        Both already fitted. Forwarded as-is to
        :func:`zoneboost.compare_models`.
    X_eval : DataFrame or array-like of shape (n_samples, n_features)
    y_eval : array-like, default=None
        Forwarded to :func:`zoneboost.compare_models` -- if given, the
        performance change is annotated on the figure.
    plot : bool, default=True
        If ``True`` (default), additionally renders the multi-panel
        figure via ``matplotlib`` (a new import, only reached when
        ``plot=True`` -- requires ``pip install zoneboost[eda]``). If
        ``False``, returns exactly :func:`zoneboost.compare_models`'s own
        dict -- a pure passthrough, useful for toggling visualization on
        or off without switching functions.
    top_n : int, default=15
        Number of terms shown in the feature-importance-change panel,
        ranked by ``|change|`` (already the sort order
        ``compare_models`` returns). Ignored if ``plot=False``.

    Returns
    -------
    data : dict
        Exactly :func:`zoneboost.compare_models`'s own return value.
    fig : matplotlib.figure.Figure
        Only returned when ``plot=True`` (in which case the return value
        is the tuple ``(data, fig)``, not just ``data``).
    """
    data = compare_models(model_old, model_new, X_eval, y_eval)
    if not plot:
        return data

    import matplotlib.pyplot as plt

    boundary_shift = data["boundary_shift"]
    population_migration = data["population_migration"]
    has_boundary_panel = len(boundary_shift) > 0
    n_panels = 2 + int(has_boundary_panel) + int(len(population_migration) > 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    axes = np.atleast_1d(axes)
    panel = iter(axes)

    # Panel 1: feature-importance-change tornado chart.
    ax = next(panel)
    fic = data["feature_importance_change"].head(top_n).iloc[::-1]
    colors = [_INCREASE_COLOR if v >= 0 else _DECREASE_COLOR for v in fic["change"]]
    ax.barh(fic.index, fic["change"], color=colors, alpha=0.85)
    ax.axvline(0, color="#898781", linewidth=1)
    ax.set_xlabel("Change in mean |contribution|")
    ax.set_title("Feature importance change")

    # Panel 2 (only if any shared continuous column): old vs. new observed
    # range, one horizontal interval per feature.
    if has_boundary_panel:
        ax = next(panel)
        features = list(boundary_shift.keys())
        for i, feature in enumerate(features):
            old_lo, old_hi = boundary_shift[feature]["old_range"]
            new_lo, new_hi = boundary_shift[feature]["new_range"]
            ax.plot([old_lo, old_hi], [i - 0.15, i - 0.15], color=_OLD_COLOR, linewidth=4, solid_capstyle="round")
            ax.plot([new_lo, new_hi], [i + 0.15, i + 0.15], color=_NEW_COLOR, linewidth=4, solid_capstyle="round")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color=_OLD_COLOR, linewidth=4, label="old"),
                plt.Line2D([0], [0], color=_NEW_COLOR, linewidth=4, label="new"),
            ],
            loc="best",
        )
        ax.set_xlabel("Observed range")
        ax.set_title("Boundary shift")

    # Panel 3 (only if any shared continuous column): population migration.
    if len(population_migration) > 0:
        ax = next(panel)
        features = list(population_migration.keys())
        values = [population_migration[f] for f in features]
        ax.bar(features, values, color=_MIGRATION_COLOR, alpha=0.85)
        ax.set_ylabel("Fraction of rows in a different zone")
        ax.set_title("Population migration")
        ax.tick_params(axis="x", rotation=30)

    # Final panel: prediction shift histogram -- the one number not
    # already summarized by compare_models beyond {"mean", "std"}, so the
    # raw per-row diff is recomputed here directly via predict(), disclosed
    # as such rather than pretending it came from compare_models itself.
    ax = next(panel)
    X_df = model_old._ensure_dataframe(X_eval)
    diff = model_new.predict(X_df) - model_old.predict(X_df)
    ax.hist(diff, bins=30, color=_MIGRATION_COLOR, alpha=0.85)
    ax.axvline(0, color="#898781", linewidth=1)
    ax.set_xlabel("predict_new(X) - predict_old(X)")
    ax.set_title("Prediction shift")

    title = "Model drift dashboard"
    if data["performance_change"] is not None:
        pc = data["performance_change"]
        title += f"  (RMSE: {pc['rmse_old']:.4g} -> {pc['rmse_new']:.4g})"
    fig.suptitle(title)
    fig.tight_layout()
    return data, fig
