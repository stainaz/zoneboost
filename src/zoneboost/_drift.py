"""Time-based / drift-aware reporting: a **stateless comparison** between
two already-fitted `ZoneBoostRegressor` models -- e.g. last quarter's
model and this quarter's -- not a new fit-time behavior. zoneboost itself
doesn't monitor anything in real time or retain a history of past fits;
the user retrains a new model each period and calls :func:`compare_models`
to compare it against the previous one, on a shared evaluation dataset
(zoneboost retains no training data after `fit`, so there is no other way
to compare "the same rows" across two models).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._weak_learner import _column_zone_index

__all__ = ["compare_models"]


def _observed_range(model, feature: str) -> tuple:
    """Min/max of ``feature``'s own zone centers across every round
    ``model`` fit it in as a continuous main effect -- what the model
    actually has information about, not the column's full theoretical
    range. Shared by :meth:`zoneboost.ZoneBoostRegressor.counterfactual`
    (via a thin wrapper) and :func:`compare_models` -- one implementation,
    not two."""
    mins, maxs = [], []
    for round_ in model.rounds_:
        if feature not in round_["main_effects"]:
            continue
        zone_info_col = round_["zone_info"][feature]
        if zone_info_col[0] != "continuous":
            continue
        centers = zone_info_col[2]
        if len(centers) > 0:
            mins.append(float(centers.min()))
            maxs.append(float(centers.max()))
    if not mins:
        raise ValueError(f"{feature!r} never appeared as a continuous main effect in any round.")
    return min(mins), max(maxs)


def _shared_continuous_columns(model_old, model_new) -> list:
    old_cols = model_old._continuous_main_effect_columns()
    new_cols = model_new._continuous_main_effect_columns()
    return sorted(old_cols & new_cols)


def _last_continuous_zone_info(model, feature):
    """The most recent round's own zone_info for ``feature``, or ``None``
    if it never appeared as a continuous main effect -- a representative
    snapshot, not a claim about every round's own boundaries (zoneboost's
    zones are re-derived fresh every round; see the module docstring)."""
    for round_ in reversed(model.rounds_):
        if feature in round_["main_effects"] and round_["zone_info"][feature][0] == "continuous":
            return round_["zone_info"][feature]
    return None


def compare_models(model_old, model_new, X_eval, y_eval=None) -> dict:
    """Compares two already-fitted :class:`zoneboost.ZoneBoostRegressor`
    models -- e.g. a model refit on a later time period -- on a shared
    evaluation dataset. Reuses existing, already-exact machinery
    throughout (``feature_importance``, ``predict``) rather than
    inventing parallel comparison logic.

    **Scope**: regressor only (classifier support deferred -- multiclass's
    per-class nested ``rounds_`` would meaningfully complicate every
    comparison below); compares exactly two snapshots (call this function
    pairwise across more periods for a longer trend); reports an aggregate
    boundary/zone summary, not each model's full per-round boundary
    provenance.

    Parameters
    ----------
    model_old, model_new : ZoneBoostRegressor
        Both already fitted.
    X_eval : DataFrame or array-like of shape (n_samples, n_features)
        A dataset both models can score -- typically the newer period's
        data, or a fixed holdout common to both fits.
    y_eval : array-like, default=None
        If given, also reports ``performance_change``.

    Returns
    -------
    dict with keys:
        ``feature_importance_change`` -- DataFrame indexed by term name,
        columns ``old``/``new``/``change`` (a term absent from one model
        contributes `0`), sorted by ``|change|`` descending.
        ``new_terms``/``disappeared_terms`` -- term names present in only
        one model's ``feature_importance`` index.
        ``boundary_shift`` -- ``{feature: {"old_range": (lo, hi),
        "new_range": (lo, hi), "center_shift": float}}`` for every
        continuous main-effect column present in both models --
        ``center_shift`` is the shift in each range's own midpoint.
        ``population_migration`` -- ``{feature: fraction}`` for the same
        shared columns: using each model's own *last* fitted round's zone
        boundaries (a representative snapshot) to assign every row of
        ``X_eval`` a hard zone index under both models, the fraction of
        rows whose zone assignment differs.
        ``performance_change`` -- ``{"rmse_old": ..., "rmse_new": ...}``,
        or ``None`` if ``y_eval`` not given.
        ``prediction_shift`` -- ``{"mean": ..., "std": ...}`` of
        ``predict_new(X_eval) - predict_old(X_eval)``.
    """
    X_eval = model_old._ensure_dataframe(X_eval)

    importance_old = model_old.feature_importance(X_eval)
    importance_new = model_new.feature_importance(X_eval)
    all_terms = sorted(set(importance_old.index) | set(importance_new.index))
    old_vals = np.array([float(importance_old.get(t, 0.0)) for t in all_terms])
    new_vals = np.array([float(importance_new.get(t, 0.0)) for t in all_terms])
    feature_importance_change = pd.DataFrame(
        {"old": old_vals, "new": new_vals, "change": new_vals - old_vals}, index=all_terms
    )
    feature_importance_change = feature_importance_change.reindex(
        feature_importance_change["change"].abs().sort_values(ascending=False).index
    )

    new_terms = sorted(set(importance_new.index) - set(importance_old.index))
    disappeared_terms = sorted(set(importance_old.index) - set(importance_new.index))

    shared_cols = _shared_continuous_columns(model_old, model_new)
    boundary_shift = {}
    population_migration = {}
    for feature in shared_cols:
        old_range = _observed_range(model_old, feature)
        new_range = _observed_range(model_new, feature)
        old_center = (old_range[0] + old_range[1]) / 2.0
        new_center = (new_range[0] + new_range[1]) / 2.0
        boundary_shift[feature] = {
            "old_range": old_range,
            "new_range": new_range,
            "center_shift": new_center - old_center,
        }

        old_zone_info = _last_continuous_zone_info(model_old, feature)
        new_zone_info = _last_continuous_zone_info(model_new, feature)
        if old_zone_info is not None and new_zone_info is not None:
            old_zones = _column_zone_index(X_eval[feature], old_zone_info)
            new_zones = _column_zone_index(X_eval[feature], new_zone_info)
            population_migration[feature] = float(np.mean(old_zones != new_zones))

    performance_change = None
    if y_eval is not None:
        y_arr = np.asarray(y_eval, dtype=float).reshape(-1)
        pred_old = model_old.predict(X_eval)
        pred_new = model_new.predict(X_eval)
        performance_change = {
            "rmse_old": float(np.sqrt(np.mean((y_arr - pred_old) ** 2))),
            "rmse_new": float(np.sqrt(np.mean((y_arr - pred_new) ** 2))),
        }

    pred_old = model_old.predict(X_eval)
    pred_new = model_new.predict(X_eval)
    diff = pred_new - pred_old
    prediction_shift = {"mean": float(diff.mean()), "std": float(diff.std())}

    return {
        "feature_importance_change": feature_importance_change,
        "new_terms": new_terms,
        "disappeared_terms": disappeared_terms,
        "boundary_shift": boundary_shift,
        "population_migration": population_migration,
        "performance_change": performance_change,
        "prediction_shift": prediction_shift,
    }
