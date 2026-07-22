"""Drift threshold/alert monitor: an active flag on top of
:func:`zoneboost.compare_models`'s stateless diff, answering "is this
drift bigger than the model's own normal residual noise" rather than
leaving that judgment to a person eyeballing the numbers.

A deliberately separate module from :mod:`zoneboost._drift` -- this calls
:func:`zoneboost.compare_models` rather than modifying it, so the
existing, already-tested comparison logic is reused unchanged rather than
edited in place.

Reuses ``ZoneBoostRegressor``'s own already-calibrated split-conformal
margin -- the exact quantity :meth:`zoneboost.ZoneBoostRegressor.
predict_interval` already uses -- as the "band" a drift must exceed to be
flagged, instead of inventing a new arbitrary threshold. The per-Mondrian-
group alert reuses :attr:`zoneboost.ZoneBoostRegressor.
conformal_scores_by_group_` the same way, with the identical
fallback-to-global-margin behavior ``predict_interval`` already applies
for a group too small at fit time or unseen at evaluation time.

This is a heuristic significance check reusing an existing calibrated
quantity, not a formal hypothesis test -- no p-value, no
multiple-comparison correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._drift import compare_models

__all__ = ["flag_drift"]


def _margin(scores: np.ndarray, alpha: float) -> float:
    """Identical formula to ``ZoneBoostRegressor.predict_interval``'s own
    ``_margin`` closure (``regressor.py``) -- the same split-conformal
    order-statistic margin, reimplemented here since that closure is
    private and not importable, rather than diverging with a different
    quantile definition."""
    n = len(scores)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(scores[k - 1])


def flag_drift(model_old, model_new, X_eval, y_eval=None, alpha: float = 0.1) -> dict:
    """Flags whether the drift between two fitted ``ZoneBoostRegressor``
    snapshots exceeds ``model_new``'s own calibrated conformal margin --
    i.e. whether the observed prediction shift is bigger than what the
    model's own normal residual noise would predict.

    Parameters
    ----------
    model_old, model_new : ZoneBoostRegressor
        Both already fitted. ``model_new`` must have been fit with
        ``validation_fraction > 0`` (the default) or
        ``calibration_fraction > 0``, so it has a ``conformal_scores_``
        band to compare against -- the same requirement
        ``predict_interval`` already has.
    X_eval : DataFrame or array-like of shape (n_samples, n_features)
        Passed straight through to :func:`zoneboost.compare_models`.
    y_eval : array-like, default=None
        Passed straight through to :func:`zoneboost.compare_models`.
    alpha : float, default=0.1
        Miscoverage rate for the conformal margin -- the same parameter
        ``predict_interval`` accepts, and the same meaning.

    Returns
    -------
    dict with keys:
        ``comparison`` -- the full :func:`zoneboost.compare_models` result.
        ``alpha`` -- the level used.
        ``global_margin`` -- ``model_new``'s own conformal margin at
        ``alpha``, computed from ``conformal_scores_``.
        ``mean_prediction_shift`` -- mean of
        ``model_new.predict(X_eval) - model_old.predict(X_eval)``.
        ``drifted`` -- ``True`` if ``|mean_prediction_shift| >
        global_margin``.
        ``group_alerts`` -- ``None`` unless ``model_new.mondrian_col_``
        was set at fit time, else ``{group_value: {"mean_shift": float,
        "margin": float, "drifted": bool, "used_group_margin": bool}}``
        for every group present in ``X_eval``. ``used_group_margin`` is
        ``False`` when a group was too small at fit time or unseen here,
        in which case ``margin`` falls back to ``global_margin`` -- the
        same fallback ``predict_interval`` already applies.

    Examples
    --------
    >>> from zoneboost import ZoneBoostRegressor, flag_drift
    >>> model_old = ZoneBoostRegressor(n_rounds=20).fit(X_q1, y_q1)
    >>> model_new = ZoneBoostRegressor(n_rounds=20).fit(X_q2, y_q2)
    >>> result = flag_drift(model_old, model_new, X_q2, y_q2)  # doctest: +SKIP
    >>> result["drifted"]  # doctest: +SKIP
    """
    if model_new.conformal_scores_ is None:
        raise ValueError(
            "flag_drift requires model_new to have been fit with validation_fraction > 0 "
            "(the default) or calibration_fraction > 0 -- no conformal_scores_ to compare "
            "the drift against."
        )
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    comparison = compare_models(model_old, model_new, X_eval, y_eval)

    X_eval_df = model_new._ensure_dataframe(X_eval)
    diff = model_new.predict(X_eval_df) - model_old.predict(X_eval_df)
    mean_shift = float(diff.mean())
    global_margin = _margin(model_new.conformal_scores_, alpha)

    group_alerts = None
    if model_new.mondrian_col_ is not None:
        group_values = X_eval_df[model_new.mondrian_col_].to_numpy()
        group_alerts = {}
        for group in pd.unique(group_values):
            mask = group_values == group
            group_scores = (
                model_new.conformal_scores_by_group_.get(group) if model_new.conformal_scores_by_group_ else None
            )
            margin = _margin(group_scores, alpha) if group_scores is not None else global_margin
            group_mean_shift = float(diff[mask].mean())
            group_alerts[group] = {
                "mean_shift": group_mean_shift,
                "margin": margin,
                "drifted": abs(group_mean_shift) > margin,
                "used_group_margin": group_scores is not None,
            }

    return {
        "comparison": comparison,
        "alpha": alpha,
        "global_margin": global_margin,
        "mean_prediction_shift": mean_shift,
        "drifted": abs(mean_shift) > global_margin,
        "group_alerts": group_alerts,
    }
