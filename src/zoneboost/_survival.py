"""Zone-native survival analysis: a piecewise-exponential hazard model
built from nothing but a person-period-expanded :class:`zoneboost.
ZoneBoostRegressor` fit with ``loss="poisson"``.

The equivalence is exact, not a metaphor: split each subject's follow-up
time into intervals, emit one row per interval reached (covariates,
whether the event happened in that interval, how much exposure time the
interval contributed), and Poisson regression with a log-exposure
``offset`` on that expanded table *is* a piecewise-exponential hazard
model -- see :func:`_expand_person_period`. Nothing about
:mod:`zoneboost.regressor`/:mod:`zoneboost._common` changes for this;
:class:`ZoneBoostSurvival` is a thin composing wrapper, the same shape as
:class:`zoneboost.ConformalizedQuantileRegressor`.

A side benefit of this reduction: the baseline hazard becomes an ordinary
**main effect over a time column**, fit by zoneboost's own adaptive
continuous zoning -- no parametric baseline-hazard assumption, unlike Cox
proportional hazards -- and covariate-by-time interactions (time-varying
effects, which Cox PH assumes away) are just interaction terms, available
whenever ``max_interaction_order=2`` on the underlying estimator.
``explain()`` on the expanded data decomposes a subject's log-hazard into
baseline-time-shape + covariate effects + interactions directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from ._common import ensure_dataframe
from .regressor import ZoneBoostRegressor

__all__ = ["ZoneBoostSurvival"]

_TIME_COL = "__zoneboost_interval_start__"


def _default_breakpoints(duration: np.ndarray, event: np.ndarray, n_intervals: int) -> np.ndarray:
    """Quantile cut points of the *event* times (not all times, censored
    included) -- standard piecewise-exponential practice, since basing
    cuts on event times keeps every interval event-dense rather than
    risking near-empty intervals wherever censoring happens to cluster.
    Always starts at ``0.0`` and ends at ``np.inf`` so the last interval
    is open-ended and covers every subject's tail risk regardless of
    ``n_intervals``."""
    event_times = duration[event == 1]
    if len(event_times) == 0:
        event_times = duration
    quantiles = np.quantile(event_times, np.linspace(0.0, 1.0, n_intervals + 1))
    breakpoints = np.unique(quantiles)
    if len(breakpoints) < 2:
        return np.array([0.0, np.inf])
    breakpoints[0] = 0.0
    breakpoints[-1] = np.inf
    return breakpoints


def _expand_person_period(
    X: pd.DataFrame, duration: np.ndarray, event: np.ndarray, breakpoints: np.ndarray
) -> tuple:
    """Person-period expansion: one row per (subject, interval reached).
    Vectorized over intervals (bounded by ``len(breakpoints) - 1``, not
    ``len(X)``), not a per-subject Python loop.
    """
    n_intervals = len(breakpoints) - 1
    X_parts, y_parts, offset_parts = [], [], []
    for j in range(n_intervals):
        lo, hi = breakpoints[j], breakpoints[j + 1]
        at_risk = duration > lo
        if not np.any(at_risk):
            break
        exposure = np.minimum(duration[at_risk], hi) - lo
        y_j = ((event[at_risk] == 1) & (duration[at_risk] <= hi)).astype(float)
        X_j = X.loc[at_risk].copy()
        X_j[_TIME_COL] = lo
        X_parts.append(X_j)
        y_parts.append(y_j)
        offset_parts.append(np.log(exposure))
    X_expanded = pd.concat(X_parts, ignore_index=True)
    y_expanded = np.concatenate(y_parts)
    offset_expanded = np.concatenate(offset_parts)
    return X_expanded, y_expanded, offset_expanded


def _concordance_index(risk_scores: np.ndarray, duration: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C-index: among every pair of subjects where the earlier
    time is an observed event (so the ordering is actually knowable), the
    fraction where the higher-risk subject really did have the earlier
    event. Self-contained (no new dependency) -- used only for this
    module's own honest "measured" benchmark, not required for the model
    itself to function."""
    n = len(duration)
    concordant = 0.0
    comparable = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        later = duration > duration[i]
        tied_event = (duration == duration[i]) & (event == 1)
        comparable_mask = later | tied_event
        comparable_mask[i] = False
        n_comp = np.sum(comparable_mask)
        if n_comp == 0:
            continue
        comparable += n_comp
        concordant += np.sum(risk_scores[i] > risk_scores[comparable_mask])
        concordant += 0.5 * np.sum(risk_scores[i] == risk_scores[comparable_mask])
    if comparable == 0:
        return float("nan")
    return float(concordant / comparable)


class ZoneBoostSurvival(BaseEstimator):
    """Zone-native piecewise-exponential survival model.

    Fits a single :class:`zoneboost.ZoneBoostRegressor` with
    ``loss="poisson"`` on a person-period expansion of the data (see the
    module docstring) -- follow-up time is split into ``n_intervals``
    (or user-supplied ``breakpoints``), each subject contributes one row
    per interval reached, and the fitted rate at ``offset=0`` for any
    (covariates, interval) combination *is* the hazard for that
    interval. No new boosting mechanism: everything here reduces to
    Poisson regression with a log-exposure offset, exactly like
    :class:`zoneboost.ZoneBoostRegressor`'s existing ``loss="poisson"``
    actuarial-losses support.

    Parameters
    ----------
    estimator : ZoneBoostRegressor, default=None
        An unfit template supplying every tuning knob *other* than
        ``loss``/``random_state`` (e.g. ``n_rounds``, ``max_zones``,
        ``max_interaction_order``, ``monotonic_constraints``) --
        cloned internally. ``None`` (default) uses a plain
        ``ZoneBoostRegressor()``. ``loss`` is always overridden to
        ``"poisson"`` regardless of what the template sets; every other
        parameter is respected as-is. Setting ``max_interaction_order=2``
        on the template lets a covariate interact with the interval-start
        column -- i.e. a time-varying covariate *effect* (not a
        time-varying covariate *value*, see Scope below).
    n_intervals : int, default=10
        Number of piecewise-constant hazard intervals, when
        ``breakpoints`` isn't given directly -- cut points are the
        quantiles of the observed *event* times (event-dense intervals),
        with the last interval always extended to cover every subject's
        tail risk.
    breakpoints : array-like, default=None
        Explicit interval boundaries (must start at ``0``, be strictly
        increasing, and typically end at or beyond the largest observed
        ``duration``) -- overrides ``n_intervals`` entirely when given.
    random_state : int, default=42
        Passed through to the underlying ``ZoneBoostRegressor``.

    Attributes
    ----------
    regressor_ : ZoneBoostRegressor
        The fitted Poisson-loss regressor on the expanded person-period
        table -- fully public; call ``explain()``/``feature_importance()``
        on it directly (with the same expanded-style ``X``, interval-start
        column included) for a transparent decomposition of the log-hazard
        into baseline-time shape + covariate effects + interactions.
    breakpoints_ : ndarray
        The interval boundaries actually used (``[0, ..., inf]``).
    max_duration_ : float
        Largest observed ``duration`` at `fit` time -- used as the
        default upper query time in :meth:`predict_survival_function`/
        :meth:`predict_cumulative_hazard` (the true last breakpoint is
        ``inf``, not a usable finite query point).

    Scope
    -----
    Right-censoring only -- no left truncation/delayed entry (every
    subject's risk period is assumed to start at ``duration=0``), no
    interval censoring, no competing risks; ``event`` is a plain 0/1
    indicator. Covariates are time-invariant: ``X`` is one row per
    subject at baseline, and can't change value mid-follow-up in this
    pass. ``sample_weight`` isn't supported, consistent with every other
    GLM loss in this package. Ties in ``duration`` need no special
    handling here -- a genuine advantage of the piecewise-exponential
    reduction over Cox's partial-likelihood tie-breaking machinery.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import ZoneBoostSurvival
    >>> X = pd.DataFrame({"age": [50, 60, 70, 40, 55, 65]})
    >>> duration = [5.0, 2.0, 1.0, 8.0, 4.0, 1.5]
    >>> event = [1, 1, 1, 0, 1, 1]
    >>> model = ZoneBoostSurvival(n_intervals=3, random_state=0).fit(X, duration, event)
    >>> model.predict_survival_function(X)
    """

    def __init__(
        self,
        estimator: ZoneBoostRegressor = None,
        n_intervals: int = 10,
        breakpoints=None,
        random_state: int = 42,
    ):
        self.estimator = estimator
        self.n_intervals = n_intervals
        self.breakpoints = breakpoints
        self.random_state = random_state

    def fit(self, X, duration, event):
        """Fit the underlying Poisson-loss regressor on a person-period
        expansion of ``(X, duration, event)``.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        duration : array-like of shape (n_samples,)
            Time to event or censoring; must be strictly positive.
        event : array-like of shape (n_samples,)
            1 if the event was observed at ``duration``, 0 if censored.

        Returns
        -------
        self : ZoneBoostSurvival
        """
        X = ensure_dataframe(X, getattr(self, "feature_names_in_", None))
        duration_arr = np.asarray(duration, dtype=float).reshape(-1)
        event_arr = np.asarray(event).reshape(-1)
        if len(X) != len(duration_arr) or len(X) != len(event_arr):
            raise ValueError(
                f"X, duration, and event have inconsistent lengths: {len(X)}, {len(duration_arr)}, {len(event_arr)}"
            )
        if np.any(duration_arr <= 0):
            raise ValueError("duration must be strictly positive.")
        if not np.all(np.isin(event_arr, [0, 1])):
            raise ValueError("event must be 0/1.")
        event_arr = event_arr.astype(float)

        if self.breakpoints is not None:
            breakpoints = np.asarray(self.breakpoints, dtype=float)
            if breakpoints[0] != 0.0 or np.any(np.diff(breakpoints) <= 0):
                raise ValueError("breakpoints must start at 0 and be strictly increasing.")
            if not np.isinf(breakpoints[-1]):
                breakpoints = np.concatenate([breakpoints, [np.inf]])
        else:
            breakpoints = _default_breakpoints(duration_arr, event_arr, self.n_intervals)

        self.feature_names_in_ = np.array(X.columns)
        self.breakpoints_ = breakpoints
        self.max_duration_ = float(duration_arr.max())

        X_expanded, y_expanded, offset_expanded = _expand_person_period(X, duration_arr, event_arr, breakpoints)

        base = self.estimator if self.estimator is not None else ZoneBoostRegressor()
        self.regressor_ = clone(base).set_params(loss="poisson", random_state=self.random_state)
        self.regressor_.fit(X_expanded, y_expanded, offset=offset_expanded)
        return self

    def _hazard_grid(self, X: pd.DataFrame) -> np.ndarray:
        """Per-row, per-finite-interval hazard rate: shape
        ``(len(X), n_intervals)``. Each column ``j`` is ``self.regressor_.
        predict(X_at_interval_j, offset=0)`` -- the Poisson rate at zero
        exposure adjustment *is* the hazard, since ``mu = exp(eta +
        offset)`` and ``offset=0`` leaves ``mu = exp(eta)``, the bare
        rate for that (covariates, interval) combination."""
        n_intervals = len(self.breakpoints_) - 1
        hazards = np.empty((len(X), n_intervals))
        for j in range(n_intervals):
            X_j = X.copy()
            X_j[_TIME_COL] = self.breakpoints_[j]
            hazards[:, j] = self.regressor_.predict(X_j, offset=np.zeros(len(X_j)))
        return hazards

    def predict_cumulative_hazard(self, X, times=None) -> pd.DataFrame:
        """Cumulative hazard :math:`H(t) = \\int_0^t h(s)\\,ds` at each
        query time, for each row of ``X``.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        times : array-like, default=None
            Query times. ``None`` defaults to the finite interval
            boundaries (the true last breakpoint is ``inf``, replaced by
            :attr:`max_duration_` for a concrete default grid).

        Returns
        -------
        DataFrame of shape (n_samples, len(times))
        """
        X = ensure_dataframe(X, self.feature_names_in_)
        if times is None:
            times = np.append(self.breakpoints_[1:-1], self.max_duration_)
        times = np.asarray(times, dtype=float).reshape(-1)

        hazards = self._hazard_grid(X)
        finite_bounds = np.append(self.breakpoints_[1:-1], np.inf)
        lowers = self.breakpoints_[:-1]

        cumhaz = np.zeros((len(X), len(times)))
        for k, t in enumerate(times):
            overlap = np.clip(np.minimum(finite_bounds, t) - lowers, 0.0, None)
            cumhaz[:, k] = hazards @ overlap
        return pd.DataFrame(cumhaz, columns=times)

    def predict_survival_function(self, X, times=None) -> pd.DataFrame:
        """Survival probability :math:`S(t) = \\exp(-H(t))` at each query
        time, for each row of ``X``. See :meth:`predict_cumulative_hazard`
        for the ``times`` default."""
        cumhaz = self.predict_cumulative_hazard(X, times)
        return np.exp(-cumhaz)
