"""ZoneBoostTimeSeries: native sequential/windowed fitting -- the notebook
pages' own "if data follows a date pattern it's broken down to follow
sequential date pattern" concern, made a first-class fitting mode instead
of something the caller has to loop over :class:`zoneboost.
ZoneBoostRegressor`/:func:`zoneboost.compare_models`/:func:`zoneboost.
flag_drift` themselves.

Invents no new modeling math: every per-period model is an ordinary
:class:`zoneboost.ZoneBoostRegressor` fit, and every pairwise diagnostic
between consecutive periods is exactly :func:`zoneboost.compare_models`/
:func:`zoneboost.flag_drift`, already shipped -- this module is purely the
period-bucketing and windowing orchestration around them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from ._common import ensure_dataframe
from ._drift import compare_models
from ._drift_alert import flag_drift
from .regressor import ZoneBoostRegressor

__all__ = ["ZoneBoostTimeSeries"]


class ZoneBoostTimeSeries(BaseEstimator, RegressorMixin):
    """Fits one :class:`zoneboost.ZoneBoostRegressor` per time period,
    walk-forward, and automatically compares each consecutive pair via
    :func:`zoneboost.compare_models`/:func:`zoneboost.flag_drift`.

    Buckets rows into periods from ``time_col`` via ``pandas``'s own
    ``.dt.to_period(freq)`` (any valid pandas offset alias -- ``"Y"``,
    ``"Q"``, ``"M"``, ``"W"``, ``"D"``, ...), then fits a model "as of"
    every period from the first one with enough history
    (``min_periods``) through the most recent one:

    - ``window="expanding"`` (default): the model as of period ``k`` is
      trained on **every** row from the first period through ``k`` --
      the training window only grows.
    - ``window="rolling"``: the model as of period ``k`` is trained on
      exactly the ``min_periods`` periods ending at ``k`` -- a
      fixed-size, sliding lookback.

    For every pair of **consecutive** fitted periods ``(k, k+1)``, this
    is a genuine walk-forward evaluation: period ``k``'s own model (which
    never saw period ``k+1``'s rows at fit time) is compared against
    period ``k+1``'s model on period ``k+1``'s own held-out data via
    :func:`zoneboost.compare_models` (always) and
    :func:`zoneboost.flag_drift` (whenever period ``k+1``'s model has a
    calibrated conformal margin to check the drift against -- true by
    default, since ``ZoneBoostRegressor``'s own ``validation_fraction``
    defaults to ``0.2``).

    ``time_col`` is a partitioning key, not a feature: it is excluded
    from every per-period model's own fitted feature space entirely (a
    raw timestamp has no meaning as a zone-averaged predictor, and would
    otherwise be auto-detected as an absurdly high-cardinality
    categorical column). A caller who wants the calendar signal itself
    fitted (day-of-year, month, a trend index, ...) engineers that as a
    **separate** column before calling `fit` -- this class doesn't derive
    calendar features on its own.

    Parameters
    ----------
    base_estimator : ZoneBoostRegressor, default=None
        An unfit template. ``None`` (default) uses a plain
        ``ZoneBoostRegressor()``. Cloned and refit once per period; only
        ``random_state`` is overridden per clone -- every other parameter
        is respected as-is, the same precedent
        :class:`zoneboost.BootstrapStability` sets.
    time_col : str or int
        Required -- the column (name, or index if ``X`` isn't a
        DataFrame) giving each row's timestamp. Coerced via
        ``pandas.to_datetime`` (a no-op if already datetime-like).
    freq : str, default="Y"
        A pandas offset alias controlling period granularity (``"Y"``
        for annual periods, ``"Q"`` for quarterly, ``"M"`` for monthly,
        etc.) -- forwarded directly to ``.dt.to_period(freq)``. Not
        validated here beyond whatever pandas itself raises for an
        invalid alias.
    window : {"expanding", "rolling"}, default="expanding"
        See above. Raises ``ValueError`` if not one of these two
        strings.
    min_periods : int, default=2
        Dual meaning by ``window``: for ``"expanding"``, the minimum
        number of periods pooled for the very first fit (every later
        period adds one more period to the pool); for ``"rolling"``, the
        **fixed** number of periods in every window. Either way, the
        first period that gets its own fitted model is
        ``periods_[min_periods - 1]`` -- a period earlier than that
        never has enough history and is simply never fitted (present in
        ``periods_`` but absent from ``models_``'s own keys). Must be
        ``>= 1``; raises ``ValueError`` if there aren't at least
        ``min_periods`` distinct periods in the training data at all.

        Deliberately an integer period **count**, not a timedelta string
        (e.g. the notebook's own ``"2Y"`` phrasing) -- ``freq`` already
        controls what one period spans, so ``min_periods=2`` with
        ``freq="Y"`` already means "2 years of history," with no second
        string-parsing convention needed on top.
    random_state : int, default=42
        Seed for each period's own derived clone seed.

    Attributes
    ----------
    time_col_ : str
        Resolved column name for ``time_col``.
    feature_names_in_ : ndarray
        Every column ``base_estimator`` was actually fit on -- ``X``'s
        own columns **excluding** ``time_col_``.
    periods_ : list
        Every distinct period observed in the training data, sorted,
        including any too early to have their own fitted model (see
        ``min_periods``).
    models_ : dict
        ``{period: fitted_model}`` -- one entry per period from
        ``periods_[min_periods - 1]`` onward. Iterate
        ``sorted(models_)`` for chronological order (pandas ``Period``
        objects sort naturally).
    comparisons_ : dict
        ``{(period_k, period_k_plus_1): compare_models(...)}`` for every
        consecutive pair of fitted periods -- period ``k_plus_1``'s own
        rows are the evaluation data, genuinely unseen by period ``k``'s
        model at its own fit time.
    drift_alerts_ : dict
        ``{(period_k, period_k_plus_1): flag_drift(...)}`` -- **only**
        for a pair where period ``k_plus_1``'s own model has a non-``None``
        ``conformal_scores_`` (i.e. was fit with ``validation_fraction >
        0`` or ``calibration_fraction > 0``); a pair missing from this
        dict (while still present in ``comparisons_``) discloses that
        precondition wasn't met for that period, rather than storing
        ``None`` there.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import ZoneBoostTimeSeries
    >>> X = pd.DataFrame({
    ...     "date": pd.date_range("2020-01-01", periods=400, freq="D"),
    ...     "x": range(400),
    ... })
    >>> y = [float(v % 10) for v in range(400)]
    >>> model = ZoneBoostTimeSeries(
    ...     time_col="date", freq="M", min_periods=2, random_state=0,
    ... ).fit(X, y)
    >>> model.predict(X).shape
    (400,)

    Notes
    -----
    **Scope**: regressor only (:func:`zoneboost.compare_models` itself is
    regressor-only). No automatic "flag when a zone's empirical Bayes
    prior should be updated" rule -- no defensible, general heuristic for
    that currently exists; :attr:`drift_alerts_`/:meth:`stability_report`
    are the closest available signal. No plotting of its own either (no
    new dependency needed here at all) -- for the notebook's own
    "boxplots over time" ask, compose with
    :func:`zoneboost.eda.drift_dashboard` directly per consecutive pair,
    e.g. ``drift_dashboard(model.models_[k], model.models_[k_plus_1],
    X_eval, y_eval)``.
    """

    def __init__(
        self,
        base_estimator=None,
        time_col=None,
        freq: str = "Y",
        window: str = "expanding",
        min_periods: int = 2,
        random_state: int = 42,
    ):
        self.base_estimator = base_estimator
        self.time_col = time_col
        self.freq = freq
        self.window = window
        self.min_periods = min_periods
        self.random_state = random_state

    def fit(self, X, y):
        """Fit one model per qualifying period, then compare every
        consecutive pair.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Must include ``time_col``.
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : ZoneBoostTimeSeries
        """
        if self.time_col is None:
            raise ValueError("time_col is required.")
        if self.window not in ("expanding", "rolling"):
            raise ValueError(f"window must be 'expanding' or 'rolling', got {self.window!r}")
        if self.min_periods < 1:
            raise ValueError(f"min_periods must be >= 1, got {self.min_periods!r}")

        X = ensure_dataframe(X, getattr(self, "_feature_names_in_full", None))
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y_arr):
            raise ValueError(f"X and y have inconsistent lengths: {len(X)} vs {len(y_arr)}")

        time_col = X.columns[self.time_col] if isinstance(self.time_col, (int, np.integer)) else self.time_col
        if time_col not in X.columns:
            raise ValueError(f"time_col={self.time_col!r} is not a column of X.")
        self.time_col_ = time_col

        timestamps = pd.to_datetime(X[time_col])
        periods = timestamps.dt.to_period(self.freq)
        unique_periods = sorted(periods.unique())
        n_periods = len(unique_periods)
        if n_periods < self.min_periods:
            raise ValueError(
                f"Only {n_periods} distinct period(s) found at freq={self.freq!r}, "
                f"fewer than min_periods={self.min_periods!r} -- no period has enough history."
            )

        feature_cols = [c for c in X.columns if c != time_col]
        self.feature_names_in_ = np.array(feature_cols)
        self._feature_names_in_full = np.array(X.columns)

        base = self.base_estimator if self.base_estimator is not None else ZoneBoostRegressor()
        rng = np.random.default_rng(self.random_state)

        self.periods_ = unique_periods
        models_ = {}
        for k in range(self.min_periods - 1, n_periods):
            period_k = unique_periods[k]
            if self.window == "expanding":
                train_periods = unique_periods[: k + 1]
            else:
                train_periods = unique_periods[k - self.min_periods + 1 : k + 1]
            mask = periods.isin(train_periods).to_numpy()
            seed = int(rng.integers(0, 2**31 - 1))
            model = clone(base).set_params(random_state=seed)
            model.fit(X.loc[mask, feature_cols].reset_index(drop=True), y_arr[mask])
            models_[period_k] = model
        self.models_ = models_

        fitted_periods = sorted(models_)
        comparisons_ = {}
        drift_alerts_ = {}
        for i in range(len(fitted_periods) - 1):
            k_old, k_new = fitted_periods[i], fitted_periods[i + 1]
            mask_new = (periods == k_new).to_numpy()
            X_eval = X.loc[mask_new, feature_cols].reset_index(drop=True)
            y_eval = y_arr[mask_new]
            comparisons_[(k_old, k_new)] = compare_models(models_[k_old], models_[k_new], X_eval, y_eval)
            if models_[k_new].conformal_scores_ is not None:
                drift_alerts_[(k_old, k_new)] = flag_drift(models_[k_old], models_[k_new], X_eval, y_eval)
        self.comparisons_ = comparisons_
        self.drift_alerts_ = drift_alerts_

        return self

    def _latest_model(self):
        check_is_fitted(self, "models_")
        if not self.models_:
            raise ValueError("No period had enough history to fit a model.")
        return self.models_[max(self.models_)]

    def predict(self, X) -> np.ndarray:
        """Predict using the most recent period's own fitted model.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Same feature set as :attr:`feature_names_in_` (``time_col``
            excluded) -- an extra ``time_col`` column, if present, is
            simply ignored by the underlying model, not an error.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        return self._latest_model().predict(X)

    def explain(self, X, **kwargs):
        """``explain(X)`` on the most recent period's own fitted model --
        see :meth:`zoneboost.ZoneBoostRegressor.explain` for the return
        shape and every keyword argument."""
        return self._latest_model().explain(X, **kwargs)

    def feature_importance(self, X, **kwargs) -> pd.Series:
        """``feature_importance(X)`` on the most recent period's own
        fitted model."""
        return self._latest_model().feature_importance(X, **kwargs)

    def stability_report(self) -> pd.DataFrame:
        """Aggregates :attr:`comparisons_` across **every** consecutive
        period pair into one "which zones are stable vs. drifting over
        time" summary per feature -- zero new statistics, a synthesis of
        numbers :func:`zoneboost.compare_models` already computed for
        each pair.

        Returns
        -------
        DataFrame
            Indexed by feature name (every continuous main-effect column
            that appeared in at least one comparison), columns
            ``n_comparisons`` (how many consecutive pairs included it),
            ``mean_population_migration``/``max_population_migration``
            (fraction of rows landing in a different zone, averaged/maxed
            across those pairs), and
            ``mean_abs_center_shift``/``max_abs_center_shift`` (shift in
            the feature's own observed-range midpoint). Sorted by
            ``mean_population_migration`` descending -- the most
            persistently drifting feature first.
        """
        check_is_fitted(self, "comparisons_")
        rows: dict = {}
        for comparison in self.comparisons_.values():
            for feature, migration in comparison["population_migration"].items():
                center_shift = abs(comparison["boundary_shift"][feature]["center_shift"])
                entry = rows.setdefault(
                    feature, {"migrations": [], "center_shifts": []}
                )
                entry["migrations"].append(migration)
                entry["center_shifts"].append(center_shift)

        table = {}
        for feature, entry in rows.items():
            migrations = np.array(entry["migrations"])
            center_shifts = np.array(entry["center_shifts"])
            table[feature] = {
                "n_comparisons": len(migrations),
                "mean_population_migration": float(migrations.mean()),
                "max_population_migration": float(migrations.max()),
                "mean_abs_center_shift": float(center_shifts.mean()),
                "max_abs_center_shift": float(center_shifts.max()),
            }
        result = pd.DataFrame.from_dict(table, orient="index")
        if result.empty:
            return result
        return result.sort_values("mean_population_migration", ascending=False)
