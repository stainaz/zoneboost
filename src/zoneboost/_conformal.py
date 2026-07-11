"""Conformalized Quantile Regression (CQR): a distribution-free prediction
interval whose width can vary with X, built from two quantile-mode
:class:`zoneboost.ZoneBoostRegressor` fits -- unlike
:meth:`zoneboost.ZoneBoostRegressor.predict_interval`'s split-conformal
margin, which is a single fixed width added uniformly to every row.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from ._common import ensure_dataframe
from .regressor import ZoneBoostRegressor

__all__ = ["ConformalizedQuantileRegressor"]


class ConformalizedQuantileRegressor(BaseEstimator):
    """Locally-adaptive prediction intervals via Conformalized Quantile
    Regression (Romano, Patterson & Candes, 2019).

    Fits two quantile-mode :class:`zoneboost.ZoneBoostRegressor` instances
    at levels ``alpha/2`` and ``1 - alpha/2`` (the raw quantile band), then
    conformalizes their gap on a genuinely held-out calibration split: the
    nonconformity score ``E_i = max(q_lo(X_i) - y_i, y_i - q_hi(X_i))`` is
    computed for every calibration row, and the same fixed additive margin
    (the finite-sample-corrected quantile of these scores) is added to both
    quantile predictions. The result still gives ``ZoneBoostRegressor.
    predict_interval``'s exact distribution-free marginal coverage guarantee
    (``P(y in interval) >= 1 - alpha``, under exchangeability) -- but
    because the *quantile* predictions themselves already vary with ``X``,
    the total interval width does too, unlike a plain split-conformal band's
    single constant-width margin.

    Not a general-purpose point regressor (there is no meaningful single
    ``predict``): use :meth:`predict_interval` for the band, or fit
    ``estimator`` directly (with ``loss="squared_error"``) alongside this
    class if a point prediction is also needed.

    Parameters
    ----------
    estimator : ZoneBoostRegressor, default=None
        An unfit template supplying every tuning knob *other* than
        ``loss``/``quantile``/``calibration_fraction`` (e.g. ``n_rounds``,
        ``max_zones``, ``shrinkage_m``, ``monotonic_constraints``,
        ``adaptive_boundary_smoothing``) -- cloned twice internally, once
        per quantile level. ``None`` (default) uses a plain
        ``ZoneBoostRegressor()``. Any ``loss``/``quantile``/
        ``calibration_fraction``/``random_state`` set on the template are
        always overridden (this class manages them itself) -- every other
        parameter is respected as-is. Quantile-mode fitting uses
        ``QuantileRegressor``'s linear-programming solver internally, which
        is substantially more expensive per round than the default loss
        (see :class:`zoneboost.ZoneBoostRegressor`'s ``loss`` parameter) --
        two such models are fit here, so consider a smaller ``n_rounds`` or
        ``n_iter_no_change`` on the template for large datasets.
    alpha : float, default=0.1
        Miscoverage rate -- e.g. ``0.1`` targets 90% coverage. The two
        internal quantile levels are ``alpha / 2`` and ``1 - alpha / 2``.
    calibration_fraction : float, default=0.2
        Fraction of training rows held out (once, at the start of `fit`)
        purely for CQR calibration -- genuinely separate from, and never
        reused by, either quantile model's own internal
        ``validation_fraction`` split (which only drives their own early
        stopping).
    random_state : int, default=42
        Seed for the calibration split and (via the two cloned estimators)
        their own internal splits/subsampling.

    Attributes
    ----------
    lo_ : ZoneBoostRegressor
        Fitted quantile model at level ``alpha / 2``.
    hi_ : ZoneBoostRegressor
        Fitted quantile model at level ``1 - alpha / 2``.
    cqr_scores_ : ndarray
        Sorted CQR nonconformity scores on the calibration split -- the
        margin :meth:`predict_interval` draws from.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import ConformalizedQuantileRegressor
    >>> X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8]})
    >>> y = [1.0, 2.1, 2.9, 4.2, 4.8, 6.1, 6.9, 8.3]
    >>> model = ConformalizedQuantileRegressor(alpha=0.2, random_state=0).fit(X, y)
    >>> lower, upper = model.predict_interval(X)
    """

    def __init__(
        self,
        estimator: ZoneBoostRegressor = None,
        alpha: float = 0.1,
        calibration_fraction: float = 0.2,
        random_state: int = 42,
    ):
        self.estimator = estimator
        self.alpha = alpha
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the two internal quantile models and calibrate the CQR margin.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : ConformalizedQuantileRegressor
        """
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha!r}")
        if not 0 < self.calibration_fraction < 1:
            raise ValueError(f"calibration_fraction must be in (0, 1), got {self.calibration_fraction!r}")

        X = ensure_dataframe(X, getattr(self, "feature_names_in_", None))
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y_arr):
            raise ValueError(f"X and y have inconsistent lengths: {len(X)} vs {len(y_arr)}")
        self.feature_names_in_ = np.array(X.columns)

        base = self.estimator if self.estimator is not None else ZoneBoostRegressor()

        rng = np.random.default_rng(self.random_state)
        n_total = len(X)
        perm = rng.permutation(n_total)
        n_cal = max(1, int(n_total * self.calibration_fraction))
        if n_cal >= n_total:
            raise ValueError("calibration_fraction leaves no rows for the training split.")
        cal_idx, train_idx = perm[:n_cal], perm[n_cal:]
        X_cal = X.iloc[cal_idx].reset_index(drop=True)
        y_cal = y_arr[cal_idx]
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y_arr[train_idx]

        self.lo_ = clone(base).set_params(
            loss="quantile", quantile=self.alpha / 2, calibration_fraction=0.0, random_state=self.random_state
        )
        self.hi_ = clone(base).set_params(
            loss="quantile", quantile=1 - self.alpha / 2, calibration_fraction=0.0, random_state=self.random_state
        )
        self.lo_.fit(X_train, y_train)
        self.hi_.fit(X_train, y_train)

        lo_cal_pred = self.lo_.predict(X_cal)
        hi_cal_pred = self.hi_.predict(X_cal)
        scores = np.maximum(lo_cal_pred - y_cal, y_cal - hi_cal_pred)
        self.cqr_scores_ = np.sort(scores)
        return self

    def predict_interval(self, X) -> tuple:
        """Locally-adaptive prediction interval.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)

        Returns
        -------
        lower, upper : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "lo_")
        X = ensure_dataframe(X, self.feature_names_in_)
        q_lo = self.lo_.predict(X)
        q_hi = self.hi_.predict(X)
        n = len(self.cqr_scores_)
        k = min(int(np.ceil((n + 1) * (1 - self.alpha))), n)
        margin = self.cqr_scores_[k - 1]
        return q_lo - margin, q_hi + margin
