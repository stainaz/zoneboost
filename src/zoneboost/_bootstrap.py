"""Bootstrap stability: refit the whole model on resampled data to answer
"if I refit on a different sample from the same population, how much would
this contribution, this term's overall importance, or this prediction
actually change?" -- genuine resampling-based uncertainty, distinct from
:mod:`zoneboost._reliability`'s single-fit diagnostics (which report how
much a *given* fit's own contribution should be trusted, not how much it
would vary across refits).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from ._common import ensure_dataframe
from .regressor import ZoneBoostRegressor

__all__ = ["BootstrapStability"]


class BootstrapStability(BaseEstimator):
    """Bootstrap-based stability and uncertainty report for any
    ``ZoneBoostRegressor``/``ZoneBoostClassifier`` configuration: refits
    ``estimator`` on ``n_bootstrap`` independent bootstrap resamples (rows
    drawn with replacement, same size as the original training set -- the
    standard nonparametric bootstrap), then reports how much a term's
    contribution, a term's overall importance, whether a term appears at
    all, or a prediction itself varies across those refits.

    Real, disclosed cost: ``n_bootstrap`` full model refits -- this is a
    separate wrapper you opt into, not a `ZoneBoostRegressor`/
    `ZoneBoostClassifier` parameter, exactly like
    :class:`zoneboost.ConformalizedQuantileRegressor`.

    Works for both estimators (unlike ``ConformalizedQuantileRegressor``,
    which is regressor-only because quantile mode is): bootstrapping the
    whole fit procedure has no such restriction. The two point-valued
    methods (:meth:`predict_confidence_interval`, :meth:`predict_diff_interval`)
    support regressors and *binary* classifiers (via
    ``predict_proba(X)[:, 1]``) -- a multiclass model has no single scalar
    per row to bootstrap there, so those two methods raise ``ValueError``;
    :meth:`contribution_interval`/:meth:`feature_importance_interval`/
    :meth:`inclusion_frequency` fully support multiclass.

    **Deferred**: boundary-position uncertainty (how much zone cut points
    themselves move across bootstrap fits) -- different bootstrap fits can
    produce a different *number* of zones for the same column, so there's
    no clean 1:1 alignment to summarize without real additional machinery.

    Parameters
    ----------
    estimator : ZoneBoostRegressor or ZoneBoostClassifier, default=None
        An unfit template. ``None`` (default) uses a plain
        ``ZoneBoostRegressor()``. Cloned and refit once per bootstrap
        resample; only ``random_state`` is overridden per clone (a fresh
        seed per resample) -- every other parameter is respected as-is.
    n_bootstrap : int, default=30
        Number of bootstrap refits. Kept modest by default since this is
        ``n_bootstrap`` full model fits, not a free diagnostic -- raise it
        for smoother interval estimates at proportionally higher cost.
    alpha : float, default=0.1
        Default miscoverage rate for every interval method below (e.g.
        ``0.1`` reports a 90% bootstrap interval); each method also accepts
        its own ``alpha`` override.
    random_state : int, default=42
        Seed for the bootstrap resampling; each resample's cloned estimator
        gets its own derived seed, so results are fully reproducible.

    Attributes
    ----------
    bootstrap_models_ : list
        The ``n_bootstrap`` fitted clones, in resampling order.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import BootstrapStability, ZoneBoostRegressor
    >>> X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8]})
    >>> y = [1.0, 2.1, 2.9, 4.2, 4.8, 6.1, 6.9, 8.3]
    >>> model = BootstrapStability(ZoneBoostRegressor(n_rounds=20), n_bootstrap=10, random_state=0).fit(X, y)
    >>> model.inclusion_frequency()  # doctest: +SKIP
    >>> lower, upper = model.predict_confidence_interval(X)
    """

    def __init__(
        self,
        estimator=None,
        n_bootstrap: int = 30,
        alpha: float = 0.1,
        random_state: int = 42,
    ):
        self.estimator = estimator
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        """Fit ``n_bootstrap`` independent bootstrap resamples.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : BootstrapStability
        """
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha!r}")
        if self.n_bootstrap < 2:
            raise ValueError(f"n_bootstrap must be >= 2, got {self.n_bootstrap!r}")

        X = ensure_dataframe(X, getattr(self, "feature_names_in_", None))
        y_arr = np.asarray(y).reshape(-1)
        if len(X) != len(y_arr):
            raise ValueError(f"X and y have inconsistent lengths: {len(X)} vs {len(y_arr)}")
        self.feature_names_in_ = np.array(X.columns)

        base = self.estimator if self.estimator is not None else ZoneBoostRegressor()
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.bootstrap_models_ = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            seed = int(rng.integers(0, 2**31 - 1))
            model_b = clone(base).set_params(random_state=seed)
            model_b.fit(X.iloc[idx].reset_index(drop=True), y_arr[idx])
            self.bootstrap_models_.append(model_b)
        return self

    def _term_set(self, model) -> set:
        """Every term (main effect, ``"A x B"`` pair, ``"A x B x C"``
        triple) that appears anywhere in ``model``'s fitted rounds --
        flattened across classes for a multiclass classifier, matching
        ``feature_importance``'s own averaging-across-classes precedent."""
        if hasattr(model, "multiclass_"):
            if not model.multiclass_:
                rounds = model.booster_.rounds_
            else:
                rounds = [
                    round_dict
                    for round_tables in model.softmax_booster_.rounds_
                    for round_dict in round_tables.values()
                ]
        else:
            rounds = model.rounds_

        terms = set()
        for r in rounds:
            terms.update(r["main_effects"].keys())
            terms.update(" x ".join(sorted(key)) for key in r["interactions"])
            terms.update(" x ".join(sorted(key)) for key in r["triples"])
        return terms

    def inclusion_frequency(self) -> pd.Series:
        """Fraction of ``n_bootstrap`` fits in which each term appeared at
        all -- the "does this term show up when the model is refit on
        resampled data" stability signal. Only terms that appeared in at
        least one bootstrap fit are listed (an always-absent term simply
        never surfaces, rather than being reported as `0`).

        Returns
        -------
        Series
            Indexed by term name, sorted descending.
        """
        check_is_fitted(self, "bootstrap_models_")
        counts: dict = {}
        for model in self.bootstrap_models_:
            for term in self._term_set(model):
                counts[term] = counts.get(term, 0) + 1
        return pd.Series(counts, dtype=float).sort_values(ascending=False) / len(self.bootstrap_models_)

    def _term_interval_from_explanations(self, X: pd.DataFrame, explanations: list, alpha: float) -> dict:
        all_terms: set = set()
        for df in explanations:
            all_terms.update(c for c in df.columns if c not in ("baseline", "_softmax_centering"))

        result = {}
        n = len(X)
        for term in all_terms:
            stacked = np.zeros((len(explanations), n))
            for i, df in enumerate(explanations):
                if term in df.columns:
                    stacked[i] = df[term].to_numpy()
            lower = np.percentile(stacked, 100 * alpha / 2, axis=0)
            upper = np.percentile(stacked, 100 * (1 - alpha / 2), axis=0)
            result[term] = pd.DataFrame({"lower": lower, "upper": upper}, index=X.index)
        return result

    def contribution_interval(self, X, alpha: float = None) -> dict:
        """Per-term, per-row bootstrap contribution interval: refits
        ``explain(X)`` on every bootstrap model, takes the union of term
        names across all fits (a term absent from a given fit contributes
        exactly `0` for every row that fit -- it was never selected, not
        merely small), and returns the ``alpha/2``/``1 - alpha/2``
        percentile of that term's bootstrap contribution distribution.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        alpha : float, default=None
            Overrides the constructor's own ``alpha`` when given.

        Returns
        -------
        dict of {term: DataFrame} with columns ``lower``/``upper``, or
        {class_label: {term: DataFrame}} for a multiclass classifier
        (mirroring :func:`zoneboost._reliability.explain_reliability`'s own
        per-class nesting).
        """
        check_is_fitted(self, "bootstrap_models_")
        alpha = alpha if alpha is not None else self.alpha
        X = ensure_dataframe(X, self.feature_names_in_)

        explanations = [model.explain(X) for model in self.bootstrap_models_]
        if isinstance(explanations[0], dict):
            classes = list(explanations[0].keys())
            return {
                k: self._term_interval_from_explanations(X, [e[k] for e in explanations], alpha) for k in classes
            }
        return self._term_interval_from_explanations(X, explanations, alpha)

    def feature_importance_interval(self, X, alpha: float = None) -> pd.DataFrame:
        """Per-term bootstrap interval on ``feature_importance(X)`` --
        always a flat table, even for a multiclass classifier, since
        ``feature_importance`` itself already averages across classes.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        alpha : float, default=None
            Overrides the constructor's own ``alpha`` when given.

        Returns
        -------
        DataFrame
            Indexed by term name, columns ``lower``/``upper``, sorted by
            ``upper`` descending.
        """
        check_is_fitted(self, "bootstrap_models_")
        alpha = alpha if alpha is not None else self.alpha
        X = ensure_dataframe(X, self.feature_names_in_)

        importances = [model.feature_importance(X) for model in self.bootstrap_models_]
        all_terms: set = set()
        for s in importances:
            all_terms.update(s.index)

        rows = []
        for term in all_terms:
            values = np.array([float(s.get(term, 0.0)) for s in importances])
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            rows.append((term, lower, upper))
        result = pd.DataFrame(rows, columns=["term", "lower", "upper"]).set_index("term")
        return result.sort_values("upper", ascending=False)

    def _point_predict(self, model, X) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.shape[1] != 2:
                raise ValueError(
                    "predict_confidence_interval/predict_diff_interval only support "
                    "regressors or binary classifiers (a single scalar per row) -- a "
                    "multiclass model has no single value to bootstrap here; use "
                    "contribution_interval/feature_importance_interval/inclusion_frequency "
                    "instead."
                )
            return proba[:, 1]
        return model.predict(X)

    def predict_confidence_interval(self, X, alpha: float = None) -> tuple:
        """Bootstrap confidence interval of the point prediction --
        genuinely different from ``ZoneBoostRegressor.predict_interval``/
        ``ConformalizedQuantileRegressor.predict_interval``: those give a
        distribution-free *coverage* guarantee for a future observation of
        `y`; this reports model/estimation uncertainty from resampling
        (how much the fitted function itself would move under a different
        sample) -- not the same statement, and not a substitute for either.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        alpha : float, default=None
            Overrides the constructor's own ``alpha`` when given.

        Returns
        -------
        lower, upper : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "bootstrap_models_")
        alpha = alpha if alpha is not None else self.alpha
        X = ensure_dataframe(X, self.feature_names_in_)
        preds = np.stack([self._point_predict(model, X) for model in self.bootstrap_models_])
        lower = np.percentile(preds, 100 * alpha / 2, axis=0)
        upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
        return lower, upper

    def predict_diff_interval(self, X_a, X_b, alpha: float = None) -> tuple:
        """Bootstrap interval of the row-wise difference in point
        predictions between two (possibly single-row) inputs -- answers
        "is this pair's predicted difference actually distinguishable
        given resampling uncertainty," with whether the interval excludes
        `0` as the natural read.

        Parameters
        ----------
        X_a, X_b : DataFrame or array-like of shape (n_samples, n_features)
            Same number of rows.
        alpha : float, default=None
            Overrides the constructor's own ``alpha`` when given.

        Returns
        -------
        lower, upper : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "bootstrap_models_")
        alpha = alpha if alpha is not None else self.alpha
        X_a = ensure_dataframe(X_a, self.feature_names_in_)
        X_b = ensure_dataframe(X_b, self.feature_names_in_)
        if len(X_a) != len(X_b):
            raise ValueError(f"X_a and X_b must have the same number of rows, got {len(X_a)} and {len(X_b)}")

        diffs = np.stack(
            [
                self._point_predict(model, X_a) - self._point_predict(model, X_b)
                for model in self.bootstrap_models_
            ]
        )
        lower = np.percentile(diffs, 100 * alpha / 2, axis=0)
        upper = np.percentile(diffs, 100 * (1 - alpha / 2), axis=0)
        return lower, upper
