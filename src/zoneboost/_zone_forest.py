"""ZoneForest: a bagged ensemble of transparent zone-based estimators.

The notebook pages behind zoneboost describe fitting on many small samples
and averaging the results -- a different paradigm from the sequential
boosting :class:`zoneboost.ZoneBoostRegressor` performs on one sample.
``ZoneForest`` is that averaging paradigm: bootstrap-resample rows,
subsample columns, fit an independent ``base_estimator`` clone on each
resample (embarrassingly parallel, unlike sequential boosting rounds), and
average predictions/per-term contributions across the ensemble. Every
number a `predict`/`explain` call returns is still a plain average of
zone-based estimators -- never a hidden weight, exactly the same moat
``ZoneBoostRegressor`` itself keeps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from ._common import ensure_dataframe
from .regressor import ZoneBoostRegressor

__all__ = ["ZoneForest"]


def _fit_one(base_estimator, X: pd.DataFrame, y: np.ndarray, row_idx: np.ndarray, cols: list, seed: int):
    model = clone(base_estimator).set_params(random_state=seed)
    model.fit(X.iloc[row_idx][cols].reset_index(drop=True), y[row_idx])
    return model


class ZoneForest(BaseEstimator, RegressorMixin):
    """Bagged ensemble of ``base_estimator`` clones -- Random Forest's
    bootstrap-aggregating formula, with interaction-aware, fully
    explainable zone-based estimators standing in for opaque trees.

    Each of ``n_estimators`` clones is fit on its own bootstrap resample
    (rows drawn **with replacement**, the standard nonparametric
    bootstrap -- same convention as :class:`zoneboost.BootstrapStability`)
    and its own fixed column subset (drawn once per estimator, **without**
    replacement) -- ``max_samples``/``max_features`` control the size of
    each. ``base_estimator``'s own internal ``row_subsample``/
    ``col_subsample`` (if it's a :class:`zoneboost.ZoneBoostRegressor`)
    then apply *on top of* that already-reduced view every boosting round,
    the same way Random Forest's per-split feature sampling compounds with
    each tree's own bootstrap sample -- the two subsampling layers are
    independent parameters, not aliases of each other.

    Fits are dispatched via ``joblib.Parallel`` -- embarrassingly
    parallel, unlike ``ZoneBoostRegressor``'s inherently sequential
    boosting rounds. All resampling draws (bootstrap indices, column
    subsets, per-estimator seeds) are made sequentially from one ``rng``
    *before* dispatch, so the fitted ensemble is identical regardless of
    ``n_jobs``.

    ``predict(X)`` is the plain mean of every estimator's own
    ``predict(X)`` (restricted to that estimator's own column subset).
    ``explain(X)`` is the plain mean of every estimator's own
    ``explain(X)``, aligned by term name -- a term some estimators never
    fit (because their column subset excluded a constituent column, or
    that round's own screening dropped it) contributes exactly ``0`` for
    those estimators, not a re-normalized share, so ``explain(X)`` still
    sums to ``predict(X)`` exactly (linearity of the mean over each
    estimator's own exact row-sum-equals-prediction guarantee).

    Parameters
    ----------
    base_estimator : estimator, default=None
        An unfit template exposing ``fit``/``predict``/``explain``
        (typically a :class:`zoneboost.ZoneBoostRegressor`). ``None``
        (default) uses a plain ``ZoneBoostRegressor()`` -- the same
        meta-estimator precedent as :class:`zoneboost.BootstrapStability`.
        Cloned and refit once per estimator; only ``random_state`` is
        overridden per clone. Since bagging already reduces variance
        across many fits, a *shallower* ``base_estimator`` (fewer
        ``n_rounds``) than you'd use standalone is the usual, though not
        required, choice.
    n_estimators : int, default=100
        Number of bootstrap resamples / fitted clones.
    max_samples : float, default=1.0
        Bootstrap draw size, as a fraction of the training set (rows
        still drawn **with replacement**, so ``1.0`` is the classic
        same-size bootstrap, not "no resampling"). Must be in ``(0, 1]``.
    max_features : float, default=1.0
        Fraction of columns drawn **without replacement** for each
        estimator's own fixed column subset (drawn once per estimator, at
        `fit` time -- not re-drawn per round). Must be in ``(0, 1]``.
        ``base_estimator.group_col``/``mondrian_col`` (when set) are
        force-included in every estimator's subset regardless of this
        draw -- the same "never subsampled away" guarantee
        ``ZoneBoostRegressor.group_col`` already makes for its own
        internal ``col_subsample``, since those two params raise
        ``ValueError`` from the base estimator's own `fit` if the
        declared column is simply absent from its view of `X`. Every
        *other* column-selecting parameter (``categorical_features``,
        ``monotonic_constraints``, ``forbidden_interactions``, ...)
        degrades silently instead if this draw excludes it for a given
        estimator (that estimator's own resolution logic already treats a
        declared-but-absent column as a harmless no-op) -- disclosed, not
        specially handled.
    n_jobs : int, default=None
        Passed to ``joblib.Parallel`` -- ``None`` means sequential
        (``1``), ``-1`` means all available cores. No memmapping
        optimization: with process-based parallelism, each worker
        receives its own copy of `X`/`y`, a real memory cost on large
        datasets, disclosed rather than optimized around here.
    random_state : int, default=42
        Seed for every bootstrap/column draw and each clone's own derived
        seed.

    Attributes
    ----------
    n_features_in_ : int
    feature_names_in_ : ndarray
    estimators_ : list
        The ``n_estimators`` fitted clones, in resampling order.
    estimators_samples_ : list of ndarray
        Each estimator's own bootstrap row indices (into the original
        training `X`), same order as ``estimators_`` -- stored for
        possible future out-of-bag scoring, unused by anything in this
        class today.
    estimators_features_ : list of list of str
        Each estimator's own column subset (as column names), same order
        as ``estimators_`` -- what `predict`/`explain` restrict `X` to
        before calling that estimator.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import ZoneForest, ZoneBoostRegressor
    >>> X = pd.DataFrame({"x": range(20)})
    >>> y = [float(v) for v in range(20)]
    >>> model = ZoneForest(
    ...     ZoneBoostRegressor(n_rounds=10), n_estimators=5, random_state=0,
    ... ).fit(X, y)
    >>> model.predict(X).shape
    (20,)

    Notes
    -----
    **Scope**: regressor-only in this pass -- a ``ZoneBoostClassifier``
    ``base_estimator`` (majority vote or averaged ``predict_proba``) is
    deferred, a separate, larger change to aggregation. No native
    prediction-interval/spread method either: uncertainty stays
    :class:`zoneboost.BootstrapStability`'s job (compose the two,
    ``BootstrapStability(ZoneForest(...))``, if both are wanted) rather
    than adding a second, overlapping API for the same kind of question.
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 100,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        n_jobs: int = None,
        random_state: int = 42,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        """Fit ``n_estimators`` independent bootstrap/column resamples.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : ZoneForest
        """
        if self.n_estimators < 2:
            raise ValueError(f"n_estimators must be >= 2, got {self.n_estimators!r}")
        if not 0 < self.max_samples <= 1:
            raise ValueError(f"max_samples must be in (0, 1], got {self.max_samples!r}")
        if not 0 < self.max_features <= 1:
            raise ValueError(f"max_features must be in (0, 1], got {self.max_features!r}")

        X = ensure_dataframe(X, getattr(self, "feature_names_in_", None))
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y_arr):
            raise ValueError(f"X and y have inconsistent lengths: {len(X)} vs {len(y_arr)}")

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)

        base = self.base_estimator if self.base_estimator is not None else ZoneBoostRegressor()
        columns = list(X.columns)

        # group_col/mondrian_col raise ValueError from the base estimator's
        # own fit if the declared column is simply absent from its view of
        # X -- force-included the same way ZoneBoostRegressor.group_col is
        # never subsampled away by its own internal col_subsample. Every
        # other column-selecting param degrades silently instead (see the
        # class docstring), so isn't handled here.
        forced_cols = []
        for attr in ("group_col", "mondrian_col"):
            declared = getattr(base, attr, None)
            if declared is not None:
                name = columns[declared] if isinstance(declared, (int, np.integer)) else declared
                if name in columns and name not in forced_cols:
                    forced_cols.append(name)

        rng = np.random.default_rng(self.random_state)
        n = len(X)
        n_row_sample = max(1, int(round(n * self.max_samples)))
        n_col_sample = max(1, min(self.n_features_in_, int(round(self.n_features_in_ * self.max_features))))
        n_col_sample = max(n_col_sample, len(forced_cols))

        plan = []
        for _ in range(self.n_estimators):
            row_idx = rng.integers(0, n, size=n_row_sample)
            remaining = [c for c in columns if c not in forced_cols]
            n_draw = n_col_sample - len(forced_cols)
            drawn = list(rng.choice(remaining, size=n_draw, replace=False)) if n_draw > 0 else []
            cols = forced_cols + drawn
            seed = int(rng.integers(0, 2**31 - 1))
            plan.append((row_idx, cols, seed))

        fitted = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(base, X, y_arr, row_idx, cols, seed) for row_idx, cols, seed in plan
        )
        self.estimators_ = fitted
        self.estimators_samples_ = [row_idx for row_idx, _, _ in plan]
        self.estimators_features_ = [cols for _, cols, _ in plan]
        return self

    def predict(self, X) -> np.ndarray:
        """Mean prediction across the ensemble.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "estimators_")
        X = ensure_dataframe(X, self.feature_names_in_)
        preds = np.stack(
            [model.predict(X[cols]) for model, cols in zip(self.estimators_, self.estimators_features_)]
        )
        return preds.mean(axis=0)

    def explain(self, X) -> pd.DataFrame:
        """Mean per-term contribution across the ensemble -- rows sum
        exactly to :meth:`predict`, up to floating-point rounding (see the
        class docstring for why: linearity of the mean over each
        estimator's own exact row-sum-equals-prediction guarantee).

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)

        Returns
        -------
        DataFrame of shape (n_samples, n_terms + 1)
            One column per term that appeared in any estimator's own
            ``explain``, plus ``"baseline"``. A term absent from a given
            estimator's ``explain`` output contributes exactly ``0`` for
            that estimator -- it was never fit, not merely small.
        """
        check_is_fitted(self, "estimators_")
        X = ensure_dataframe(X, self.feature_names_in_)
        explanations = [
            model.explain(X[cols]) for model, cols in zip(self.estimators_, self.estimators_features_)
        ]
        all_terms: set = set()
        for df in explanations:
            all_terms.update(df.columns)

        n = len(X)
        result = {}
        for term in all_terms:
            stacked = np.zeros((len(explanations), n))
            for i, df in enumerate(explanations):
                if term in df.columns:
                    stacked[i] = df[term].to_numpy()
            result[term] = stacked.mean(axis=0)
        ordered = [c for c in result if c != "baseline"] + ["baseline"]
        return pd.DataFrame(result, index=X.index)[ordered]

    def feature_importance(self, X) -> pd.Series:
        """Global importance: each term's mean absolute contribution over
        the rows in `X`, derived directly from :meth:`explain`.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)

        Returns
        -------
        Series
            Indexed by term name, sorted descending.
        """
        contributions = self.explain(X).drop(columns=["baseline"])
        return contributions.abs().mean().sort_values(ascending=False)
