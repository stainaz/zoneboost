"""ZoneBoostRegressor: a fully transparent, zone-based gradient boosting
regressor built entirely from descriptive statistics.

No decision trees, no gradient descent, no neural weights. Every number
in a prediction traces back to a quantile, a group count, or a group
average -- inspectable directly from the fitted attributes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._common import ensure_dataframe, resolve_categorical_features, resolve_monotonic_constraints
from ._explain import explain_rounds
from ._weak_learner import _fit_lasso_weights, weak_learner_contributions, weak_learner_fit

__all__ = ["ZoneBoostRegressor"]


class ZoneBoostRegressor(BaseEstimator, RegressorMixin):
    """Gradient boosting with zone-grid weak learners instead of trees.

    Each boosting round fits a "weak learner" made of two transparent
    pieces, both derived by splitting each predictor's axis into a small
    number of data-driven zones and averaging the current residual within
    each zone (or zone pair):

    - **Main effects**: for each predictor, a 1D lookup from zone to
      average residual.
    - **Interactions**: for every pair of predictors, a 2D lookup from
      their joint zones to average residual -- captures effects neither
      variable explains alone.
    - **3-way interactions** (opt-in via ``max_interaction_order=3``): a
      small, adaptively-selected set of 3D lookups, added only where main
      effects and pairwise interactions leave a genuine higher-order
      pattern unexplained -- see ``max_interaction_order`` below.

    Continuous predictors get *adaptive* zone boundaries, found the way a
    regression tree finds a split (the cut that most reduces the target's
    within-zone variance), re-derived fresh every round from that round's
    residual. Nominal categorical predictors (declared via
    ``categorical_features``, or auto-detected from ``object``/``category``/
    ``bool`` dtype) skip that search entirely: each distinct category gets
    its own zone, since there is no meaningful "adjacent" to search between
    for a nominal variable.

    Every zone's contribution is weighted by density confidence -- its
    supporting row count relative to the best-supported zone in that
    round -- so sparsely-populated zones contribute less. Each round's
    weak learner is applied at a small, shrunk step
    (``learning_rate``) and added to a running prediction, exactly like
    standard gradient boosting; ``row_subsample`` / ``col_subsample``
    add stochastic-gradient-boosting-style regularization by fitting each
    round on a random subsample of rows and columns.

    Parameters
    ----------
    n_rounds : int, default=300
        Maximum number of boosting rounds.
    learning_rate : float, default=0.1
        Shrinkage applied to each round's correction before it is added
        to the running prediction.
    row_subsample : float, default=0.7
        Fraction of training rows randomly sampled to fit each round's
        weak learner (a fresh draw every round).
    col_subsample : float, default=0.7
        Fraction of predictor columns randomly sampled for each round.
    max_zones : int, default=7
        Upper bound on the number of zones for *continuous* columns only
        (``min(max_zones, n_unique_values)`` is the actual cap per
        column). Categorical columns are unaffected by this parameter --
        they always get one zone per distinct category. Keep this
        conservative: raising it for every continuous column adds
        per-round fitting flexibility that mostly helps a variable overfit
        noise rather than capture real structure, unless that variable
        genuinely has many distinct meaningful regimes -- which is exactly
        what ``categorical_features`` is for.
    min_zone_frac : float, default=0.02
        Minimum fraction of (subsampled) rows required on each side of a
        candidate zone split.
    categorical_features : list of str or int, default=None
        Columns to treat as nominal categories (one zone per distinct
        value, no ordering assumption) rather than continuous. Accepts
        column names (if ``X`` is a DataFrame) or integer positions.
        Columns of ``object``, ``category``, or ``bool`` dtype are always
        treated as categorical automatically, in addition to any
        columns named here.
    validation_fraction : float, default=0.2
        Fraction of training rows held out (once, at the start of `fit`)
        to pick the best number of rounds via early stopping. Set to 0 to
        disable early stopping and always use exactly ``n_rounds`` rounds.
    n_iter_no_change : int, default=None
        If set, stop adding rounds once the held-out score has not
        improved for this many consecutive rounds (only used when
        ``validation_fraction > 0``). If None, all ``n_rounds`` are run
        and the best-scoring round is used.
    max_interaction_order : int, default=2
        ``2`` fits main effects + pairwise interactions only (the behavior
        of every prior release). ``3`` additionally attempts a bounded,
        adaptive search for 3-way interactions each round: candidates are
        seeded from columns already showing strong pairwise signal (not
        every possible triple), and a candidate is only kept if a joint
        3-way zone grouping still explains meaningful residual variance
        after subtracting what main effects and its three constituent
        pairwise interactions would already predict -- see
        ``max_triple_interactions`` and ``triple_min_gain``.
    max_triple_interactions : int, default=5
        Cap on how many 3-way terms a single round may add. Only relevant
        when ``max_interaction_order=3``.
    triple_min_gain : float, default=0.05
        Minimum residual-explained magnitude a candidate 3-way interaction
        must retain after subtracting the main-effect + pairwise fit for
        its three columns, expressed as a fraction of its strongest
        constituent pair's own importance (a like-for-like comparison, not
        one against the residual's raw scale) -- to be judged genuine
        higher-order structure rather than something pairwise interactions
        already explain. Only relevant when ``max_interaction_order=3``.
    cross_fit_folds : int, default=5
        Every zone's cell mean is otherwise computed from the same rows a
        round then scores -- each row's own residual partly determines the
        zone mean it's then judged against, a leakage that biases the
        boosting trajectory optimistic about sparse zones (small
        ``min_zone_frac`` continuous zones, high-cardinality categoricals).
        Each round instead splits its (already row/column-subsampled) rows
        into this many folds and scores each fold only with zone tables
        built from the *other* folds -- the same fix CatBoost's ordered
        boosting applies to target statistics. Only the training signal is
        affected; the tables stored in ``rounds_`` and used by `predict`
        still use every available row. Falls back to no cross-fitting if a
        round's row count is smaller than 2 folds.
    shrinkage_m : float, default=10.0
        Every zone's mean is shrunk toward a hierarchical prior via an
        empirical-Bayes (m-estimate) fit --
        ``shrunk_mean = (n * cell_mean + m * prior) / (n + m)`` -- rather
        than the flat ``counts / counts.max()`` confidence discount used by
        every prior release. A zone needs about ``shrinkage_m`` rows of its
        own before it's trusted as much as its prior; for a main effect the
        prior is the global mean, for an interaction/triple it's the
        additive combination of its already-shrunk lower-order marginals
        (row+column, or main effects+pairs), not the flat global mean --
        a materially better guess for a sparse cell than the overall
        average of everything.
    stacking_alpha : float, default=0.01
        Every prior release combined a round's terms by averaging every
        contribution equally, then fit one shared scale for the whole
        blend. This is replaced by a **Lasso** fit treating each term's own
        (cross-fitted) contribution as its own feature: an irrelevant term
        gets its weight zeroed by the L1 penalty, a strong term gets its
        own learned weight instead of a diluted ``1/n_terms`` share, and
        the fitted weights become a real interaction-importance ranking
        that flows straight through ``feature_importance``/``explain``.
        ``stacking_alpha`` is the L1 regularization strength, in a
        standardized space so it's comparable across rounds/datasets
        regardless of scale -- see :func:`zoneboost._weak_learner._fit_lasso_weights`.
    monotonic_constraints : dict, default=None
        ``{column: +1 or -1}`` -- forces a continuous column's *main
        effect* to be non-decreasing (+1) or non-increasing (-1) across
        its zones, via isotonic regression weighted by each zone's own row
        count. Interaction terms are never constrained. Accepts column
        names (if ``X`` is a DataFrame) or integer positions, the same
        convention as ``categorical_features``. A constraint declared on a
        categorical column is silently dropped -- there's no meaningful
        order to constrain for a nominal category. Unlike every other
        parameter here, this is **opt-in**: it encodes domain knowledge the
        model can't infer on its own (e.g. "take-up must not decrease as
        affordability rises"), not a general improvement, so the default
        (no constraints) reproduces the exact same predictions as if this
        parameter didn't exist.
    random_state : int, default=42
        Seed controlling the validation split and the per-round row/column
        subsampling.

    Attributes
    ----------
    n_features_in_ : int
        Number of predictor columns seen during `fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Column names seen during `fit`.
    categorical_features_ : set
        Resolved set of columns treated as categorical (declared union
        auto-detected).
    monotonic_constraints_ : dict
        Resolved ``{column: +1 or -1}`` constraints actually in effect
        (declared constraints on a categorical column are dropped here).
    baseline_ : float
        The target's mean on the training split -- the starting prediction
        before any boosting round is applied.
    rounds_ : list
        One entry per fitted boosting round, each a plain dict with keys
        ``"zone_info"``, ``"main_effects"``, ``"interactions"``,
        ``"triples"`` (empty unless ``max_interaction_order=3``), and
        ``"intercept"``/``"weights"`` -- the round's fitted Lasso intercept
        and one weight per term (``fitted_residual = intercept +
        contributions @ weights``, in the same order as
        ``main_effects``/``interactions``/``triples`` are themselves
        iterated). Every value is plain data (dicts/arrays of numpy arrays
        or floats) -- fully inspectable, nothing hidden in an opaque model
        object.
    best_n_rounds_ : int
        The number of rounds actually used by `predict` (the early-stopped
        count, or ``n_rounds`` if early stopping was disabled or never
        found a better round than the last).
    val_rmse_ : list of float
        Held-out RMSE after each round (empty if ``validation_fraction=0``).

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import ZoneBoostRegressor
    >>> X = pd.DataFrame({
    ...     "rooms": [3, 4, 2, 5, 3, 4],
    ...     "distance_km": [5.0, 2.0, 8.0, 1.0, 6.0, 3.0],
    ...     "neighborhood": ["a", "b", "a", "b", "a", "b"],
    ... })
    >>> y = [300, 450, 220, 520, 310, 470]
    >>> model = ZoneBoostRegressor(
    ...     n_rounds=20, categorical_features=["neighborhood"], random_state=0,
    ... )
    >>> model.fit(X, y).predict(X).shape
    (6,)

    Notes
    -----
    This estimator targets practical scikit-learn compatibility --
    ``get_params``/``set_params``/``clone``, use inside a ``Pipeline``,
    and scoring via ``cross_val_score`` -- rather than full compliance
    with ``sklearn.utils.estimator_checks.check_estimator`` (which checks
    many edge cases, e.g. sparse-matrix input, not exercised here).
    """

    def __init__(
        self,
        n_rounds: int = 300,
        learning_rate: float = 0.1,
        row_subsample: float = 0.7,
        col_subsample: float = 0.7,
        max_zones: int = 7,
        min_zone_frac: float = 0.02,
        categorical_features=None,
        validation_fraction: float = 0.2,
        n_iter_no_change: int = None,
        max_interaction_order: int = 2,
        max_triple_interactions: int = 5,
        triple_min_gain: float = 0.05,
        cross_fit_folds: int = 5,
        shrinkage_m: float = 10.0,
        stacking_alpha: float = 0.01,
        monotonic_constraints=None,
        random_state: int = 42,
    ):
        # scikit-learn convention: __init__ only assigns parameters as-is,
        # with no validation or derived computation -- required for
        # get_params()/set_params()/clone() to work correctly.
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.row_subsample = row_subsample
        self.col_subsample = col_subsample
        self.max_zones = max_zones
        self.min_zone_frac = min_zone_frac
        self.categorical_features = categorical_features
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.max_interaction_order = max_interaction_order
        self.max_triple_interactions = max_triple_interactions
        self.triple_min_gain = triple_min_gain
        self.cross_fit_folds = cross_fit_folds
        self.shrinkage_m = shrinkage_m
        self.stacking_alpha = stacking_alpha
        self.monotonic_constraints = monotonic_constraints
        self.random_state = random_state

    def _ensure_dataframe(self, X) -> pd.DataFrame:
        return ensure_dataframe(X, getattr(self, "feature_names_in_", None))

    def fit(self, X, y):
        """Fit the boosted zone-grid ensemble.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Training predictors. Passing a DataFrame is recommended when
            using ``categorical_features`` by name.
        y : array-like of shape (n_samples,)
            Training target.

        Returns
        -------
        self : ZoneBoostRegressor
            The fitted estimator.
        """
        # Not sklearn's check_X_y: it assumes a homogeneous numeric array,
        # which is the wrong tool here -- categorical columns are first-class
        # input (string/object dtype), not something to coerce to float.
        X = self._ensure_dataframe(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y_arr):
            raise ValueError(f"X and y have inconsistent lengths: {len(X)} vs {len(y_arr)}")

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)
        self.predictor_names_ = list(X.columns)
        self.categorical_features_ = resolve_categorical_features(X, self.categorical_features)
        self.monotonic_constraints_ = resolve_monotonic_constraints(
            X, self.monotonic_constraints, self.categorical_features_
        )

        rng = np.random.default_rng(self.random_state)
        has_val = self.validation_fraction and self.validation_fraction > 0
        if has_val:
            n_total = len(X)
            perm = rng.permutation(n_total)
            n_val = max(1, int(n_total * self.validation_fraction))
            val_idx, fit_idx = perm[:n_val], perm[n_val:]
            X_fit = X.iloc[fit_idx].reset_index(drop=True)
            y_fit = y_arr[fit_idx]
            X_val = X.iloc[val_idx].reset_index(drop=True)
            y_val = y_arr[val_idx]
        else:
            X_fit, y_fit = X, y_arr
            X_val = y_val = None

        self.baseline_ = float(y_fit.mean())

        n = len(y_fit)
        n_row_sample = min(n, max(min(20, n), int(n * self.row_subsample)))
        n_predictors = len(self.predictor_names_)
        n_col_sample = min(n_predictors, max(min(2, n_predictors), int(n_predictors * self.col_subsample)))

        current_pred = np.full(n, self.baseline_)
        if has_val:
            current_val_pred = np.full(len(y_val), self.baseline_)

        self.rounds_, self.train_rmse_, self.val_rmse_ = [], [], []
        no_improve_streak = 0

        for _ in range(self.n_rounds):
            residual = y_fit - current_pred

            row_idx = rng.choice(n, size=n_row_sample, replace=False)
            col_subset = list(rng.choice(self.predictor_names_, size=n_col_sample, replace=False))
            X_sub = X_fit.iloc[row_idx][col_subset]
            residual_sub = residual[row_idx]

            zone_info, main_effects, interactions, triples, oof_contributions = weak_learner_fit(
                X_sub,
                residual_sub,
                col_subset,
                self.categorical_features_,
                rng,
                max_zones=self.max_zones,
                min_zone_frac=self.min_zone_frac,
                max_interaction_order=self.max_interaction_order,
                max_triple_interactions=self.max_triple_interactions,
                triple_min_gain=self.triple_min_gain,
                cross_fit_folds=self.cross_fit_folds,
                shrinkage_m=self.shrinkage_m,
                monotonic_constraints=self.monotonic_constraints_,
            )
            contributions = weak_learner_contributions(X_fit, zone_info, main_effects, interactions, triples)
            # The round's own (sub)sampled rows would otherwise be scored by a
            # table partly built from their own residual -- replace exactly
            # those rows with their honest, cross-fitted contributions. Rows
            # this round didn't sample were never part of the table, so
            # they're already leak-free and left untouched.
            contributions[row_idx, :] = oof_contributions
            # Lasso, not a shared equal-weight average: an irrelevant term's
            # weight gets zeroed by the L1 penalty, a strong term gets its
            # own learned weight instead of a diluted 1/n_terms share.
            intercept, weights = _fit_lasso_weights(contributions, residual, self.stacking_alpha)
            fitted_residual = intercept + contributions @ weights

            current_pred = current_pred + self.learning_rate * fitted_residual
            self.rounds_.append(
                {
                    "zone_info": zone_info,
                    "main_effects": main_effects,
                    "interactions": interactions,
                    "triples": triples,
                    "intercept": intercept,
                    "weights": weights,
                }
            )
            self.train_rmse_.append(float(np.sqrt(np.mean((y_fit - current_pred) ** 2))))

            if has_val:
                val_contributions = weak_learner_contributions(X_val, zone_info, main_effects, interactions, triples)
                val_fitted = intercept + val_contributions @ weights
                current_val_pred = current_val_pred + self.learning_rate * val_fitted
                self.val_rmse_.append(float(np.sqrt(np.mean((y_val - current_val_pred) ** 2))))

                if self.n_iter_no_change is not None:
                    best_so_far = min(self.val_rmse_)
                    if self.val_rmse_[-1] <= best_so_far + 1e-12:
                        no_improve_streak = 0
                    else:
                        no_improve_streak += 1
                    if no_improve_streak >= self.n_iter_no_change:
                        break

        self.best_n_rounds_ = int(np.argmin(self.val_rmse_)) + 1 if has_val and self.val_rmse_ else len(self.rounds_)
        return self

    def _raw_predict(self, X: pd.DataFrame, n_rounds: int) -> np.ndarray:
        pred = np.full(len(X), self.baseline_)
        for round_ in self.rounds_[:n_rounds]:
            contributions = weak_learner_contributions(
                X, round_["zone_info"], round_["main_effects"], round_["interactions"], round_["triples"]
            )
            fitted_residual = round_["intercept"] + contributions @ round_["weights"]
            pred = pred + self.learning_rate * fitted_residual
        return pred

    def predict(self, X, n_rounds: int = None) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        n_rounds : int, default=None
            Use only the first ``n_rounds`` boosting rounds instead of the
            fitted ``best_n_rounds_``. Useful for inspecting how the
            prediction evolves round by round (analogous to
            ``staged_predict`` on sklearn's own boosting estimators).

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "rounds_")
        n_rounds = n_rounds if n_rounds is not None else self.best_n_rounds_
        X = self._ensure_dataframe(X)
        return self._raw_predict(X, n_rounds)

    def explain(self, X, n_rounds: int = None) -> pd.DataFrame:
        """Exact per-row, per-term prediction attribution -- not a SHAP/LIME
        -style approximation, but an algebraic decomposition of the same
        computation `predict` performs, so results sum exactly to the
        prediction (see :mod:`zoneboost._explain` for the derivation).

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        n_rounds : int, default=None

        Returns
        -------
        DataFrame of shape (n_samples, n_terms + 1)
            One column per term that appeared in any round (a predictor's
            own name for its main effect, ``"A x B"`` for an interaction
            pair, ``"A x B x C"`` for a 3-way interaction) plus
            ``"baseline"``. Row sums equal ``predict(X)`` exactly, up to
            floating-point rounding.
        """
        check_is_fitted(self, "rounds_")
        n_rounds = n_rounds if n_rounds is not None else self.best_n_rounds_
        X = self._ensure_dataframe(X)
        return explain_rounds(X, self.rounds_[:n_rounds], self.baseline_, self.learning_rate)

    def feature_importance(self, X, n_rounds: int = None) -> pd.Series:
        """Global importance: each term's mean absolute contribution over
        the rows in `X`, derived directly from :meth:`explain` (not a
        split-count or permutation proxy).

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        n_rounds : int, default=None

        Returns
        -------
        Series
            Indexed by term name, sorted descending.
        """
        contributions = self.explain(X, n_rounds).drop(columns=["baseline"])
        return contributions.abs().mean().sort_values(ascending=False)
