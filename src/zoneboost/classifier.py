"""ZoneBoostClassifier: the same zone-grid weak learner as
ZoneBoostRegressor, boosted in log-odds space with a sigmoid link instead
of boosting the raw target directly -- the standard way gradient boosting
generalizes from regression to classification.

Binary is a single log-odds booster. 3+ classes uses one-vs-rest: an
independent log-odds booster is fit per class (that class vs. everything
else, sharing one validation split across all of them), and their
probabilities are combined at predict time by normalizing so each row
sums to 1. No new mechanism for multiclass -- it's K independent copies of
the exact same binary booster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_is_fitted

from ._common import ensure_dataframe, resolve_categorical_features, resolve_monotonic_constraints
from ._explain import explain_rounds
from ._weak_learner import _fit_lasso_weights, weak_learner_contributions, weak_learner_fit

__all__ = ["ZoneBoostClassifier"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _log_loss(y_true: np.ndarray, p_hat: np.ndarray, eps: float = 1e-15) -> float:
    p_hat = np.clip(p_hat, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p_hat) + (1 - y_true) * np.log(1 - p_hat)))


class _LogOddsBooster:
    """One binary log-odds booster -- the actual boosting loop. Takes
    already-split fit/val data directly (no validation-fraction logic of
    its own); ZoneBoostClassifier does one shared split and reuses it
    across all K one-vs-rest boosters, so class distributions in the
    validation set stay consistent across classes."""

    def __init__(self, n_rounds, learning_rate, row_subsample, col_subsample,
                 max_zones, min_zone_frac, categorical_features, n_iter_no_change,
                 max_interaction_order, max_triple_interactions, triple_min_gain,
                 cross_fit_folds, shrinkage_m, stacking_alpha, monotonic_constraints,
                 max_pair_interactions, calibrate, random_state):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.row_subsample = row_subsample
        self.col_subsample = col_subsample
        self.max_zones = max_zones
        self.min_zone_frac = min_zone_frac
        self.categorical_features = categorical_features
        self.n_iter_no_change = n_iter_no_change
        self.max_interaction_order = max_interaction_order
        self.max_triple_interactions = max_triple_interactions
        self.triple_min_gain = triple_min_gain
        self.cross_fit_folds = cross_fit_folds
        self.shrinkage_m = shrinkage_m
        self.stacking_alpha = stacking_alpha
        self.monotonic_constraints = monotonic_constraints
        self.max_pair_interactions = max_pair_interactions
        self.calibrate = calibrate
        self.random_state = random_state
        self.calibrator_ = None

    def fit(self, X_fit: pd.DataFrame, y_fit: np.ndarray, X_val=None, y_val=None):
        predictor_names = list(X_fit.columns)
        n = len(y_fit)
        p0 = np.clip(y_fit.mean(), 1e-6, 1 - 1e-6)
        self.baseline_ = float(np.log(p0 / (1 - p0)))

        n_row_sample = min(n, max(min(20, n), int(n * self.row_subsample)))
        n_predictors = len(predictor_names)
        n_col_sample = min(n_predictors, max(min(2, n_predictors), int(n_predictors * self.col_subsample)))
        rng = np.random.default_rng(self.random_state)

        current_score = np.full(n, self.baseline_)
        has_val = X_val is not None and y_val is not None
        if has_val:
            current_val_score = np.full(len(y_val), self.baseline_)

        self.rounds_, self.val_logloss_ = [], []
        no_improve_streak = 0

        for _ in range(self.n_rounds):
            p_hat = _sigmoid(current_score)
            residual = y_fit - p_hat  # negative gradient of log-loss

            row_idx = rng.choice(n, size=n_row_sample, replace=False)
            col_subset = list(rng.choice(predictor_names, size=n_col_sample, replace=False))
            X_sub = X_fit.iloc[row_idx][col_subset]
            residual_sub = residual[row_idx]

            zone_info, main_effects, interactions, triples, oof_contributions = weak_learner_fit(
                X_sub, residual_sub, col_subset, self.categorical_features, rng,
                max_zones=self.max_zones, min_zone_frac=self.min_zone_frac,
                max_interaction_order=self.max_interaction_order,
                max_triple_interactions=self.max_triple_interactions,
                triple_min_gain=self.triple_min_gain,
                cross_fit_folds=self.cross_fit_folds,
                shrinkage_m=self.shrinkage_m,
                monotonic_constraints=self.monotonic_constraints,
                max_pair_interactions=self.max_pair_interactions,
            )
            contributions = weak_learner_contributions(X_fit, zone_info, main_effects, interactions, triples)
            contributions[row_idx, :] = oof_contributions
            intercept, weights = _fit_lasso_weights(contributions, residual, self.stacking_alpha)
            fitted_residual = intercept + contributions @ weights

            current_score = current_score + self.learning_rate * fitted_residual
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

            if has_val:
                val_contributions = weak_learner_contributions(X_val, zone_info, main_effects, interactions, triples)
                val_fitted = intercept + val_contributions @ weights
                current_val_score = current_val_score + self.learning_rate * val_fitted
                self.val_logloss_.append(_log_loss(y_val, _sigmoid(current_val_score)))

                if self.n_iter_no_change is not None:
                    best_so_far = min(self.val_logloss_)
                    no_improve_streak = 0 if self.val_logloss_[-1] <= best_so_far + 1e-12 else no_improve_streak + 1
                    if no_improve_streak >= self.n_iter_no_change:
                        break

        self.best_n_rounds_ = int(np.argmin(self.val_logloss_)) + 1 if has_val and self.val_logloss_ else len(self.rounds_)

        # Isotonic probability calibration: fit on the same held-out
        # validation split already used for early stopping (never training
        # rows), mapping this booster's own raw held-out probability to the
        # actual label -- the standard isotonic-calibration recipe (same
        # idea sklearn.calibration.CalibratedClassifierCV(method="isotonic")
        # uses). Only changes predict_proba's output, not explain()'s raw
        # log-odds decomposition.
        if self.calibrate and has_val:
            raw_p_val = self._raw_predict_proba(X_val, self.best_n_rounds_)
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_p_val, y_val)

        return self

    def _raw_predict_proba(self, X: pd.DataFrame, n_rounds: int) -> np.ndarray:
        score = np.full(len(X), self.baseline_)
        for round_ in self.rounds_[:n_rounds]:
            contributions = weak_learner_contributions(
                X, round_["zone_info"], round_["main_effects"], round_["interactions"], round_["triples"]
            )
            fitted_residual = round_["intercept"] + contributions @ round_["weights"]
            score = score + self.learning_rate * fitted_residual
        return _sigmoid(score)

    def predict_proba(self, X: pd.DataFrame, n_rounds: int = None) -> np.ndarray:
        n_rounds = n_rounds if n_rounds is not None else self.best_n_rounds_
        raw_p = self._raw_predict_proba(X, n_rounds)
        if self.calibrator_ is not None:
            return self.calibrator_.predict(raw_p)
        return raw_p


class ZoneBoostClassifier(BaseEstimator, ClassifierMixin):
    """Gradient boosting classifier with zone-grid weak learners instead
    of trees -- the classification counterpart of
    :class:`~zoneboost.ZoneBoostRegressor`, using the identical weak
    learner (main effects + pairwise interactions, density-confidence
    weighted). The only change from the regressor: each round is fit
    against the residual in **log-odds space**
    (``y - sigmoid(current_score)``, the standard logistic-loss gradient)
    instead of the raw target, and predictions are squashed through a
    sigmoid.

    Binary targets (2 distinct classes) fit a single log-odds booster.
    3+ classes use one-vs-rest: an independent log-odds booster is fit per
    class against "is this class vs. everything else", sharing one
    validation split across all of them, and their probabilities are
    normalized to sum to 1 at predict time. Multiclass is not a different
    mechanism -- it's K independent copies of the same binary booster.

    Parameters
    ----------
    n_rounds : int, default=300
        Maximum number of boosting rounds per class.
    learning_rate : float, default=0.1
        Shrinkage applied to each round's correction (in log-odds space).
    row_subsample : float, default=0.7
        Fraction of training rows randomly sampled to fit each round.
    col_subsample : float, default=0.7
        Fraction of predictor columns randomly sampled for each round.
    max_zones : int, default=7
        Upper bound on the number of zones for *continuous* columns only.
        See :class:`~zoneboost.ZoneBoostRegressor` for why this is kept
        conservative and why high-cardinality nominal variables should use
        ``categorical_features`` instead of a larger cap.
    min_zone_frac : float, default=0.02
        Minimum fraction of (subsampled) rows required on each side of a
        candidate zone split.
    categorical_features : list of str or int, default=None
        Columns to treat as nominal categories rather than continuous.
        Columns of ``object``, ``category``, or ``bool`` dtype are always
        treated as categorical automatically, in addition to any named
        here.
    validation_fraction : float, default=0.2
        Fraction of training rows held out (once, shared across every
        one-vs-rest booster) to pick each booster's best round count via
        early stopping. Set to 0 to disable and always use ``n_rounds``.
    n_iter_no_change : int, default=None
        If set, a booster stops adding rounds once its held-out log-loss
        has not improved for this many consecutive rounds.
    max_interaction_order : int, default=2
        ``2`` fits main effects + pairwise interactions only (the behavior
        of every prior release). ``3`` additionally attempts a bounded,
        adaptive search for 3-way interactions each round. See
        :class:`~zoneboost.ZoneBoostRegressor` for the full description --
        the mechanism is identical here.
    max_triple_interactions : int, default=5
        Cap on how many 3-way terms a single round may add. Only relevant
        when ``max_interaction_order=3``.
    max_pair_interactions : int, default=None
        Cap on how many pairwise interactions a single round keeps, ranked
        by mean absolute contribution, applied after 3-way interaction
        selection so it never changes which triples are found. ``None``
        (default) keeps every pair -- see
        :class:`~zoneboost.ZoneBoostRegressor` for the full description.
    triple_min_gain : float, default=0.05
        Minimum residual-explained magnitude a candidate 3-way interaction
        must retain after subtracting the main-effect + pairwise fit for
        its three columns, expressed as a fraction of its strongest
        constituent pair's own importance. Only relevant when
        ``max_interaction_order=3``.
    cross_fit_folds : int, default=5
        Each round splits its rows into this many folds and scores each
        fold only with zone tables built from the other folds, so no row's
        own residual leaks into the zone mean it's judged against -- see
        :class:`~zoneboost.ZoneBoostRegressor` for the full description.
    shrinkage_m : float, default=10.0
        Every zone's mean is shrunk toward a hierarchical prior via an
        empirical-Bayes (m-estimate) fit, replacing the flat confidence
        discount used by every prior release -- see
        :class:`~zoneboost.ZoneBoostRegressor` for the full description.
    stacking_alpha : float, default=0.01
        Terms are combined via a Lasso fit on each term's own (cross-fitted)
        contribution rather than an equal-weight average, so an irrelevant
        term's weight gets zeroed and a strong term gets its own learned
        weight -- see :class:`~zoneboost.ZoneBoostRegressor` for the full
        description.
    monotonic_constraints : dict, default=None
        ``{column: +1 or -1}`` -- forces a continuous column's main effect
        to be non-decreasing/non-increasing across its zones. Opt-in
        (encodes domain knowledge, not a general improvement) -- see
        :class:`~zoneboost.ZoneBoostRegressor` for the full description.
    calibrate : bool, default=False
        If ``True``, recalibrate each booster's raw probability with an
        isotonic regression fit on the held-out validation split (the same
        split used for early stopping), so predicted probabilities better
        match empirical frequencies -- the standard isotonic-calibration
        recipe (``sklearn.calibration.CalibratedClassifierCV(method=
        "isotonic")`` uses the same idea). Only changes :meth:`predict_proba`
        -- :meth:`explain`/:meth:`feature_importance` still decompose the raw
        log-odds score, unaffected. Requires ``validation_fraction > 0``;
        raises ``ValueError`` at `fit` otherwise. This is the one parameter
        that differs from :class:`~zoneboost.ZoneBoostRegressor` -- every
        other parameter is identical across both estimators.
    random_state : int, default=42
        Seed controlling the validation split and per-round subsampling.

    Attributes
    ----------
    classes_ : ndarray
        Distinct class labels seen during `fit`, sorted.
    n_features_in_ : int
    feature_names_in_ : ndarray
    categorical_features_ : set
        Resolved set of columns treated as categorical.
    monotonic_constraints_ : dict
        Resolved ``{column: +1 or -1}`` constraints actually in effect.
    multiclass_ : bool
        Whether one-vs-rest (3+ classes) was used.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import ZoneBoostClassifier
    >>> X = pd.DataFrame({
    ...     "score": [1, 2, 8, 9, 3, 7],
    ...     "group": ["a", "a", "b", "b", "a", "b"],
    ... })
    >>> y = [0, 0, 1, 1, 0, 1]
    >>> model = ZoneBoostClassifier(n_rounds=20, categorical_features=["group"], random_state=0)
    >>> model.fit(X, y).predict(X).shape
    (6,)

    Notes
    -----
    This estimator targets practical scikit-learn compatibility --
    ``get_params``/``set_params``/``clone``, use inside a ``Pipeline``,
    scoring via ``cross_val_score`` -- rather than full compliance with
    ``sklearn.utils.estimator_checks.check_estimator``.
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
        max_pair_interactions=None,
        calibrate: bool = False,
        random_state: int = 42,
    ):
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
        self.max_pair_interactions = max_pair_interactions
        self.calibrate = calibrate
        self.random_state = random_state

    def _ensure_dataframe(self, X) -> pd.DataFrame:
        return ensure_dataframe(X, getattr(self, "feature_names_in_", None))

    def _make_booster(self) -> _LogOddsBooster:
        return _LogOddsBooster(
            n_rounds=self.n_rounds,
            learning_rate=self.learning_rate,
            row_subsample=self.row_subsample,
            col_subsample=self.col_subsample,
            max_zones=self.max_zones,
            min_zone_frac=self.min_zone_frac,
            categorical_features=self.categorical_features_,
            n_iter_no_change=self.n_iter_no_change,
            max_interaction_order=self.max_interaction_order,
            max_triple_interactions=self.max_triple_interactions,
            triple_min_gain=self.triple_min_gain,
            cross_fit_folds=self.cross_fit_folds,
            shrinkage_m=self.shrinkage_m,
            stacking_alpha=self.stacking_alpha,
            monotonic_constraints=self.monotonic_constraints_,
            max_pair_interactions=self.max_pair_interactions,
            calibrate=self.calibrate,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        """Fit the boosted zone-grid ensemble (one booster if binary, one
        per class via one-vs-rest if 3+ classes).

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels -- at least 2 distinct values.

        Returns
        -------
        self : ZoneBoostClassifier
        """
        X = self._ensure_dataframe(X)
        y_arr = np.asarray(y).reshape(-1)
        if len(X) != len(y_arr):
            raise ValueError(f"X and y have inconsistent lengths: {len(X)} vs {len(y_arr)}")

        self.classes_ = np.unique(y_arr)
        if len(self.classes_) < 2:
            raise ValueError(f"Need at least 2 classes, got {len(self.classes_)}")
        self.multiclass_ = len(self.classes_) > 2

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)
        self.predictor_names_ = list(X.columns)
        self.categorical_features_ = resolve_categorical_features(X, self.categorical_features)
        self.monotonic_constraints_ = resolve_monotonic_constraints(
            X, self.monotonic_constraints, self.categorical_features_
        )

        rng = np.random.default_rng(self.random_state)
        has_val = self.validation_fraction and self.validation_fraction > 0
        if self.calibrate and not has_val:
            raise ValueError("calibrate=True requires validation_fraction > 0 (no held-out data to calibrate against).")
        if has_val:
            n_total = len(X)
            perm = rng.permutation(n_total)
            n_val = max(1, int(n_total * self.validation_fraction))
            val_idx, fit_idx = perm[:n_val], perm[n_val:]
            X_fit = X.iloc[fit_idx].reset_index(drop=True)
            X_val = X.iloc[val_idx].reset_index(drop=True)
            y_fit_raw, y_val_raw = y_arr[fit_idx], y_arr[val_idx]
        else:
            X_fit, y_fit_raw = X, y_arr
            X_val = y_val_raw = None

        def fit_one(positive_class):
            y_fit_bin = (y_fit_raw == positive_class).astype(float)
            y_val_bin = (y_val_raw == positive_class).astype(float) if has_val else None
            return self._make_booster().fit(X_fit, y_fit_bin, X_val=X_val, y_val=y_val_bin)

        if not self.multiclass_:
            self.booster_ = fit_one(self.classes_[-1])
        else:
            self.boosters_ = {k: fit_one(k) for k in self.classes_}
        return self

    def predict_proba(self, X, n_rounds: int = None) -> np.ndarray:
        """Class probabilities.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        n_rounds : int, default=None
            Use only the first ``n_rounds`` boosting rounds per booster
            instead of each one's fitted best-round count.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, "classes_")
        X = self._ensure_dataframe(X)

        if not self.multiclass_:
            p1 = self.booster_.predict_proba(X, n_rounds)
            return np.column_stack([1 - p1, p1])

        raw = np.column_stack([self.boosters_[k].predict_proba(X, n_rounds) for k in self.classes_])
        row_sums = raw.sum(axis=1, keepdims=True)
        return np.divide(raw, row_sums, out=np.full_like(raw, 1.0 / len(self.classes_)), where=row_sums > 0)

    def predict(self, X, n_rounds: int = None) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        n_rounds : int, default=None

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X, n_rounds)
        return self.classes_[np.argmax(proba, axis=1)]

    def _explain_one(self, booster: _LogOddsBooster, X: pd.DataFrame, n_rounds: int) -> pd.DataFrame:
        n_rounds = n_rounds if n_rounds is not None else booster.best_n_rounds_
        return explain_rounds(X, booster.rounds_[:n_rounds], booster.baseline_, self.learning_rate)

    def explain(self, X, n_rounds: int = None):
        """Exact per-row, per-term attribution of each booster's log-odds
        score -- not a SHAP/LIME-style approximation, but an algebraic
        decomposition of the same computation `predict_proba` performs
        (see :mod:`zoneboost._explain`). Row sums equal the **log-odds**
        score exactly, not the probability directly -- the same convention
        SHAP itself uses for logistic/margin-based models, since
        probability contributions don't add linearly through a sigmoid.

        For 3+ classes: ``sigmoid(explain(X)[k].sum(axis=1))`` reproduces
        that class's booster's *raw*, pre-calibration, pre-normalization
        one-vs-rest probability -- equal to ``boosters_[k].predict_proba(X)``
        only when ``calibrate=False`` (the default); with ``calibrate=True``,
        ``predict_proba`` additionally applies the fitted isotonic calibrator
        on top of this raw value, and then normalizes across all K boosters
        so ``predict_proba(X)[:, k]`` sums to 1 across classes. ``explain``
        always reflects the raw, uncalibrated log-odds score either way.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        n_rounds : int, default=None

        Returns
        -------
        DataFrame, or dict of {class_label: DataFrame} if 3+ classes
            One column per term plus ``"baseline"``. Binary: a single
            DataFrame for the positive class's (``classes_[-1]``) log-odds
            score. Multiclass: one DataFrame per class, each for that
            class's own one-vs-rest booster.
        """
        check_is_fitted(self, "classes_")
        X = self._ensure_dataframe(X)
        if not self.multiclass_:
            return self._explain_one(self.booster_, X, n_rounds)
        return {k: self._explain_one(self.boosters_[k], X, n_rounds) for k in self.classes_}

    def feature_importance(self, X, n_rounds: int = None) -> pd.Series:
        """Global importance: each term's mean absolute log-odds
        contribution over the rows in `X`, derived directly from
        :meth:`explain`. For multiclass, averaged across the per-class
        booster importances.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        n_rounds : int, default=None

        Returns
        -------
        Series
            Indexed by term name, sorted descending.
        """
        explanation = self.explain(X, n_rounds)
        if not self.multiclass_:
            return explanation.drop(columns=["baseline"]).abs().mean().sort_values(ascending=False)
        per_class = [df.drop(columns=["baseline"]).abs().mean() for df in explanation.values()]
        return pd.concat(per_class, axis=1).mean(axis=1).sort_values(ascending=False)
