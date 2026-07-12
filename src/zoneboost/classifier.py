"""ZoneBoostClassifier: the same zone-grid weak learner as
ZoneBoostRegressor, boosted in log-odds space with a sigmoid link instead
of boosting the raw target directly -- the standard way gradient boosting
generalizes from regression to classification.

Binary is a single log-odds booster (``_LogOddsBooster``) -- already a
single, principled sigmoid with no one-vs-rest heuristic involved, so it's
untouched by everything below.

3+ classes uses **native multinomial (softmax) boosting** (``_SoftmaxBooster``):
one booster maintains all K logits jointly per row, using row-wise softmax
instead of K independent sigmoids, so each round's residual
(``1(y==k) - softmax(scores)[:, k]``) reflects genuine competition between
classes through the shared softmax denominator -- unlike one-vs-rest, where
each class's booster never knows about the other K-1 classes' current
scores at all. A separate weak learner is still fit per class per round
(reusing :func:`zoneboost._weak_learner.weak_learner_fit` unchanged), but the
K resulting corrections are centered to sum to zero per row before being
added to the running scores -- a no-op for predictions (softmax is
shift-invariant to a constant added equally to every class's logit), needed
only so each class's own contribution is uniquely defined for
``explain()`` rather than ambiguous up to an arbitrary shared shift. See
:class:`_SoftmaxBooster` and ``ZoneBoostClassifier.explain`` for the full
mechanism, including the ``"_softmax_centering"`` column that reconciles
this centering back into an exact per-class attribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.utils.validation import check_is_fitted

from ._common import (
    ensure_dataframe,
    resolve_bounded_effects,
    resolve_categorical_features,
    resolve_forbidden_interactions,
    resolve_monotonic_constraints,
)
from ._explain import explain_rounds
from ._reliability import evidence_report, explain_reliability
from ._weak_learner import _fit_lasso_weights, weak_learner_contributions, weak_learner_fit

__all__ = ["ZoneBoostClassifier"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _log_loss(y_true: np.ndarray, p_hat: np.ndarray, eps: float = 1e-15) -> float:
    p_hat = np.clip(p_hat, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p_hat) + (1 - y_true) * np.log(1 - p_hat)))


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shape ``(n, K)`` -- shift-invariant (subtracting
    each row's own max before exponentiating changes no output, only
    numerical stability), the same invariance the sum-to-zero
    identifiability centering in ``_SoftmaxBooster`` relies on."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _multinomial_log_loss(y_onehot: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p_clipped = np.clip(p, eps, 1 - eps)
    return float(-np.mean(np.sum(y_onehot * np.log(p_clipped), axis=1)))


class _LogOddsBooster:
    """One binary log-odds booster -- the actual boosting loop, used only
    for binary targets. Takes already-split fit/val data directly (no
    validation-fraction logic of its own); ZoneBoostClassifier owns that
    split. 3+ classes use ``_SoftmaxBooster`` instead -- see the module
    docstring."""

    def __init__(self, n_rounds, learning_rate, row_subsample, col_subsample,
                 max_zones, min_zone_frac, categorical_features, n_iter_no_change,
                 max_interaction_order, max_triple_interactions, triple_min_gain,
                 cross_fit_folds, shrinkage_m, stacking_alpha, monotonic_constraints,
                 max_pair_interactions, adaptive_boundary_smoothing, boundary_shrinkage_m,
                 convexity_constraints, bounded_effects, forbidden_interactions,
                 track_reliability, calibrate, refit_on_full_data, random_state):
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
        self.adaptive_boundary_smoothing = adaptive_boundary_smoothing
        self.boundary_shrinkage_m = boundary_shrinkage_m
        self.convexity_constraints = convexity_constraints
        self.bounded_effects = bounded_effects
        self.forbidden_interactions = forbidden_interactions
        self.track_reliability = track_reliability
        self.calibrate = calibrate
        self.refit_on_full_data = refit_on_full_data
        self.random_state = random_state
        self.calibrator_ = None

    def fit(self, X_fit: pd.DataFrame, y_fit: np.ndarray, X_val=None, y_val=None, X_cal=None, y_cal=None):
        predictor_names = list(X_fit.columns)
        rng = np.random.default_rng(self.random_state)
        has_val = X_val is not None and y_val is not None

        # Phase 1 (selection): fit on X_fit, track val_logloss_ on X_val (if
        # any) with early stopping, to decide best_n_rounds_.
        rounds, baseline, val_logloss = self._boost_rounds(
            X_fit, y_fit, predictor_names, rng, self.n_rounds, X_val, y_val, early_stopping=True
        )
        self.val_logloss_ = val_logloss
        self.best_n_rounds_ = int(np.argmin(val_logloss)) + 1 if has_val and val_logloss else len(rounds)

        if self.refit_on_full_data and has_val:
            # Phase 2 (refit): the deployed model is retrained on fit+
            # validation combined, for exactly best_n_rounds_ rounds --
            # already decided, so no early stopping this time. Fresh rng so
            # reproducibility doesn't depend on how much of phase 1's random
            # stream got consumed. val_logloss_ above stays as the phase-1
            # selection diagnostic; it isn't recomputed here.
            X_refit = pd.concat([X_fit, X_val], ignore_index=True)
            y_refit = np.concatenate([y_fit, y_val])
            refit_rng = np.random.default_rng(self.random_state)
            rounds, baseline, _ = self._boost_rounds(X_refit, y_refit, predictor_names, refit_rng, self.best_n_rounds_)

        self.rounds_ = rounds
        self.baseline_ = baseline

        # Isotonic probability calibration: fit on a genuinely held-out
        # split -- the dedicated calibration split if provided, else the
        # same validation split used for early stopping (the default,
        # disclosed tradeoff), mapping this booster's own raw held-out
        # probability to the actual label -- the standard isotonic-
        # calibration recipe (same idea sklearn.calibration.
        # CalibratedClassifierCV(method="isotonic") uses). Only changes
        # predict_proba's output, not explain()'s raw log-odds decomposition.
        has_cal = X_cal is not None and y_cal is not None
        cal_X, cal_y = (X_cal, y_cal) if has_cal else (X_val, y_val)
        if self.calibrate and cal_X is not None:
            raw_p_cal = self._raw_predict_proba(cal_X, self.best_n_rounds_)
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_p_cal, cal_y)

        return self

    def _boost_rounds(self, X_train, y_train, predictor_names, rng, n_rounds, X_val=None, y_val=None, early_stopping=False):
        """Core boosting loop -- extracted so `fit` can run it twice: once
        for round-count selection (on the fit split, tracking
        ``val_logloss_`` with early stopping), and optionally again for a
        final refit (fit+validation combined, exact round count, no early
        stopping) -- see `fit` and ``refit_on_full_data``.

        Returns
        -------
        rounds, baseline, val_logloss
        """
        n = len(y_train)
        p0 = np.clip(y_train.mean(), 1e-6, 1 - 1e-6)
        baseline = float(np.log(p0 / (1 - p0)))

        n_row_sample = min(n, max(min(20, n), int(n * self.row_subsample)))
        n_predictors = len(predictor_names)
        n_col_sample = min(n_predictors, max(min(2, n_predictors), int(n_predictors * self.col_subsample)))

        current_score = np.full(n, baseline)
        has_val = X_val is not None
        if has_val:
            current_val_score = np.full(len(y_val), baseline)

        rounds, val_logloss = [], []
        no_improve_streak = 0

        for _ in range(n_rounds):
            p_hat = _sigmoid(current_score)
            residual = y_train - p_hat  # negative gradient of log-loss

            row_idx = rng.choice(n, size=n_row_sample, replace=False)
            col_subset = list(rng.choice(predictor_names, size=n_col_sample, replace=False))
            X_sub = X_train.iloc[row_idx][col_subset]
            residual_sub = residual[row_idx]

            zone_info, main_effects, interactions, triples, oof_contributions, diagnostics = weak_learner_fit(
                X_sub, residual_sub, col_subset, self.categorical_features, rng,
                max_zones=self.max_zones, min_zone_frac=self.min_zone_frac,
                max_interaction_order=self.max_interaction_order,
                max_triple_interactions=self.max_triple_interactions,
                triple_min_gain=self.triple_min_gain,
                cross_fit_folds=self.cross_fit_folds,
                shrinkage_m=self.shrinkage_m,
                monotonic_constraints=self.monotonic_constraints,
                max_pair_interactions=self.max_pair_interactions,
                adaptive_boundary_smoothing=self.adaptive_boundary_smoothing,
                boundary_shrinkage_m=self.boundary_shrinkage_m,
                convexity_constraints=self.convexity_constraints,
                bounded_effects=self.bounded_effects,
                forbidden_interactions=self.forbidden_interactions,
                track_reliability=self.track_reliability,
            )
            contributions = weak_learner_contributions(X_train, zone_info, main_effects, interactions, triples)
            contributions[row_idx, :] = oof_contributions
            intercept, weights = _fit_lasso_weights(contributions, residual, self.stacking_alpha)
            fitted_residual = intercept + contributions @ weights

            current_score = current_score + self.learning_rate * fitted_residual
            rounds.append(
                {
                    "zone_info": zone_info,
                    "main_effects": main_effects,
                    "interactions": interactions,
                    "triples": triples,
                    "intercept": intercept,
                    "weights": weights,
                    "diagnostics": diagnostics,
                }
            )

            if has_val:
                val_contributions = weak_learner_contributions(X_val, zone_info, main_effects, interactions, triples)
                val_fitted = intercept + val_contributions @ weights
                current_val_score = current_val_score + self.learning_rate * val_fitted
                val_logloss.append(_log_loss(y_val, _sigmoid(current_val_score)))

                if early_stopping and self.n_iter_no_change is not None:
                    best_so_far = min(val_logloss)
                    no_improve_streak = 0 if val_logloss[-1] <= best_so_far + 1e-12 else no_improve_streak + 1
                    if no_improve_streak >= self.n_iter_no_change:
                        break

        return rounds, baseline, val_logloss

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


class _SoftmaxBooster:
    """The joint, K-class booster used for 3+ classes -- native multinomial
    (softmax) boosting instead of K independent one-vs-rest
    ``_LogOddsBooster``s. See the module docstring for the full mechanism.
    Works entirely in terms of integer class indices ``0..K-1`` (not class
    labels) -- ``ZoneBoostClassifier`` owns the label <-> index mapping.

    Unlike ``_LogOddsBooster``, ``fit`` needs an explicit ``n_classes``: with
    K independent boosters there was never a question of how many classes
    exist, but here the score matrix and one-hot encoding need to be sized
    correctly regardless of which labels happen to appear in a particular
    split.
    """

    def __init__(self, n_rounds, learning_rate, row_subsample, col_subsample,
                 max_zones, min_zone_frac, categorical_features, n_iter_no_change,
                 max_interaction_order, max_triple_interactions, triple_min_gain,
                 cross_fit_folds, shrinkage_m, stacking_alpha, monotonic_constraints,
                 max_pair_interactions, adaptive_boundary_smoothing, boundary_shrinkage_m,
                 convexity_constraints, bounded_effects, forbidden_interactions,
                 track_reliability, calibrate, refit_on_full_data, random_state):
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
        self.adaptive_boundary_smoothing = adaptive_boundary_smoothing
        self.boundary_shrinkage_m = boundary_shrinkage_m
        self.convexity_constraints = convexity_constraints
        self.bounded_effects = bounded_effects
        self.forbidden_interactions = forbidden_interactions
        self.track_reliability = track_reliability
        self.calibrate = calibrate
        self.refit_on_full_data = refit_on_full_data
        self.random_state = random_state
        self.calibrators_ = None

    def fit(self, X_fit, y_fit_idx, n_classes, X_val=None, y_val_idx=None, X_cal=None, y_cal_idx=None):
        predictor_names = list(X_fit.columns)
        rng = np.random.default_rng(self.random_state)
        has_val = X_val is not None and y_val_idx is not None

        # Phase 1 (selection): fit on X_fit, track val_logloss_ on X_val (if
        # any) with early stopping, to decide best_n_rounds_.
        rounds, baseline, val_logloss = self._boost_rounds(
            X_fit, y_fit_idx, n_classes, predictor_names, rng, self.n_rounds, X_val, y_val_idx, early_stopping=True
        )
        self.val_logloss_ = val_logloss
        self.best_n_rounds_ = int(np.argmin(val_logloss)) + 1 if has_val and val_logloss else len(rounds)
        self.n_classes_ = n_classes

        if self.refit_on_full_data and has_val:
            # Phase 2 (refit): mirrors _LogOddsBooster -- deployed model
            # retrained on fit+validation combined, exact round count,
            # fresh rng, no early stopping. val_logloss_ above stays the
            # phase-1 selection diagnostic.
            X_refit = pd.concat([X_fit, X_val], ignore_index=True)
            y_refit_idx = np.concatenate([y_fit_idx, y_val_idx])
            refit_rng = np.random.default_rng(self.random_state)
            rounds, baseline, _ = self._boost_rounds(
                X_refit, y_refit_idx, n_classes, predictor_names, refit_rng, self.best_n_rounds_
            )

        self.rounds_ = rounds
        self.baseline_ = baseline

        # Isotonic probability calibration: one calibrator per class,
        # fit on that class's own marginal softmax probability on a
        # genuinely held-out split -- mirrors _LogOddsBooster's recipe,
        # applied per class since softmax's K outputs are calibrated
        # independently, then renormalized back to sum to 1 (independent
        # per-class calibration can otherwise break that).
        has_cal = X_cal is not None and y_cal_idx is not None
        cal_X, cal_y_idx = (X_cal, y_cal_idx) if has_cal else (X_val, y_val_idx)
        if self.calibrate and cal_X is not None:
            raw_p_cal = self._raw_predict_proba(cal_X, self.best_n_rounds_)
            self.calibrators_ = {
                k_idx: IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                    raw_p_cal[:, k_idx], (cal_y_idx == k_idx).astype(float)
                )
                for k_idx in range(n_classes)
            }

        return self

    def _boost_rounds(
        self, X_train, y_train_idx, n_classes, predictor_names, rng, n_rounds,
        X_val=None, y_val_idx=None, early_stopping=False,
    ):
        """Core multinomial boosting loop -- mirrors
        ``_LogOddsBooster._boost_rounds``, extracted so `fit` can run it
        twice: once for round-count selection, optionally again for a
        final refit. Returns ``rounds, baseline, val_logloss``.

        Each round fits one weak learner *per class* against that class's
        own softmax-cross-entropy residual (``weak_learner_fit`` reused
        unchanged, just called K times), then centers the K raw outputs to
        sum to zero per row before scaling by ``learning_rate`` and adding
        to the running scores -- see the module docstring for why.
        """
        n = len(y_train_idx)
        K = n_classes
        class_counts = np.bincount(y_train_idx, minlength=K).astype(float)
        class_freq = np.clip(class_counts / n, 1e-6, 1 - 1e-6)
        baseline = np.log(class_freq)  # (K,) -- multinomial analogue of log(p/(1-p))

        n_row_sample = min(n, max(min(20, n), int(n * self.row_subsample)))
        n_predictors = len(predictor_names)
        n_col_sample = min(n_predictors, max(min(2, n_predictors), int(n_predictors * self.col_subsample)))

        Y_onehot = np.eye(K)[y_train_idx]
        current_scores = np.tile(baseline, (n, 1))
        has_val = X_val is not None
        if has_val:
            Y_val_onehot = np.eye(K)[y_val_idx]
            current_val_scores = np.tile(baseline, (len(y_val_idx), 1))

        rounds, val_logloss = [], []
        no_improve_streak = 0

        for _ in range(n_rounds):
            p = _softmax(current_scores)
            residual = Y_onehot - p  # (n, K), the joint softmax cross-entropy gradient

            row_idx = rng.choice(n, size=n_row_sample, replace=False)
            col_subset = list(rng.choice(predictor_names, size=n_col_sample, replace=False))
            X_sub = X_train.iloc[row_idx][col_subset]

            round_tables = {}
            fitted_residual = np.zeros((n, K))
            for k_idx in range(K):
                residual_sub_k = residual[row_idx, k_idx]
                zone_info, main_effects, interactions, triples, oof_contributions, diagnostics = weak_learner_fit(
                    X_sub, residual_sub_k, col_subset, self.categorical_features, rng,
                    max_zones=self.max_zones, min_zone_frac=self.min_zone_frac,
                    max_interaction_order=self.max_interaction_order,
                    max_triple_interactions=self.max_triple_interactions,
                    triple_min_gain=self.triple_min_gain,
                    cross_fit_folds=self.cross_fit_folds,
                    shrinkage_m=self.shrinkage_m,
                    monotonic_constraints=self.monotonic_constraints,
                    max_pair_interactions=self.max_pair_interactions,
                    adaptive_boundary_smoothing=self.adaptive_boundary_smoothing,
                    boundary_shrinkage_m=self.boundary_shrinkage_m,
                    convexity_constraints=self.convexity_constraints,
                    bounded_effects=self.bounded_effects,
                    forbidden_interactions=self.forbidden_interactions,
                    track_reliability=self.track_reliability,
                )
                contributions = weak_learner_contributions(X_train, zone_info, main_effects, interactions, triples)
                contributions[row_idx, :] = oof_contributions
                intercept, weights = _fit_lasso_weights(contributions, residual[:, k_idx], self.stacking_alpha)
                fitted_residual[:, k_idx] = intercept + contributions @ weights
                round_tables[k_idx] = {
                    "zone_info": zone_info,
                    "main_effects": main_effects,
                    "interactions": interactions,
                    "triples": triples,
                    "intercept": intercept,
                    "weights": weights,
                    "diagnostics": diagnostics,
                }

            # Identifiability: center this round's K raw outputs to sum to
            # zero per row -- a no-op for predictions (softmax is
            # shift-invariant), needed only so each class's own contribution
            # is uniquely defined -- see module docstring.
            fitted_residual -= fitted_residual.mean(axis=1, keepdims=True)

            current_scores = current_scores + self.learning_rate * fitted_residual
            rounds.append(round_tables)

            if has_val:
                val_fitted = np.zeros((len(y_val_idx), K))
                for k_idx in range(K):
                    t = round_tables[k_idx]
                    val_contributions = weak_learner_contributions(
                        X_val, t["zone_info"], t["main_effects"], t["interactions"], t["triples"]
                    )
                    val_fitted[:, k_idx] = t["intercept"] + val_contributions @ t["weights"]
                val_fitted -= val_fitted.mean(axis=1, keepdims=True)
                current_val_scores = current_val_scores + self.learning_rate * val_fitted
                p_val = _softmax(current_val_scores)
                val_logloss.append(_multinomial_log_loss(Y_val_onehot, p_val))

                if early_stopping and self.n_iter_no_change is not None:
                    best_so_far = min(val_logloss)
                    no_improve_streak = 0 if val_logloss[-1] <= best_so_far + 1e-12 else no_improve_streak + 1
                    if no_improve_streak >= self.n_iter_no_change:
                        break

        return rounds, baseline, val_logloss

    def _raw_predict_proba(self, X: pd.DataFrame, n_rounds: int) -> np.ndarray:
        K = self.n_classes_
        scores = np.tile(self.baseline_, (len(X), 1))
        for round_tables in self.rounds_[:n_rounds]:
            fitted_residual = np.zeros((len(X), K))
            for k_idx in range(K):
                t = round_tables[k_idx]
                contributions = weak_learner_contributions(
                    X, t["zone_info"], t["main_effects"], t["interactions"], t["triples"]
                )
                fitted_residual[:, k_idx] = t["intercept"] + contributions @ t["weights"]
            fitted_residual -= fitted_residual.mean(axis=1, keepdims=True)
            scores = scores + self.learning_rate * fitted_residual
        return _softmax(scores)

    def predict_proba(self, X: pd.DataFrame, n_rounds: int = None) -> np.ndarray:
        n_rounds = n_rounds if n_rounds is not None else self.best_n_rounds_
        raw_p = self._raw_predict_proba(X, n_rounds)
        if self.calibrators_ is None:
            return raw_p
        calibrated = np.column_stack(
            [self.calibrators_[k_idx].predict(raw_p[:, k_idx]) for k_idx in range(raw_p.shape[1])]
        )
        row_sums = calibrated.sum(axis=1, keepdims=True)
        return np.divide(
            calibrated, row_sums, out=np.full_like(calibrated, 1.0 / raw_p.shape[1]), where=row_sums > 0
        )


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

    Binary targets (2 distinct classes) fit a single log-odds booster --
    already a principled, single sigmoid with no heuristic involved. 3+
    classes use **native multinomial (softmax) boosting**: one joint
    booster maintains all K classes' logits together and optimizes the
    true softmax cross-entropy, rather than K independent one-vs-rest
    boosters normalized after the fact -- see the module docstring and
    :class:`_SoftmaxBooster` for the full mechanism.

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
        Fraction of training rows held out to pick the booster's best round
        count via early stopping. Set to 0 to disable and always use
        ``n_rounds``.
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
    adaptive_boundary_smoothing : bool, default=False
        If ``True``, each continuous column's soft zone lookup is scaled by
        a per-column, per-round mixing weight estimated honestly
        (cross-fitted), shrunk toward full smoothness absent strong
        out-of-fold evidence a hard lookup fits better. ``False`` (default)
        reproduces every prior release's behavior exactly -- opt-in, see
        :class:`~zoneboost.ZoneBoostRegressor` for the full description.
    boundary_shrinkage_m : float, default=10.0
        Shrinkage strength for ``adaptive_boundary_smoothing`` -- see
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
        to be non-decreasing/non-increasing across its zones, inherited by
        every interaction that column participates in. Opt-in (encodes
        domain knowledge, not a general improvement) -- see
        :class:`~zoneboost.ZoneBoostRegressor` for the full description.
    convexity_constraints : dict, default=None
        ``{column: +1 convex, -1 concave}`` -- forces a continuous column's
        main effect onto a convex/concave sequence. Main effects only,
        opt-in -- see :class:`~zoneboost.ZoneBoostRegressor` for the full
        description.
    bounded_effects : dict, default=None
        ``{column: (lower, upper)}`` -- clips a continuous column's main
        effect to this range, **per boosting round, not cumulatively**
        across all rounds. Main effects only, opt-in -- see
        :class:`~zoneboost.ZoneBoostRegressor` for the full description.
    forbidden_interactions : list, default=None
        A list of 2-column name/index pairs that must never be fit as
        pairwise (or 3-way) interactions. Opt-in -- see
        :class:`~zoneboost.ZoneBoostRegressor` for the full description.
    track_reliability : bool, default=False
        If ``True``, each round additionally records per-term support
        counts and cross-fold variability, consumed by ``explain(X,
        include_reliability=True)``. Opt-in -- see
        :class:`~zoneboost.ZoneBoostRegressor` for the full description.
    calibrate : bool, default=False
        If ``True``, recalibrate each booster's raw probability with an
        isotonic regression fit on a held-out split, so predicted
        probabilities better match empirical frequencies -- the standard
        isotonic-calibration recipe (``sklearn.calibration.
        CalibratedClassifierCV(method="isotonic")`` uses the same idea).
        Only changes :meth:`predict_proba` -- :meth:`explain`/
        :meth:`feature_importance` still decompose the raw log-odds score,
        unaffected. Requires ``validation_fraction > 0`` or
        ``calibration_fraction > 0``; raises ``ValueError`` at `fit`
        otherwise. This is the one parameter that differs from
        :class:`~zoneboost.ZoneBoostRegressor` -- every other parameter is
        identical across both estimators.
    calibration_fraction : float, default=0.0
        Fraction of training rows held out in a **third**, dedicated split
        purely for calibration -- distinct from ``validation_fraction``'s
        split, which drives early stopping. ``0.0`` (default) reproduces
        every prior release's behavior exactly: calibration reuses the
        validation split. When set, a value never seen by either the fit or
        validation split is used instead -- see
        :class:`~zoneboost.ZoneBoostRegressor` for the full description.
    refit_on_full_data : bool, default=False
        If ``True``, once each booster's own best round count is chosen from
        the validation split, the deployed model is refit on fit+validation
        data combined. Requires ``calibration_fraction > 0`` (raises
        ``ValueError`` otherwise) -- see
        :class:`~zoneboost.ZoneBoostRegressor` for the full description.
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
    convexity_constraints_ : dict
        Resolved ``{column: +1 or -1}`` convexity/concavity constraints
        actually in effect.
    bounded_effects_ : dict
        Resolved ``{column: (lower, upper)}`` bounds actually in effect.
    forbidden_interactions_ : set
        Resolved ``set`` of 2-element column-name ``frozenset``s actually
        excluded from interaction discovery.
    multiclass_ : bool
        Whether native multinomial boosting (3+ classes) was used.

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
        adaptive_boundary_smoothing: bool = False,
        boundary_shrinkage_m: float = 10.0,
        convexity_constraints=None,
        bounded_effects=None,
        forbidden_interactions=None,
        track_reliability: bool = False,
        calibrate: bool = False,
        calibration_fraction: float = 0.0,
        refit_on_full_data: bool = False,
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
        self.adaptive_boundary_smoothing = adaptive_boundary_smoothing
        self.boundary_shrinkage_m = boundary_shrinkage_m
        self.convexity_constraints = convexity_constraints
        self.bounded_effects = bounded_effects
        self.forbidden_interactions = forbidden_interactions
        self.track_reliability = track_reliability
        self.calibrate = calibrate
        self.calibration_fraction = calibration_fraction
        self.refit_on_full_data = refit_on_full_data
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
            adaptive_boundary_smoothing=self.adaptive_boundary_smoothing,
            boundary_shrinkage_m=self.boundary_shrinkage_m,
            convexity_constraints=self.convexity_constraints_,
            bounded_effects=self.bounded_effects_,
            forbidden_interactions=self.forbidden_interactions_,
            track_reliability=self.track_reliability,
            calibrate=self.calibrate,
            refit_on_full_data=self.refit_on_full_data,
            random_state=self.random_state,
        )

    def _make_softmax_booster(self) -> _SoftmaxBooster:
        return _SoftmaxBooster(
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
            adaptive_boundary_smoothing=self.adaptive_boundary_smoothing,
            boundary_shrinkage_m=self.boundary_shrinkage_m,
            convexity_constraints=self.convexity_constraints_,
            bounded_effects=self.bounded_effects_,
            forbidden_interactions=self.forbidden_interactions_,
            track_reliability=self.track_reliability,
            calibrate=self.calibrate,
            refit_on_full_data=self.refit_on_full_data,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        """Fit the boosted zone-grid ensemble (a single log-odds booster if
        binary, one joint multinomial softmax booster if 3+ classes).

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
        self.convexity_constraints_ = resolve_monotonic_constraints(
            X, self.convexity_constraints, self.categorical_features_
        )
        self.bounded_effects_ = resolve_bounded_effects(X, self.bounded_effects, self.categorical_features_)
        self.forbidden_interactions_ = resolve_forbidden_interactions(X, self.forbidden_interactions)

        has_val = self.validation_fraction and self.validation_fraction > 0
        has_cal = self.calibration_fraction and self.calibration_fraction > 0
        if self.calibrate and not has_val and not has_cal:
            raise ValueError(
                "calibrate=True requires validation_fraction > 0 or calibration_fraction > 0 "
                "(no held-out data to calibrate against)."
            )
        if self.refit_on_full_data and has_val and not has_cal:
            raise ValueError(
                "refit_on_full_data=True folds the validation split into the final "
                "model's own training data, so calibration_fraction > 0 is required "
                "for a genuinely held-out calibration set (otherwise calibrate=True "
                "would calibrate against rows the model was just trained on)."
            )

        rng = np.random.default_rng(self.random_state)
        n_total = len(X)
        if has_val or has_cal:
            perm = rng.permutation(n_total)
            n_cal = int(n_total * self.calibration_fraction) if has_cal else 0
            n_val = max(1, int(n_total * self.validation_fraction)) if has_val else 0
            if n_cal + n_val >= n_total:
                raise ValueError("validation_fraction + calibration_fraction leave no rows for the fit split.")
            cal_idx, val_idx, fit_idx = perm[:n_cal], perm[n_cal : n_cal + n_val], perm[n_cal + n_val :]
            X_fit = X.iloc[fit_idx].reset_index(drop=True)
            y_fit_raw = y_arr[fit_idx]
            X_val = X.iloc[val_idx].reset_index(drop=True) if has_val else None
            y_val_raw = y_arr[val_idx] if has_val else None
            X_cal = X.iloc[cal_idx].reset_index(drop=True) if has_cal else None
            y_cal_raw = y_arr[cal_idx] if has_cal else None
        else:
            X_fit, y_fit_raw = X, y_arr
            X_val = y_val_raw = X_cal = y_cal_raw = None

        if not self.multiclass_:
            y_fit_bin = (y_fit_raw == self.classes_[-1]).astype(float)
            y_val_bin = (y_val_raw == self.classes_[-1]).astype(float) if has_val else None
            y_cal_bin = (y_cal_raw == self.classes_[-1]).astype(float) if has_cal else None
            self.booster_ = self._make_booster().fit(
                X_fit, y_fit_bin, X_val=X_val, y_val=y_val_bin, X_cal=X_cal, y_cal=y_cal_bin
            )
        else:
            # Native multinomial (softmax) boosting -- one joint booster
            # working in integer class indices (0..K-1), not one-vs-rest.
            # See the module docstring.
            y_fit_idx = np.searchsorted(self.classes_, y_fit_raw)
            y_val_idx = np.searchsorted(self.classes_, y_val_raw) if has_val else None
            y_cal_idx = np.searchsorted(self.classes_, y_cal_raw) if has_cal else None
            self.softmax_booster_ = self._make_softmax_booster().fit(
                X_fit, y_fit_idx, n_classes=len(self.classes_),
                X_val=X_val, y_val_idx=y_val_idx, X_cal=X_cal, y_cal_idx=y_cal_idx,
            )
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

        return self.softmax_booster_.predict_proba(X, n_rounds)

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

    def _softmax_centering_column(self, X: pd.DataFrame, rounds: list) -> np.ndarray:
        """The single column that reconciles ``_SoftmaxBooster``'s per-round
        sum-to-zero identifiability centering back into an exact per-class
        attribution -- see the module docstring. Replays each round's K raw
        per-class totals (``intercept + contributions @ weights``, via
        :func:`weak_learner_contributions` -- not ``explain_rounds``'s own
        per-term blend logic, since only the row total is needed here) to
        reconstruct the exact row-wise mean subtracted during training,
        identical for every class."""
        n = len(X)
        K = self.softmax_booster_.n_classes_
        cumulative = np.zeros(n)
        for round_tables in rounds:
            round_raw = np.zeros((n, K))
            for k_idx in range(K):
                t = round_tables[k_idx]
                contributions = weak_learner_contributions(
                    X, t["zone_info"], t["main_effects"], t["interactions"], t["triples"]
                )
                round_raw[:, k_idx] = t["intercept"] + contributions @ t["weights"]
            cumulative += self.learning_rate * round_raw.mean(axis=1)
        return -cumulative

    def explain(self, X, n_rounds: int = None, include_reliability: bool = False):
        """Exact per-row, per-term attribution of each booster's log-odds
        score -- not a SHAP/LIME-style approximation, but an algebraic
        decomposition of the same computation `predict_proba` performs
        (see :mod:`zoneboost._explain`). Row sums equal the **log-odds**
        score exactly, not the probability directly -- the same convention
        SHAP itself uses for logistic/margin-based models, since
        probability contributions don't add linearly through a sigmoid.

        For 3+ classes (native multinomial boosting, see the module
        docstring): each class's DataFrame includes one extra column,
        ``"_softmax_centering"`` -- the cumulative sum-to-zero
        identifiability adjustment ``_SoftmaxBooster`` applies every round
        (mathematically a no-op for ``predict_proba``, needed only so each
        class's own contribution is uniquely defined). With it included,
        ``softmax(explain(X)[classes_[0]].sum(axis=1), ..., explain(X)[classes_[K-1]].sum(axis=1))``
        reproduces ``predict_proba(X)`` exactly when ``calibrate=False``
        (the default); with ``calibrate=True``, ``predict_proba``
        additionally applies each class's fitted isotonic calibrator and
        renormalizes. ``explain`` always reflects the raw, uncalibrated
        scores either way.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        n_rounds : int, default=None
        include_reliability : bool, default=False
            If ``True``, also returns a reliability report (see
            :func:`zoneboost._reliability.explain_reliability`). Requires
            ``track_reliability=True`` at `fit` time (raises `ValueError`
            otherwise). Multiclass: one reliability dict per class (same
            nesting as the contributions themselves), since each class's
            own softmax booster fits its own zones per round.

        Returns
        -------
        DataFrame, or dict of {class_label: DataFrame} if 3+ classes
            One column per term plus ``"baseline"`` (and, for 3+ classes,
            ``"_softmax_centering"``). Binary: a single DataFrame for the
            positive class's (``classes_[-1]``) log-odds score. Multiclass:
            one DataFrame per class, from the shared joint softmax booster.
            If ``include_reliability=True``, instead returns a tuple
            ``(contributions, reliability)``.
        """
        check_is_fitted(self, "classes_")
        X = self._ensure_dataframe(X)
        if not self.multiclass_:
            contributions = self._explain_one(self.booster_, X, n_rounds)
            if not include_reliability:
                return contributions
            if not self.track_reliability:
                raise ValueError(
                    "include_reliability=True requires track_reliability=True at fit time "
                    "(support/shrinkage_fraction/cross_fold_std are only computed then)."
                )
            nr = n_rounds if n_rounds is not None else self.booster_.best_n_rounds_
            reliability = explain_reliability(X, self.booster_.rounds_[:nr], self.shrinkage_m)
            return contributions, reliability

        booster = self.softmax_booster_
        n_rounds = n_rounds if n_rounds is not None else booster.best_n_rounds_
        rounds = booster.rounds_[:n_rounds]
        centering = self._softmax_centering_column(X, rounds)

        result = {}
        for k_idx, k in enumerate(self.classes_):
            per_class_rounds = [round_tables[k_idx] for round_tables in rounds]
            df = explain_rounds(X, per_class_rounds, float(booster.baseline_[k_idx]), self.learning_rate)
            df["_softmax_centering"] = centering
            result[k] = df
        if not include_reliability:
            return result

        if not self.track_reliability:
            raise ValueError(
                "include_reliability=True requires track_reliability=True at fit time "
                "(support/shrinkage_fraction/cross_fold_std are only computed then)."
            )
        reliability = {}
        for k_idx, k in enumerate(self.classes_):
            per_class_rounds = [round_tables[k_idx] for round_tables in rounds]
            reliability[k] = explain_reliability(X, per_class_rounds, self.shrinkage_m)
        return result, reliability

    def feature_importance(self, X, n_rounds: int = None) -> pd.Series:
        """Global importance: each term's mean absolute log-odds
        contribution over the rows in `X`, derived directly from
        :meth:`explain`. For multiclass, averaged across the per-class
        contributions (excluding the bookkeeping ``"_softmax_centering"``
        column, which isn't a predictor-derived term).

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
        per_class = [
            df.drop(columns=["baseline", "_softmax_centering"]).abs().mean() for df in explanation.values()
        ]
        return pd.concat(per_class, axis=1).mean(axis=1).sort_values(ascending=False)

    def evidence_report(self, X, n_rounds: int = None, sparse_threshold: float = None):
        """Per-prediction "evidence quality" summary: combines every term's
        own reliability (:meth:`explain`'s ``include_reliability=True``)
        into a single per-row signal for whether *this specific
        prediction* should be trusted, rather than reporting each term's
        own reliability separately.

        Requires ``track_reliability=True`` at `fit` time (raises
        ``ValueError`` otherwise, same precondition as ``explain(X,
        include_reliability=True)``).

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        n_rounds : int, default=None
        sparse_threshold : float, default=None
            Average-``support`` cutoff below which a term's contribution
            counts as coming from a "sparse" cell. Defaults to
            ``shrinkage_m`` itself -- the empirical-Bayes half-trust
            point, where a zone's ``shrinkage_fraction`` is exactly `0.5`.

        Returns
        -------
        DataFrame, or dict of {class_label: DataFrame} if 3+ classes
            Columns ``extrapolating`` (bool), ``unobserved_cell`` (bool),
            ``pct_contribution_from_sparse_cells`` (float, 0-1),
            ``evidence_score`` (float, 0-1, an honestly disclosed
            heuristic combination -- not a calibrated statistical score),
            and ``evidence_quality`` (``"Low"``/``"Medium"``/``"High"``).
            Multiclass: nested per class, since each class has its own
            softmax booster and its own zones per round -- unlike
            ``feature_importance``, which averages across classes.
            See :func:`zoneboost._reliability.evidence_report`.
        """
        if not self.track_reliability:
            raise ValueError(
                "evidence_report requires track_reliability=True at fit time "
                "(support/shrinkage_fraction/cross_fold_std are only computed then)."
            )
        contrib, reliability = self.explain(X, n_rounds, include_reliability=True)
        if not self.multiclass_:
            return evidence_report(contrib, reliability, self.shrinkage_m, sparse_threshold)
        return {
            k: evidence_report(contrib[k], reliability[k], self.shrinkage_m, sparse_threshold) for k in contrib
        }
