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

from ._common import (
    ensure_dataframe,
    resolve_bounded_effects,
    resolve_categorical_features,
    resolve_forbidden_interactions,
    resolve_group_col,
    resolve_monotonic_constraints,
)
from ._drift import _observed_range as _drift_observed_range
from ._explain import explain_rounds
from ._reliability import evidence_report, explain_reliability
from ._weak_learner import _column_zone_index, _fit_lasso_weights, weak_learner_contributions, weak_learner_fit

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
    max_pair_interactions : int, default=None
        Cap on how many pairwise interactions a single round keeps, ranked
        by mean absolute contribution. Applied after 3-way interaction
        selection, so it never changes which triples are found. ``None``
        (default) keeps every pair -- identical to every prior release; only
        relevant once the number of candidate pairs (``C(p, 2)`` for ``p``
        columns) is large enough that cross-fitting and Lasso-stacking every
        one of them becomes the per-round bottleneck.
    adaptive_boundary_smoothing : bool, default=False
        If ``True``, each continuous column's soft zone lookup is scaled by
        a per-column, per-round mixing weight estimated honestly
        (cross-fitted), shrunk toward full smoothness absent strong
        out-of-fold evidence a hard, single-zone lookup fits better -- so a
        column with a genuine sharp threshold can represent it instead of
        the boundary always being blurred by interpolation. ``False``
        (default) reproduces every prior release's behavior exactly (always
        fully smooth) -- a real approximation/judgment tradeoff, not a free
        correctness fix, so this is opt-in.
    boundary_shrinkage_m : float, default=10.0
        Shrinkage strength for ``adaptive_boundary_smoothing`` -- a
        column's own boundary needs about this many held-out rows near it
        before its cross-fitted hard-vs-smooth evidence is trusted as much
        as the full-smoothness prior. Only used when
        ``adaptive_boundary_smoothing=True``.
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
        count. **Inherited by interactions**: every pairwise/triple term
        that column participates in is also projected along that column's
        own axis (holding the other axis/axes fixed), so the column's
        *total* dependence on the target can't come out non-monotonic
        overall just because an interaction term wasn't constrained --
        automatic whenever this is declared, no separate opt-in. Accepts
        column names (if ``X`` is a DataFrame) or integer positions, the
        same convention as ``categorical_features``. A constraint declared
        on a categorical column is silently dropped -- there's no
        meaningful order to constrain for a nominal category. This is
        **opt-in**: it encodes domain knowledge the model can't infer on
        its own (e.g. "take-up must not decrease as affordability rises"),
        not a general improvement, so the default (no constraints)
        reproduces the exact same predictions as if this parameter didn't
        exist.
    loss : str, default="squared_error"
        ``"squared_error"`` (default) targets the conditional mean, exactly
        as every prior release -- zero change to today's behavior or cost.
        ``"quantile"`` targets a single conditional quantile of ``y``
        instead (see ``quantile`` below) -- every zone's fitted value
        becomes a shrunk *quantile* of the residual rather than a shrunk
        mean, and each round's term-combination step switches from an
        ordinary Lasso to ``sklearn.linear_model.QuantileRegressor``
        (pinball loss + L1 penalty) so the combination stays consistent
        with the same loss every term's own value was fit against -- not
        optional, since combining quantile-shrunk terms via a squared-error
        Lasso would silently re-center every round's output back toward the
        mean/median. The raw residual still drives zone-split search,
        cross-fitting, and pair screening's cheap proxy identically either
        way (a disclosed approximation: those stay squared-error-flavored
        regardless of loss). ``QuantileRegressor``'s linear-programming
        solver is substantially more expensive per round than ``Lasso`` --
        measured roughly 30x slower end-to-end in one benchmark -- a real,
        disclosed cost of ``loss="quantile"``, not a free option. Raises
        ``ValueError`` at `fit` if not one of these two strings.
    quantile : float, default=0.5
        The target quantile level when ``loss="quantile"`` (ignored
        otherwise). Must be in ``(0, 1)``; raises ``ValueError`` at `fit`
        otherwise. Fit several instances at different levels (e.g. ``0.05``,
        ``0.5``, ``0.95``) to get a full conditional distribution, or see
        :class:`zoneboost.ConformalizedQuantileRegressor` for a
        distribution-free, locally-adaptive prediction interval built from
        exactly two such quantile fits.
    convexity_constraints : dict, default=None
        ``{column: +1 convex, -1 concave}`` -- forces a continuous column's
        *main effect* onto a convex/concave sequence across its zones:
        isotonic-regresses the sequence's own first differences (a convex
        sequence has non-decreasing first differences) rather than the
        values themselves, weighted by each gap's neighboring row count,
        then reconstructs and re-centers to the original level. Same
        declaration convention as ``monotonic_constraints`` (categorical
        columns dropped); main effects only -- not inherited by
        interactions. Combining this with ``monotonic_constraints`` on the
        same column is a heuristic ordering (monotonic projection happens
        first), not guaranteed to keep the result strictly monotonic
        afterward. **Opt-in**: the default (no constraints) reproduces the
        exact same predictions as if this parameter didn't exist.
    bounded_effects : dict, default=None
        ``{column: (lower, upper)}`` -- clips a continuous column's *main
        effect* deviation to this range, applied after any monotonic/
        convexity projection. Main effects only. **Bounds each boosting
        round's own contribution, not the cumulative multi-round total**:
        with ``learning_rate`` shrinkage and many rounds, the column's
        summed contribution across all of ``rounds_`` can still exceed
        ``(lower, upper)`` even though no single round's own stored value
        ever does -- a real regularization (no single round's zone-fitting
        can produce an extreme outlier value for that term), not a
        business-rule guarantee on the final prediction's total range.
        **Opt-in**: the default (no bounds) reproduces the exact same
        predictions as if this parameter didn't exist.
    forbidden_interactions : list, default=None
        A list of 2-column name/index pairs (same convention as
        ``categorical_features``) that must never be fit as pairwise
        interactions -- applies to both the exhaustive and screened
        (``max_pair_interactions``) discovery paths. Any 3-way candidate
        (when ``max_interaction_order=3``) whose three constituent pairs
        include a forbidden one is skipped too. Raises ``ValueError`` at
        `fit` if an entry doesn't name exactly 2 distinct columns.
        **Opt-in**: the default (``None``) reproduces the exact same
        predictions as if this parameter didn't exist.
    track_reliability : bool, default=False
        If ``True``, each round additionally records, per term, its zone/
        cell row counts and the standard deviation of its shrunk value
        across the ``cross_fit_folds`` cross-fitting folds -- consumed by
        ``explain(X, include_reliability=True)`` (see below) to report how
        much to trust each contribution, not just its value. Real
        per-round memory/compute cost, so opt-in; ``False`` (default) is
        bit-identical to every prior release.
    group_col : str or int, default=None
        Designates a column (name or index, same convention as
        ``categorical_features``) as a hierarchical grouping key for
        partial pooling across grouped data (patients within hospitals,
        customers within regions): every ``(feature, group_col)`` pair is
        guaranteed to be fit as a pairwise interaction every round -- the
        column is never dropped by ``col_subsample``, and the pair is
        never dropped by ``max_pair_interactions`` screening (an explicit
        ``forbidden_interactions`` entry still wins). The joint (zone,
        group) cell then already shrinks toward "what the zone alone
        predicts + what the group alone predicts" via
        :func:`_pair_shrunk_deviation`'s existing hierarchical-prior
        shrinkage -- local (zone x group) <- regional (zone's own
        marginal deviation) <- global (overall mean) partial pooling,
        reusing the identical empirical-Bayes machinery every other term
        uses, not new math. Read the decomposition directly off
        ``explain(X)``: the ``feature`` column is the regional/global
        pooled effect, ``"feature x group_col"`` is the local deviation a
        specific group adds on top of it.

        **Scope**: a single grouping column (nested/multi-level grouping,
        e.g. hospital within region, is not supported); pairs, not
        triples (a forced pair still only participates in
        ``max_interaction_order=3``'s triple-candidate seeding on the
        same footing as any other pair, never force-included in a
        triple); reuses ``shrinkage_m`` rather than a dedicated
        group-level shrinkage constant. All deferred, disclosed.
        **Opt-in**: the default (``None``) reproduces the exact same
        predictions as if this parameter didn't exist.
    calibration_fraction : float, default=0.0
        Fraction of training rows held out in a **third**, dedicated split
        purely for calibration (:attr:`conformal_scores_`) -- distinct from
        ``validation_fraction``'s split, which drives early stopping.
        ``0.0`` (default) reproduces every prior release's behavior exactly:
        calibration reuses the validation split (a disclosed tradeoff, see
        "Prediction intervals" in the docs). When set, a value never seen by
        either the fit or validation split is used instead, so
        ``best_n_rounds_``'s own selection process can no longer bias the
        calibration margin.
    refit_on_full_data : bool, default=False
        If ``True``, once ``best_n_rounds_`` is chosen from the validation
        split, the *deployed* model (:attr:`rounds_`/:attr:`baseline_`) is
        refit on fit+validation data combined, running exactly
        ``best_n_rounds_`` rounds -- recovering validation data that would
        otherwise be permanently withheld from the model that actually
        predicts. Requires ``calibration_fraction > 0``: folding the
        validation split into training means it can no longer double as a
        calibration set, so a genuinely separate one is required (raises
        ``ValueError`` at `fit` otherwise). :attr:`train_rmse_`/
        :attr:`val_rmse_` still reflect the *original* selection-phase
        curves, not the refit pass, since the refit's own training dynamics
        on a different dataset aren't a meaningful continuation of them.
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
    convexity_constraints_ : dict
        Resolved ``{column: +1 or -1}`` convexity/concavity constraints
        actually in effect (same resolution as ``monotonic_constraints_``).
    bounded_effects_ : dict
        Resolved ``{column: (lower, upper)}`` bounds actually in effect
        (declared bounds on a categorical column are dropped here).
    forbidden_interactions_ : set
        Resolved ``set`` of 2-element column-name ``frozenset``s actually
        excluded from pairwise/triple interaction discovery.
    group_col_ : str or None
        Resolved column name for ``group_col`` (``None`` if not set).
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
        Held-out score after each round from the *selection* phase (empty
        if ``validation_fraction=0``) -- always reflects how
        ``best_n_rounds_`` was chosen, even when ``refit_on_full_data=True``
        retrains :attr:`rounds_` on more data afterward. Despite the name,
        this tracks whatever loss is actually being minimized: RMSE for
        ``loss="squared_error"`` (the default), mean pinball loss at
        ``quantile`` for ``loss="quantile"``.
    conformal_scores_ : ndarray or None
        Sorted absolute residuals on the calibration split, at
        ``best_n_rounds_`` -- the nonconformity scores :meth:`predict_interval`
        draws its margin from. Computed from the dedicated
        ``calibration_fraction`` split if set, otherwise the validation
        split (see ``calibration_fraction`` above). ``None`` if neither is
        available.
    effect_overrides_ : list
        Every edit made via :meth:`edit_effect`, in application order
        (``[]`` until the first edit) -- ``{"feature": col, "lo": float,
        "hi": float, "contribution": float}`` per entry. `predict`/
        `explain` apply these automatically; :meth:`reset_overrides`
        clears them.

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
        max_pair_interactions=None,
        adaptive_boundary_smoothing: bool = False,
        boundary_shrinkage_m: float = 10.0,
        loss: str = "squared_error",
        quantile: float = 0.5,
        convexity_constraints=None,
        bounded_effects=None,
        forbidden_interactions=None,
        track_reliability: bool = False,
        group_col=None,
        calibration_fraction: float = 0.0,
        refit_on_full_data: bool = False,
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
        self.max_pair_interactions = max_pair_interactions
        self.adaptive_boundary_smoothing = adaptive_boundary_smoothing
        self.boundary_shrinkage_m = boundary_shrinkage_m
        self.loss = loss
        self.quantile = quantile
        self.convexity_constraints = convexity_constraints
        self.bounded_effects = bounded_effects
        self.forbidden_interactions = forbidden_interactions
        self.track_reliability = track_reliability
        self.group_col = group_col
        self.calibration_fraction = calibration_fraction
        self.refit_on_full_data = refit_on_full_data
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
        if self.loss not in ("squared_error", "quantile"):
            raise ValueError(f"loss must be 'squared_error' or 'quantile', got {self.loss!r}")
        if self.loss == "quantile" and not 0 < self.quantile < 1:
            raise ValueError(f"quantile must be in (0, 1), got {self.quantile!r}")

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
        self.group_col_ = resolve_group_col(X, self.group_col)

        has_val = self.validation_fraction and self.validation_fraction > 0
        has_cal = self.calibration_fraction and self.calibration_fraction > 0
        if self.refit_on_full_data and has_val and not has_cal:
            raise ValueError(
                "refit_on_full_data=True folds the validation split into the final "
                "model's own training data, so calibration_fraction > 0 is required "
                "for a genuinely held-out calibration set (otherwise conformal_scores_ "
                "would be computed from rows the model was just trained on)."
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
            y_fit = y_arr[fit_idx]
            X_val = X.iloc[val_idx].reset_index(drop=True) if has_val else None
            y_val = y_arr[val_idx] if has_val else None
            X_cal = X.iloc[cal_idx].reset_index(drop=True) if has_cal else None
            y_cal = y_arr[cal_idx] if has_cal else None
        else:
            X_fit, y_fit = X, y_arr
            X_val = y_val = X_cal = y_cal = None

        # Phase 1 (selection): fit on X_fit, track val_rmse_ on X_val (if
        # any) with early stopping, to decide best_n_rounds_.
        rounds, baseline, train_rmse, val_rmse = self._boost_rounds(
            X_fit, y_fit, rng, self.n_rounds, X_val, y_val, early_stopping=True
        )
        self.train_rmse_ = train_rmse
        self.val_rmse_ = val_rmse
        self.best_n_rounds_ = int(np.argmin(val_rmse)) + 1 if has_val and val_rmse else len(rounds)

        if self.refit_on_full_data and has_val:
            # Phase 2 (refit): the deployed model is retrained on fit+
            # validation combined, for exactly best_n_rounds_ rounds --
            # already decided, so no early stopping this time. Fresh rng so
            # reproducibility doesn't depend on how much of phase 1's random
            # stream got consumed. train_rmse_/val_rmse_ above stay as the
            # phase-1 selection diagnostics; they aren't recomputed here.
            X_refit = pd.concat([X_fit, X_val], ignore_index=True)
            y_refit = np.concatenate([y_fit, y_val])
            refit_rng = np.random.default_rng(self.random_state)
            rounds, baseline, _, _ = self._boost_rounds(X_refit, y_refit, refit_rng, self.best_n_rounds_)

        self.rounds_ = rounds
        self.baseline_ = baseline

        # Split-conformal calibration: nonconformity scores from a genuinely
        # held-out split -- the dedicated calibration split if
        # calibration_fraction > 0, else the same validation split used for
        # early stopping (the default, disclosed tradeoff -- see
        # `predict_interval`). Never computed from rows the final model
        # trained on: the ValueError above guarantees calibration_fraction >
        # 0 whenever refit_on_full_data folds the validation split into
        # training.
        cal_X, cal_y = (X_cal, y_cal) if has_cal else (X_val, y_val)
        if cal_X is not None:
            cal_pred = self._raw_predict(cal_X, self.best_n_rounds_)
            self.conformal_scores_ = np.sort(np.abs(cal_y - cal_pred))
        else:
            self.conformal_scores_ = None

        self.effect_overrides_ = []

        return self

    def _baseline_stat(self, y_train: np.ndarray) -> float:
        """The starting prediction before any boosting round -- the best
        constant predictor for whichever loss is active: the mean for
        ``loss="squared_error"``, the target quantile for
        ``loss="quantile"``."""
        if self.loss == "quantile":
            return float(np.quantile(y_train, self.quantile))
        return float(y_train.mean())

    def _score(self, y_true: np.ndarray, pred: np.ndarray) -> float:
        """The loss actually being minimized, evaluated on ``pred``: RMSE
        for ``loss="squared_error"``, mean pinball loss at ``self.quantile``
        for ``loss="quantile"`` -- used identically for
        ``train_rmse_``/``val_rmse_``/early stopping/``best_n_rounds_``
        selection regardless of which loss is active."""
        if self.loss == "quantile":
            diff = y_true - pred
            return float(np.mean(np.maximum(self.quantile * diff, (self.quantile - 1) * diff)))
        return float(np.sqrt(np.mean((y_true - pred) ** 2)))

    def _boost_rounds(self, X_train, y_train, rng, n_rounds, X_val=None, y_val=None, early_stopping=False):
        """Core boosting loop -- extracted so `fit` can run it twice: once
        for round-count selection (on the fit split, tracking ``val_rmse_``
        with early stopping), and optionally again for a final refit
        (fit+validation combined, exact round count, no early stopping) --
        see `fit` and ``refit_on_full_data``.

        Returns
        -------
        rounds, baseline, train_rmse, val_rmse
        """
        baseline = self._baseline_stat(y_train)
        n = len(y_train)
        n_row_sample = min(n, max(min(20, n), int(n * self.row_subsample)))
        n_predictors = len(self.predictor_names_)
        n_col_sample = min(n_predictors, max(min(2, n_predictors), int(n_predictors * self.col_subsample)))

        current_pred = np.full(n, baseline)
        has_val = X_val is not None
        if has_val:
            current_val_pred = np.full(len(y_val), baseline)

        rounds, train_rmse, val_rmse = [], [], []
        no_improve_streak = 0

        for _ in range(n_rounds):
            residual = y_train - current_pred

            row_idx = rng.choice(n, size=n_row_sample, replace=False)
            if self.group_col_ is not None:
                # Guarantees group_col is never subsampled away, so every
                # feature that IS sampled this round reliably gets a chance
                # to pair with it -- see the group_col docstring above.
                # n_col_sample <= n_predictors always (line above), so
                # n_col_sample - 1 <= len(other_cols) always holds.
                other_cols = [c for c in self.predictor_names_ if c != self.group_col_]
                sampled = rng.choice(other_cols, size=n_col_sample - 1, replace=False)
                col_subset = list(sampled) + [self.group_col_]
            else:
                col_subset = list(rng.choice(self.predictor_names_, size=n_col_sample, replace=False))
            X_sub = X_train.iloc[row_idx][col_subset]
            residual_sub = residual[row_idx]

            zone_info, main_effects, interactions, triples, oof_contributions, diagnostics = weak_learner_fit(
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
                max_pair_interactions=self.max_pair_interactions,
                adaptive_boundary_smoothing=self.adaptive_boundary_smoothing,
                boundary_shrinkage_m=self.boundary_shrinkage_m,
                quantile=self.quantile if self.loss == "quantile" else None,
                convexity_constraints=self.convexity_constraints_,
                bounded_effects=self.bounded_effects_,
                forbidden_interactions=self.forbidden_interactions_,
                track_reliability=self.track_reliability,
                group_col=self.group_col_,
            )
            contributions = weak_learner_contributions(X_train, zone_info, main_effects, interactions, triples)
            # The round's own (sub)sampled rows would otherwise be scored by a
            # table partly built from their own residual -- replace exactly
            # those rows with their honest, cross-fitted contributions. Rows
            # this round didn't sample were never part of the table, so
            # they're already leak-free and left untouched.
            contributions[row_idx, :] = oof_contributions
            # Lasso, not a shared equal-weight average: an irrelevant term's
            # weight gets zeroed by the L1 penalty, a strong term gets its
            # own learned weight instead of a diluted 1/n_terms share.
            intercept, weights = _fit_lasso_weights(
                contributions, residual, self.stacking_alpha,
                quantile=self.quantile if self.loss == "quantile" else None,
            )
            fitted_residual = intercept + contributions @ weights

            current_pred = current_pred + self.learning_rate * fitted_residual
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
            train_rmse.append(self._score(y_train, current_pred))

            if has_val:
                val_contributions = weak_learner_contributions(X_val, zone_info, main_effects, interactions, triples)
                val_fitted = intercept + val_contributions @ weights
                current_val_pred = current_val_pred + self.learning_rate * val_fitted
                val_rmse.append(self._score(y_val, current_val_pred))

                if early_stopping and self.n_iter_no_change is not None:
                    best_so_far = min(val_rmse)
                    if val_rmse[-1] <= best_so_far + 1e-12:
                        no_improve_streak = 0
                    else:
                        no_improve_streak += 1
                    if no_improve_streak >= self.n_iter_no_change:
                        break

        return rounds, baseline, train_rmse, val_rmse

    def _raw_predict(self, X: pd.DataFrame, n_rounds: int) -> np.ndarray:
        pred = np.full(len(X), self.baseline_)
        for round_ in self.rounds_[:n_rounds]:
            contributions = weak_learner_contributions(
                X, round_["zone_info"], round_["main_effects"], round_["interactions"], round_["triples"]
            )
            fitted_residual = round_["intercept"] + contributions @ round_["weights"]
            pred = pred + self.learning_rate * fitted_residual
        return pred

    def _apply_overrides(self, X: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
        """Replaces an overridden main effect's contribution with its
        override's constant for every row whose raw value falls inside
        that override's ``[lo, hi]`` -- see :meth:`edit_effect`. Overrides
        are applied in insertion order, so the *last* override covering a
        given row wins (supports incremental/corrective edits). Guarantees
        ``explain(X)`` still sums exactly to ``predict(X)``: the same
        replacement is applied to the one term's own column, nothing else
        changes."""
        if not self.effect_overrides_:
            return contributions
        contributions = contributions.copy()
        touched_features = {ov["feature"] for ov in self.effect_overrides_}
        for feature in touched_features:
            raw = np.asarray(X[feature], dtype=float)
            values = contributions[feature].to_numpy(dtype=float).copy()
            for ov in self.effect_overrides_:
                if ov["feature"] != feature:
                    continue
                in_range = (raw >= ov["lo"]) & (raw <= ov["hi"])
                values[in_range] = ov["contribution"]
            contributions[feature] = values
        return contributions

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
        if self.effect_overrides_:
            return self.explain(X, n_rounds).sum(axis=1).to_numpy()
        return self._raw_predict(X, n_rounds)

    def predict_interval(self, X, alpha: float = 0.1) -> tuple:
        """Split-conformal prediction interval: a constant-width margin
        around ``predict(X)`` with a distribution-free marginal coverage
        guarantee -- ``P(y in interval) >= 1 - alpha``, assuming the held-out
        validation rows and future ``X`` rows are exchangeable (the standard
        split-conformal assumption; see Vovk et al. / Lei et al.).

        The margin is a *fixed* quantile of nonconformity scores (absolute
        residuals) measured on a genuinely held-out split -- never training
        rows, so the margin isn't optimistic about how well the model fits
        its own training data. By default (``calibration_fraction=0``) that
        split is the same ``validation_fraction`` split already used for
        early stopping -- a disclosed simplification, since the round count
        `predict` uses was itself chosen to minimize error on this exact
        set, which can understate the true margin slightly. Set
        ``calibration_fraction > 0`` at `fit` time for a dedicated,
        genuinely separate calibration split instead. Requires
        ``validation_fraction > 0`` or ``calibration_fraction > 0`` at `fit`
        time. Not available when ``loss="quantile"`` -- a constant-width
        margin around a single conditional quantile isn't a meaningful
        coverage interval the same way it is around a mean; use
        :class:`zoneboost.ConformalizedQuantileRegressor` instead for a
        locally-adaptive interval built from two quantile fits.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        alpha : float, default=0.1
            Miscoverage rate -- e.g. ``0.1`` targets 90% coverage.

        Returns
        -------
        lower, upper : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "rounds_")
        if self.loss == "quantile":
            raise ValueError(
                "predict_interval is not available when loss='quantile' -- a constant-width margin "
                "around a single conditional quantile isn't a meaningful coverage interval the same "
                "way it is around a mean. Use zoneboost.ConformalizedQuantileRegressor instead."
            )
        if self.conformal_scores_ is None:
            raise ValueError(
                "predict_interval requires validation_fraction > 0 or calibration_fraction > 0 "
                "at fit time (no held-out data to calibrate a conformal margin against)."
            )
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

        point_pred = self.predict(X)
        n = len(self.conformal_scores_)
        k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        margin = self.conformal_scores_[k - 1]
        return point_pred - margin, point_pred + margin

    def explain(self, X, n_rounds: int = None, include_reliability: bool = False):
        """Exact per-row, per-term prediction attribution -- not a SHAP/LIME
        -style approximation, but an algebraic decomposition of the same
        computation `predict` performs, so results sum exactly to the
        prediction (see :mod:`zoneboost._explain` for the derivation).

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        n_rounds : int, default=None
        include_reliability : bool, default=False
            If ``True``, also returns a reliability report: how much to
            trust each contribution, not just its value. Requires
            ``track_reliability=True`` at `fit` time for the ``support``/
            ``shrinkage_fraction``/``cross_fold_std`` columns (raises
            ``ValueError`` otherwise) -- see :func:`zoneboost._reliability.
            explain_reliability` for exactly what each column means.

        Returns
        -------
        DataFrame of shape (n_samples, n_terms + 1)
            One column per term that appeared in any round (a predictor's
            own name for its main effect, ``"A x B"`` for an interaction
            pair, ``"A x B x C"`` for a 3-way interaction) plus
            ``"baseline"``. Row sums equal ``predict(X)`` exactly, up to
            floating-point rounding. If ``include_reliability=True``,
            instead returns a tuple ``(contributions, reliability)`` where
            ``reliability`` is a ``{term_name: DataFrame}`` dict, one row
            per row of ``X``, columns ``support``, ``shrinkage_fraction``,
            ``cross_fold_std``, ``n_rounds_present``, and (continuous main
            effects only) ``boundary_weight``/``extrapolation_frac``.
        """
        check_is_fitted(self, "rounds_")
        n_rounds = n_rounds if n_rounds is not None else self.best_n_rounds_
        X = self._ensure_dataframe(X)
        contributions = explain_rounds(X, self.rounds_[:n_rounds], self.baseline_, self.learning_rate)
        contributions = self._apply_overrides(X, contributions)
        if not include_reliability:
            return contributions
        if not self.track_reliability:
            raise ValueError(
                "include_reliability=True requires track_reliability=True at fit time "
                "(support/shrinkage_fraction/cross_fold_std are only computed then)."
            )
        reliability = explain_reliability(X, self.rounds_[:n_rounds], self.shrinkage_m)
        return contributions, reliability

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

    def evidence_report(self, X, n_rounds: int = None, sparse_threshold: float = None) -> pd.DataFrame:
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
        DataFrame indexed like ``X``, columns ``extrapolating`` (bool),
        ``unobserved_cell`` (bool), ``pct_contribution_from_sparse_cells``
        (float, 0-1), ``evidence_score`` (float, 0-1, an honestly
        disclosed heuristic combination -- not a calibrated statistical
        score), and ``evidence_quality`` (``"Low"``/``"Medium"``/
        ``"High"``). See :func:`zoneboost._reliability.evidence_report`.
        """
        if not self.track_reliability:
            raise ValueError(
                "evidence_report requires track_reliability=True at fit time "
                "(support/shrinkage_fraction/cross_fold_std are only computed then)."
            )
        contrib, reliability = self.explain(X, n_rounds, include_reliability=True)
        return evidence_report(contrib, reliability, self.shrinkage_m, sparse_threshold)

    def _continuous_main_effect_columns(self) -> set:
        cols = set()
        for round_ in self.rounds_:
            for col in round_["main_effects"]:
                if round_["zone_info"][col][0] == "continuous":
                    cols.add(col)
        return cols

    def edit_effect(self, feature, value_range: tuple, contribution: float, X_eval=None, y_eval=None) -> dict:
        """Override this column's main-effect contribution to a fixed
        constant, for every row whose raw value falls in ``value_range``
        -- "governed" human editing: the edit takes effect immediately
        (``predict``/``explain`` reflect it from this call onward) and
        this method returns a report on its consequences, rather than
        editing silently.

        zoneboost's zones are re-derived fresh every boosting round from
        that round's own row/column subsample -- there is no single,
        stable "zone" per column to name across the whole model the way
        EBM's fixed bins allow. Editing is instead defined on the **raw
        feature value range**, which is always well-defined regardless of
        how any given round happened to bin it.

        **Scope**: main effects only (not interactions/triples), and this
        column must be continuous and have appeared as a main effect in at
        least one round. Multiple overrides on overlapping ranges: the
        *last* one applied wins for affected rows.

        Parameters
        ----------
        feature : str
            Must be a continuous column that appeared as a main effect in
            at least one round (raises ``ValueError`` otherwise).
        value_range : tuple of (lower, upper)
            Inclusive on both ends. Raises ``ValueError`` if
            ``lower > upper``.
        contribution : float
            The new, fixed contribution for every affected row.
        X_eval, y_eval : optional
            If given, the report includes affected-population and
            predictive-performance diagnostics measured on this data
            (never training data the model doesn't retain).

        Returns
        -------
        dict with keys:
            ``affected_rows``/``affected_fraction`` -- count/fraction of
            ``X_eval`` in ``value_range`` (``None`` if ``X_eval`` not
            given).
            ``original_contribution_mean``/``original_contribution_range``
            -- this term's own contribution stats for affected rows,
            *before* this edit (``None`` if ``X_eval`` not given).
            ``contribution_change`` -- ``contribution -
            original_contribution_mean`` (``None`` if ``X_eval`` not
            given).
            ``exceeds_uncertainty`` -- ``True`` if the change exceeds
            ``2 * mean(cross_fold_std)`` for affected rows -- a disclosed
            heuristic threshold, not a calibrated test. Requires
            ``track_reliability=True`` and ``X_eval`` (``None``
            otherwise); see :class:`zoneboost.BootstrapStability`'s own
            ``contribution_interval`` for a more rigorous, refit-based
            check to compare this edit against manually.
            ``constraint_violation`` -- ``True`` if ``bounded_effects_``
            declares a bound for this column and ``contribution`` falls
            outside it (``None`` if no bound is declared for this
            column). Monotonic-constraint violations are not checked --
            zoneboost's adaptive, per-round zones don't have a stable
            enough "neighboring region" concept to check cheaply and
            honestly.
            ``rmse_before``/``rmse_after`` -- only if ``y_eval`` given.
            ``prediction_mean_shift`` -- ``mean(predict_after) -
            mean(predict_before)`` on ``X_eval`` -- a simple
            distribution-shift proxy (``None`` if ``X_eval`` not given).

        Deferred: fairness impact (needs a protected-attribute and
        fairness-metric design of its own) and calibration change
        (classifier-specific -- ``edit_effect`` is regressor-only for
        now).
        """
        check_is_fitted(self, "rounds_")
        lower, upper = value_range
        if lower > upper:
            raise ValueError(f"value_range's lower bound must be <= upper bound, got {value_range!r}")
        if feature not in self._continuous_main_effect_columns():
            raise ValueError(
                f"{feature!r} must be a continuous column that appeared as a main effect in "
                "at least one round -- edit_effect only supports continuous main effects."
            )

        report = {
            "affected_rows": None,
            "affected_fraction": None,
            "original_contribution_mean": None,
            "original_contribution_range": None,
            "contribution_change": None,
            "exceeds_uncertainty": None,
            "constraint_violation": None,
            "rmse_before": None,
            "rmse_after": None,
            "prediction_mean_shift": None,
        }

        lower_bound = self.bounded_effects_.get(feature)
        if lower_bound is not None:
            b_lower, b_upper = lower_bound
            report["constraint_violation"] = bool(contribution < b_lower or contribution > b_upper)

        if X_eval is not None:
            X_eval = self._ensure_dataframe(X_eval)
            raw = np.asarray(X_eval[feature], dtype=float)
            mask = (raw >= lower) & (raw <= upper)
            affected_rows = int(mask.sum())
            report["affected_rows"] = affected_rows
            report["affected_fraction"] = affected_rows / len(X_eval)

            pred_before = self.predict(X_eval)
            if affected_rows > 0:
                contrib_before = self.explain(X_eval)[feature].to_numpy()[mask]
                original_mean = float(contrib_before.mean())
                report["original_contribution_mean"] = original_mean
                report["original_contribution_range"] = (float(contrib_before.min()), float(contrib_before.max()))
                report["contribution_change"] = contribution - original_mean

                if self.track_reliability:
                    _, reliability_before = self.explain(X_eval, include_reliability=True)
                    fold_std = reliability_before[feature]["cross_fold_std"].to_numpy()[mask]
                    if not np.all(np.isnan(fold_std)):
                        threshold = 2 * float(np.nanmean(fold_std))
                        report["exceeds_uncertainty"] = bool(abs(report["contribution_change"]) > threshold)

        self.effect_overrides_.append(
            {"feature": feature, "lo": float(lower), "hi": float(upper), "contribution": float(contribution)}
        )

        if X_eval is not None:
            pred_after = self.predict(X_eval)
            report["prediction_mean_shift"] = float(pred_after.mean() - pred_before.mean())
            if y_eval is not None:
                y_arr = np.asarray(y_eval, dtype=float).reshape(-1)
                report["rmse_before"] = float(np.sqrt(np.mean((y_arr - pred_before) ** 2)))
                report["rmse_after"] = float(np.sqrt(np.mean((y_arr - pred_after) ** 2)))

        return report

    def reset_overrides(self) -> None:
        """Clears every edit made via :meth:`edit_effect` -- a cheap,
        reversible safety net, since edits are meant to be governed, not
        permanent by accident. ``predict``/``explain`` immediately
        reflect the original, unedited model again."""
        check_is_fitted(self, "rounds_")
        self.effect_overrides_ = []

    def _observed_range(self, feature) -> tuple:
        """Min/max of ``feature``'s own zone centers across every round it
        appeared in as a continuous main effect -- what the model
        actually has information about, not the column's full theoretical
        range (the same range :func:`zoneboost._reliability.
        evidence_report`'s extrapolation check is built on). Thin wrapper
        around the standalone :func:`zoneboost._drift._observed_range`,
        shared with :func:`zoneboost._drift.compare_models`."""
        return _drift_observed_range(self, feature)

    def _zone_transition_frequency(self, feature, old_value: float, new_value: float) -> float:
        """Of the rounds that included ``feature`` as a continuous main
        effect, the fraction where its (hard) zone index actually differs
        between ``old_value`` and ``new_value`` -- the honest "zone
        transitions" statistic :meth:`counterfactual` reports, given
        zoneboost's zones are re-derived fresh every round (no single
        stable zone index to compare across the whole model)."""
        n_with_feature = 0
        n_transitions = 0
        old_series = pd.Series([old_value])
        new_series = pd.Series([new_value])
        for round_ in self.rounds_:
            if feature not in round_["main_effects"]:
                continue
            zone_info_col = round_["zone_info"][feature]
            if zone_info_col[0] != "continuous":
                continue
            n_with_feature += 1
            old_zone = _column_zone_index(old_series, zone_info_col)[0]
            new_zone = _column_zone_index(new_series, zone_info_col)[0]
            if old_zone != new_zone:
                n_transitions += 1
        return n_transitions / n_with_feature if n_with_feature > 0 else 0.0

    def counterfactual(
        self, X, target: float, actionable: list, immutable: list = None, tol: float = None,
        n_candidates: int = 200,
    ) -> dict:
        """Zone-native counterfactual search: "what's the smallest change
        to these actionable features that gets the prediction to
        `target`?" -- computed by directly evaluating the model's own
        *exact*, known `predict()` function over a dense grid spanning
        each actionable feature's own observed range, not by training a
        surrogate or searching an opaque black box the way generic
        counterfactual-explanation tools must.

        Single-feature solutions are always preferred when one alone can
        reach `target` within `tol` (the fewest "zone transitions"); a
        greedy, coordinate-descent-style multi-feature search is the
        fallback otherwise -- a disclosed heuristic, not a guaranteed
        joint-optimal minimal change when actionable features interact.

        **Scope**: a single row; actionable features must be continuous
        columns that appeared as a main effect in at least one round
        (raises ``ValueError`` otherwise).

        Parameters
        ----------
        X : DataFrame or array-like of shape (1, n_features)
        target : float
            The desired prediction value.
        actionable : list of str
            Columns allowed to change.
        immutable : list of str, default=None
            Purely informational -- everything not in ``actionable`` is
            already implicitly immutable. Raises ``ValueError`` if it
            overlaps ``actionable``.
        tol : float, default=None
            How close to `target` counts as "close enough". Defaults to
            ``max(1e-6, 0.05 * abs(target - original_prediction))`` --
            self-scaling to the size of the gap being closed.
        n_candidates : int, default=200
            Grid resolution per actionable feature.

        Returns
        -------
        dict with keys:
            ``feasible`` (bool), ``original_prediction``,
            ``counterfactual_prediction``, ``prediction_change``.
            ``changes`` -- ``{feature: (original_value, new_value)}``,
            only features that actually changed.
            ``zone_transition_frequency`` -- ``{feature: fraction}``, only
            for changed features.
            ``interaction_consequences`` -- ``{term: contribution_change}``
            from `explain(X)` vs. `explain(X_cf)`, for every term whose
            contribution changed by more than a tiny epsilon -- the
            *exact* decomposition `explain` already provides, separating a
            changed feature's own main effect from any interaction it
            participates in.
            ``extrapolating`` -- whether the *original* row already had a
            changed feature sitting outside its own observed range.
            ``evidence`` -- :meth:`evidence_report`'s output for the
            counterfactual row if ``track_reliability=True``, else
            ``None`` -- "confidence in the counterfactual."
        """
        check_is_fitted(self, "rounds_")
        X = self._ensure_dataframe(X)
        if len(X) != 1:
            raise ValueError(f"counterfactual expects a single row, got {len(X)}")

        continuous_cols = self._continuous_main_effect_columns()
        for feat in actionable:
            if feat not in continuous_cols:
                raise ValueError(
                    f"{feat!r} must be a continuous column that appeared as a main effect in "
                    "at least one round -- counterfactual only supports continuous main effects."
                )
        if immutable:
            overlap = set(actionable) & set(immutable)
            if overlap:
                raise ValueError(f"actionable and immutable overlap: {sorted(overlap)}")

        original_prediction = float(self.predict(X)[0])
        gap = target - original_prediction
        tol = tol if tol is not None else max(1e-6, 0.05 * abs(gap))

        working_row = X.copy()

        if abs(gap) > tol:
            grids = {feat: np.linspace(*self._observed_range(feat), n_candidates) for feat in actionable}

            best_single = None  # (feature, value, relative_change)
            for feat in actionable:
                candidates = pd.concat([working_row] * n_candidates, ignore_index=True)
                candidates[feat] = grids[feat]
                preds = self.predict(candidates)
                within_tol = np.abs(preds - target) <= tol
                if not np.any(within_tol):
                    continue
                current_val = float(working_row[feat].iloc[0])
                grid_range = grids[feat].max() - grids[feat].min()
                grid_range = grid_range if grid_range > 0 else 1.0
                candidate_vals = grids[feat][within_tol]
                rel_changes = np.abs(candidate_vals - current_val) / grid_range
                best_idx = int(np.argmin(rel_changes))
                if best_single is None or rel_changes[best_idx] < best_single[2]:
                    best_single = (feat, float(candidate_vals[best_idx]), float(rel_changes[best_idx]))

            if best_single is not None:
                feat, val, _ = best_single
                working_row[feat] = val
            else:
                remaining = list(actionable)
                for _ in range(len(actionable)):
                    current_pred = float(self.predict(working_row)[0])
                    if abs(target - current_pred) <= tol or not remaining:
                        break
                    best_this_round = None  # (feature, value, resulting_gap)
                    for feat in remaining:
                        candidates = pd.concat([working_row] * n_candidates, ignore_index=True)
                        candidates[feat] = grids[feat]
                        preds = self.predict(candidates)
                        idx = int(np.argmin(np.abs(preds - target)))
                        resulting_gap = float(abs(preds[idx] - target))
                        if best_this_round is None or resulting_gap < best_this_round[2]:
                            best_this_round = (feat, float(grids[feat][idx]), resulting_gap)
                    feat, val, resulting_gap = best_this_round
                    working_row[feat] = val
                    remaining.remove(feat)

        X_cf = working_row
        counterfactual_prediction = float(self.predict(X_cf)[0])
        feasible = bool(abs(target - counterfactual_prediction) <= tol)

        changes = {}
        zone_transition_frequency = {}
        extrapolating = False
        for feat in actionable:
            old_val = float(X[feat].iloc[0])
            new_val = float(X_cf[feat].iloc[0])
            obs_lo, obs_hi = self._observed_range(feat)
            if old_val < obs_lo or old_val > obs_hi:
                extrapolating = True
            if new_val != old_val:
                changes[feat] = (old_val, new_val)
                zone_transition_frequency[feat] = self._zone_transition_frequency(feat, old_val, new_val)

        contrib_orig = self.explain(X)
        contrib_cf = self.explain(X_cf)
        interaction_consequences = {}
        for term in contrib_orig.columns:
            if term == "baseline":
                continue
            delta = float(contrib_cf[term].iloc[0] - contrib_orig[term].iloc[0])
            if abs(delta) > 1e-9:
                interaction_consequences[term] = delta

        evidence = self.evidence_report(X_cf) if self.track_reliability else None

        return {
            "feasible": feasible,
            "original_prediction": original_prediction,
            "counterfactual_prediction": counterfactual_prediction,
            "prediction_change": counterfactual_prediction - original_prediction,
            "changes": changes,
            "zone_transition_frequency": zone_transition_frequency,
            "interaction_consequences": interaction_consequences,
            "extrapolating": extrapolating,
            "evidence": evidence,
        }
