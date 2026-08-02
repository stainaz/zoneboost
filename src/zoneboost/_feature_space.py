"""ZoneFeatureSpace: zoneboost as a feature space, not just a model.

The notebook pages behind zoneboost list "the ado-transformer" as step one,
ahead of any modeling -- feature engineering first. ``ZoneProfileEncoder``,
``DepthTransformer``, ``CategoricalDepthTransformer``, and
``ConditionalZoneGrid`` already turn raw columns into transparent,
zone-based engineered features usable ahead of *any* downstream estimator
(each is already an ``sklearn`` ``TransformerMixin``, composable via a
plain ``FeatureUnion`` today). ``ZoneFeatureSpace`` doesn't reinvent that
machinery -- it's a thin, convenience-first orchestrator over the same
transformers, plus two things a bare ``FeatureUnion`` doesn't give you:
rolling a downstream model's coefficients/importances back to each
*original* raw column (:meth:`ZoneFeatureSpace.explain`), and a cheap,
disclosed-as-crude candidate-pair suggester for
:class:`zoneboost.ConditionalZoneGrid`
(:meth:`ZoneFeatureSpace.suggest_interactions`).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._categorical_depth import CategoricalDepthTransformer
from ._common import ensure_dataframe, resolve_categorical_features
from ._conditional_grid import ConditionalZoneGrid
from ._depth import DepthTransformer
from ._zone_profile import ZoneProfileEncoder

__all__ = ["ZoneFeatureSpace"]


class ZoneFeatureSpace(BaseEstimator, TransformerMixin):
    """Compose zoneboost's transformers into one transparent feature space.

    Fits, in a fixed order, whichever of the following are enabled:
    :class:`zoneboost.ZoneProfileEncoder` (``zone_profiles``), one
    :class:`zoneboost.DepthTransformer` per declared group
    (``depth_scores``), one :class:`zoneboost.CategoricalDepthTransformer`
    per declared group (``categorical_depth_scores``), and one
    :class:`zoneboost.ConditionalZoneGrid` per declared pair
    (``conditional_grids``) -- then concatenates their ``transform(X)``
    output. Every emitted column is still a plain zone average/count from
    one of those transformers; nothing new is computed here.

    Not a replacement for hand-building a ``FeatureUnion`` of the same
    transformers -- which already works today, with zero new code. What
    this adds: convenience defaults, and :meth:`explain`/
    :meth:`suggest_interactions` below.

    **Deferred, disclosed**: :class:`zoneboost.DepthCrowd` (aggregates
    *already-built* depth outputs -- an aggregator-of-aggregators, not a
    ``columns -> features`` step this class's flat toggles can represent)
    and :class:`zoneboost.LaplaceHistoryTransformer` (needs a second
    ``event_history`` table and a per-row ``asof_col`` at transform, a
    fundamentally different signature). Build either separately and
    combine with this class's own output via ``FeatureUnion`` if wanted.

    Parameters
    ----------
    zone_profiles : bool or list of str/int, default=True
        ``True`` fits one :class:`zoneboost.ZoneProfileEncoder` over every
        column of ``X`` (its own default). A list instead encodes only
        those columns. ``False`` disables it. Supervised -- requires
        ``y`` at `fit` whenever enabled.
    categorical_features : list of str or int, default=None
        Forwarded to the internal ``ZoneProfileEncoder`` only (the other
        three transformers below auto-detect categorical columns
        themselves and don't accept an override).
    depth_scores : bool or list, default=False
        ``True`` fits one :class:`zoneboost.DepthTransformer` over every
        numeric column of ``X`` (its own default: one joint group).
        A list instead declares one or more explicit groups -- each entry
        is either a plain ``list`` of columns (auto-named, joining column
        names with ``"_"``, the same convention ``DepthTransformer``
        itself uses) or a ``tuple`` ``(name, columns)`` for an explicit
        name. ``False`` disables it. Unsupervised.
    categorical_depth_scores : bool or list, default=False
        Same shape as ``depth_scores``, but fits
        :class:`zoneboost.CategoricalDepthTransformer` instances over
        categorical/binary columns instead. Unsupervised.
    conditional_grids : list of (col_a, col_b, segment_columns), default=None
        One :class:`zoneboost.ConditionalZoneGrid` per declared 3-tuple --
        explicit only, never auto-discovered (see
        :meth:`suggest_interactions` for a suggestion helper that still
        leaves the final choice, and ``segment_columns``, to you).
        Supervised -- requires ``y`` at `fit` whenever non-empty.
    max_zones : int, default=7
        Forwarded to every internal ``ZoneProfileEncoder``/
        ``ConditionalZoneGrid``.
    min_zone_frac : float, default=0.02
        Forwarded to every internal ``ZoneProfileEncoder``/
        ``ConditionalZoneGrid``.
    min_zone_abs : int, default=20
        Forwarded to every internal ``ZoneProfileEncoder``/
        ``ConditionalZoneGrid``.
    min_segment_size : int, default=50
        Forwarded to every internal ``ConditionalZoneGrid``.
    shrinkage : bool, default=True
        Forwarded to every internal ``ZoneProfileEncoder``/
        ``ConditionalZoneGrid``.
    depth_ridge : float, default=1e-6
        Forwarded to every internal ``DepthTransformer`` as its own
        ``ridge`` parameter.
    random_state : int, default=42
        Forwarded to every internal transformer.

    Attributes
    ----------
    feature_names_in_ : ndarray
    transformers_ : list
        The fitted sub-transformer instances, in output order.

    Notes
    -----
    Using more than one entry in ``depth_scores``/``categorical_depth_scores``
    without giving each an explicit, distinct name can produce duplicate
    output column names if their auto-derived names collide (e.g. two
    groups whose joined column names happen to match) -- not
    deduplicated; give explicit names to avoid this.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import ZoneFeatureSpace
    >>> X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8]})
    >>> y = [0, 0, 0, 1, 0, 1, 1, 1]
    >>> space = ZoneFeatureSpace(max_zones=2, min_zone_abs=2).fit(X, y)
    >>> space.transform(X).columns.tolist()
    ['x__zone_mean', 'x__zone_count', 'x__zone_var']
    """

    def __init__(
        self,
        zone_profiles=True,
        categorical_features=None,
        depth_scores=False,
        categorical_depth_scores=False,
        conditional_grids=None,
        max_zones: int = 7,
        min_zone_frac: float = 0.02,
        min_zone_abs: int = 20,
        min_segment_size: int = 50,
        shrinkage: bool = True,
        depth_ridge: float = 1e-6,
        random_state: int = 42,
    ):
        self.zone_profiles = zone_profiles
        self.categorical_features = categorical_features
        self.depth_scores = depth_scores
        self.categorical_depth_scores = categorical_depth_scores
        self.conditional_grids = conditional_grids
        self.max_zones = max_zones
        self.min_zone_frac = min_zone_frac
        self.min_zone_abs = min_zone_abs
        self.min_segment_size = min_segment_size
        self.shrinkage = shrinkage
        self.depth_ridge = depth_ridge
        self.random_state = random_state

    def _resolve_groups(self, spec, label: str) -> list:
        if not spec:
            return []
        if spec is True:
            return [(None, None)]
        groups = []
        for entry in spec:
            if isinstance(entry, tuple):
                if len(entry) != 2 or not isinstance(entry[0], str):
                    raise ValueError(f"{label} tuple entries must be (name, columns), got {entry!r}")
                groups.append((entry[0], list(entry[1])))
            elif isinstance(entry, list):
                groups.append((None, entry))
            else:
                raise ValueError(
                    f"{label} entries must be a list of columns or a (name, columns) tuple, got {entry!r}"
                )
        return groups

    def fit(self, X, y=None, sample_weight=None):
        """Fit every enabled sub-transformer.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,), default=None
            Required whenever ``zone_profiles`` or ``conditional_grids``
            is enabled (both are supervised); ignored by ``depth_scores``/
            ``categorical_depth_scores`` either way.
        sample_weight : array-like of shape (n_samples,), default=None
            Forwarded to every sub-transformer that accepts one
            (``ZoneProfileEncoder``, ``DepthTransformer``,
            ``ConditionalZoneGrid``) -- ``CategoricalDepthTransformer``
            has no such parameter, so it's not passed there.

        Returns
        -------
        self : ZoneFeatureSpace
        """
        X = ensure_dataframe(X, getattr(self, "feature_names_in_", None))
        self.feature_names_in_ = np.array(X.columns)

        needs_y = bool(self.zone_profiles) or bool(self.conditional_grids)
        if needs_y and y is None:
            raise ValueError(
                "y is required when zone_profiles or conditional_grids is enabled "
                "(both fit a supervised, target-based statistic)."
            )
        y_arr = np.asarray(y, dtype=float).reshape(-1) if y is not None else None

        built = []  # list of (fitted_transformer, {output_col: (source_cols...)})

        if self.zone_profiles:
            cols = None if self.zone_profiles is True else list(self.zone_profiles)
            enc = ZoneProfileEncoder(
                columns=cols,
                categorical_features=self.categorical_features,
                max_zones=self.max_zones,
                min_zone_frac=self.min_zone_frac,
                min_zone_abs=self.min_zone_abs,
                shrinkage=self.shrinkage,
                random_state=self.random_state,
            ).fit(X, y_arr, sample_weight=sample_weight)
            sources = {}
            for c in enc.columns_:
                for suffix in ("zone_mean", "zone_count", "zone_var"):
                    sources[f"{c}__{suffix}"] = (c,)
            built.append((enc, sources))

        for name, cols in self._resolve_groups(self.depth_scores, "depth_scores"):
            dep = DepthTransformer(
                columns=cols, group_name=name, ridge=self.depth_ridge, random_state=self.random_state,
            ).fit(X, sample_weight=sample_weight)
            sources = {out: tuple(dep.columns_) for out in dep.get_feature_names_out()}
            built.append((dep, sources))

        for name, cols in self._resolve_groups(self.categorical_depth_scores, "categorical_depth_scores"):
            cdep = CategoricalDepthTransformer(columns=cols, group_name=name, random_state=self.random_state).fit(X)
            sources = {out: tuple(cdep.columns_) for out in cdep.get_feature_names_out()}
            built.append((cdep, sources))

        for col_a, col_b, segment_columns in (self.conditional_grids or []):
            grid = ConditionalZoneGrid(
                columns=[col_a, col_b],
                segment_columns=segment_columns,
                max_zones=self.max_zones,
                min_zone_frac=self.min_zone_frac,
                min_zone_abs=self.min_zone_abs,
                min_segment_size=self.min_segment_size,
                shrinkage=self.shrinkage,
                random_state=self.random_state,
            ).fit(X, y_arr, sample_weight=sample_weight)
            all_sources = tuple(grid.columns_) + tuple(grid.segment_columns_)
            sources = {out: all_sources for out in grid.get_feature_names_out()}
            built.append((grid, sources))

        if not built:
            raise ValueError(
                "ZoneFeatureSpace has nothing enabled -- set at least one of zone_profiles/"
                "depth_scores/categorical_depth_scores/conditional_grids."
            )

        self.transformers_ = [t for t, _ in built]
        output_sources = {}
        for _, sources in built:
            output_sources.update(sources)
        self._output_sources_ = output_sources
        return self

    def transform(self, X) -> pd.DataFrame:
        """Apply every fitted sub-transformer and concatenate their output.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)

        Returns
        -------
        DataFrame
            One block of columns per enabled sub-transformer, in the
            order documented on the class itself; column names exactly
            as that sub-transformer's own ``transform`` emits them.
        """
        check_is_fitted(self, "transformers_")
        X = ensure_dataframe(X, self.feature_names_in_)
        blocks = [t.transform(X) for t in self.transformers_]
        return pd.concat(blocks, axis=1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Output column names, in the same order :meth:`transform`
        emits them."""
        check_is_fitted(self, "transformers_")
        names = []
        for t in self.transformers_:
            names.extend(t.get_feature_names_out())
        return np.array(names)

    def explain(self, values) -> pd.Series:
        """Roll a downstream model's per-output-column coefficient or
        importance array back to each **original raw column** it
        depends on -- "the ado-transformer... explain back through the
        zone space."

        Parameters
        ----------
        values : array-like of shape (n_output_features,)
            Aligned to :meth:`get_feature_names_out` -- e.g. a fitted
            ``LogisticRegression.coef_[0]`` or a
            ``RandomForestRegressor.feature_importances_`` array, fit on
            this instance's own :meth:`transform` output.

        Returns
        -------
        Series
            Indexed by original raw column name, sorted descending. An
            output column with more than one source column (an
            interaction, or a joint depth/grid group) contributes its
            full, undivided ``abs(value)`` to **every** one of its
            source columns -- never split between them, the same "never
            divided back between parent variables" rule
            :meth:`zoneboost.ZoneBoostRegressor.explain` itself follows.
        """
        check_is_fitted(self, "transformers_")
        names = self.get_feature_names_out()
        values_arr = np.asarray(values, dtype=float).reshape(-1)
        if len(values_arr) != len(names):
            raise ValueError(
                f"values must have length {len(names)} (== get_feature_names_out()), got {len(values_arr)}"
            )
        totals: dict = {}
        for name, v in zip(names, np.abs(values_arr)):
            for col in self._output_sources_[name]:
                totals[col] = totals.get(col, 0.0) + float(v)
        return pd.Series(totals).sort_values(ascending=False)

    def suggest_interactions(self, X, y, columns=None, top_k: int = 10) -> list:
        """Cheap, disclosed-as-crude candidate ``(col_a, col_b)`` pairs
        worth building a :class:`zoneboost.ConditionalZoneGrid` for --
        suggestions only, never auto-fit, and never guesses
        ``segment_columns`` (left entirely to you).

        For every candidate pair, fits ``y ~ a + b`` by plain least
        squares and scores the pair by the absolute correlation between
        that fit's residual and the centered product
        ``(a - mean(a)) * (b - mean(b))`` -- a fast proxy for "is there
        multiplicative interaction signal left over after each column's
        own additive effect is removed." This is a materially cruder
        heuristic than :class:`zoneboost.ZoneBoostRegressor`'s own
        cross-fitted, zone-based pair screening (``max_pair_interactions``)
        -- not a replacement for it, just cheap enough to run before any
        model exists at all.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        columns : list of str/int, default=None
            Candidate continuous columns to consider. ``None`` (default)
            uses every auto-detected non-categorical column of ``X``.
        top_k : int, default=10
            Number of top-scoring pairs to return.

        Returns
        -------
        list of (col_a, col_b) tuples, ranked highest-scoring first.
        """
        X = ensure_dataframe(X, None)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y_arr):
            raise ValueError(f"X and y have inconsistent lengths: {len(X)} vs {len(y_arr)}")

        if columns is not None:
            cols = [X.columns[c] if isinstance(c, (int, np.integer)) else c for c in columns]
        else:
            categorical = resolve_categorical_features(X, None)
            cols = [c for c in X.columns if c not in categorical]
        if len(cols) < 2:
            raise ValueError(f"suggest_interactions needs at least 2 candidate columns, got {cols!r}")

        n = len(X)
        design_ones = np.ones(n)
        scores = {}
        for a, b in itertools.combinations(cols, 2):
            a_vals = X[a].to_numpy(dtype=float)
            b_vals = X[b].to_numpy(dtype=float)
            design = np.column_stack([design_ones, a_vals, b_vals])
            coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
            residual = y_arr - design @ coef
            product = (a_vals - a_vals.mean()) * (b_vals - b_vals.mean())
            if product.std() == 0 or residual.std() == 0:
                scores[(a, b)] = 0.0
                continue
            scores[(a, b)] = abs(float(np.corrcoef(residual, product)[0, 1]))

        ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return ranked
