"""Zone grids nested within discrete segments: fit a separate 2D zone
grid over two continuous columns *within* each distinct combination of
one or more segment columns, rather than relying on the core boosting
estimator's own combinatorial 3-way interaction search
(``max_interaction_order=3``) to discover the same structure.

Built as a standalone transformer -- like :class:`zoneboost.
ZoneProfileEncoder` and :class:`zoneboost.DepthTransformer`, it emits new
feature columns for an external downstream model and never touches
``rounds_``/``explain()``, so there is no attribution to canonicalize or
double-count the way there would be if this were folded into the boosting
loop itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._common import ensure_dataframe, resolve_categorical_features
from ._shrinkage import _estimate_shrinkage_m
from ._zones import adaptive_zone_boundaries, zone_index

__all__ = ["ConditionalZoneGrid"]


def _fit_grid(a: np.ndarray, b: np.ndarray, y: np.ndarray, sw: np.ndarray, params: dict) -> dict:
    """Fit one 2D zone grid (own boundaries for both columns, plus
    shrunk per-cell mean/count/variance of ``y``) over whatever rows are
    passed in -- either the full training set (the global grid) or one
    segment's own rows.
    """
    boundaries_a = adaptive_zone_boundaries(
        a, y, params["max_zones"], params["min_zone_frac"], params["min_zone_abs"], sample_weight=sw
    )
    boundaries_b = adaptive_zone_boundaries(
        b, y, params["max_zones"], params["min_zone_frac"], params["min_zone_abs"], sample_weight=sw
    )
    za = zone_index(a, boundaries_a)
    zb = zone_index(b, boundaries_b)
    n_a = len(boundaries_a) + 2  # + missing
    n_b = len(boundaries_b) + 2  # + missing
    combined = za * n_b + zb
    n_cells = n_a * n_b

    counts = np.bincount(combined, weights=sw, minlength=n_cells)
    sums = np.bincount(combined, weights=y * sw, minlength=n_cells)
    sq_sums = np.bincount(combined, weights=(y**2) * sw, minlength=n_cells)

    grand_mean = float(np.average(y, weights=sw))
    has_support = counts > 0
    raw_means = np.where(has_support, np.divide(sums, counts, out=np.zeros_like(sums), where=has_support), grand_mean)
    raw_var = np.where(
        has_support,
        np.maximum(0.0, np.divide(sq_sums, counts, out=np.zeros_like(sq_sums), where=has_support) - raw_means**2),
        0.0,
    )

    shrunk_means = raw_means
    if params["shrinkage"]:
        deviations = [(raw_means - grand_mean, counts)]
        residual_var = float(np.average((y - grand_mean) ** 2, weights=sw))
        m = _estimate_shrinkage_m(deviations, residual_var, fallback_m=10.0)
        weight = counts / (counts + m)
        shrunk_means = grand_mean + weight * (raw_means - grand_mean)

    return {
        "boundaries_a": boundaries_a,
        "boundaries_b": boundaries_b,
        "n_b": n_b,
        "counts": counts,
        "means": shrunk_means,
        "variances": raw_var,
    }


def _lookup_grid(grid: dict, a: np.ndarray, b: np.ndarray) -> tuple:
    za = zone_index(a, grid["boundaries_a"])
    zb = zone_index(b, grid["boundaries_b"])
    combined = za * grid["n_b"] + zb
    return grid["means"][combined], grid["counts"][combined], grid["variances"][combined]


class ConditionalZoneGrid(BaseEstimator, TransformerMixin):
    """2D zone grid over two continuous columns, fit separately within
    each discrete segment.

    For each distinct combination of ``segment_columns`` values with at
    least ``min_segment_size`` training rows, independently fits its own
    adaptive zone boundaries for both ``columns`` (exactly like
    :class:`zoneboost.ZoneProfileEncoder`) using only that segment's own
    rows, then the joint cell's (empirical-Bayes-shrunk, by default)
    mean, count, and variance of ``y``. A segment below
    ``min_segment_size``, or never seen at ``fit`` time, falls back to a
    single pooled grid fit on every row regardless of segment -- an
    unstable, independently-fit grid on a handful of rows is worse than a
    disclosed fallback to the pooled estimate.

    One instance encodes one pair of continuous columns nested within one
    segment definition. For more than one such pair, use multiple
    ``ConditionalZoneGrid`` instances inside a
    ``ColumnTransformer``/``FeatureUnion`` -- the same
    compose-rather-than-build-in precedent :class:`zoneboost.
    ZoneProfileEncoder`/:class:`zoneboost.DepthTransformer` already set.

    **Deferred**: only a single level of shrinkage (a cell toward its own
    grid's grand mean) -- no second, hierarchical level additionally
    pulling a segment's cell toward the global grid's corresponding cell.
    A real future refinement, not built here.

    Parameters
    ----------
    columns : list of str or int
        Exactly two continuous columns to grid jointly. Must not be
        categorical -- raises ``ValueError`` otherwise.
    segment_columns : list of str or int
        One or more columns whose distinct value combination defines a
        segment. May be categorical or numeric; grouped by exact value,
        not zone-binned.
    max_zones : int, default=7
        Zone cap per column, per grid (segment or global). See
        :func:`zoneboost._zones.adaptive_zone_boundaries`.
    min_zone_frac : float, default=0.02
        Minimum row fraction required on each side of a zone split,
        within whichever grid (segment or global) is being fit.
    min_zone_abs : int, default=20
        Minimum absolute row count required on each side of a zone split.
    min_segment_size : int, default=50
        A segment with fewer training rows than this doesn't get its own
        grid -- falls back to the global grid instead. Higher than a
        single column's own ``min_zone_abs`` since a whole 2D grid needs
        more rows to trust than one 1D zone split does.
    shrinkage : bool, default=True
        Empirical-Bayes-shrink each grid's own cell means toward that
        grid's own grand mean. ``False`` emits raw, unshrunk cell means.
    random_state : int, default=42
        Accepted for interface consistency; fitting is fully
        deterministic given ``X``/``y``, so this is currently unused.

    Attributes
    ----------
    segment_grids_ : dict
        ``{segment_key: grid}`` for every segment that met
        ``min_segment_size``, one entry per distinct
        ``segment_columns`` value combination seen at ``fit`` time.
    global_grid_ : dict
        The pooled fallback grid, fit on every row regardless of segment.
    feature_names_in_ : ndarray of shape (n_features_in,)

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import ConditionalZoneGrid
    >>> X = pd.DataFrame({
    ...     "x": [1, 2, 3, 4] * 20, "y": [1, 2, 3, 4] * 20,
    ...     "region": (["north"] * 40 + ["south"] * 40),
    ... })
    >>> target = [1.0, 2.0, 3.0, 4.0] * 20
    >>> grid = ConditionalZoneGrid(columns=["x", "y"], segment_columns=["region"]).fit(X, target)
    >>> grid.transform(X).columns.tolist()
    ['x_y__cell_mean', 'x_y__cell_count', 'x_y__cell_var', 'x_y__used_segment_grid']
    """

    def __init__(
        self,
        columns,
        segment_columns,
        max_zones: int = 7,
        min_zone_frac: float = 0.02,
        min_zone_abs: int = 20,
        min_segment_size: int = 50,
        shrinkage: bool = True,
        random_state: int = 42,
    ):
        self.columns = columns
        self.segment_columns = segment_columns
        self.max_zones = max_zones
        self.min_zone_frac = min_zone_frac
        self.min_zone_abs = min_zone_abs
        self.min_segment_size = min_segment_size
        self.shrinkage = shrinkage
        self.random_state = random_state

    def _resolve_names(self, X: pd.DataFrame, declared, label: str) -> list:
        names = [X.columns[c] if isinstance(c, (int, np.integer)) else c for c in declared]
        unknown = [c for c in names if c not in X.columns]
        if unknown:
            raise ValueError(f"{label} not found in X: {unknown}")
        return names

    def fit(self, X, y, sample_weight=None):
        """Fit the global grid, then each qualifying segment's own grid.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), default=None

        Returns
        -------
        self : ConditionalZoneGrid
        """
        X = ensure_dataframe(X, getattr(self, "feature_names_in_", None))
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y_arr):
            raise ValueError(f"X and y have inconsistent lengths: {len(X)} vs {len(y_arr)}")
        sw = np.asarray(sample_weight, dtype=float).reshape(-1) if sample_weight is not None else np.ones(len(y_arr))
        self.feature_names_in_ = np.array(X.columns)

        if len(self.columns) != 2:
            raise ValueError(f"columns must name exactly 2 continuous columns, got {self.columns!r}")
        col_a, col_b = self._resolve_names(X, self.columns, "columns")
        categorical = resolve_categorical_features(X, None)
        declared_categorical = [c for c in (col_a, col_b) if c in categorical]
        if declared_categorical:
            raise ValueError(
                f"ConditionalZoneGrid requires numeric columns; {declared_categorical} are categorical."
            )
        self.columns_ = [col_a, col_b]
        self.segment_columns_ = self._resolve_names(X, self.segment_columns, "segment_columns")

        a_vals = X[col_a].to_numpy(dtype=float)
        b_vals = X[col_b].to_numpy(dtype=float)
        params = {
            "max_zones": self.max_zones,
            "min_zone_frac": self.min_zone_frac,
            "min_zone_abs": self.min_zone_abs,
            "shrinkage": self.shrinkage,
        }

        self.global_grid_ = _fit_grid(a_vals, b_vals, y_arr, sw, params)

        segment_keys = list(X[self.segment_columns_].itertuples(index=False, name=None))
        segment_keys_arr = np.array(segment_keys, dtype=object)
        unique_keys = pd.unique(pd.Series(segment_keys, dtype=object))

        segment_grids_ = {}
        for key in unique_keys:
            mask = np.array([k == key for k in segment_keys])
            total_weight = sw[mask].sum()
            if total_weight < self.min_segment_size:
                continue
            segment_grids_[key] = _fit_grid(a_vals[mask], b_vals[mask], y_arr[mask], sw[mask], params)
        self.segment_grids_ = segment_grids_

        pair_name = f"{col_a}_{col_b}"
        self._pair_name = pair_name
        return self

    def transform(self, X) -> pd.DataFrame:
        """Look up each row's segment grid (or the global fallback) and
        that grid's cell for the row's own ``columns`` values.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)

        Returns
        -------
        DataFrame of shape (n_samples, 4)
            ``"<pair>__cell_mean"``, ``"<pair>__cell_count"``,
            ``"<pair>__cell_var"``, and ``"<pair>__used_segment_grid"``
            (1 when the row's segment had its own fitted grid, 0 when it
            fell back to the global grid).
        """
        check_is_fitted(self, "global_grid_")
        X = ensure_dataframe(X, self.feature_names_in_)

        col_a, col_b = self.columns_
        a_vals = X[col_a].to_numpy(dtype=float)
        b_vals = X[col_b].to_numpy(dtype=float)
        segment_keys = list(X[self.segment_columns_].itertuples(index=False, name=None))

        n = len(X)
        means = np.empty(n)
        counts = np.empty(n)
        variances = np.empty(n)
        used_segment = np.zeros(n, dtype=int)

        by_grid = {}
        for i, key in enumerate(segment_keys):
            grid = self.segment_grids_.get(key)
            by_grid.setdefault(id(grid) if grid is not None else "global", []).append(i)

        for bucket, idx in by_grid.items():
            idx = np.array(idx)
            key = segment_keys[idx[0]]
            grid = self.segment_grids_.get(key, self.global_grid_)
            m, c, v = _lookup_grid(grid, a_vals[idx], b_vals[idx])
            means[idx], counts[idx], variances[idx] = m, c, v
            used_segment[idx] = 1 if key in self.segment_grids_ else 0

        return pd.DataFrame(
            {
                f"{self._pair_name}__cell_mean": means,
                f"{self._pair_name}__cell_count": counts,
                f"{self._pair_name}__cell_var": variances,
                f"{self._pair_name}__used_segment_grid": used_segment,
            },
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Output column names, in the same order :meth:`transform`
        emits them."""
        check_is_fitted(self, "global_grid_")
        return np.array(
            [
                f"{self._pair_name}__cell_mean",
                f"{self._pair_name}__cell_count",
                f"{self._pair_name}__cell_var",
                f"{self._pair_name}__used_segment_grid",
            ]
        )
