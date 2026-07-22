"""Zone profile encoding: per-zone target statistics (a shrunk mean, a raw
variance, and a support count) turned into standalone engineered features,
rather than something only usable inside zoneboost's own boosting loop.

Where :class:`~zoneboost.ZoneBoostRegressor`/:class:`~zoneboost.
ZoneBoostClassifier` use a column's zones as one weak learner among many,
boosted together, :class:`ZoneProfileEncoder` fits the same zone
construction once per column and emits the resulting statistics as new
feature columns -- usable ahead of *any* downstream estimator (a plain
``LogisticRegression``, ``XGBoost``, anything scikit-learn can fit), not
only zoneboost's own estimators. A zone's mean of ``y`` is exactly
``P(outcome | zone)`` when ``y`` is 0/1, and a plain conditional mean
otherwise -- one code path serves both framings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._common import ensure_dataframe, resolve_categorical_features
from ._shrinkage import _estimate_shrinkage_m
from ._zones import adaptive_zone_boundaries, categorical_zone_index, categorical_zone_map, zone_index

__all__ = ["ZoneProfileEncoder"]


def _column_zone_stats(zone_idx: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, n_zones: int) -> dict:
    """Per-zone count, raw mean, and raw (population) variance of ``y``
    via ``bincount`` -- O(n), the same complexity the core estimator's own
    per-round zone means cost. A zone with zero weight falls back to the
    column's grand mean/zero variance rather than producing NaN, exactly
    the "carries zero confidence" precedent :func:`zoneboost._zones.
    categorical_zone_index`'s missing/unknown zones already set.
    """
    counts = np.bincount(zone_idx, weights=sample_weight, minlength=n_zones)
    sums = np.bincount(zone_idx, weights=y * sample_weight, minlength=n_zones)
    sq_sums = np.bincount(zone_idx, weights=(y**2) * sample_weight, minlength=n_zones)

    grand_mean = float(np.average(y, weights=sample_weight))
    has_support = counts > 0
    raw_means = np.where(has_support, np.divide(sums, counts, out=np.zeros_like(sums), where=has_support), grand_mean)
    raw_var = np.where(
        has_support,
        np.maximum(0.0, np.divide(sq_sums, counts, out=np.zeros_like(sq_sums), where=has_support) - raw_means**2),
        0.0,
    )
    return {"counts": counts, "raw_means": raw_means, "raw_var": raw_var, "grand_mean": grand_mean}


class ZoneProfileEncoder(BaseEstimator, TransformerMixin):
    """Per-column zone statistics as standalone engineered features.

    For every fitted column, splits it into zones exactly the way
    :class:`~zoneboost.ZoneBoostRegressor` does (adaptive variance-reducing
    boundaries for continuous columns, one zone per distinct value for
    categorical ones), then emits each row's zone mean, zone variance, and
    zone support count of ``y`` as three new columns. Every emitted number
    is a stated, auditable conditional statistic -- "the mean outcome in
    this zone is 0.12" -- not a hidden model weight.

    Only per-column (main-effect) profiles are emitted -- no automatic
    pairwise zone-grid profiling. Pairwise profiling raises the same
    combinatorial pair-selection question :mod:`zoneboost._weak_learner`
    already solves for boosting rounds, and a v1 encoder doesn't need it:
    combine :class:`ZoneProfileEncoder` output with the raw features via
    ``ColumnTransformer``/``FeatureUnion`` and let the downstream model
    find cross-column effects itself.

    Emits only the new zone-profile columns (standard scikit-learn
    transformer behavior) -- it does not pass through the original
    features itself.

    Parameters
    ----------
    columns : list of str or int, default=None
        Columns to encode. ``None`` (default) encodes every column of
        ``X``. Names or positional indices, same convention as
        ``categorical_features``.
    categorical_features : list of str or int, default=None
        Columns to treat as categorical (each distinct value its own
        zone), in addition to any auto-detected via dtype (see
        :func:`zoneboost._common.resolve_categorical_features`) -- the
        same convention :class:`~zoneboost.ZoneBoostRegressor` uses.
    max_zones : int, default=7
        Upper bound on zones per continuous column. See
        :func:`zoneboost._zones.adaptive_zone_boundaries`.
    min_zone_frac : float, default=0.02
        Minimum fraction of rows (or weight) required on each side of a
        candidate split, for continuous columns.
    min_zone_abs : int, default=20
        Minimum absolute row count (or weight) required on each side of a
        candidate split, for continuous columns.
    shrinkage : bool, default=True
        Shrink each zone's raw mean toward the column's own grand mean via
        empirical Bayes (:func:`zoneboost._shrinkage._estimate_shrinkage_m`,
        the same DerSimonian-Laird estimator the core boosting estimator
        uses), so a sparse zone's emitted mean leans toward the grand mean
        instead of overfitting a handful of rows. ``False`` emits the raw,
        unshrunk zone mean.
    random_state : int, default=42
        Accepted for interface consistency with the rest of the package;
        fitting is fully deterministic given ``X``/``y``, so this is
        currently unused.

    Attributes
    ----------
    zone_stats_ : dict
        ``{column: {"kind": "continuous" or "categorical", "boundaries" or
        "category_map": ..., "counts": ndarray, "raw_means": ndarray,
        "shrunk_means": ndarray, "variances": ndarray, "grand_mean": float}}``
        -- one entry per encoded column, every value plain inspectable data.
    columns_ : list of str
        The columns actually encoded, in output order.
    feature_names_in_ : ndarray of shape (n_features_in,)
        Every column name seen in ``X`` at ``fit`` time.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import ZoneProfileEncoder
    >>> X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8]})
    >>> y = [0, 0, 0, 1, 0, 1, 1, 1]
    >>> encoder = ZoneProfileEncoder(max_zones=2, min_zone_abs=2).fit(X, y)
    >>> encoder.transform(X).columns.tolist()
    ['x__zone_mean', 'x__zone_count', 'x__zone_var']
    """

    def __init__(
        self,
        columns=None,
        categorical_features=None,
        max_zones: int = 7,
        min_zone_frac: float = 0.02,
        min_zone_abs: int = 20,
        shrinkage: bool = True,
        random_state: int = 42,
    ):
        self.columns = columns
        self.categorical_features = categorical_features
        self.max_zones = max_zones
        self.min_zone_frac = min_zone_frac
        self.min_zone_abs = min_zone_abs
        self.shrinkage = shrinkage
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Fit zone boundaries/maps and per-zone statistics for every
        encoded column.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), default=None
            Per-row weight for zone construction and every emitted
            statistic. ``None`` (default) weighs every row equally.

        Returns
        -------
        self : ZoneProfileEncoder
        """
        X = ensure_dataframe(X, getattr(self, "feature_names_in_", None))
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y_arr):
            raise ValueError(f"X and y have inconsistent lengths: {len(X)} vs {len(y_arr)}")
        sw = np.asarray(sample_weight, dtype=float).reshape(-1) if sample_weight is not None else np.ones(len(y_arr))

        self.feature_names_in_ = np.array(X.columns)
        if self.columns is not None:
            columns_ = [X.columns[c] if isinstance(c, (int, np.integer)) else c for c in self.columns]
            unknown = [c for c in columns_ if c not in X.columns]
            if unknown:
                raise ValueError(f"columns not found in X: {unknown}")
        else:
            columns_ = list(X.columns)
        self.columns_ = columns_

        categorical = resolve_categorical_features(X, self.categorical_features)
        zone_stats_ = {}
        for col in columns_:
            series = X[col]
            if col in categorical:
                category_map = categorical_zone_map(series)
                zone_idx = categorical_zone_index(series, category_map)
                n_zones = len(category_map) + 2  # + missing + unknown
                entry = {"kind": "categorical", "category_map": category_map}
            else:
                boundaries = adaptive_zone_boundaries(
                    series, y_arr, self.max_zones, self.min_zone_frac, self.min_zone_abs, sample_weight=sw
                )
                zone_idx = zone_index(series, boundaries)
                n_zones = len(boundaries) + 2  # + missing
                entry = {"kind": "continuous", "boundaries": boundaries}

            stats = _column_zone_stats(zone_idx, y_arr, sw, n_zones)
            shrunk_means = stats["raw_means"]
            if self.shrinkage:
                deviations = [(stats["raw_means"] - stats["grand_mean"], stats["counts"])]
                residual_var = float(np.average((y_arr - stats["grand_mean"]) ** 2, weights=sw))
                m = _estimate_shrinkage_m(deviations, residual_var, fallback_m=10.0)
                weight = stats["counts"] / (stats["counts"] + m)
                shrunk_means = stats["grand_mean"] + weight * (stats["raw_means"] - stats["grand_mean"])

            entry.update(
                counts=stats["counts"],
                raw_means=stats["raw_means"],
                shrunk_means=shrunk_means,
                variances=stats["raw_var"],
                grand_mean=stats["grand_mean"],
            )
            zone_stats_[col] = entry

        self.zone_stats_ = zone_stats_
        return self

    def transform(self, X) -> pd.DataFrame:
        """Map each row to its per-column zone and look up that zone's
        statistics.

        An unseen category or an out-of-fit-range/missing value lands in
        the dedicated missing/unknown zone reserved at ``fit`` time (see
        :func:`zoneboost._zones.categorical_zone_index`/:func:`zoneboost.
        _zones.zone_index`) -- already near-zero support, so its emitted
        mean already leans toward the column's grand mean with no further
        special-casing needed here.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)

        Returns
        -------
        DataFrame of shape (n_samples, 3 * len(columns_))
            Three columns per encoded column: ``"<col>__zone_mean"``,
            ``"<col>__zone_count"``, ``"<col>__zone_var"``.
        """
        check_is_fitted(self, "zone_stats_")
        X = ensure_dataframe(X, self.feature_names_in_)

        out = {}
        for col in self.columns_:
            entry = self.zone_stats_[col]
            series = X[col]
            if entry["kind"] == "categorical":
                zone_idx = categorical_zone_index(series, entry["category_map"])
            else:
                zone_idx = zone_index(series, entry["boundaries"])

            out[f"{col}__zone_mean"] = entry["shrunk_means"][zone_idx]
            out[f"{col}__zone_count"] = entry["counts"][zone_idx]
            out[f"{col}__zone_var"] = entry["variances"][zone_idx]

        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Output column names, in the same order :meth:`transform`
        emits them."""
        check_is_fitted(self, "zone_stats_")
        names = []
        for col in self.columns_:
            names += [f"{col}__zone_mean", f"{col}__zone_count", f"{col}__zone_var"]
        return np.array(names)
