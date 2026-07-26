"""A ``DepthTransformer``-shaped typicality score for a categorical/binary
zone, rather than a continuous one.

``DepthTransformer``'s Mahalanobis-distance geometry assumes roughly
continuous, elliptical structure: it explicitly rejects a declared
categorical/``bool`` column, and silently accepts an int-coded 0/1 column
but produces a coarse, barely-discriminating ``coreness`` for it (mixing
even one binary column into a joint Mahalanobis distance collapses most of
its resolution). This transformer fills that gap with the discrete
analogue: how common is this exact combination of category values,
relative to every combination seen during training.

Reuses existing, already-tested primitives rather than inventing new
categorical-bucketing logic: :func:`zoneboost._zones.categorical_zone_map`/
:func:`zoneboost._zones.categorical_zone_index` already give each distinct
value its own zone, with two separately-reserved fallback zones -- missing
(``NaN``/``None``) and unseen-but-real (present at ``transform`` time, not
at ``fit`` time) -- exactly the distinction this needs, for free. Multiple
declared columns are combined into one joint cell index via mixed-radix
encoding, the same ``combined = za * n_b + zb`` trick
:func:`zoneboost._conditional_grid._fit_grid` already uses for 2 columns,
generalized here to however many columns are declared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._common import ensure_dataframe, resolve_categorical_features
from ._zones import categorical_zone_index, categorical_zone_map

__all__ = ["CategoricalDepthTransformer"]


class CategoricalDepthTransformer(BaseEstimator, TransformerMixin):
    """Typicality score for a categorical/binary column group -- the
    discrete sibling of :class:`zoneboost.DepthTransformer`.

    Emits, for each row, how many training rows shared its exact
    combination of ``columns`` values (``"<group>__count"``) and that
    count as a fraction of the training set (``"<group>__coreness"``, the
    empirical joint probability of this exact combination, bounded in
    ``[0, 1]``, higher = more typical) -- the same raw-then-bounded
    disclosure pattern, polarity, and column-suffix convention
    ``DepthTransformer`` uses for ``depth_distance``/``coreness``, so a
    caller building a :class:`zoneboost.DepthCrowd` ``columns=[...]`` list
    doesn't need to remember two different naming schemes for a
    continuous expert vs. a categorical one.

    Unsupervised, same as ``DepthTransformer``: typicality is a property
    of ``X``'s own joint distribution, not of any target. No
    shrinkage/smoothing toward a prior -- unlike
    :class:`zoneboost.ConditionalZoneGrid`'s cell *mean of y*, which
    genuinely needs enough support to be a trustworthy estimate, a raw
    count is already an honest, correct answer even at ``count = 1``
    ("this exact combination appeared once in training") -- there's
    nothing to shrink toward.

    One instance encodes one categorical/binary zone. For more than one
    such zone -- or to mix continuous and categorical experts together --
    use multiple instances (this one and/or ``DepthTransformer``) and feed
    their ``__coreness`` columns into one :class:`zoneboost.DepthCrowd`,
    the same compose-rather-than-build-in precedent every transformer in
    this package already sets.

    **Deferred**: no automatic cap or fallback for combinatorial sparsity
    (many columns, or high-cardinality columns, multiply out to a huge
    joint cell space with mostly singleton cells). Unlike
    ``ConditionalZoneGrid``'s ``min_segment_size`` fallback -- which
    exists because a *mean* needs enough support to trust -- a raw count
    doesn't have that problem, it's just less discriminating as cells get
    sparser; disclosed as a caveat (keep column groups small, 2-4 columns)
    rather than built around. No Laplace/empirical-Bayes smoothing of the
    count itself -- a future refinement, not needed for an honest raw
    count. No ordinal awareness -- every distinct combination is its own
    bucket, same "no meaningful adjacent" precedent
    ``categorical_zone_map``'s own docstring already states for a single
    nominal column.

    Parameters
    ----------
    columns : list of str or int, default=None
        The categorical/binary zone. ``None`` (default) uses every
        auto-detected categorical column of ``X`` (see
        :func:`zoneboost._common.resolve_categorical_features`). Unlike
        ``DepthTransformer``, explicitly declaring a column here is
        always accepted regardless of dtype (including a ``bool`` or
        int-coded binary column) -- treating a column as discrete-by-
        exact-value is this class's entire point, not something only some
        dtypes can support. At least 1 column is required.
    group_name : str, default=None
        Name used in the two emitted output columns. ``None`` joins the
        encoded column names with ``"_"``.
    random_state : int, default=42
        Accepted for interface consistency; fitting is fully
        deterministic given ``X``, so this is currently unused.

    Attributes
    ----------
    columns_ : list of str
        The columns actually encoded, in fitted order.
    feature_names_in_ : ndarray of shape (n_features_in,)

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import CategoricalDepthTransformer
    >>> X = pd.DataFrame({
    ...     "treatment": ["A", "A", "A", "A", "B", "C"],
    ... })
    >>> cat_depth = CategoricalDepthTransformer(columns=["treatment"]).fit(X)
    >>> out = cat_depth.transform(X)
    >>> out.loc[0, "treatment__coreness"] > out.loc[4, "treatment__coreness"]
    True
    """

    def __init__(
        self,
        columns=None,
        group_name: str = None,
        random_state: int = 42,
    ):
        self.columns = columns
        self.group_name = group_name
        self.random_state = random_state

    def _resolve_columns(self, X: pd.DataFrame) -> list:
        categorical = resolve_categorical_features(X, None)
        if self.columns is not None:
            columns_ = [X.columns[c] if isinstance(c, (int, np.integer)) else c for c in self.columns]
            unknown = [c for c in columns_ if c not in X.columns]
            if unknown:
                raise ValueError(f"columns not found in X: {unknown}")
        else:
            columns_ = [c for c in X.columns if c in categorical]
        if not columns_:
            raise ValueError("CategoricalDepthTransformer needs at least 1 column, got none.")
        return columns_

    def _combined_index(self, X: pd.DataFrame) -> np.ndarray:
        combined = np.zeros(len(X), dtype=int)
        multiplier = 1
        for c in self.columns_:
            n_zones = len(self._category_maps_[c]) + 2  # + missing + unseen
            zone_idx = categorical_zone_index(X[c], self._category_maps_[c])
            combined = combined + zone_idx * multiplier
            multiplier *= n_zones
        return combined

    def fit(self, X, y=None):
        """Fit each column's category map and every joint cell's training
        support count.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        y : ignored
            Accepted for ``Pipeline`` compatibility only -- typicality is
            a property of ``X``'s own joint distribution.

        Returns
        -------
        self : CategoricalDepthTransformer
        """
        X = ensure_dataframe(X, getattr(self, "feature_names_in_", None))
        self.feature_names_in_ = np.array(X.columns)

        self.columns_ = self._resolve_columns(X)
        self._group_name = self.group_name if self.group_name is not None else "_".join(self.columns_)
        self._category_maps_ = {c: categorical_zone_map(X[c]) for c in self.columns_}

        combined = self._combined_index(X)
        n_cells = 1
        for c in self.columns_:
            n_cells *= len(self._category_maps_[c]) + 2
        self._counts_ = np.bincount(combined, minlength=n_cells)
        self._n_train_ = len(X)
        return self

    def transform(self, X) -> pd.DataFrame:
        """Look up each row's joint cell and its training support count.

        A combination unseen at ``fit`` time (in any column) gets
        ``count = 0``, ``coreness = 0.0`` -- an unseen combination is
        exactly as atypical as that gets, not an error.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)

        Returns
        -------
        DataFrame of shape (n_samples, 2)
            ``"<group>__count"`` (raw training support count for this
            row's joint category combination) and ``"<group>__coreness"``
            (``count / n_train``, bounded in ``[0, 1]``).
        """
        check_is_fitted(self, "_counts_")
        X = ensure_dataframe(X, self.feature_names_in_)
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"columns not found in X: {missing}")

        combined = self._combined_index(X)
        counts = self._counts_[combined].astype(float)
        coreness = counts / self._n_train_

        group = self._group_name
        return pd.DataFrame({f"{group}__count": counts, f"{group}__coreness": coreness}, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Output column names, in the same order :meth:`transform` emits
        them."""
        check_is_fitted(self, "_counts_")
        group = self._group_name
        return np.array([f"{group}__count", f"{group}__coreness"])
