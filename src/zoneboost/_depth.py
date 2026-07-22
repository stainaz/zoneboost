"""Statistical depth: a continuous "how typical is this observation"
score over a group of numeric columns, generalizing a discrete inner-
core/outer-core/outlier grouping into one bounded number instead of a
handful of hand-drawn rings.

Uses **Mahalanobis distance** -- a point's distance from the joint mean
of a column group, scaled by their covariance -- rather than Tukey
halfspace depth or convex-hull peeling. Both alternatives were considered
and rejected: exact halfspace depth has no simple closed form and is
combinatorially expensive past ~2 dimensions; convex-hull peeling needs
``scipy.spatial.ConvexHull``, a dependency this package doesn't otherwise
have. Mahalanobis distance is closed-form, numpy-only, and already
accounts for correlation between the group's columns the same way a
"region of interest" jointly defined over more than one variable would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._common import ensure_dataframe, resolve_categorical_features

__all__ = ["DepthTransformer"]


class DepthTransformer(BaseEstimator, TransformerMixin):
    """Continuous "coreness" score for a group of numeric columns.

    Fits one mean vector and one (ridge-regularized) covariance matrix
    over the declared column group, then emits each row's Mahalanobis
    distance from that center and a bounded rescaling of it. Unsupervised
    -- depth is a property of ``X``'s own joint distribution, not of any
    target.

    One instance encodes one column group. To get depth scores for more
    than one group, use several ``DepthTransformer`` instances inside a
    ``ColumnTransformer``/``FeatureUnion`` -- the same
    compose-rather-than-build-in precedent :class:`zoneboost.
    ZoneProfileEncoder` already sets.

    **Deferred**: no discrete region labels (no "inner core"/"outlier"
    bucketing) -- a continuous score composes with any downstream model,
    which can bin or threshold it itself if discrete regions are still
    wanted.

    Parameters
    ----------
    columns : list of str or int, default=None
        The column group to compute joint depth over. ``None`` (default)
        uses every numeric column of ``X`` (columns auto-detected or
        declared as categorical are excluded -- Mahalanobis distance has
        no meaning over a nominal category). Declaring a categorical
        column explicitly raises ``ValueError``.
    group_name : str, default=None
        Name used in the two emitted output columns. ``None`` (default)
        joins the encoded column names with ``"_"``.
    ridge : float, default=1e-6
        Added to the covariance matrix's diagonal before inverting, so a
        singular or ill-conditioned covariance (perfectly correlated
        columns, or more columns than rows) degrades gracefully rather
        than raising. The inverse itself is computed via
        ``np.linalg.pinv``, a further safeguard for the same case.
    random_state : int, default=42
        Accepted for interface consistency; fitting is fully
        deterministic given ``X``, so this is currently unused.

    Attributes
    ----------
    columns_ : list of str
        The columns actually encoded, in fitted order.
    mean_ : ndarray of shape (n_columns,)
        Fitted (weighted) mean vector.
    covariance_ : ndarray of shape (n_columns, n_columns)
        Fitted (weighted) covariance matrix, ridge-regularized.
    feature_names_in_ : ndarray of shape (n_features_in,)
        Every column name seen in ``X`` at ``fit`` time.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import DepthTransformer
    >>> X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100], "y": [1, 2, 3, 4, 5, 100]})
    >>> depth = DepthTransformer().fit(X)
    >>> out = depth.transform(X)
    >>> out["x_y__coreness"].iloc[-1] < out["x_y__coreness"].iloc[0]
    True
    """

    def __init__(
        self,
        columns=None,
        group_name: str = None,
        ridge: float = 1e-6,
        random_state: int = 42,
    ):
        self.columns = columns
        self.group_name = group_name
        self.ridge = ridge
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        """Fit the column group's mean vector and covariance matrix.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        y : ignored
            Accepted for ``Pipeline`` compatibility only -- depth is a
            property of ``X``'s own joint distribution.
        sample_weight : array-like of shape (n_samples,), default=None
            Per-row weight for the fitted mean/covariance. ``None``
            (default) weighs every row equally.

        Returns
        -------
        self : DepthTransformer
        """
        X = ensure_dataframe(X, getattr(self, "feature_names_in_", None))
        self.feature_names_in_ = np.array(X.columns)

        categorical = resolve_categorical_features(X, None)
        if self.columns is not None:
            columns_ = [X.columns[c] if isinstance(c, (int, np.integer)) else c for c in self.columns]
            unknown = [c for c in columns_ if c not in X.columns]
            if unknown:
                raise ValueError(f"columns not found in X: {unknown}")
            declared_categorical = [c for c in columns_ if c in categorical]
            if declared_categorical:
                raise ValueError(
                    f"DepthTransformer requires numeric columns; {declared_categorical} "
                    "are categorical. Mahalanobis distance has no meaning over a nominal "
                    "category."
                )
        else:
            columns_ = [c for c in X.columns if c not in categorical]
            if not columns_:
                raise ValueError("No numeric columns found in X to compute depth over.")
        self.columns_ = columns_

        values = X[columns_].to_numpy(dtype=float)
        sw = np.asarray(sample_weight, dtype=float).reshape(-1) if sample_weight is not None else np.ones(len(values))
        if len(values) != len(sw):
            raise ValueError(f"X and sample_weight have inconsistent lengths: {len(values)} vs {len(sw)}")

        mean = np.average(values, axis=0, weights=sw)
        centered = values - mean
        covariance = (centered * sw[:, None]).T @ centered / sw.sum()
        covariance = covariance + self.ridge * np.eye(len(columns_))

        self.mean_ = mean
        self.covariance_ = covariance
        self._inv_covariance = np.linalg.pinv(covariance)
        return self

    def transform(self, X) -> pd.DataFrame:
        """Compute each row's Mahalanobis distance from the fitted center
        and its bounded coreness rescaling.

        A missing value in any encoded column is mean-imputed with that
        column's own fitted mean before computing distance -- it
        contributes nothing to the distance along that axis, the same
        "carries zero confidence, doesn't crash" precedent
        :func:`zoneboost._zones.categorical_zone_index`'s dedicated
        missing zone already sets.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)

        Returns
        -------
        DataFrame of shape (n_samples, 2)
            ``"<group_name>__depth_distance"`` (raw Mahalanobis distance,
            0 at the fitted center, unbounded above) and
            ``"<group_name>__coreness"`` (``1 / (1 + distance)`` -- a
            bounded, monotonically-decreasing rescaling, **not** a
            calibrated percentile or probability).
        """
        check_is_fitted(self, "covariance_")
        X = ensure_dataframe(X, self.feature_names_in_)

        values = X[self.columns_].to_numpy(dtype=float)
        missing = np.isnan(values)
        if missing.any():
            values = np.where(missing, self.mean_, values)

        centered = values - self.mean_
        distance = np.sqrt(np.einsum("ij,jk,ik->i", centered, self._inv_covariance, centered))
        coreness = 1.0 / (1.0 + distance)

        name = self.group_name if self.group_name is not None else "_".join(self.columns_)
        return pd.DataFrame(
            {f"{name}__depth_distance": distance, f"{name}__coreness": coreness}, index=X.index
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Output column names, in the same order :meth:`transform`
        emits them."""
        check_is_fitted(self, "covariance_")
        name = self.group_name if self.group_name is not None else "_".join(self.columns_)
        return np.array([f"{name}__depth_distance", f"{name}__coreness"])
