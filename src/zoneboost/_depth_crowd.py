"""Aggregate several already-computed per-expert typicality scores -- e.g.
:class:`zoneboost.DepthTransformer`'s ``*__coreness`` columns from a
``FeatureUnion`` of experts over different column groups -- into a single
crowd signal: central tendency, disagreement, a vote count, and which
expert is driving an outlier.

Doesn't know or care that its input columns came from ``DepthTransformer``
specifically -- it treats them as an arbitrary set of comparable, bounded
per-expert scores, the same decoupling precedent :class:`zoneboost.
LLMZoneNamer` already set for ``zone_summaries``.

``DepthTransformer.coreness`` is explicitly documented as "not a
calibrated percentile" -- a bounded rescaling, not a distribution-
referenced score. Different ``DepthTransformer`` instances (different
column counts, different covariance structure) produce ``coreness`` on
different effective scales, so a raw cross-expert mean isn't apples-to-
apples. ``rank_normalize`` (default on) fixes this: at ``fit``, each input
column's training values are stored; at ``transform``, each value is
converted to its percentile within *that column's own fitted training
distribution* -- a fitted reference, not a percentile computed against
whatever rows happen to be in the current ``transform`` call, so it
doesn't shift under a small or skewed scoring batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._common import ensure_dataframe, resolve_categorical_features

__all__ = ["DepthCrowd"]


class DepthCrowd(BaseEstimator, TransformerMixin):
    """Crowd-level summary of several per-expert typicality scores.

    Each input column is treated as one "expert" opinion of how typical a
    row is (higher = more typical, e.g. :class:`zoneboost.DepthTransformer`'s
    ``coreness``). Reports the crowd's mean, median, standard deviation
    (disagreement), minimum (the most-alarmed single expert), a vote count/
    share of experts below ``vote_threshold``, and which expert is driving
    the lowest reading for that row.

    Fits the plain ``fit(X, y=None)`` / ``transform(X)`` shape every other
    transformer in this package uses, because its input *is* a normal
    per-row feature table -- typically several ``DepthTransformer``
    instances' ``*__coreness`` columns, produced upstream by a
    ``ColumnTransformer``/``FeatureUnion``. Unlike :class:`zoneboost.
    LaplaceHistoryTransformer`, it composes automatically inside a
    ``Pipeline`` the same way ``DepthTransformer`` itself does; no extra
    per-call arguments needed.

    **Deferred**: no per-expert learned reliability weights -- that needs
    held-out performance labels, a materially separate, larger feature. No
    supervised soft-voting/stacking over probability outputs -- already
    solved by sklearn's ``VotingClassifier``/``StackingClassifier``, not
    duplicated here. No bootstrap-covariance crowd (refitting one expert
    across resamples) -- a distinct future feature in the shape of
    :class:`zoneboost.BootstrapStability`, specifically around
    ``DepthTransformer``, not this class's job. No random-subspace zone
    generator -- the caller already builds ``DepthTransformer(columns=[...])``
    instances by hand or with their own random selection.

    Parameters
    ----------
    columns : list of str or int, default=None
        Input columns to treat as crowd experts. ``None`` (default) uses
        every numeric column of ``X`` (categorical columns auto-excluded;
        declaring one explicitly raises ``ValueError``), the same
        convention :class:`zoneboost.DepthTransformer`'s own
        ``columns=None`` sets. At least 2 columns are required -- a crowd
        of one isn't a crowd.
    rank_normalize : bool, default=True
        Convert each column to its percentile within that column's own
        fitted training distribution before aggregating. ``False`` uses
        the raw input values directly, which is only apples-to-apples
        across experts whose raw scores already share a scale.
    vote_threshold : float, default=0.05
        An expert "votes" atypical for a row when its (rank-normalized, if
        enabled) score is ``<= vote_threshold``. Calibrated for
        ``rank_normalize=True`` (bottom 5% of that expert's own training
        distribution); needs to be raised substantially when
        ``rank_normalize=False``, since a raw bounded score like
        ``coreness`` compresses toward 0 quickly and rarely dips below
        0.05 for realistic distances.
    group_name : str, default="crowd"
        Prefix on every emitted output column.

    Attributes
    ----------
    columns_ : list of str
        The columns actually treated as experts, in fitted order.
    feature_names_in_ : ndarray of shape (n_features_in,)

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import DepthCrowd
    >>> X = pd.DataFrame({
    ...     "viral__coreness": [0.9, 0.9, 0.1],
    ...     "immune__coreness": [0.8, 0.85, 0.8],
    ...     "treatment__coreness": [0.7, 0.75, 0.7],
    ... })
    >>> crowd = DepthCrowd(rank_normalize=False).fit(X)
    >>> out = crowd.transform(X)
    >>> out.loc[2, "crowd__most_atypical_expert"]
    'viral__coreness'
    """

    def __init__(
        self,
        columns=None,
        rank_normalize: bool = True,
        vote_threshold: float = 0.05,
        group_name: str = "crowd",
    ):
        self.columns = columns
        self.rank_normalize = rank_normalize
        self.vote_threshold = vote_threshold
        self.group_name = group_name

    def fit(self, X, y=None):
        """Store each expert column's training distribution.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
        y : ignored
            Accepted for ``Pipeline`` compatibility only.

        Returns
        -------
        self : DepthCrowd
        """
        if not 0.0 <= self.vote_threshold <= 1.0:
            raise ValueError(f"vote_threshold must be in [0, 1], got {self.vote_threshold!r}")

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
                    f"DepthCrowd requires numeric columns; {declared_categorical} are categorical."
                )
        else:
            columns_ = [c for c in X.columns if c not in categorical]

        if len(columns_) < 2:
            raise ValueError(f"DepthCrowd needs at least 2 columns to form a crowd, got {columns_!r}")
        self.columns_ = columns_

        values = X[columns_].to_numpy(dtype=float)
        self._reference_ = {c: np.sort(values[:, i]) for i, c in enumerate(columns_)}
        return self

    def _scores(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.columns_].to_numpy(dtype=float)
        if not self.rank_normalize:
            return values
        scores = np.empty_like(values)
        for i, c in enumerate(self.columns_):
            ref = self._reference_[c]
            scores[:, i] = np.searchsorted(ref, values[:, i], side="right") / len(ref)
        return scores

    def transform(self, X) -> pd.DataFrame:
        """Aggregate the crowd's expert columns into a single set of
        crowd-level scores.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Must contain every column in ``columns_``.

        Returns
        -------
        DataFrame of shape (n_samples, 7)
            ``"<group>__mean"``, ``"<group>__median"``, ``"<group>__std"``
            (disagreement), ``"<group>__min"`` (the most-alarmed single
            expert's reading), ``"<group>__vote_count"``/
            ``"<group>__vote_share"`` (experts at or below
            ``vote_threshold``), and ``"<group>__most_atypical_expert"``
            (the input column name with the lowest score for that row --
            a string label meant for human review/audit, droppable before
            feeding the rest into a model that needs purely numeric
            input).
        """
        check_is_fitted(self, "_reference_")
        X = ensure_dataframe(X, self.feature_names_in_)
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"columns not found in X: {missing}")

        scores = self._scores(X)
        votes = scores <= self.vote_threshold
        vote_count = votes.sum(axis=1)
        argmin = scores.argmin(axis=1)

        group = self.group_name
        return pd.DataFrame(
            {
                f"{group}__mean": scores.mean(axis=1),
                f"{group}__median": np.median(scores, axis=1),
                f"{group}__std": scores.std(axis=1),
                f"{group}__min": scores.min(axis=1),
                f"{group}__vote_count": vote_count,
                f"{group}__vote_share": vote_count / scores.shape[1],
                f"{group}__most_atypical_expert": np.array(self.columns_)[argmin],
            },
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Output column names, in the same order :meth:`transform` emits
        them."""
        check_is_fitted(self, "_reference_")
        group = self.group_name
        return np.array(
            [
                f"{group}__mean",
                f"{group}__median",
                f"{group}__std",
                f"{group}__min",
                f"{group}__vote_count",
                f"{group}__vote_share",
                f"{group}__most_atypical_expert",
            ]
        )
