"""Exponential (half-life) decay-weighted aggregation of a per-entity event
history -- claims, purchases, fraud events, sensor readings, anything with
an entity id, a timestamp, and (optionally) a value. Gives a model memory of
"what happened before now" without a recurrent/attention architecture.

Structurally different from :class:`zoneboost.ZoneProfileEncoder`,
:class:`zoneboost.DepthTransformer`, and :class:`zoneboost.
ConditionalZoneGrid`: those three all turn *static* columns of ``X`` into
new columns of the same ``X``, and their ``transform(X)`` takes no argument
beyond ``X`` -- so they drop into a ``ColumnTransformer``/``FeatureUnion``/
``Pipeline`` and get called automatically at both fit and predict time. This
transformer needs a second, long-format table (one row per historical
event, not one row per prediction instance) at fit time -- supplied as a
dedicated ``fit(X, y=None, event_history=...)`` keyword rather than folded
into ``X`` itself or bound into ``__init__`` (binding a DataFrame into
``__init__`` would break sklearn's own ``clone()``/``get_params()``
convention that constructors hold hyperparameters only) -- and, more
consequentially, its ``transform`` also requires a per-call ``asof_col`` or
``cutoff_date`` that a plain ``transform(X)`` has nowhere to carry. That
means it does **not** drop into an automatic ``ColumnTransformer``/
``Pipeline`` step the way the other three do: call ``fit``/``transform``
directly and concatenate the resulting columns onto ``X`` before handing
the combined table to a ``Pipeline`` or model, the same manual-merge usage
the feature was originally proposed with.

The critical correctness rule for any point-in-time feature: a row may only
see events strictly at or before its own "as of" moment, never after --
otherwise future events leak into a training row's features. A single global
cutoff shared by every row (as a naive ``cutoff_date`` parameter would
imply) is only safe when every row really is being scored at one shared
moment (e.g. a production batch score). It is not safe for a historical
training set built from rows at many different past dates, so the primary
``transform`` API takes a **per-row** ``asof_col`` instead, filtering each
row's own entity history to that row's own asof independently. A constant
``cutoff_date`` remains available as a convenience for the single-moment
case, implemented as a broadcast of one asof value to every row -- it does
not bypass the per-row filter, it just gives every row the same one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._common import ensure_dataframe

__all__ = ["LaplaceHistoryTransformer"]


def _resolve_name(df: pd.DataFrame, declared, label: str) -> str:
    name = df.columns[declared] if isinstance(declared, (int, np.integer)) else declared
    if name not in df.columns:
        raise ValueError(f"{label} not found: {name!r}")
    return name


def _to_numeric_time(values, label: str) -> np.ndarray:
    """Coerce a datetime-like or already-numeric series to float days (for
    datetimes: days since the Unix epoch; for numeric input: passed through
    unchanged, on the assumption it's already expressed in the same unit as
    ``half_lives`` -- disclosed in the class docstring rather than silently
    guessed at).
    """
    series = pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        result = series.to_numpy(dtype=float)
    else:
        parsed = pd.to_datetime(series, errors="coerce")
        as_days = parsed.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / 1e9 / 86400.0
        result = np.where(parsed.isna().to_numpy(), np.nan, as_days)
    if np.isnan(result).any():
        raise ValueError(f"{label} contains missing or unparseable time values.")
    return result


def _fmt_half_life(h: float) -> str:
    return str(int(h)) if float(h).is_integer() else str(h)


class LaplaceHistoryTransformer(BaseEstimator, TransformerMixin):
    """Decay-weighted per-entity event history, evaluated at each row's own
    point in time.

    For each output row (identified by ``entity_col`` and its own asof
    moment), looks up that entity's events strictly at or before the asof,
    and for every declared half-life computes an exponential-decay-weighted
    sum of each ``value_columns`` entry plus a decay-weighted event count --
    the amount and frequency halves of a standard RFM-style feature, sharing
    one computation. A half-life of ``h`` means an event exactly ``h`` time
    units in the past carries half the weight of an event right at the
    asof moment; weight decays as ``exp(-ln(2) * age / h)``.

    Unlike :class:`zoneboost.ZoneProfileEncoder`/:class:`zoneboost.
    DepthTransformer`/:class:`zoneboost.ConditionalZoneGrid`, ``fit`` does
    not compute any decayed aggregate -- decay depends on ``asof -
    event_time``, which is only known per output row at ``transform`` time.
    ``fit`` only validates and indexes the raw ``event_history`` (sorted per
    entity) so ``transform`` can look it up efficiently. Because the filter
    to "events at or before this row's own asof" is applied independently
    per row at ``transform`` time, the same fitted ``event_history`` --
    which may well contain events later than some rows' asof -- is safe to
    reuse across an entire historical training set without slicing it per
    fold or per row first; the row-level filter is the actual leakage
    guard, not how much of the table happens to be in memory.

    One instance encodes one event log against one set of value columns and
    half-lives. For more than one event log (e.g. claims and separately
    purchases), use multiple ``LaplaceHistoryTransformer`` instances and
    concatenate their outputs -- **not** automatically composed inside a
    ``ColumnTransformer``/``FeatureUnion`` the way :class:`zoneboost.
    ZoneProfileEncoder`/:class:`zoneboost.DepthTransformer`/:class:`zoneboost.
    ConditionalZoneGrid` are, since ``transform`` needs a per-call
    ``asof_col``/``cutoff_date`` those tools have no way to pass through;
    call each instance's ``fit``/``transform`` directly instead.
    ``group_name`` still disambiguates their output columns from each
    other, the same role it plays for ``DepthTransformer``.

    **Deferred**: a single ``entity_col`` only -- no multi-key entity joins
    (e.g. customer x product history), a materially larger feature. No
    automatic half-life selection -- caller declares them explicitly, same
    as ``max_zones``/``min_zone_frac`` elsewhere in this codebase. No
    integration into ``ZoneBoostRegressor``/``ZoneBoostClassifier``'s own
    ``fit``/``explain()`` -- a standalone transformer whose output columns
    are fed in like any other feature.

    Parameters
    ----------
    entity_col : str or int
        Column in ``event_history`` (and in ``X``) identifying an event's
        owner.
    time_col : str or int
        Column in ``event_history`` giving each event's timestamp.
        Datetime-like columns are converted to days since the Unix epoch;
        already-numeric columns are used as-is and must already be
        expressed in the same unit as ``half_lives``.
    value_columns : list of str or int, default=None
        Numeric columns in ``event_history`` to amount-decay. ``None``
        (default) or ``[]`` produces no amount columns -- the decay-
        weighted event count is still emitted, a pure recency/frequency
        signal.
    half_lives : list of float
        Positive half-lives, in the same time unit as ``time_col`` (days,
        for a datetime ``time_col``). One amount-decay column per
        ``(value_column, half_life)`` pair, plus one decay-weighted event
        count column per half-life.
    group_name : str, default=None
        Name used in every emitted output column, disambiguating multiple
        ``LaplaceHistoryTransformer`` instances (e.g. one per event log)
        used together. ``None`` (default) uses ``time_col`` -- distinct
        event logs almost always have distinctly-named timestamp columns,
        even when they happen to share a ``value_columns`` name (e.g. two
        logs both tracking an "amount").
    random_state : int, default=42
        Accepted for interface consistency; fitting and transforming are
        both fully deterministic given their inputs, so this is currently
        unused.

    Attributes
    ----------
    entity_col_ : str
    time_col_ : str
    value_columns_ : list of str
    feature_names_in_ : ndarray of shape (n_features_in,)
        Every column name seen in ``X`` at ``fit`` time.

    Examples
    --------
    >>> import pandas as pd
    >>> from zoneboost import LaplaceHistoryTransformer
    >>> events = pd.DataFrame({
    ...     "customer_id": [1, 1],
    ...     "claim_date": pd.to_datetime(["2026-01-01", "2026-06-01"]),
    ...     "claim_amount": [1000.0, 2000.0],
    ... })
    >>> X = pd.DataFrame({
    ...     "customer_id": [1, 1],
    ...     "snapshot_date": pd.to_datetime(["2026-03-01", "2026-07-01"]),
    ... })
    >>> laplace = LaplaceHistoryTransformer(
    ...     entity_col="customer_id", time_col="claim_date",
    ...     value_columns=["claim_amount"], half_lives=[30, 365],
    ... ).fit(X, event_history=events)
    >>> out = laplace.transform(X, asof_col="snapshot_date")
    >>> out.loc[0, "claim_date__claim_amount__laplace_30d"] < out.loc[1, "claim_date__claim_amount__laplace_30d"]
    True
    """

    def __init__(
        self,
        entity_col,
        time_col,
        half_lives,
        value_columns=None,
        group_name: str = None,
        random_state: int = 42,
    ):
        self.entity_col = entity_col
        self.time_col = time_col
        self.half_lives = half_lives
        self.value_columns = value_columns
        self.group_name = group_name
        self.random_state = random_state

    def fit(self, X, y=None, event_history=None):
        """Validate and index ``event_history`` per entity.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Accepted for ``Pipeline``/``ColumnTransformer`` interface
            compatibility; only its columns are recorded as
            ``feature_names_in_``.
        y : ignored
        event_history : DataFrame
            Required. Long-format event log: one row per event, with
            ``entity_col``, ``time_col``, and every ``value_columns`` entry.

        Returns
        -------
        self : LaplaceHistoryTransformer
        """
        if event_history is None:
            raise ValueError("event_history is required.")
        X = ensure_dataframe(X, getattr(self, "feature_names_in_", None))
        self.feature_names_in_ = np.array(X.columns)

        if not self.half_lives or any(h <= 0 for h in self.half_lives):
            raise ValueError(f"half_lives must be a non-empty list of positive numbers, got {self.half_lives!r}")
        self.half_lives_ = list(self.half_lives)

        event_history = ensure_dataframe(event_history)
        entity_col_ = _resolve_name(event_history, self.entity_col, "entity_col")
        time_col_ = _resolve_name(event_history, self.time_col, "time_col")
        declared_values = self.value_columns or []
        value_columns_ = [_resolve_name(event_history, c, "value_columns") for c in declared_values]
        non_numeric = [c for c in value_columns_ if not pd.api.types.is_numeric_dtype(event_history[c])]
        if non_numeric:
            raise ValueError(f"value_columns must be numeric; {non_numeric} are not.")

        self.entity_col_ = entity_col_
        self.time_col_ = time_col_
        self.value_columns_ = value_columns_
        self._group_name = self.group_name if self.group_name is not None else time_col_

        times = _to_numeric_time(event_history[time_col_], "time_col")
        entities = event_history[entity_col_].to_numpy()
        values_by_col = {c: event_history[c].fillna(0.0).to_numpy(dtype=float) for c in value_columns_}

        entity_index = {}
        for entity_value, idx in pd.Series(np.arange(len(event_history))).groupby(entities):
            idx = idx.to_numpy()
            order = np.argsort(times[idx], kind="stable")
            sorted_idx = idx[order]
            entity_index[entity_value] = {
                "times": times[sorted_idx],
                "values": {c: values_by_col[c][sorted_idx] for c in value_columns_},
            }
        self._entity_index = entity_index
        return self

    def transform(self, X, asof_col=None, cutoff_date=None) -> pd.DataFrame:
        """Look up each row's entity history, restrict to events at or
        before that row's own asof, and compute decay-weighted amount and
        count features per half-life.

        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Must contain ``entity_col`` and (when used) ``asof_col``.
        asof_col : str or int, default=None
            Column in ``X`` giving each row's own point-in-time cutoff.
            Exactly one of ``asof_col``/``cutoff_date`` must be given.
        cutoff_date : default=None
            A single value (datetime-like or numeric, matching ``time_col``)
            broadcast as every row's asof -- a convenience for scoring an
            entire batch "as of" one shared moment. Still filtered per row
            through the same entity-history lookup as ``asof_col``.

        Returns
        -------
        DataFrame of shape (n_samples, len(value_columns) * len(half_lives) + len(half_lives) + 1)
            ``"<group_name>__<value_col>__laplace_<half_life>d"`` per
            ``(value_column, half_life)`` pair, ``"<group_name>__event_decay_count_<half_life>d"``
            per half-life, and ``"<group_name>__had_history"`` (1 when the
            entity had at least one qualifying event before its own asof, 0
            for a cold-start entity -- whose decay columns are then exactly
            ``0.0``, an empty sum, not ``NaN``).
        """
        check_is_fitted(self, "_entity_index")
        if (asof_col is None) == (cutoff_date is None):
            raise ValueError("transform requires exactly one of asof_col or cutoff_date.")
        X = ensure_dataframe(X, self.feature_names_in_)
        if self.entity_col_ not in X.columns:
            raise ValueError(f"entity_col {self.entity_col_!r} not found in X.")

        if asof_col is not None:
            asof_name = X.columns[asof_col] if isinstance(asof_col, (int, np.integer)) else asof_col
            if asof_name not in X.columns:
                raise ValueError(f"asof_col {asof_name!r} not found in X.")
            asof_values = _to_numeric_time(X[asof_name], "asof_col")
        else:
            asof_values = np.full(len(X), _to_numeric_time(pd.Series([cutoff_date]), "cutoff_date")[0])

        entities = X[self.entity_col_].to_numpy()
        n = len(X)
        n_half = len(self.half_lives_)
        s_values = np.array([np.log(2.0) / h for h in self.half_lives_])

        amount_out = {c: np.zeros((n, n_half)) for c in self.value_columns_}
        count_out = np.zeros((n, n_half))
        had_history = np.zeros(n, dtype=int)

        for entity_value, idx in pd.Series(np.arange(n)).groupby(entities):
            entry = self._entity_index.get(entity_value)
            if entry is None:
                continue
            times = entry["times"]
            for i in idx.to_numpy():
                cutoff_idx = np.searchsorted(times, asof_values[i], side="right")
                if cutoff_idx == 0:
                    continue
                had_history[i] = 1
                age = asof_values[i] - times[:cutoff_idx]
                weights = np.exp(-np.outer(s_values, age))  # (n_half, cutoff_idx)
                count_out[i] = weights.sum(axis=1)
                for c in self.value_columns_:
                    amount_out[c][i] = weights @ entry["values"][c][:cutoff_idx]

        group = self._group_name
        data = {}
        for c in self.value_columns_:
            for j, h in enumerate(self.half_lives_):
                data[f"{group}__{c}__laplace_{_fmt_half_life(h)}d"] = amount_out[c][:, j]
        for j, h in enumerate(self.half_lives_):
            data[f"{group}__event_decay_count_{_fmt_half_life(h)}d"] = count_out[:, j]
        data[f"{group}__had_history"] = had_history
        return pd.DataFrame(data, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Output column names, in the same order :meth:`transform` emits
        them."""
        check_is_fitted(self, "_entity_index")
        group = self._group_name
        names = [
            f"{group}__{c}__laplace_{_fmt_half_life(h)}d" for c in self.value_columns_ for h in self.half_lives_
        ]
        names += [f"{group}__event_decay_count_{_fmt_half_life(h)}d" for h in self.half_lives_]
        names.append(f"{group}__had_history")
        return np.array(names)
