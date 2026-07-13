"""Compile a fitted :class:`zoneboost.ZoneBoostRegressor` to a single SQL
``SELECT`` statement -- for in-warehouse scoring with no Python runtime,
no model-serving infrastructure, and no external dependency at query time.

The reduction is genuine, not a metaphor: every round's main effects and
pairwise interactions are already plain lookup tables (``rounds_``), and
``predict(X)`` is already just ``baseline_ + sum of learning_rate *
(round intercept + weighted term lookups)`` (see :func:`zoneboost.
regressor.ZoneBoostRegressor._raw_predict`). The one real subtlety:
production scoring uses a **soft, linearly-interpolated** zone lookup for
every continuous column (:func:`zoneboost._weak_learner.
_column_soft_zone_index`), not a plain hard lookup -- so an honestly
"lossless" SQL export must replicate that interpolation arithmetic too
(``CASE`` for the hard zone/centroid dispatch, then plain arithmetic for
the blend), not just emit a bare ``CASE WHEN x < b THEN v`` per zone.
That is exactly what this module does; see :func:`compile_to_sql`'s own
docstring for the full scope and disclosed limitations.
"""

from __future__ import annotations

import json

from sklearn.utils.validation import check_is_fitted

from ._evidence_card import evidence_card

__all__ = ["compile_to_sql"]


def _sql_num(x: float) -> str:
    """Format a float as a SQL numeric literal that round-trips exactly
    (``repr``-quality precision, not the default ``str`` rounding).
    Negative values are parenthesized -- ``a - -1.0`` written without a
    space (``a--1.0``) would otherwise be lexed as ``a`` followed by a
    ``--`` line comment, silently truncating the rest of the statement."""
    value = float(x)
    text = repr(value)
    return f"({text})" if value < 0 else text


def _sql_str(value) -> str:
    """Format a category value as a SQL literal -- single-quoted with
    internal quotes doubled for strings, a bare numeric literal
    otherwise (categorical columns may be declared on a numeric dtype)."""
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def _quote_ident(name: str) -> str:
    """Double-quoted SQL identifier, doubling any internal double quote --
    safe for column/table/alias names containing spaces or punctuation."""
    return '"' + str(name).replace('"', '""') + '"'


def _dispatch_zone(col_sql: str, zone_info: tuple, leaf_fn) -> str:
    """Emits one nested ``CASE`` expression dispatching a column's raw SQL
    value to whichever zone (and, for continuous columns, blend
    direction) it falls into -- the SQL mirror of
    :func:`zoneboost._weak_learner._column_soft_zone_index`.

    ``leaf_fn(zone_idx: int, neighbor_idx: int | None) -> str`` is called
    once per reachable leaf and must return the already-fully-resolved
    SQL expression for that leaf; ``neighbor_idx`` is ``None`` for
    missing values and categorical zones (no blend direction, matching
    ``weight_hi == 0`` there), and the compile-time-known neighbor zone
    index otherwise (a continuous real zone always has at least one
    neighbor unless it's the column's only zone, in which case both
    directions degenerate to ``neighbor_idx=None``).
    """
    kind = zone_info[0]
    if kind == "categorical":
        cat_map = zone_info[1]
        missing_idx = len(cat_map)
        unseen_idx = len(cat_map) + 1
        whens = " ".join(
            f"WHEN {col_sql} = {_sql_str(cat)} THEN {leaf_fn(idx, None)}" for cat, idx in cat_map.items()
        )
        return (
            f"(CASE WHEN {col_sql} IS NULL THEN {leaf_fn(missing_idx, None)} "
            f"{whens} ELSE {leaf_fn(unseen_idx, None)} END)"
        )

    boundaries, centers = zone_info[1], zone_info[2]
    n_real = len(centers)
    missing_idx = n_real

    def zone_direction_branch(i: int) -> str:
        own = centers[i]
        right_neighbor = i + 1 if i < n_real - 1 else None
        left_neighbor = i - 1 if i > 0 else None
        right_expr = leaf_fn(i, right_neighbor)
        left_expr = leaf_fn(i, left_neighbor)
        return f"(CASE WHEN {col_sql} > {_sql_num(own)} THEN {right_expr} ELSE {left_expr} END)"

    whens = " ".join(f"WHEN {col_sql} < {_sql_num(b)} THEN {zone_direction_branch(i)}" for i, b in enumerate(boundaries))
    return (
        f"(CASE WHEN {col_sql} IS NULL THEN {leaf_fn(missing_idx, None)} "
        f"{whens} ELSE {zone_direction_branch(n_real - 1)} END)"
    )


def _main_effect_leaf_fn(col_sql: str, deviation, zone_info: tuple):
    """``leaf_fn`` for :func:`_dispatch_zone` reproducing
    :func:`zoneboost._weak_learner._blend_1d` exactly: ``D[own] + w *
    (D[neighbor] - D[own])``, ``w`` the clipped, ``lam``-scaled distance
    from the value to its own zone's centroid."""
    lam = zone_info[3] if zone_info[0] == "continuous" and len(zone_info) > 3 else 1.0

    def leaf_fn(zone_idx: int, neighbor_idx) -> str:
        d_own = deviation[zone_idx]
        if neighbor_idx is None:
            return _sql_num(d_own)
        own = zone_info[2][zone_idx]
        neighbor_center = zone_info[2][neighbor_idx]
        denom = neighbor_center - own
        return (
            f"({_sql_num(d_own)} + MIN(MAX(({col_sql}-{_sql_num(own)})/{_sql_num(denom)},0.0),1.0) "
            f"* {_sql_num(lam)} * ({_sql_num(deviation[neighbor_idx])}-{_sql_num(d_own)}))"
        )

    return leaf_fn


def _pair_sql(col_a_sql: str, zone_info_a: tuple, col_b_sql: str, zone_info_b: tuple, deviation2d) -> str:
    """Nested ``CASE`` (column A's zone/direction dispatch, then column
    B's) whose leaf reproduces :func:`zoneboost._weak_learner._blend_2d`'s
    bilinear combination of the four surrounding cells exactly. A
    categorical/missing axis always has ``neighbor_idx=None`` (weight 0),
    so this collapses to plain 1D interpolation along whichever axis is
    actually continuous -- the identical degeneracy `_blend_2d` itself
    documents."""
    lam_a = zone_info_a[3] if zone_info_a[0] == "continuous" and len(zone_info_a) > 3 else 1.0
    lam_b = zone_info_b[3] if zone_info_b[0] == "continuous" and len(zone_info_b) > 3 else 1.0

    def leaf_a(za: int, neighbor_a):
        def leaf_b(zb: int, neighbor_b) -> str:
            d00 = deviation2d[za, zb]
            if neighbor_a is None:
                wa_expr = "0.0"
                d10 = d00
            else:
                own_a = zone_info_a[2][za]
                nbr_center_a = zone_info_a[2][neighbor_a]
                denom_a = nbr_center_a - own_a
                wa_expr = f"(MIN(MAX(({col_a_sql}-{_sql_num(own_a)})/{_sql_num(denom_a)},0.0),1.0)*{_sql_num(lam_a)})"
                d10 = deviation2d[neighbor_a, zb]
            if neighbor_b is None:
                wb_expr = "0.0"
                d01 = d00
            else:
                own_b = zone_info_b[2][zb]
                nbr_center_b = zone_info_b[2][neighbor_b]
                denom_b = nbr_center_b - own_b
                wb_expr = f"(MIN(MAX(({col_b_sql}-{_sql_num(own_b)})/{_sql_num(denom_b)},0.0),1.0)*{_sql_num(lam_b)})"
                d01 = deviation2d[za, neighbor_b]
            d11 = deviation2d[neighbor_a, neighbor_b] if (neighbor_a is not None and neighbor_b is not None) else d00
            return (
                f"(((1.0-{wa_expr})*(1.0-{wb_expr})*{_sql_num(d00)}) + "
                f"({wa_expr}*(1.0-{wb_expr})*{_sql_num(d10)}) + "
                f"((1.0-{wa_expr})*{wb_expr}*{_sql_num(d01)}) + "
                f"({wa_expr}*{wb_expr}*{_sql_num(d11)}))"
            )

        return _dispatch_zone(col_b_sql, zone_info_b, leaf_b)

    return _dispatch_zone(col_a_sql, zone_info_a, leaf_a)


def compile_to_sql(
    model,
    table_name: str = "input_table",
    score_alias: str = "score",
    offset_expr: str = "0",
    include_evidence_card: bool = False,
    dialect: str = "sqlite",
) -> str:
    """Compile a fitted :class:`zoneboost.ZoneBoostRegressor` to a single
    SQL ``SELECT`` statement computing the same prediction ``predict(X)``
    would, reading raw feature columns directly from ``table_name``.

    Reproduces the model's own soft, centroid-interpolated zone lookup
    (see the module docstring) via nested ``CASE`` plus arithmetic, not
    an approximate hard-zone lookup -- verified by literally executing
    the generated SQL (via Python's built-in ``sqlite3``) against real
    data and comparing to ``predict(X)`` directly, see
    ``tests/test_sql_export.py``.

    Parameters
    ----------
    model : ZoneBoostRegressor
        An already-fitted model.
    table_name : str, default="input_table"
        Source table name the generated ``SELECT`` reads from -- must
        have one column per predictor `model` was fit on, matching name
        and (compatible) type.
    score_alias : str, default="score"
        Output column alias for the computed prediction.
    offset_expr : str, default="0"
        A raw SQL expression for the per-row ``offset`` term (only
        meaningful for ``loss in ("poisson", "gamma", "tweedie")``) --
        e.g. ``"LN(exposure)"``. ``offset`` is not a fitted attribute
        (must be resupplied fresh, exactly like at `predict` time), so
        this can't be inferred; the default ``"0"`` matches "no exposure
        adjustment."
    include_evidence_card : bool, default=False
        Prepend ``model.evidence_card()``'s JSON as a leading ``/* ...
        */`` SQL comment block -- the model-risk artifact attached
        alongside the deployable SQL, unchanged from
        :func:`zoneboost.evidence_card`.
    dialect : str, default="sqlite"
        The only value currently accepted. Clipping uses SQLite's scalar
        (2-argument) ``MIN``/``MAX`` idiom (also supported by DuckDB,
        MySQL 8+); PostgreSQL/Snowflake/BigQuery/Redshift use
        ``LEAST``/``GREATEST`` instead and would need a small
        dialect-specific rewrite -- not attempted in this pass.

    Returns
    -------
    str
        A complete, executable SQL statement.

    Scope
    -----
    **Main effects and pairwise interactions only** -- raises
    ``ValueError`` if any deployed round has a non-empty 3-way
    interaction (``triples``), rather than silently dropping that
    signal; refit with ``max_interaction_order=2`` (the default) to stay
    in scope. **Regressor only** -- ``ZoneBoostClassifier`` is not
    supported (binary reuses an identical ``rounds_`` shape but adds a
    sigmoid; native multiclass softmax is a materially different,
    deferred problem). ``effect_overrides_`` (audited human editing) is
    not supported -- raises ``ValueError`` if any are set. ``sample_
    weight`` is not applicable (not supported by any loss yet).

    SQL size scales with ``(rounds actually deployed) x (main effects +
    pairs) x (zones, or zones**2 for a pair)`` -- every round
    independently re-derives its own continuous-column zone boundaries
    from that round's residual, so there is no way to consolidate
    lookups across rounds into one shared table per column. This mirrors
    the same size characteristic any gradient-boosted-ensemble-to-SQL
    compiler has. In practice this feature suits the traditional
    "scorecard" use case directly -- a small, curated model (few rounds,
    few columns, often with `max_pair_interactions` capped) -- rather
    than a deep, wide, default-configured ensemble.
    """
    check_is_fitted(model, "rounds_")
    if dialect != "sqlite":
        raise ValueError(f"dialect must be 'sqlite', got {dialect!r}")
    if model.effect_overrides_:
        raise ValueError(
            "compile_to_sql does not support effect_overrides_ (deferred) -- "
            "clear overrides before compiling."
        )
    deployed_rounds = model.rounds_[: model.best_n_rounds_]
    for round_ in deployed_rounds:
        if round_["triples"]:
            raise ValueError(
                "compile_to_sql does not support 3-way interactions (deferred) -- "
                "refit with max_interaction_order=2 (the default) to stay in scope."
            )

    round_terms_sql = []
    for round_ in deployed_rounds:
        zone_info = round_["zone_info"]
        weights = round_["weights"]
        weight_idx = 0
        term_pieces = []
        for col, deviation in round_["main_effects"].items():
            col_sql = _quote_ident(col)
            leaf_fn = _main_effect_leaf_fn(col_sql, deviation, zone_info[col])
            expr = _dispatch_zone(col_sql, zone_info[col], leaf_fn)
            term_pieces.append(f"({_sql_num(weights[weight_idx])} * {expr})")
            weight_idx += 1
        for (a, b), deviation2d in round_["interactions"].items():
            expr = _pair_sql(_quote_ident(a), zone_info[a], _quote_ident(b), zone_info[b], deviation2d)
            term_pieces.append(f"({_sql_num(weights[weight_idx])} * {expr})")
            weight_idx += 1

        round_body = " + ".join(term_pieces) if term_pieces else "0.0"
        round_contrib = f"({_sql_num(round_['intercept'])} + {round_body})"
        round_terms_sql.append(f"({_sql_num(model.learning_rate)} * {round_contrib})")

    link_sum = f"({_sql_num(model.baseline_)} + {' + '.join(round_terms_sql)})" if round_terms_sql else _sql_num(
        model.baseline_
    )

    if model.loss in ("poisson", "gamma", "tweedie"):
        score_sql = f"EXP({link_sum} + ({offset_expr}))"
    else:
        score_sql = link_sum

    header = ""
    if include_evidence_card:
        card_json = json.dumps(evidence_card(model), indent=2)
        header = f"/*\nzoneboost evidence card:\n{card_json}\n*/\n"

    return f"{header}SELECT {score_sql} AS {_quote_ident(score_alias)} FROM {_quote_ident(table_name)};"
