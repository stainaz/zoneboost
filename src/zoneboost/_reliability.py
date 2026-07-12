"""Explanation reliability diagnostics: for every term `explain()` reports
a contribution for, how much training support actually backed it, how much
empirical-Bayes shrinkage pulled it toward its prior, how stable it was
across cross-fitting folds, and whether a row sits beyond the zones the
model ever fit -- consumed by ``explain(X, include_reliability=True)``.

``support``/``shrinkage_fraction``/``cross_fold_std`` require the model to
have been fit with ``track_reliability=True`` (they read the ``counts``/
``fold_std`` arrays :func:`zoneboost._weak_learner.weak_learner_fit` only
computes then, stored per round as ``round_["diagnostics"]``).
``n_rounds_present``/``boundary_weight``/``extrapolation_frac`` need no
such flag -- they're derived from ``zone_info``, which is always retained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._weak_learner import _column_soft_zone_index, _column_zone_index

__all__ = ["explain_reliability", "evidence_report"]


def _aggregate_main_effect(X: pd.DataFrame, rounds: list, col: str, shrinkage_m: float) -> pd.DataFrame:
    n = len(X)
    support_sum = np.zeros(n)
    shrink_sum = np.zeros(n)
    fold_std_sum = np.zeros(n)
    fold_std_rounds = 0
    boundary_sum = np.zeros(n)
    extrap_sum = np.zeros(n)
    n_present = 0
    n_continuous_present = 0

    for round_ in rounds:
        if col not in round_["main_effects"]:
            continue
        n_present += 1
        zone_info_col = round_["zone_info"][col]
        z = _column_zone_index(X[col], zone_info_col)
        counts = round_["diagnostics"]["main_effects"][col]["counts"]
        c = counts[z]
        support_sum += c
        shrink_sum += shrinkage_m / (c + shrinkage_m)
        fstd = round_["diagnostics"]["main_effects"][col]["fold_std"]
        if fstd is not None:
            fold_std_sum += fstd[z]
            fold_std_rounds += 1

        if zone_info_col[0] == "continuous":
            n_continuous_present += 1
            _, _, weight_hi = _column_soft_zone_index(X[col], zone_info_col)
            boundary_sum += weight_hi
            centers = zone_info_col[2]
            x_arr = np.asarray(X[col], dtype=float)
            out_of_range = (x_arr < centers.min()) | (x_arr > centers.max())
            extrap_sum += out_of_range.astype(float)

    denom = max(n_present, 1)
    result = pd.DataFrame(index=X.index)
    result["support"] = support_sum / denom
    result["shrinkage_fraction"] = shrink_sum / denom
    result["cross_fold_std"] = fold_std_sum / fold_std_rounds if fold_std_rounds > 0 else np.nan
    result["n_rounds_present"] = n_present
    if n_continuous_present > 0:
        result["boundary_weight"] = boundary_sum / n_continuous_present
        result["extrapolation_frac"] = extrap_sum / n_continuous_present
    return result


def _find_term_key(term_dict: dict, term_id: frozenset):
    return next((k for k in term_dict if frozenset(k) == term_id), None)


def _aggregate_pair(X: pd.DataFrame, rounds: list, pair_id: frozenset, shrinkage_m: float) -> pd.DataFrame:
    n = len(X)
    support_sum = np.zeros(n)
    shrink_sum = np.zeros(n)
    fold_std_sum = np.zeros(n)
    fold_std_rounds = 0
    n_present = 0

    for round_ in rounds:
        key = _find_term_key(round_["interactions"], pair_id)
        if key is None:
            continue
        a, b = key
        n_present += 1
        za = _column_zone_index(X[a], round_["zone_info"][a])
        zb = _column_zone_index(X[b], round_["zone_info"][b])
        counts = round_["diagnostics"]["interactions"][key]["counts"]
        c = counts[za, zb]
        support_sum += c
        shrink_sum += shrinkage_m / (c + shrinkage_m)
        fstd = round_["diagnostics"]["interactions"][key]["fold_std"]
        if fstd is not None:
            fold_std_sum += fstd[za, zb]
            fold_std_rounds += 1

    denom = max(n_present, 1)
    result = pd.DataFrame(index=X.index)
    result["support"] = support_sum / denom
    result["shrinkage_fraction"] = shrink_sum / denom
    result["cross_fold_std"] = fold_std_sum / fold_std_rounds if fold_std_rounds > 0 else np.nan
    result["n_rounds_present"] = n_present
    return result


def _aggregate_triple(X: pd.DataFrame, rounds: list, triple_id: frozenset, shrinkage_m: float) -> pd.DataFrame:
    n = len(X)
    support_sum = np.zeros(n)
    shrink_sum = np.zeros(n)
    fold_std_sum = np.zeros(n)
    fold_std_rounds = 0
    n_present = 0

    for round_ in rounds:
        key = _find_term_key(round_["triples"], triple_id)
        if key is None:
            continue
        a, b, c = key
        n_present += 1
        za = _column_zone_index(X[a], round_["zone_info"][a])
        zb = _column_zone_index(X[b], round_["zone_info"][b])
        zc = _column_zone_index(X[c], round_["zone_info"][c])
        counts = round_["diagnostics"]["triples"][key]["counts"]
        cnt = counts[za, zb, zc]
        support_sum += cnt
        shrink_sum += shrinkage_m / (cnt + shrinkage_m)
        fstd = round_["diagnostics"]["triples"][key]["fold_std"]
        if fstd is not None:
            fold_std_sum += fstd[za, zb, zc]
            fold_std_rounds += 1

    denom = max(n_present, 1)
    result = pd.DataFrame(index=X.index)
    result["support"] = support_sum / denom
    result["shrinkage_fraction"] = shrink_sum / denom
    result["cross_fold_std"] = fold_std_sum / fold_std_rounds if fold_std_rounds > 0 else np.nan
    result["n_rounds_present"] = n_present
    return result


def explain_reliability(X: pd.DataFrame, rounds: list, shrinkage_m: float) -> dict:
    """One reliability DataFrame per term appearing in any of ``rounds`` --
    keyed identically to :func:`zoneboost._explain.explain_rounds`'s own
    column names (a predictor's own name for main effects, ``"A x B"``/
    ``"A x B x C"`` for interactions/triples, sorted alphabetically), so a
    caller can line a term's contribution and its reliability report up by
    name directly.

    Every round may fit a different subset of columns/pairs/triples (row/
    column subsampling, pair screening, adaptive triple selection) -- a
    term's own reliability is aggregated (averaged) only over the rounds
    that actually included it; ``n_rounds_present`` reports how many of the
    model's total rounds that was.
    """
    reliability = {}

    main_cols = set()
    for round_ in rounds:
        main_cols.update(round_["main_effects"].keys())
    for col in sorted(main_cols):
        reliability[col] = _aggregate_main_effect(X, rounds, col, shrinkage_m)

    pair_ids = {}
    for round_ in rounds:
        for key in round_["interactions"]:
            pair_ids.setdefault(frozenset(key), key)
    for pair_id, key in pair_ids.items():
        reliability[" x ".join(sorted(key))] = _aggregate_pair(X, rounds, pair_id, shrinkage_m)

    triple_ids = {}
    for round_ in rounds:
        for key in round_["triples"]:
            triple_ids.setdefault(frozenset(key), key)
    for triple_id, key in triple_ids.items():
        reliability[" x ".join(sorted(key))] = _aggregate_triple(X, rounds, triple_id, shrinkage_m)

    return reliability


def evidence_report(
    contrib: pd.DataFrame, reliability: dict, shrinkage_m: float, sparse_threshold: float = None
) -> pd.DataFrame:
    """Per-prediction "evidence quality" summary: a single row-level report
    combining every term's own reliability (see :func:`explain_reliability`)
    into one signal for "should this specific prediction be trusted" --
    distinct from that function's per-*term* detail.

    ``sparse_threshold`` (default ``shrinkage_m`` itself -- the empirical-
    Bayes half-trust point, where a zone's ``shrinkage_fraction`` is
    exactly `0.5`) is the average-``support`` cutoff below which a term's
    contribution counts as coming from a "sparse" cell.

    Parameters
    ----------
    contrib : DataFrame
        ``explain(X)``'s own output (one column per term plus
        ``"baseline"``, and ``"_softmax_centering"`` for a multiclass
        class's own table -- both excluded from the calculations below).
    reliability : dict
        ``explain_reliability(X, rounds, shrinkage_m)``'s own output.
    shrinkage_m : float
    sparse_threshold : float, default=None

    Returns
    -------
    DataFrame indexed like ``contrib``, columns:
        ``extrapolating`` (bool), ``unobserved_cell`` (bool),
        ``pct_contribution_from_sparse_cells`` (float, 0-1),
        ``evidence_score`` (float, 0-1) and ``evidence_quality``
        (categorical ``"Low"``/``"Medium"``/``"High"``) -- an honestly
        disclosed heuristic combination of the above into one number, not
        a calibrated statistical score.
    """
    threshold = sparse_threshold if sparse_threshold is not None else shrinkage_m
    terms = [c for c in contrib.columns if c not in ("baseline", "_softmax_centering")]

    n = len(contrib)
    total_abs = np.zeros(n)
    sparse_abs = np.zeros(n)
    extrapolating = np.zeros(n, dtype=bool)
    unobserved_cell = np.zeros(n, dtype=bool)

    for term in terms:
        rel = reliability[term]
        abs_contrib = contrib[term].to_numpy(dtype=float)
        abs_contrib = np.abs(abs_contrib)
        support = rel["support"].to_numpy(dtype=float)

        total_abs += abs_contrib
        sparse_abs += np.where(support < threshold, abs_contrib, 0.0)
        unobserved_cell |= support < 1.0
        if "extrapolation_frac" in rel.columns:
            extrapolating |= rel["extrapolation_frac"].to_numpy(dtype=float) > 0

    pct_sparse = np.divide(sparse_abs, total_abs, out=np.zeros(n), where=total_abs > 0)
    evidence_score = 1.0 - pct_sparse
    evidence_score = np.where(extrapolating, evidence_score * 0.5, evidence_score)

    result = pd.DataFrame(
        {
            "extrapolating": extrapolating,
            "unobserved_cell": unobserved_cell,
            "pct_contribution_from_sparse_cells": pct_sparse,
            "evidence_score": evidence_score,
        },
        index=contrib.index,
    )
    result["evidence_quality"] = pd.cut(
        result["evidence_score"], bins=[-np.inf, 0.5, 0.8, np.inf], labels=["Low", "Medium", "High"]
    )
    return result
