"""One boosting round's weak learner: per-column zone info, main effects
(single-variable zone -> average residual), interactions (variable-pair
zone grid -> average residual), an adaptively-selected small set of 3-way
interactions, and density-based confidence weighting.

Every "weak learner" in ZoneBoostRegressor is built from this module alone
-- no decision tree, no gradient computation beyond a plain residual, no
external model of any kind. What changes round to round is only the target
these functions are pointed at (the current residual) and which rows/
columns were sampled for that round.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ._zones import adaptive_zone_boundaries, categorical_zone_index, categorical_zone_map, zone_index

__all__ = ["weak_learner_fit", "weak_learner_score"]


def _zone_deviation_confidence(zone_values: np.ndarray, target_values: np.ndarray, overall_mean: float, n_zones: int):
    """For each zone: the actual average target among fit-rows in that
    zone (minus the overall mean), and a density-confidence weight
    relative to the best-supported zone. O(n) via bincount."""
    counts = np.bincount(zone_values, minlength=n_zones).astype(float)
    sums = np.bincount(zone_values, weights=target_values, minlength=n_zones)
    cell_mean = np.where(counts > 0, np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0), overall_mean)
    deviation = cell_mean - overall_mean
    base = counts.max()
    confidence = counts / base if base > 0 else counts
    return deviation, confidence


def _pair_deviation_confidence(
    za: np.ndarray, zb: np.ndarray, target_values: np.ndarray, overall_mean: float, n_zones_a: int, n_zones_b: int
):
    """Same idea, gridded over two variables' zones jointly. Combines both
    zone indices into one flat index and does a single bincount pass."""
    combined = za * n_zones_b + zb
    size = n_zones_a * n_zones_b
    counts = np.bincount(combined, minlength=size).astype(float).reshape(n_zones_a, n_zones_b)
    sums = np.bincount(combined, weights=target_values, minlength=size).reshape(n_zones_a, n_zones_b)
    cell_mean = np.where(counts > 0, np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0), overall_mean)
    deviation = cell_mean - overall_mean
    base = counts.max()
    confidence = counts / base if base > 0 else counts
    return deviation, confidence


def _triple_deviation_confidence(
    za: np.ndarray,
    zb: np.ndarray,
    zc: np.ndarray,
    target_values: np.ndarray,
    overall_mean: float,
    n_zones_a: int,
    n_zones_b: int,
    n_zones_c: int,
):
    """Same idea as :func:`_pair_deviation_confidence`, gridded over three
    variables' zones jointly -- one more flattened axis, still a single
    bincount pass."""
    combined = (za * n_zones_b + zb) * n_zones_c + zc
    size = n_zones_a * n_zones_b * n_zones_c
    counts = np.bincount(combined, minlength=size).astype(float).reshape(n_zones_a, n_zones_b, n_zones_c)
    sums = np.bincount(combined, weights=target_values, minlength=size).reshape(n_zones_a, n_zones_b, n_zones_c)
    cell_mean = np.where(counts > 0, np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0), overall_mean)
    deviation = cell_mean - overall_mean
    base = counts.max()
    confidence = counts / base if base > 0 else counts
    return deviation, confidence


def _term_importance(deviation: np.ndarray, confidence: np.ndarray) -> float:
    """A term's own confidence-weighted average magnitude -- used to rank
    pairs/triples by how much signal they carry, independent of ndim."""
    return float(np.mean(np.abs(deviation) * confidence))


def _ols_scale(raw: np.ndarray, residual: np.ndarray) -> tuple:
    """Ordinary-least-squares fit of ``residual`` on ``raw`` (a single
    predictor): returns ``(alpha, beta)`` minimizing
    ``sum((residual - (alpha + beta*raw))**2)``.

    This replaces a std-ratio rescale (``resid_mean + (raw - raw_mean) *
    (resid_std / raw_std)``), which forces ``raw``'s spread to match
    ``residual``'s regardless of how well the two actually correlate.  That
    is safe as long as ``raw`` is itself in-sample-inflated (which it always
    was before cross-fitting), but once ``raw`` is honestly cross-fitted and
    carries little real signal, its variance can legitimately collapse
    toward zero -- and dividing by a near-zero ``raw_std`` explodes the
    correction instead of correctly producing "this round found nothing,
    barely move the prediction." OLS doesn't have this failure mode: with
    weak or no correlation, ``beta`` is naturally small, not amplified.
    """
    raw_mean, raw_var = float(raw.mean()), float(raw.var())
    if raw_var <= 0:
        return float(residual.mean()), 0.0
    beta = float(np.mean((raw - raw_mean) * (residual - residual.mean()))) / raw_var
    alpha = float(residual.mean()) - beta * raw_mean
    return alpha, beta


def _residualize(raw: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """The part of ``residual`` left over after regressing out ``raw``
    (see :func:`_ols_scale`) -- a proper "partial out" step, whose output
    variance is guaranteed no larger than ``residual``'s. Used to measure
    how much of a candidate triple's columns' signal is *not already
    captured* by main effects + pairwise interactions alone."""
    alpha, beta = _ols_scale(raw, residual)
    return residual - (alpha + beta * raw)


def _make_folds(rng: np.random.Generator, n: int, n_folds: int) -> np.ndarray:
    """Randomly assign each of ``n`` rows to one of ``n_folds`` folds, as
    evenly as possible. Every fold index is guaranteed non-empty as long as
    ``n_folds <= n``."""
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=int)
    fold_ids[perm] = np.arange(n) % n_folds
    return fold_ids


def _cross_fitted_raw(
    zones: dict,
    n_zones: dict,
    residual: np.ndarray,
    main_effect_keys: list,
    interaction_keys: list,
    triple_keys: list,
    fold_ids: np.ndarray,
    n_folds: int,
) -> np.ndarray:
    """Leakage-free version of what ``weak_learner_score`` would compute for
    the exact rows used to build this round's tables: for each fold, every
    term's ``(deviation, confidence)`` is recomputed from the *other* folds'
    rows only (reusing ``_zone_deviation_confidence``/
    ``_pair_deviation_confidence``/``_triple_deviation_confidence`` unchanged
    on fold-restricted slices), then used to score that fold's own held-out
    rows. No row is ever scored with a table that included its own value --
    the CatBoost ordered-target-statistics fix, applied to zoneboost's zone
    grids. Which terms exist (main effects / which pairs / which triples) is
    decided once from the full subsample elsewhere; this only recomputes the
    numeric cell means used to score training rows honestly.
    """
    n = len(residual)
    n_terms = len(main_effect_keys) + len(interaction_keys) + len(triple_keys)
    contributions = np.empty((n, n_terms))

    for k in range(n_folds):
        out_mask = fold_ids != k
        in_mask = fold_ids == k
        if not np.any(in_mask):
            continue
        overall_mean_k = float(residual[out_mask].mean())

        col = 0
        for name in main_effect_keys:
            dev, conf = _zone_deviation_confidence(
                zones[name][out_mask], residual[out_mask], overall_mean_k, n_zones[name]
            )
            z_in = zones[name][in_mask]
            contributions[in_mask, col] = dev[z_in] * conf[z_in]
            col += 1

        for a, b in interaction_keys:
            dev, conf = _pair_deviation_confidence(
                zones[a][out_mask], zones[b][out_mask], residual[out_mask], overall_mean_k, n_zones[a], n_zones[b]
            )
            za_in, zb_in = zones[a][in_mask], zones[b][in_mask]
            contributions[in_mask, col] = dev[za_in, zb_in] * conf[za_in, zb_in]
            col += 1

        for a, b, c in triple_keys:
            dev, conf = _triple_deviation_confidence(
                zones[a][out_mask],
                zones[b][out_mask],
                zones[c][out_mask],
                residual[out_mask],
                overall_mean_k,
                n_zones[a],
                n_zones[b],
                n_zones[c],
            )
            za_in, zb_in, zc_in = zones[a][in_mask], zones[b][in_mask], zones[c][in_mask]
            contributions[in_mask, col] = dev[za_in, zb_in, zc_in] * conf[za_in, zb_in, zc_in]
            col += 1

    return contributions.mean(axis=1)


def _get_pair(interactions: dict, x: str, y: str):
    """Fetch a pair's (deviation, confidence) regardless of which of the two
    key orders ``interactions`` happened to store it under (pair keys follow
    ``predictor_subset``'s order, not alphabetical)."""
    return interactions[(x, y)] if (x, y) in interactions else interactions[(y, x)]


def _select_triples(
    predictor_subset: list,
    zones: dict,
    n_zones: dict,
    main_effects: dict,
    interactions: dict,
    residual: np.ndarray,
    max_triple_interactions: int,
    triple_min_gain: float,
):
    """Adaptive 3-way interaction selection for one round: start from main
    effects + pairwise interactions (already fit), and only add a small
    number of 3-way terms where there's evidence of genuine higher-order
    structure that main effects and pairwise interactions alone don't
    already explain -- rather than trying every possible triple.

    Candidates are seeded from the columns that appear in this round's
    strongest pairs (not the full C(p, 3) space), then each candidate is
    kept only if, after subtracting what main effects + its three
    constituent pairs would already predict, a joint 3-way zone grouping
    still carries a confidence-weighted signal worth at least
    ``triple_min_gain`` times its strongest constituent pair's own
    importance -- comparing the triple's leftover signal to a pair's signal
    (both computed identically via ``_term_importance``) rather than to the
    residual's raw scale, since a zone-averaged, confidence-discounted
    importance score is not on the same scale as a raw standard deviation.
    """
    if len(predictor_subset) < 3 or not interactions:
        return {}
    if float(residual.std()) <= 0:
        return {}

    pair_importance = {pair: _term_importance(*interactions[pair]) for pair in interactions}
    k_pairs = min(len(pair_importance), max(2 * max_triple_interactions, 6))
    top_pairs = sorted(pair_importance, key=pair_importance.get, reverse=True)[:k_pairs]

    col_scores: dict = {}
    for a, b in top_pairs:
        col_scores[a] = max(col_scores.get(a, 0.0), pair_importance[(a, b)])
        col_scores[b] = max(col_scores.get(b, 0.0), pair_importance[(a, b)])
    candidate_cols = sorted(col_scores, key=col_scores.get, reverse=True)[:10]

    scored = []
    for a, b, c in itertools.combinations(candidate_cols, 3):
        za, zb, zc = zones[a], zones[b], zones[c]
        dev_a, conf_a = main_effects[a]
        dev_b, conf_b = main_effects[b]
        dev_c, conf_c = main_effects[c]
        dev_ab, conf_ab = _get_pair(interactions, a, b)
        dev_ac, conf_ac = _get_pair(interactions, a, c)
        dev_bc, conf_bc = _get_pair(interactions, b, c)
        max_pair_importance = max(
            _get_pair(pair_importance, a, b), _get_pair(pair_importance, a, c), _get_pair(pair_importance, b, c)
        )
        lower_order_raw = np.column_stack(
            [
                dev_a[za] * conf_a[za],
                dev_b[zb] * conf_b[zb],
                dev_c[zc] * conf_c[zc],
                dev_ab[za, zb] * conf_ab[za, zb],
                dev_ac[za, zc] * conf_ac[za, zc],
                dev_bc[zb, zc] * conf_bc[zb, zc],
            ]
        ).mean(axis=1)
        double_residual = _residualize(lower_order_raw, residual)

        gain_dev, gain_conf = _triple_deviation_confidence(
            za, zb, zc, double_residual, float(double_residual.mean()), n_zones[a], n_zones[b], n_zones[c]
        )
        gain = _term_importance(gain_dev, gain_conf)
        if gain >= triple_min_gain * max_pair_importance:
            dev_abc, conf_abc = _triple_deviation_confidence(
                za, zb, zc, residual, float(residual.mean()), n_zones[a], n_zones[b], n_zones[c]
            )
            scored.append(((a, b, c), gain, dev_abc, conf_abc))

    scored.sort(key=lambda item: item[1], reverse=True)
    return {key: (dev, conf) for key, _, dev, conf in scored[:max_triple_interactions]}


def _column_zone_info(x_col: pd.Series, residual: np.ndarray, is_categorical: bool, max_zones: int, min_zone_frac: float):
    """Returns a ``("continuous", boundaries)`` or ``("categorical",
    category_map)`` tagged tuple -- the one place that decides which zone
    mechanism a column uses."""
    if is_categorical:
        return ("categorical", categorical_zone_map(x_col))
    col_cap = min(max_zones, x_col.nunique())
    bounds = adaptive_zone_boundaries(x_col, residual, max_zones=col_cap, min_zone_frac=min_zone_frac)
    return ("continuous", bounds)


def _column_zone_index(x_col: pd.Series, info: tuple) -> np.ndarray:
    kind, payload = info
    if kind == "categorical":
        return categorical_zone_index(x_col, payload)
    return zone_index(x_col, payload)


def _column_n_zones(info: tuple) -> int:
    kind, payload = info
    if kind == "categorical":
        return len(payload) + 2  # +2: dedicated "missing" zone, dedicated "unseen category" zone
    return len(payload) + 2  # +2: last cut point, dedicated "missing" zone


def weak_learner_fit(
    X: pd.DataFrame,
    residual: np.ndarray,
    predictor_subset: list,
    categorical_features: set,
    rng: np.random.Generator,
    max_zones: int = 7,
    min_zone_frac: float = 0.02,
    max_interaction_order: int = 2,
    max_triple_interactions: int = 5,
    triple_min_gain: float = 0.05,
    cross_fit_folds: int = 5,
):
    """Fit one boosting round's weak learner: zone info (adaptive-continuous
    or exact-categorical per column), main effects, interactions, and
    (optionally) a small number of adaptively-selected 3-way interactions --
    ALL derived fresh from this round's (already row/column-subsampled)
    residual.

    Also returns ``oof_raw``: an honest, cross-fitted version of this
    round's own raw score for exactly these rows, used by the caller to
    replace the (leaky, in-sample) score it would otherwise compute for the
    same rows when updating the running prediction -- see
    :func:`_cross_fitted_raw`.

    Parameters
    ----------
    rng : numpy.random.Generator
        Used only to assign rows to cross-fitting folds -- the same
        generator the caller already uses for row/column subsampling, so
        results stay fully reproducible under a fixed ``random_state``.
    max_interaction_order : int, default=2
        ``2`` fits main effects + pairwise interactions only (identical to
        every prior release). ``3`` additionally attempts a bounded search
        for 3-way interactions -- see :func:`_select_triples`.
    max_triple_interactions : int, default=5
        Cap on how many 3-way terms a single round may add (only relevant
        when ``max_interaction_order >= 3``).
    triple_min_gain : float, default=0.05
        Minimum confidence-weighted residual-explained magnitude a
        candidate triple must retain after subtracting the main-effect +
        pairwise fit for its three columns, expressed as a fraction of its
        strongest constituent pair's own importance (both measured
        identically, so this is a like-for-like comparison rather than one
        against the residual's raw scale) -- to be judged genuine
        higher-order structure rather than something pairwise interactions
        already explain.
    cross_fit_folds : int, default=5
        Number of folds used to compute ``oof_raw`` honestly (see above).
        Falls back to the in-sample score (no cross-fitting) if the round's
        row count is smaller than 2 folds.

    Returns
    -------
    zone_info : dict
        column -> ``("continuous", boundaries)`` or ``("categorical", map)``.
    main_effects : dict
        column -> ``(deviation, confidence)`` arrays.
    interactions : dict
        ``(col_a, col_b)`` -> ``(deviation, confidence)`` 2D arrays.
    triples : dict
        ``(col_a, col_b, col_c)`` -> ``(deviation, confidence)`` 3D arrays.
        Empty unless ``max_interaction_order >= 3`` and evidence clears
        ``triple_min_gain``.
    oof_raw : ndarray
        Cross-fitted raw score for this round's own rows, aligned to
        ``residual``'s order.
    """
    zone_info = {
        c: _column_zone_info(X[c], residual, c in categorical_features, max_zones, min_zone_frac)
        for c in predictor_subset
    }
    n_zones = {c: _column_n_zones(zone_info[c]) for c in predictor_subset}
    zones = {c: _column_zone_index(X[c], zone_info[c]) for c in predictor_subset}
    overall_mean = float(residual.mean())

    main_effects = {
        col: _zone_deviation_confidence(zones[col], residual, overall_mean, n_zones[col]) for col in predictor_subset
    }
    interactions = {
        (a, b): _pair_deviation_confidence(zones[a], zones[b], residual, overall_mean, n_zones[a], n_zones[b])
        for a, b in itertools.combinations(predictor_subset, 2)
    }
    triples = (
        _select_triples(
            predictor_subset,
            zones,
            n_zones,
            main_effects,
            interactions,
            residual,
            max_triple_interactions,
            triple_min_gain,
        )
        if max_interaction_order >= 3
        else {}
    )

    n = len(residual)
    effective_folds = min(cross_fit_folds, n)
    if effective_folds < 2:
        oof_raw = weak_learner_score(X, zone_info, main_effects, interactions, triples)
    else:
        fold_ids = _make_folds(rng, n, effective_folds)
        oof_raw = _cross_fitted_raw(
            zones,
            n_zones,
            residual,
            list(main_effects.keys()),
            list(interactions.keys()),
            list(triples.keys()),
            fold_ids,
            effective_folds,
        )
    return zone_info, main_effects, interactions, triples, oof_raw


def weak_learner_score(
    X: pd.DataFrame, zone_info: dict, main_effects: dict, interactions: dict, triples: dict = None
) -> np.ndarray:
    """Score rows with an already-fit weak learner. Self-sufficient from
    main_effects/interactions/triples' own keys, so it works whether that
    round used every predictor or only a random subset of them."""
    triples = triples or {}
    needed_cols = set(main_effects.keys())
    for a, b in interactions.keys():
        needed_cols.add(a)
        needed_cols.add(b)
    for a, b, c in triples.keys():
        needed_cols.add(a)
        needed_cols.add(b)
        needed_cols.add(c)
    zones = {c: _column_zone_index(X[c], zone_info[c]) for c in needed_cols}

    contributions = []
    for col, (deviation, confidence) in main_effects.items():
        z = zones[col]
        contributions.append(deviation[z] * confidence[z])
    for (a, b), (deviation, confidence) in interactions.items():
        za, zb = zones[a], zones[b]
        contributions.append(deviation[za, zb] * confidence[za, zb])
    for (a, b, c), (deviation, confidence) in triples.items():
        za, zb, zc = zones[a], zones[b], zones[c]
        contributions.append(deviation[za, zb, zc] * confidence[za, zb, zc])
    return np.column_stack(contributions).mean(axis=1)
