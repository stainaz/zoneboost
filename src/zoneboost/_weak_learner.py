"""One boosting round's weak learner: per-column zone info, main effects
(single-variable zone -> average residual), interactions (variable-pair
zone grid -> average residual), an adaptively-selected small set of 3-way
interactions, and empirical-Bayes (m-estimate) shrinkage of every zone's
own mean toward a hierarchical prior (cell -> marginal -> global).

Every "weak learner" in ZoneBoostRegressor is built from this module alone
-- no decision tree, no gradient computation beyond a plain residual, no
external model of any kind. What changes round to round is only the target
these functions are pointed at (the current residual) and which rows/
columns were sampled for that round.

Fit order is a single cyclic-backfitting pass -- main effects first, then
pairs, then triples -- rather than fitting every term against the same raw
residual independently: each interaction's stored deviation is computed
against the residual *after* subtracting its own lower-order terms (its
main effects for a pair, its main effects for a triple -- pairs are then
handled automatically inside a triple's own recursive prior computation,
see :func:`_pair_shrunk_deviation`/:func:`_triple_shrunk_deviation`). This
keeps a pair/triple's stored value genuinely interaction-only rather than
redundantly re-encoding signal a lower-order term already captures -- on by
default (a correctness/attribution fix, not a tunable knob).

Pair discovery is exhaustive by default (every C(p, 2) pair is fully fit),
but when ``max_pair_interactions`` is set, ``weak_learner_fit`` switches to
cheap-then-exact hierarchical discovery: every candidate pair is scored with
:func:`_pair_interaction_score` (a fast ANOVA-style statistic, not the full
shrinkage machinery) on an honest, cross-fitted main-effects-only residual,
and only the survivors ever pay the cost of a full :func:`_pair_shrunk_
deviation` fit -- see :func:`_seed_candidate_columns` and the parameter's own
docstring for how this stays consistent with 3-way interaction discovery.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Lasso

from ._zones import adaptive_zone_boundaries, categorical_zone_index, categorical_zone_map, zone_centers, zone_index

__all__ = ["weak_learner_fit", "weak_learner_contributions"]


def _zone_shrunk_deviation(
    zone_values: np.ndarray,
    target_values: np.ndarray,
    overall_mean: float,
    n_zones: int,
    m: float,
    monotonic: int = 0,
):
    """For each zone: an empirical-Bayes (m-estimate) shrunk average target
    among fit-rows in that zone, minus the overall mean.

    ``shrunk_mean = (counts * cell_mean + m * overall_mean) / (counts + m)``
    -- a zone needs about ``m`` rows of its own before it's trusted as much
    as the prior (here, the global mean); fewer rows lean toward the prior,
    more rows lean toward its own data. When ``counts == 0``,
    ``counts * cell_mean == 0`` regardless of the placeholder cell_mean
    there, so this naturally reduces to ``deviation = 0`` (the prior) with
    no special-casing needed -- replaces a flat ``counts / counts.max()``
    confidence discount with a principled, hierarchical estimate. O(n) via
    bincount.

    ``monotonic`` (0 = none, +1 = non-decreasing, -1 = non-increasing) is
    only ever passed for a column's own main effect (continuous columns,
    whose zones are meaningfully ordered by construction) -- never from
    ``_pair_shrunk_deviation``/``_triple_shrunk_deviation``'s internal calls
    for their own hierarchical priors, so interaction terms stay
    unconstrained. When set, the *real* zones (all but the last index --
    for a continuous column that's always the dedicated missing-value
    bucket, which isn't part of the ordered continuum and is left alone)
    are projected onto the nearest monotonic sequence via
    ``sklearn.isotonic.IsotonicRegression``, weighted by each zone's own
    row count so sparse zones don't distort the fit -- the same
    density-aware spirit as the shrinkage above.
    """
    counts = np.bincount(zone_values, minlength=n_zones).astype(float)
    sums = np.bincount(zone_values, weights=target_values, minlength=n_zones)
    cell_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    shrunk_mean = (counts * cell_mean + m * overall_mean) / (counts + m)
    deviation = shrunk_mean - overall_mean

    if monotonic != 0:
        n_real = n_zones - 1
        real_counts = counts[:n_real]
        if real_counts.sum() > 0:
            iso = IsotonicRegression(increasing=monotonic > 0, out_of_bounds="clip")
            deviation[:n_real] = iso.fit_transform(np.arange(n_real), deviation[:n_real], sample_weight=real_counts)

    return deviation


def _pair_shrunk_deviation(
    za: np.ndarray,
    zb: np.ndarray,
    target_values: np.ndarray,
    overall_mean: float,
    n_zones_a: int,
    n_zones_b: int,
    m: float,
):
    """Same idea, gridded over two variables' zones jointly -- but shrunk
    toward a *hierarchical* prior, not the flat global mean: each column's
    own shrunk marginal deviation (a direct recursive call to
    :func:`_zone_shrunk_deviation`) is combined additively
    (``overall_mean + dev_a + dev_b``) into a row+column prior, and the
    joint cell is shrunk toward *that*. Absent enough of its own data,
    "what row A's zone alone predicts, plus what column B's zone alone
    predicts" is a far better guess for a sparse cell than the overall
    average of everything."""
    dev_a = _zone_shrunk_deviation(za, target_values, overall_mean, n_zones_a, m)
    dev_b = _zone_shrunk_deviation(zb, target_values, overall_mean, n_zones_b, m)

    combined = za * n_zones_b + zb
    size = n_zones_a * n_zones_b
    counts = np.bincount(combined, minlength=size).astype(float).reshape(n_zones_a, n_zones_b)
    sums = np.bincount(combined, weights=target_values, minlength=size).reshape(n_zones_a, n_zones_b)
    cell_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    prior = overall_mean + dev_a[:, None] + dev_b[None, :]
    shrunk_mean = (counts * cell_mean + m * prior) / (counts + m)
    return shrunk_mean - overall_mean


def _triple_shrunk_deviation(
    za: np.ndarray,
    zb: np.ndarray,
    zc: np.ndarray,
    target_values: np.ndarray,
    overall_mean: float,
    n_zones_a: int,
    n_zones_b: int,
    n_zones_c: int,
    m: float,
):
    """Same recursive pattern as :func:`_pair_shrunk_deviation`, one level
    deeper: the three main effects and three pairwise interactions
    (themselves already shrunk) combine additively into the joint 3D cell's
    prior, and the cell is shrunk toward that."""
    dev_a = _zone_shrunk_deviation(za, target_values, overall_mean, n_zones_a, m)
    dev_b = _zone_shrunk_deviation(zb, target_values, overall_mean, n_zones_b, m)
    dev_c = _zone_shrunk_deviation(zc, target_values, overall_mean, n_zones_c, m)
    dev_ab = _pair_shrunk_deviation(za, zb, target_values, overall_mean, n_zones_a, n_zones_b, m)
    dev_ac = _pair_shrunk_deviation(za, zc, target_values, overall_mean, n_zones_a, n_zones_c, m)
    dev_bc = _pair_shrunk_deviation(zb, zc, target_values, overall_mean, n_zones_b, n_zones_c, m)

    combined = (za * n_zones_b + zb) * n_zones_c + zc
    size = n_zones_a * n_zones_b * n_zones_c
    counts = np.bincount(combined, minlength=size).astype(float).reshape(n_zones_a, n_zones_b, n_zones_c)
    sums = np.bincount(combined, weights=target_values, minlength=size).reshape(n_zones_a, n_zones_b, n_zones_c)
    cell_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    prior = (
        overall_mean
        + dev_a[:, None, None]
        + dev_b[None, :, None]
        + dev_c[None, None, :]
        + dev_ab[:, :, None]
        + dev_ac[:, None, :]
        + dev_bc[None, :, :]
    )
    shrunk_mean = (counts * cell_mean + m * prior) / (counts + m)
    return shrunk_mean - overall_mean


def _term_importance(deviation: np.ndarray) -> float:
    """A term's own average magnitude -- used to rank pairs/triples by how
    much signal they carry, independent of ndim."""
    return float(np.mean(np.abs(deviation)))


def _pair_interaction_score(
    za: np.ndarray, zb: np.ndarray, residual: np.ndarray, n_zones_a: int, n_zones_b: int
) -> float:
    """Cheap screening proxy for how much genuine pairwise interaction signal
    a candidate pair carries -- a classic weighted two-way ANOVA interaction
    sum-of-squares, via a single 2D ``bincount`` pass. Deliberately *not*
    :func:`_pair_shrunk_deviation`: no recursive marginal-prior calls, no
    per-cell empirical-Bayes shrinkage -- just "does this joint cell's mean
    deviate from what the row/column marginals alone would predict," weighted
    by each cell's own row count so sparse cells don't dominate. Used to
    screen every ``C(p, 2)`` candidate pair cheaply, before paying the much
    higher cost of :func:`_pair_shrunk_deviation` on only the survivors --
    see the module docstring.
    """
    combined = za * n_zones_b + zb
    size = n_zones_a * n_zones_b
    counts = np.bincount(combined, minlength=size).astype(float).reshape(n_zones_a, n_zones_b)
    sums = np.bincount(combined, weights=residual, minlength=size).reshape(n_zones_a, n_zones_b)
    cell_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    row_counts, col_counts = counts.sum(axis=1), counts.sum(axis=0)
    row_mean = np.divide(sums.sum(axis=1), row_counts, out=np.zeros(n_zones_a), where=row_counts > 0)
    col_mean = np.divide(sums.sum(axis=0), col_counts, out=np.zeros(n_zones_b), where=col_counts > 0)

    additive = row_mean[:, None] + col_mean[None, :] - float(residual.mean())
    return float(np.sum(counts * (cell_mean - additive) ** 2) / len(residual))


def _blend_1d(deviation: np.ndarray, z_lo, z_hi, w: np.ndarray) -> np.ndarray:
    """Linear interpolation between a zone's own value and its neighbor's
    (see ``_column_soft_zone_index``) -- degenerates to the plain hard
    lookup ``deviation[z_lo]`` whenever ``w`` is 0 (categorical columns,
    missing values, or a continuous value at/beyond its own centroid)."""
    return (1 - w) * deviation[z_lo] + w * deviation[z_hi]


def _blend_2d(deviation: np.ndarray, za_lo, za_hi, wa, zb_lo, zb_hi, wb) -> np.ndarray:
    """Standard bilinear interpolation across the four surrounding cells.
    A categorical/missing axis has ``w == 0`` there, so this collapses to
    plain 1D interpolation along whichever axis is actually continuous."""
    return (
        (1 - wa) * (1 - wb) * deviation[za_lo, zb_lo]
        + wa * (1 - wb) * deviation[za_hi, zb_lo]
        + (1 - wa) * wb * deviation[za_lo, zb_hi]
        + wa * wb * deviation[za_hi, zb_hi]
    )


def _blend_3d(deviation: np.ndarray, za_lo, za_hi, wa, zb_lo, zb_hi, wb, zc_lo, zc_hi, wc) -> np.ndarray:
    """Trilinear interpolation across the eight surrounding cells."""
    return (
        (1 - wa) * (1 - wb) * (1 - wc) * deviation[za_lo, zb_lo, zc_lo]
        + wa * (1 - wb) * (1 - wc) * deviation[za_hi, zb_lo, zc_lo]
        + (1 - wa) * wb * (1 - wc) * deviation[za_lo, zb_hi, zc_lo]
        + (1 - wa) * (1 - wb) * wc * deviation[za_lo, zb_lo, zc_hi]
        + wa * wb * (1 - wc) * deviation[za_hi, zb_hi, zc_lo]
        + wa * (1 - wb) * wc * deviation[za_hi, zb_lo, zc_hi]
        + (1 - wa) * wb * wc * deviation[za_lo, zb_hi, zc_hi]
        + wa * wb * wc * deviation[za_hi, zb_hi, zc_hi]
    )


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


def _fit_lasso_weights(contributions: np.ndarray, residual: np.ndarray, alpha: float) -> tuple:
    """Fit a Lasso relating each term's own contribution (one column per
    term) to the residual, replacing the old "average every term, then fit
    one shared scale" combination with a learned per-term weight: an
    irrelevant term's weight gets zeroed by the L1 penalty, a strong term
    gets its own weight instead of a diluted ``1/n_terms`` share, and the
    fitted weights themselves become a real interaction-importance ranking.

    Both sides are standardized before fitting (each contribution column by
    its own std, the residual by its own std) so ``alpha`` is a unitless
    regularization strength, comparable across rounds/datasets regardless
    of scale -- then un-standardized so the returned weights apply directly
    to raw (unstandardized) contributions. A column with zero variance
    (e.g. every row landed in the same zone) gets a weight of exactly 0.

    Returns
    -------
    intercept : float
    weights : ndarray of shape (n_terms,)
    """
    resid_mean, resid_std = float(residual.mean()), float(residual.std())
    if resid_std <= 0:
        return resid_mean, np.zeros(contributions.shape[1])

    col_std = contributions.std(axis=0)
    safe_col_std = np.where(col_std > 0, col_std, 1.0)
    X_std = contributions / safe_col_std
    y_std = (residual - resid_mean) / resid_std

    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    model.fit(X_std, y_std)

    weights = np.where(col_std > 0, model.coef_ * (resid_std / safe_col_std), 0.0)
    intercept = resid_mean + float(model.intercept_) * resid_std
    return intercept, weights


def _make_folds(rng: np.random.Generator, n: int, n_folds: int) -> np.ndarray:
    """Randomly assign each of ``n`` rows to one of ``n_folds`` folds, as
    evenly as possible. Every fold index is guaranteed non-empty as long as
    ``n_folds <= n``."""
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=int)
    fold_ids[perm] = np.arange(n) % n_folds
    return fold_ids


def _cross_fitted_contributions(
    zones: dict,
    soft: dict,
    n_zones: dict,
    residual: np.ndarray,
    main_effect_keys: list,
    interaction_keys: list,
    triple_keys: list,
    fold_ids: np.ndarray,
    n_folds: int,
    m: float,
    monotonic_constraints: dict = None,
) -> np.ndarray:
    """Leakage-free version of what ``weak_learner_contributions`` would
    compute for the exact rows used to build this round's tables: for each
    fold, every term's shrunk deviation is recomputed from the *other*
    folds' rows only (reusing ``_zone_shrunk_deviation``/
    ``_pair_shrunk_deviation``/``_triple_shrunk_deviation`` unchanged on
    fold-restricted slices), then used to score that fold's own held-out
    rows -- via the same soft, interpolated blend ``weak_learner_
    contributions`` uses (zone boundaries/centroids are fixed for the
    round, so a row's interpolation weights don't depend on which fold
    computed the values being blended). No row is ever scored with a table
    that included its own value -- the CatBoost ordered-target-statistics
    fix, applied to zoneboost's zone grids. Which terms exist (main
    effects / which pairs / which triples) is decided once from the full
    subsample elsewhere; this only recomputes the numeric cell means used
    to score training rows honestly. Returns the full ``(n, n_terms)``
    per-term matrix (not pooled), so the caller can fit per-term stacking
    weights on it.

    Mirrors ``weak_learner_fit``'s own cyclic-backfitting order per fold:
    each fold's main effects are computed first, then used to backfit that
    fold's pairs (mains-only subtraction; pairs handled automatically inside
    ``_triple_shrunk_deviation``'s own recursive calls for triples) -- every
    pair/triple's constituent columns are always a subset of
    ``main_effect_keys`` (both are only ever built from the same
    ``predictor_subset``), so this is always well-defined regardless of
    which pairs a caller's own screening step kept.
    """
    monotonic_constraints = monotonic_constraints or {}
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
        main_dev_k = {}
        for name in main_effect_keys:
            dev = _zone_shrunk_deviation(
                zones[name][out_mask],
                residual[out_mask],
                overall_mean_k,
                n_zones[name],
                m,
                monotonic_constraints.get(name, 0),
            )
            main_dev_k[name] = dev
            z_lo, z_hi, w = soft[name]
            contributions[in_mask, col] = _blend_1d(dev, z_lo[in_mask], z_hi[in_mask], w[in_mask])
            col += 1

        for a, b in interaction_keys:
            partial = (
                residual[out_mask]
                - main_dev_k[a][zones[a][out_mask]]
                - main_dev_k[b][zones[b][out_mask]]
            )
            dev = _pair_shrunk_deviation(
                zones[a][out_mask], zones[b][out_mask], partial, float(partial.mean()), n_zones[a], n_zones[b], m
            )
            za_lo, za_hi, wa = soft[a]
            zb_lo, zb_hi, wb = soft[b]
            contributions[in_mask, col] = _blend_2d(
                dev, za_lo[in_mask], za_hi[in_mask], wa[in_mask], zb_lo[in_mask], zb_hi[in_mask], wb[in_mask]
            )
            col += 1

        for a, b, c in triple_keys:
            partial = (
                residual[out_mask]
                - main_dev_k[a][zones[a][out_mask]]
                - main_dev_k[b][zones[b][out_mask]]
                - main_dev_k[c][zones[c][out_mask]]
            )
            dev = _triple_shrunk_deviation(
                zones[a][out_mask],
                zones[b][out_mask],
                zones[c][out_mask],
                partial,
                float(partial.mean()),
                n_zones[a],
                n_zones[b],
                n_zones[c],
                m,
            )
            za_lo, za_hi, wa = soft[a]
            zb_lo, zb_hi, wb = soft[b]
            zc_lo, zc_hi, wc = soft[c]
            contributions[in_mask, col] = _blend_3d(
                dev,
                za_lo[in_mask],
                za_hi[in_mask],
                wa[in_mask],
                zb_lo[in_mask],
                zb_hi[in_mask],
                wb[in_mask],
                zc_lo[in_mask],
                zc_hi[in_mask],
                wc[in_mask],
            )
            col += 1

    return contributions


def _get_pair(interactions: dict, x: str, y: str):
    """Fetch a pair's value regardless of which of the two key orders
    ``interactions`` happened to store it under (pair keys follow
    ``predictor_subset``'s order, not alphabetical)."""
    return interactions[(x, y)] if (x, y) in interactions else interactions[(y, x)]


def _fit_pairs(pairs, zones: dict, n_zones: dict, main_effects: dict, residual: np.ndarray, m: float) -> dict:
    """Fully fit (backfit against mains, see module docstring) exactly the
    given ``pairs`` -- shared by ``weak_learner_fit``'s screened and
    unscreened paths so there's a single place that does the expensive
    per-pair work."""
    interactions = {}
    for a, b in pairs:
        partial = residual - main_effects[a][zones[a]] - main_effects[b][zones[b]]
        interactions[(a, b)] = _pair_shrunk_deviation(
            zones[a], zones[b], partial, float(partial.mean()), n_zones[a], n_zones[b], m
        )
    return interactions


def _seed_candidate_columns(pair_importance: dict, max_triple_interactions: int) -> list:
    """Extract up to 10 columns worth trying together as 3-way candidates,
    seeded from the strongest pairs by ``pair_importance`` -- either the
    cheap screening score (:func:`_pair_interaction_score`) or a full fit's
    own :func:`_term_importance`, so :func:`_select_triples` always receives
    a candidate set whose own ``C(candidate_cols, 2)`` pairs are guaranteed
    already present in whatever ``interactions`` dict it's given (the caller
    is responsible for that guarantee -- see ``weak_learner_fit``)."""
    if not pair_importance:
        return []
    k_pairs = min(len(pair_importance), max(2 * max_triple_interactions, 6))
    top_pairs = sorted(pair_importance, key=pair_importance.get, reverse=True)[:k_pairs]
    col_scores: dict = {}
    for a, b in top_pairs:
        col_scores[a] = max(col_scores.get(a, 0.0), pair_importance[(a, b)])
        col_scores[b] = max(col_scores.get(b, 0.0), pair_importance[(a, b)])
    return sorted(col_scores, key=col_scores.get, reverse=True)[:10]


def _select_triples(
    candidate_cols: list,
    zones: dict,
    n_zones: dict,
    main_effects: dict,
    interactions: dict,
    residual: np.ndarray,
    max_triple_interactions: int,
    triple_min_gain: float,
    m: float,
):
    """Adaptive 3-way interaction selection for one round: start from main
    effects + pairwise interactions (already fit), and only add a small
    number of 3-way terms where there's evidence of genuine higher-order
    structure that main effects and pairwise interactions alone don't
    already explain -- rather than trying every possible triple.

    ``candidate_cols`` (up to 10 columns) is computed by the caller via
    :func:`_seed_candidate_columns`, seeded from the columns that appear in
    this round's strongest pairs (not the full C(p, 3) space) -- not derived
    here, so both ``weak_learner_fit``'s screened and unscreened paths can
    guarantee every ``C(candidate_cols, 2)`` pair the loop below needs is
    already present in ``interactions`` before calling this function. Each
    candidate is kept only if, after subtracting what main effects + its
    three constituent pairs would already predict, a joint 3-way zone
    grouping still carries a signal worth at least ``triple_min_gain`` times
    its strongest constituent pair's own importance -- comparing the
    triple's leftover signal to a pair's signal (both computed identically
    via ``_term_importance``) rather than to the residual's raw scale, since
    a zone-averaged, shrunk importance score is not on the same scale as a
    raw standard deviation.

    An accepted triple's *stored* value is backfit against mains only
    (``residual`` minus its three main effects) before being handed to
    :func:`_triple_shrunk_deviation` -- distinct from the ``double_residual``
    above, which is an OLS-based proxy used only for the accept/reject
    decision, not a fitting target. Pairs don't need a separate subtraction
    step here: :func:`_triple_shrunk_deviation`'s own internal
    :func:`_pair_shrunk_deviation` calls perform that automatically once fed
    a mains-removed target, so the accepted triple's coefficients end up
    interaction-only rather than re-encoding lower-order signal.
    """
    if len(candidate_cols) < 3 or not interactions:
        return {}
    if float(residual.std()) <= 0:
        return {}

    pair_importance = {pair: _term_importance(interactions[pair]) for pair in interactions}

    scored = []
    for a, b, c in itertools.combinations(candidate_cols, 3):
        za, zb, zc = zones[a], zones[b], zones[c]
        dev_a = main_effects[a]
        dev_b = main_effects[b]
        dev_c = main_effects[c]
        dev_ab = _get_pair(interactions, a, b)
        dev_ac = _get_pair(interactions, a, c)
        dev_bc = _get_pair(interactions, b, c)
        max_pair_importance = max(
            _get_pair(pair_importance, a, b), _get_pair(pair_importance, a, c), _get_pair(pair_importance, b, c)
        )
        lower_order_raw = np.column_stack(
            [
                dev_a[za],
                dev_b[zb],
                dev_c[zc],
                dev_ab[za, zb],
                dev_ac[za, zc],
                dev_bc[zb, zc],
            ]
        ).mean(axis=1)
        double_residual = _residualize(lower_order_raw, residual)

        gain_dev = _triple_shrunk_deviation(
            za, zb, zc, double_residual, float(double_residual.mean()), n_zones[a], n_zones[b], n_zones[c], m
        )
        gain = _term_importance(gain_dev)
        if gain >= triple_min_gain * max_pair_importance:
            # Backfit the accepted triple's *stored* value against mains only
            # (not the double_residual/_residualize proxy above, which stays
            # reserved for the accept/reject gain test) -- pairs are then
            # handled automatically inside _triple_shrunk_deviation's own
            # recursive _pair_shrunk_deviation calls, so dev_abc comes out
            # interaction-only rather than re-encoding mains+pairs signal.
            mains_removed = residual - dev_a[za] - dev_b[zb] - dev_c[zc]
            dev_abc = _triple_shrunk_deviation(
                za, zb, zc, mains_removed, float(mains_removed.mean()), n_zones[a], n_zones[b], n_zones[c], m
            )
            scored.append(((a, b, c), gain, dev_abc))

    scored.sort(key=lambda item: item[1], reverse=True)
    return {key: dev for key, _, dev in scored[:max_triple_interactions]}


def _column_zone_info(x_col: pd.Series, residual: np.ndarray, is_categorical: bool, max_zones: int, min_zone_frac: float):
    """Returns a ``("continuous", boundaries, centers)`` or ``("categorical",
    category_map)`` tagged tuple -- the one place that decides which zone
    mechanism a column uses. ``centers`` (one per real zone) is what lets
    a continuous column's lookup interpolate between neighboring zones
    instead of hard-assigning a value to exactly one -- see
    :func:`_column_soft_zone_index`."""
    if is_categorical:
        return ("categorical", categorical_zone_map(x_col))
    col_cap = min(max_zones, x_col.nunique())
    bounds = adaptive_zone_boundaries(x_col, residual, max_zones=col_cap, min_zone_frac=min_zone_frac)
    centers = zone_centers(x_col, bounds)
    return ("continuous", bounds, centers)


def _column_zone_index(x_col: pd.Series, info: tuple) -> np.ndarray:
    """Hard (single-zone) lookup -- still used for zone construction itself
    and by :func:`_select_triples`'s self-contained candidate-gain test.
    Production scoring uses the soft, interpolated lookup instead; see
    :func:`_column_soft_zone_index`."""
    kind = info[0]
    if kind == "categorical":
        return categorical_zone_index(x_col, info[1])
    return zone_index(x_col, info[1])


def _column_soft_zone_index(x_col: pd.Series, info: tuple):
    """Interpolated lookup for a continuous column: instead of hard-
    assigning a value to exactly one zone, find its own zone's centroid
    and the neighboring zone in whichever direction the value sits from
    that centroid, and return how far toward that neighbor to blend.

    Categorical columns and missing continuous values are a trivial
    no-op case (``z_lo == z_hi``, ``weight_hi == 0``) so the *same* blend
    formula works uniformly downstream regardless of column type -- see
    ``weak_learner_contributions``.

    Returns
    -------
    z_lo, z_hi : ndarray of int
        The value's own zone, and the neighboring zone to blend toward
        (equal to ``z_lo`` when there's nothing to blend into).
    weight_hi : ndarray of float
        0 at ``z_lo``'s own centroid, 1 at ``z_hi``'s centroid, linear
        between, clamped to ``[0, 1]`` past either end (leftmost/rightmost
        zone, or a single-zone column) so it never reaches past a
        non-existent neighbor.
    """
    if info[0] == "categorical":
        z = categorical_zone_index(x_col, info[1])
        zero = np.zeros(len(z))
        return z, z, zero

    boundaries, centers = info[1], info[2]
    x_arr = np.asarray(x_col, dtype=float)
    is_missing = np.isnan(x_arr)
    n_real = len(centers)

    z_lo = np.clip(zone_index(x_arr, boundaries), 0, n_real - 1)
    own_center = centers[z_lo]
    go_right = x_arr > own_center
    z_hi = np.where(go_right, np.minimum(z_lo + 1, n_real - 1), np.maximum(z_lo - 1, 0))
    neighbor_center = centers[z_hi]

    denom = neighbor_center - own_center
    weight_hi = np.divide(x_arr - own_center, denom, out=np.zeros_like(x_arr), where=denom != 0)
    weight_hi = np.clip(weight_hi, 0.0, 1.0)

    missing_idx = n_real  # one past the last real zone, matching zone_index's convention
    z_lo = np.where(is_missing, missing_idx, z_lo)
    z_hi = np.where(is_missing, missing_idx, z_hi)
    weight_hi = np.where(is_missing, 0.0, weight_hi)
    return z_lo, z_hi, weight_hi


def _column_n_zones(info: tuple) -> int:
    kind, payload = info[0], info[1]
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
    shrinkage_m: float = 10.0,
    monotonic_constraints: dict = None,
    max_pair_interactions: int = None,
):
    """Fit one boosting round's weak learner: zone info (adaptive-continuous
    or exact-categorical per column), main effects, interactions, and
    (optionally) a small number of adaptively-selected 3-way interactions --
    ALL derived fresh from this round's (already row/column-subsampled)
    residual.

    Terms are fit via a single cyclic-backfitting pass -- main effects
    first, then pairs (backfit against their own two main effects), then
    triples (backfit against their own three main effects, with pairs
    handled automatically inside :func:`_triple_shrunk_deviation`'s own
    recursive prior) -- so a pair/triple's stored deviation is genuinely
    interaction-only rather than re-encoding signal a lower-order term
    already captures. On by default; see the module docstring.

    Also returns ``oof_contributions``: an honest, cross-fitted version of
    this round's own per-term contributions for exactly these rows, used
    by the caller to replace the (leaky, in-sample) contributions it would
    otherwise compute for the same rows when fitting that round's stacking
    weights -- see :func:`_cross_fitted_contributions`.

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
        Minimum residual-explained magnitude a candidate triple must retain
        after subtracting the main-effect + pairwise fit for its three
        columns, expressed as a fraction of its strongest constituent
        pair's own importance (both measured identically, so this is a
        like-for-like comparison rather than one against the residual's
        raw scale) -- to be judged genuine higher-order structure rather
        than something pairwise interactions already explain.
    cross_fit_folds : int, default=5
        Number of folds used to compute ``oof_contributions`` honestly (see
        above). Falls back to in-sample contributions (no cross-fitting) if
        the round's row count is smaller than 2 folds.
    shrinkage_m : float, default=10.0
        Empirical-Bayes shrinkage strength -- a zone needs about this many
        rows of its own before it's trusted as much as its (hierarchical)
        prior; see :func:`_zone_shrunk_deviation`.
    monotonic_constraints : dict, default=None
        ``{column: +1 or -1}`` -- forces a continuous column's *main
        effect* to be non-decreasing (+1) or non-increasing (-1) across
        its zones, via isotonic regression (see
        :func:`_zone_shrunk_deviation`). Interaction terms are never
        constrained. Unlike every other tuning knob here, this encodes
        domain knowledge the model can't infer on its own, so there's no
        default direction -- an unlisted column is simply unconstrained.
    max_pair_interactions : int, default=None
        Cap on how many pairwise interactions a round keeps. ``None``
        (default) fits every pair exhaustively, identical to every prior
        release. When set, pairs are found via cheap, hierarchical
        discovery rather than fitting every ``C(len(predictor_subset), 2)``
        pair in full: every candidate pair is scored with
        :func:`_pair_interaction_score` (a cheap ANOVA-style statistic) on an
        honest, cross-fitted main-effects-only residual, and only the
        top-scoring pairs (plus whatever pairs :func:`_select_triples`'s own
        candidate columns need, when ``max_interaction_order >= 3``) are
        ever fully fit via :func:`_pair_shrunk_deviation` -- turning an
        ``O(p²)`` expensive-fit problem into ``O(p²)`` cheap screening plus
        ``O(k)`` expensive fits, ``k`` bounded by ``max_pair_interactions``
        and the triple-candidate search, not by ``p``. Falls back to fitting
        every pair (then trimming by the full fit's own importance) when the
        round has too few rows for cross-fitting to screen against.

    Returns
    -------
    zone_info : dict
        column -> ``("continuous", boundaries)`` or ``("categorical", map)``.
    main_effects : dict
        column -> shrunk deviation array.
    interactions : dict
        ``(col_a, col_b)`` -> shrunk deviation 2D array.
    triples : dict
        ``(col_a, col_b, col_c)`` -> shrunk deviation 3D array. Empty unless
        ``max_interaction_order >= 3`` and evidence clears ``triple_min_gain``.
    oof_contributions : ndarray of shape (n_rows, n_terms)
        Cross-fitted per-term contributions for this round's own rows,
        aligned to ``residual``'s row order and to the column order
        ``weak_learner_contributions`` would produce for the same tables.
    """
    monotonic_constraints = monotonic_constraints or {}
    zone_info = {
        c: _column_zone_info(X[c], residual, c in categorical_features, max_zones, min_zone_frac)
        for c in predictor_subset
    }
    n_zones = {c: _column_n_zones(zone_info[c]) for c in predictor_subset}
    zones = {c: _column_zone_index(X[c], zone_info[c]) for c in predictor_subset}
    overall_mean = float(residual.mean())

    main_effects = {
        col: _zone_shrunk_deviation(
            zones[col], residual, overall_mean, n_zones[col], shrinkage_m, monotonic_constraints.get(col, 0)
        )
        for col in predictor_subset
    }

    n = len(residual)
    effective_folds = min(cross_fit_folds, n)
    if effective_folds < 2:
        fold_ids = None
        soft = None
    else:
        fold_ids = _make_folds(rng, n, effective_folds)
        soft = {c: _column_soft_zone_index(X[c], zone_info[c]) for c in predictor_subset}

    screen = max_pair_interactions is not None and fold_ids is not None
    if screen:
        # Cheap-then-exact hierarchical discovery: score every C(p, 2)
        # candidate pair with a fast ANOVA-style statistic on an honest,
        # cross-fitted main-effects-only residual, and only pay the full
        # backfitting cost for the survivors (plus whatever pairs the triple
        # candidate search needs) -- see module docstring.
        oof_main_pred = _cross_fitted_contributions(
            zones,
            soft,
            n_zones,
            residual,
            list(main_effects.keys()),
            [],
            [],
            fold_ids,
            effective_folds,
            shrinkage_m,
            monotonic_constraints,
        ).sum(axis=1)
        screening_residual = residual - oof_main_pred

        pair_scores = {
            (a, b): _pair_interaction_score(zones[a], zones[b], screening_residual, n_zones[a], n_zones[b])
            for a, b in itertools.combinations(predictor_subset, 2)
        }
        kept_pairs = set(sorted(pair_scores, key=pair_scores.get, reverse=True)[:max_pair_interactions])

        candidate_cols = (
            _seed_candidate_columns(pair_scores, max_triple_interactions) if max_interaction_order >= 3 else []
        )
        fit_pairs = set(kept_pairs)
        for a, b in itertools.combinations(candidate_cols, 2):
            fit_pairs.add((a, b) if (a, b) in pair_scores else (b, a))
    else:
        kept_pairs = None
        fit_pairs = list(itertools.combinations(predictor_subset, 2))

    interactions_full = _fit_pairs(fit_pairs, zones, n_zones, main_effects, residual, shrinkage_m)

    if max_interaction_order >= 3:
        if not screen:
            pair_importance_for_seeding = {p: _term_importance(d) for p, d in interactions_full.items()}
            candidate_cols = _seed_candidate_columns(pair_importance_for_seeding, max_triple_interactions)
        triples = _select_triples(
            candidate_cols,
            zones,
            n_zones,
            main_effects,
            interactions_full,
            residual,
            max_triple_interactions,
            triple_min_gain,
            shrinkage_m,
        )
    else:
        triples = {}

    if kept_pairs is None and max_pair_interactions is not None and len(interactions_full) > max_pair_interactions:
        # max_pair_interactions was requested but the round had too few rows
        # for cross-fitting (no honest residual to screen against) -- fall
        # back to trimming the already-fully-fit pairs by their own
        # importance, same as every pre-0.14 release did unconditionally.
        ranked = sorted(interactions_full, key=lambda p: _term_importance(interactions_full[p]), reverse=True)
        kept_pairs = set(ranked[:max_pair_interactions])

    interactions = (
        {pair: interactions_full[pair] for pair in kept_pairs} if kept_pairs is not None else interactions_full
    )

    if effective_folds < 2:
        oof_contributions = weak_learner_contributions(X, zone_info, main_effects, interactions, triples)
    else:
        oof_contributions = _cross_fitted_contributions(
            zones,
            soft,
            n_zones,
            residual,
            list(main_effects.keys()),
            list(interactions.keys()),
            list(triples.keys()),
            fold_ids,
            effective_folds,
            shrinkage_m,
            monotonic_constraints,
        )
    return zone_info, main_effects, interactions, triples, oof_contributions


def weak_learner_contributions(
    X: pd.DataFrame, zone_info: dict, main_effects: dict, interactions: dict, triples: dict = None
) -> np.ndarray:
    """Per-term contributions for each row with an already-fit weak
    learner: one column per term, in the fixed order
    ``main_effects`` -> ``interactions`` -> ``triples`` (each dict's own
    Python-guaranteed insertion order). Self-sufficient from
    main_effects/interactions/triples' own keys, so it works whether that
    round used every predictor or only a random subset of them.

    Column order matters: a round's fitted stacking weights (see
    :func:`_fit_lasso_weights`) are aligned to this same order, since the
    caller always passes the identical stored dicts back in at predict
    time -- never re-derive or reorder these dicts independently of how
    they were fit.
    """
    triples = triples or {}
    needed_cols = set(main_effects.keys())
    for a, b in interactions.keys():
        needed_cols.add(a)
        needed_cols.add(b)
    for a, b, c in triples.keys():
        needed_cols.add(a)
        needed_cols.add(b)
        needed_cols.add(c)
    soft = {c: _column_soft_zone_index(X[c], zone_info[c]) for c in needed_cols}

    contributions = []
    for col, deviation in main_effects.items():
        z_lo, z_hi, w = soft[col]
        contributions.append(_blend_1d(deviation, z_lo, z_hi, w))
    for (a, b), deviation in interactions.items():
        za_lo, za_hi, wa = soft[a]
        zb_lo, zb_hi, wb = soft[b]
        contributions.append(_blend_2d(deviation, za_lo, za_hi, wa, zb_lo, zb_hi, wb))
    for (a, b, c), deviation in triples.items():
        za_lo, za_hi, wa = soft[a]
        zb_lo, zb_hi, wb = soft[b]
        zc_lo, zc_hi, wc = soft[c]
        contributions.append(_blend_3d(deviation, za_lo, za_hi, wa, zb_lo, zb_hi, wb, zc_lo, zc_hi, wc))
    return np.column_stack(contributions)
