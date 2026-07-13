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

Every zone/cell's stored value is a shrunk *mean* by default (``quantile=
None`` everywhere in this module). Passing a ``quantile`` level instead
switches every one of those values to a shrunk *quantile* of the residual at
that level (see :func:`_zone_raw_stat`/:func:`_zone_shrunk_deviation`) --
zone construction, cross-fitting, and pair screening's cheap proxy stay
squared-error-flavored regardless (disclosed approximations), but the
combination step (:func:`_fit_lasso_weights`) switches from ``Lasso`` to
``QuantileRegressor`` in quantile mode -- not optional, since combining
quantile-shrunk terms via an ordinary (squared-error) Lasso would silently
re-center the round's output back toward the mean/median every round.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Lasso, QuantileRegressor

from ._shrinkage import _estimate_shrinkage_m
from ._zones import adaptive_zone_boundaries, categorical_zone_index, categorical_zone_map, zone_centers, zone_index

__all__ = ["weak_learner_fit", "weak_learner_contributions"]


def _zone_raw_stat(
    zone_values: np.ndarray,
    target_values: np.ndarray,
    n_zones: int,
    quantile: float = None,
):
    """Each zone's raw (unshrunk) statistic and row count -- a mean via
    ``bincount`` (``quantile=None``, O(n)), or a given quantile level
    (sorts rows by zone once via ``argsort``/``searchsorted``, then
    ``np.quantile`` per zone, O(n log n) -- negligible next to the round's
    own ``O(p^2)`` pair loop). Shared by :func:`_zone_shrunk_deviation` and
    its pair/triple counterparts so both loss modes reuse the identical
    shrinkage arithmetic downstream -- only how the *raw* per-zone number
    is computed differs.
    """
    counts = np.bincount(zone_values, minlength=n_zones).astype(float)
    if quantile is None:
        sums = np.bincount(zone_values, weights=target_values, minlength=n_zones)
        stat = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        return stat, counts

    stat = np.zeros(n_zones)
    order = np.argsort(zone_values, kind="stable")
    sorted_zones = zone_values[order]
    sorted_targets = target_values[order]
    edges = np.searchsorted(sorted_zones, np.arange(n_zones + 1))
    for z in range(n_zones):
        lo, hi = edges[z], edges[z + 1]
        if hi > lo:
            stat[z] = np.quantile(sorted_targets[lo:hi], quantile)
    return stat, counts


def _overall_stat(values: np.ndarray, quantile: float = None) -> float:
    """The scalar prior/baseline a round's terms are shrunk toward: the
    mean (``quantile=None``, bit-identical to every prior release) or the
    given quantile level -- shared by every call site that needs "the
    overall statistic of this residual" (:func:`weak_learner_fit`'s main
    effects, :func:`_fit_pairs`/:func:`_select_triples`'s mains-removed
    backfitting prior, :func:`_cross_fitted_contributions`'s per-fold
    prior)."""
    return float(values.mean()) if quantile is None else float(np.quantile(values, quantile))


def _zone_shrunk_deviation(
    zone_values: np.ndarray,
    target_values: np.ndarray,
    overall_stat: float,
    n_zones: int,
    m: float,
    monotonic: int = 0,
    quantile: float = None,
):
    """For each zone: an empirical-Bayes (m-estimate) shrunk statistic among
    fit-rows in that zone, minus the overall statistic.

    ``shrunk_stat = (counts * cell_stat + m * overall_stat) / (counts + m)``
    -- a zone needs about ``m`` rows of its own before it's trusted as much
    as the prior; fewer rows lean toward the prior, more rows lean toward
    its own data. When ``counts == 0``, ``counts * cell_stat == 0``
    regardless of the placeholder cell_stat there, so this naturally
    reduces to ``deviation = 0`` (the prior) with no special-casing needed
    -- replaces a flat ``counts / counts.max()`` confidence discount with a
    principled, hierarchical estimate.

    ``quantile`` (default ``None``) selects the raw per-zone statistic via
    :func:`_zone_raw_stat`: ``None`` is the mean (``overall_stat`` is then
    the overall mean, bit-identical to every prior release); a quantile
    level shrinks the zone's own empirical quantile toward the *overall*
    quantile instead -- the same m-estimate shrinkage pattern applied to a
    different raw statistic, not a rigorous Bayesian quantile posterior
    (there isn't a simple closed form for that), but a defensible extension
    of the identical formula used everywhere else in zoneboost.

    ``monotonic`` (0 = none, +1 = non-decreasing, -1 = non-increasing) is
    only ever passed for a column's own main effect (continuous columns,
    whose zones are meaningfully ordered by construction) -- never from
    ``_pair_shrunk_deviation``/``_triple_shrunk_deviation``'s internal calls
    for their own marginal *priors* (``dev_a``/``dev_b``/``dev_c`` stay
    unconstrained there, exactly as before). The joint cell array those two
    functions return can still end up monotonic along a constrained column's
    own axis -- via a separate mechanism, :func:`_project_monotonic_axis`,
    applied to the *joint* deviation itself, not to this function's own
    marginal calls. When set here, the *real* zones (all but the last index
    -- for a continuous column that's always the dedicated missing-value
    bucket, which isn't part of the ordered continuum and is left alone)
    are projected onto the nearest monotonic sequence via
    ``sklearn.isotonic.IsotonicRegression``, weighted by each zone's own
    row count so sparse zones don't distort the fit -- the same
    density-aware spirit as the shrinkage above.
    """
    cell_stat, counts = _zone_raw_stat(zone_values, target_values, n_zones, quantile)
    shrunk_stat = (counts * cell_stat + m * overall_stat) / (counts + m)
    deviation = shrunk_stat - overall_stat

    if monotonic != 0:
        n_real = n_zones - 1
        real_counts = counts[:n_real]
        if real_counts.sum() > 0:
            iso = IsotonicRegression(increasing=monotonic > 0, out_of_bounds="clip")
            deviation[:n_real] = iso.fit_transform(np.arange(n_real), deviation[:n_real], sample_weight=real_counts)

    return deviation


def _project_monotonic_axis(deviation: np.ndarray, counts: np.ndarray, axis: int, direction: int) -> np.ndarray:
    """Multi-dimensional generalization of :func:`_zone_shrunk_deviation`'s
    own isotonic projection: projects one ``axis`` of a 2D/3D joint
    ``deviation`` array onto a monotonic sequence, holding every other axis
    fixed -- one independent ``sklearn.isotonic.IsotonicRegression`` fit
    per fiber along ``axis``, weighted by that fiber's own row counts (via
    ``counts``, the same shape as ``deviation``). ``axis``'s own last index
    (its dedicated missing-value zone) is excluded and left untouched, same
    convention as the main effect.

    A single, independent pass over ``axis`` -- not a jointly-optimal
    multi-dimensional isotone regression. When a term has more than one
    constrained axis, the caller applies this once per axis in a fixed
    order, and a later axis's projection can slightly disturb an earlier
    axis's own monotonicity -- a disclosed heuristic, consistent with
    cyclic backfitting's own single-pass approximation elsewhere in this
    module.
    """
    n_real = deviation.shape[axis] - 1
    if n_real <= 0:
        return deviation
    dev_moved = np.moveaxis(deviation, axis, 0)
    counts_moved = np.moveaxis(counts, axis, 0)
    out = dev_moved.copy()
    for idx in np.ndindex(dev_moved.shape[1:]):
        key = (slice(0, n_real),) + idx
        fiber_counts = counts_moved[key]
        if fiber_counts.sum() <= 0:
            continue
        iso = IsotonicRegression(increasing=direction > 0, out_of_bounds="clip")
        out[key] = iso.fit_transform(np.arange(n_real), dev_moved[key], sample_weight=fiber_counts)
    return np.moveaxis(out, 0, axis)


def _project_convexity(deviation: np.ndarray, counts: np.ndarray, centers: np.ndarray, direction: int) -> np.ndarray:
    """Project a continuous main effect's per-zone deviation onto a convex
    (``direction=+1``) or concave (``direction=-1``) sequence, as evaluated
    against the actual piecewise-*linear* interpolant :func:`_blend_1d`
    reconstructs between zone centroids -- not against the raw zone-index
    sequence. Zones are rarely evenly spaced (adaptive zone boundaries), so
    a sequence with non-decreasing *index-to-index* differences is not
    generally convex once interpolated against irregular real-valued
    centroids: convexity of a piecewise-linear function through points
    ``(center_i, y_i)`` requires non-decreasing *slopes*
    ``(y[i+1]-y[i]) / (center[i+1]-center[i])`` (divided differences), not
    non-decreasing raw differences.

    This isotonic-regresses those slopes (real zones only -- the last
    index, the dedicated missing-value zone, is excluded and left
    untouched, same convention as monotonic projection), weighted by each
    gap's average neighboring row count, reconstructs via a cumulative sum
    of ``slope * gap``, then re-centers the result to the *original*
    count-weighted mean -- the projection changes shape, not overall
    level, the same "project shape, preserve level" spirit as the existing
    monotonic projection.

    Guarantees convexity of *this round's own* stored deviation only, not
    the boosted ensemble's cumulative multi-round main effect: a sum of
    convex functions is itself convex only when every term is combined
    with a non-negative weight, but each round's own Lasso-stacking weight
    for this term (see :func:`_fit_lasso_weights`) can be negative --
    flipping a convex round's contribution to concave in the combined
    output. A real, disclosed limitation of layering a per-round shape
    constraint on top of signed Lasso stacking, not a free guarantee.

    Combining this with a monotonic constraint on the same column is a
    heuristic ordering (monotonic projection happens first, inside
    :func:`_zone_shrunk_deviation`, before this is applied) -- not
    guaranteed to keep the result strictly monotonic afterward.
    """
    n_real = len(deviation) - 1
    if n_real <= 2:
        return deviation
    real = deviation[:n_real]
    real_counts = counts[:n_real]
    real_centers = centers[:n_real]
    if real_counts.sum() <= 0:
        return deviation
    gaps = np.diff(real_centers)
    if np.any(gaps <= 0):
        return deviation
    slopes = np.diff(real) / gaps
    slope_weights = (real_counts[:-1] + real_counts[1:]) / 2.0
    if slope_weights.sum() <= 0:
        return deviation
    iso = IsotonicRegression(increasing=direction > 0, out_of_bounds="clip")
    fitted_slopes = iso.fit_transform(np.arange(n_real - 1), slopes, sample_weight=slope_weights)
    reconstructed = np.concatenate([[0.0], np.cumsum(fitted_slopes * gaps)])
    orig_mean = np.average(real, weights=real_counts)
    new_mean = np.average(reconstructed, weights=real_counts)
    reconstructed = reconstructed + (orig_mean - new_mean)
    out = deviation.copy()
    out[:n_real] = reconstructed
    return out


def _pair_shrunk_deviation(
    za: np.ndarray,
    zb: np.ndarray,
    target_values: np.ndarray,
    overall_stat: float,
    n_zones_a: int,
    n_zones_b: int,
    m: float,
    quantile: float = None,
    monotonic_a: int = 0,
    monotonic_b: int = 0,
):
    """Same idea, gridded over two variables' zones jointly -- but shrunk
    toward a *hierarchical* prior, not the flat overall statistic: each
    column's own shrunk marginal deviation (a direct recursive call to
    :func:`_zone_shrunk_deviation`) is combined additively
    (``overall_stat + dev_a + dev_b``) into a row+column prior, and the
    joint cell is shrunk toward *that*. Absent enough of its own data,
    "what row A's zone alone predicts, plus what column B's zone alone
    predicts" is a far better guess for a sparse cell than the overall
    average of everything.

    ``quantile`` (default ``None``) is forwarded to every call below,
    including the joint cell's own raw statistic (via
    :func:`_zone_raw_stat`) -- an additive sum of two quantile deviations
    isn't an exact quantile identity the way it is for means, but reuses
    the identical hierarchical-prior shrinkage pattern as a defensible
    heuristic (see :func:`_zone_shrunk_deviation`).

    ``monotonic_a``/``monotonic_b`` (default 0, unconstrained) project the
    *joint* returned deviation along that axis via
    :func:`_project_monotonic_axis` -- so a column with a declared
    monotonic main effect gets its interactions constrained too ("inherited
    monotonicity"), not just its own main effect. Applied in a fixed order
    (``a`` then ``b``) when both are set -- a disclosed heuristic, see
    :func:`_project_monotonic_axis`."""
    dev_a = _zone_shrunk_deviation(za, target_values, overall_stat, n_zones_a, m, quantile=quantile)
    dev_b = _zone_shrunk_deviation(zb, target_values, overall_stat, n_zones_b, m, quantile=quantile)

    combined = za * n_zones_b + zb
    size = n_zones_a * n_zones_b
    cell_stat, counts = _zone_raw_stat(combined, target_values, size, quantile)
    cell_stat = cell_stat.reshape(n_zones_a, n_zones_b)
    counts = counts.reshape(n_zones_a, n_zones_b)
    prior = overall_stat + dev_a[:, None] + dev_b[None, :]
    shrunk_stat = (counts * cell_stat + m * prior) / (counts + m)
    deviation = shrunk_stat - overall_stat

    if monotonic_a != 0:
        deviation = _project_monotonic_axis(deviation, counts, axis=0, direction=monotonic_a)
    if monotonic_b != 0:
        deviation = _project_monotonic_axis(deviation, counts, axis=1, direction=monotonic_b)
    return deviation


def _triple_shrunk_deviation(
    za: np.ndarray,
    zb: np.ndarray,
    zc: np.ndarray,
    target_values: np.ndarray,
    overall_stat: float,
    n_zones_a: int,
    n_zones_b: int,
    n_zones_c: int,
    m: float,
    quantile: float = None,
    monotonic_a: int = 0,
    monotonic_b: int = 0,
    monotonic_c: int = 0,
):
    """Same recursive pattern as :func:`_pair_shrunk_deviation`, one level
    deeper: the three main effects and three pairwise interactions
    (themselves already shrunk) combine additively into the joint 3D cell's
    prior, and the cell is shrunk toward that. ``quantile`` is forwarded to
    every recursive call and the joint cell's own raw statistic, same
    heuristic-extension caveat as :func:`_pair_shrunk_deviation`.

    ``monotonic_a``/``monotonic_b``/``monotonic_c`` project the *joint*
    returned deviation along that axis, same "inherited monotonicity"
    mechanism and fixed-order caveat as :func:`_pair_shrunk_deviation` (not
    passed to the ``dev_ab``/``dev_ac``/``dev_bc`` recursive calls above --
    those stay unconstrained pairwise priors, only this function's own
    final joint cell is projected)."""
    dev_a = _zone_shrunk_deviation(za, target_values, overall_stat, n_zones_a, m, quantile=quantile)
    dev_b = _zone_shrunk_deviation(zb, target_values, overall_stat, n_zones_b, m, quantile=quantile)
    dev_c = _zone_shrunk_deviation(zc, target_values, overall_stat, n_zones_c, m, quantile=quantile)
    dev_ab = _pair_shrunk_deviation(za, zb, target_values, overall_stat, n_zones_a, n_zones_b, m, quantile=quantile)
    dev_ac = _pair_shrunk_deviation(za, zc, target_values, overall_stat, n_zones_a, n_zones_c, m, quantile=quantile)
    dev_bc = _pair_shrunk_deviation(zb, zc, target_values, overall_stat, n_zones_b, n_zones_c, m, quantile=quantile)

    combined = (za * n_zones_b + zb) * n_zones_c + zc
    size = n_zones_a * n_zones_b * n_zones_c
    cell_stat, counts = _zone_raw_stat(combined, target_values, size, quantile)
    cell_stat = cell_stat.reshape(n_zones_a, n_zones_b, n_zones_c)
    counts = counts.reshape(n_zones_a, n_zones_b, n_zones_c)
    prior = (
        overall_stat
        + dev_a[:, None, None]
        + dev_b[None, :, None]
        + dev_c[None, None, :]
        + dev_ab[:, :, None]
        + dev_ac[:, None, :]
        + dev_bc[None, :, :]
    )
    shrunk_stat = (counts * cell_stat + m * prior) / (counts + m)
    deviation = shrunk_stat - overall_stat

    if monotonic_a != 0:
        deviation = _project_monotonic_axis(deviation, counts, axis=0, direction=monotonic_a)
    if monotonic_b != 0:
        deviation = _project_monotonic_axis(deviation, counts, axis=1, direction=monotonic_b)
    if monotonic_c != 0:
        deviation = _project_monotonic_axis(deviation, counts, axis=2, direction=monotonic_c)
    return deviation


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


def _batched_pair_scores(
    zones: dict,
    n_zones: dict,
    columns: list,
    residual: np.ndarray,
    forbidden_interactions: frozenset = frozenset(),
) -> dict:
    """Matrix-batched alternative to calling :func:`_pair_interaction_score`
    once per candidate pair in a Python loop. Mathematically identical, not
    an approximation -- every column's own one-hot zone-indicator block is
    concatenated into one big dense matrix ``Z`` (``n_rows x total_zones``),
    and ``Z.T @ Z`` / ``Z.T @ (Z scaled by residual)`` (both BLAS ``dgemm``
    calls) are computed *once*; each pair's joint-cell table is then a block
    slice of these two precomputed matrices, fed through the same
    cell_mean/row_mean/col_mean/additive/score arithmetic
    :func:`_pair_interaction_score` uses.

    **Not wired into the default screening path.** A first version used a
    sparse ``scipy.sparse`` matmul to keep memory down, but benchmarked
    *slower* than the plain per-pair loop at every size tried (0.2x-0.5x) --
    sparse-sparse matmul on a one-hot matrix pays for every ``(row, col_a,
    col_b)`` triple regardless of output sparsity, so it does the same
    O(n_rows * p^2) work as the loop with more overhead, not less. Switching
    to a dense BLAS matmul (this version) recovers a real ~1.4x-1.8x speedup,
    but *only* for wide, fairly shallow problems (benchmarked: reliable from
    roughly 80-120+ columns at a few thousand rows); at more rows per column
    (e.g. 8000 rows, 40-70 columns) it was measured up to ~3x *slower*, since
    building the dense ``n_rows x total_zones`` matrix and its full
    ``total_zones x total_zones`` product has fixed costs that don't
    reliably pay off. There is no cheap, reliable way to predict which side
    of that crossover a given fit falls on without risking a real regression
    for some users -- so this function is kept available (tested, exact) for
    advanced callers who have benchmarked their own workload, but
    :func:`weak_learner_fit`'s screening path keeps using the plain
    per-pair loop unconditionally. See the "Pair Screening" docs section for
    the full measured numbers.
    """
    if len(columns) < 2:
        return {}

    n_rows = len(residual)
    offsets = {}
    offset = 0
    for col in columns:
        offsets[col] = offset
        offset += n_zones[col]
    total_zones = offset

    Z = np.zeros((n_rows, total_zones), dtype=np.float64)
    rows = np.arange(n_rows)
    for col in columns:
        Z[rows, zones[col] + offsets[col]] = 1.0

    counts_mat = Z.T @ Z
    sums_mat = Z.T @ (Z * residual[:, None])

    overall_mean = float(residual.mean())
    scores = {}
    for a, b in itertools.combinations(columns, 2):
        if frozenset((a, b)) in forbidden_interactions:
            continue
        oa, ob = offsets[a], offsets[b]
        na, nb = n_zones[a], n_zones[b]
        counts = counts_mat[oa : oa + na, ob : ob + nb]
        sums = sums_mat[oa : oa + na, ob : ob + nb]
        cell_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

        row_counts, col_counts = counts.sum(axis=1), counts.sum(axis=0)
        row_mean = np.divide(sums.sum(axis=1), row_counts, out=np.zeros(na), where=row_counts > 0)
        col_mean = np.divide(sums.sum(axis=0), col_counts, out=np.zeros(nb), where=col_counts > 0)

        additive = row_mean[:, None] + col_mean[None, :] - overall_mean
        scores[(a, b)] = float(np.sum(counts * (cell_mean - additive) ** 2) / n_rows)
    return scores


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


def _estimate_boundary_lambda(
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    weight_hi: np.ndarray,
    residual: np.ndarray,
    fold_ids: np.ndarray,
    n_folds: int,
    n_zones: int,
    m: float,
    boundary_shrinkage_m: float,
    monotonic: int = 0,
    quantile: float = None,
) -> float:
    """Cross-fitted estimate of how much a continuous column's soft
    (centroid-interpolated) zone lookup should be trusted versus its hard
    (single-zone) lookup -- the ``lam`` :func:`_column_soft_zone_index`
    scales ``weight_hi`` by. Shrunk toward full smoothness (``lam=1``)
    absent strong, honest evidence that the hard lookup fits held-out rows
    better (a real, local discontinuity), mirroring the same m-estimate
    shrinkage pattern used everywhere else in zoneboost.

    ``z_lo``/``z_hi``/``weight_hi`` are the column's own *unscaled*
    (``lam=1``) soft lookup -- i.e. ``_column_soft_zone_index``'s raw
    output before any ``lam`` has been applied.

    ``quantile`` (default ``None``) is forwarded to the deviation this
    compares hard vs. smooth against, so quantile-mode rounds judge the
    lookup shape against the same values they actually fit -- but the
    comparison itself stays squared-error-based regardless (a deliberate
    simplification: it's judging which interpolation *shape* fits held-out
    rows better, not re-deriving the training loss itself).
    """
    hard_sq_err = 0.0
    smooth_sq_err = 0.0
    n_boundary_rows = 0

    for k in range(n_folds):
        out_mask = fold_ids != k
        in_mask = fold_ids == k
        if not np.any(in_mask):
            continue
        overall_stat_k = _overall_stat(residual[out_mask], quantile)
        dev_k = _zone_shrunk_deviation(
            z_lo[out_mask], residual[out_mask], overall_stat_k, n_zones, m, monotonic, quantile=quantile
        )

        r_in = residual[in_mask]
        hard_pred = dev_k[z_lo[in_mask]]
        smooth_pred = _blend_1d(dev_k, z_lo[in_mask], z_hi[in_mask], weight_hi[in_mask])
        hard_sq_err += float(np.sum((r_in - hard_pred) ** 2))
        smooth_sq_err += float(np.sum((r_in - smooth_pred) ** 2))
        n_boundary_rows += int(np.sum(weight_hi[in_mask] > 0))

    smooth_advantage = max(0.0, hard_sq_err - smooth_sq_err)
    hard_advantage = max(0.0, smooth_sq_err - hard_sq_err)
    total_advantage = smooth_advantage + hard_advantage
    raw_lambda = smooth_advantage / total_advantage if total_advantage > 0 else 1.0

    return (n_boundary_rows * raw_lambda + boundary_shrinkage_m * 1.0) / (n_boundary_rows + boundary_shrinkage_m)


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


def _fit_lasso_weights(
    contributions: np.ndarray,
    residual: np.ndarray,
    alpha: float,
    quantile: float = None,
    positive_mask: np.ndarray = None,
) -> tuple:
    """Fit an L1-penalized model relating each term's own contribution (one
    column per term) to the residual, replacing the old "average every term,
    then fit one shared scale" combination with a learned per-term weight: an
    irrelevant term's weight gets zeroed by the L1 penalty, a strong term
    gets its own weight instead of a diluted ``1/n_terms`` share, and the
    fitted weights themselves become a real interaction-importance ranking.

    ``quantile=None`` (default) fits an ordinary ``Lasso`` (squared error) --
    bit-identical to every prior release. This is **not** just a cosmetic
    choice for ``loss="quantile"`` rounds: ``Lasso``'s intercept is the
    residual's own *mean* (an OLS identity), which would silently re-center
    every round's combined output back toward the conditional mean/median
    regardless of each term's own carefully quantile-shrunk deviation --
    actively destroying the quantile target, not merely approximating it.
    When ``quantile`` is set, ``sklearn.linear_model.QuantileRegressor``
    (pinball loss + L1 penalty) is used instead, so the combination step
    stays consistent with the same loss every term's own value was fit
    against.

    Both sides are standardized before fitting (each contribution column by
    its own std, the residual by its own std) so ``alpha`` is a unitless
    regularization strength, comparable across rounds/datasets regardless
    of scale -- then un-standardized so the returned weights apply directly
    to raw (unstandardized) contributions. A column with zero variance
    (e.g. every row landed in the same zone) gets a weight of exactly 0.
    Standardization is a monotonic affine transform, so the same
    ``quantile`` level applies unchanged on the standardized scale.

    ``positive_mask`` (default ``None``, meaning "no constraint," bit-
    identical to every prior release): a boolean array, aligned to
    ``contributions``'s own columns, marking which terms' weights must be
    non-negative -- so that a non-negative-weighted sum of
    individually-monotonic (or individually-convex) per-round terms stays
    monotonic (or convex) in the aggregate, rather than a negative round
    weight silently flipping its sign. ``sklearn.linear_model.Lasso``
    only supports ``positive=True`` for *every* coefficient, not a
    per-term subset, so unconstrained columns are represented as the
    difference of two non-negative variables (``w_free = w_free+ -
    w_free-``, appending ``-X[:, free]`` as extra columns) and a single
    ``Lasso(positive=True)`` is fit on the expanded matrix. At the
    L1-optimal solution ``w_free+``/``w_free-`` are never both positive
    for the same term (reducing both by ``min(w_free+, w_free-)`` leaves
    the fit unchanged but strictly shrinks the penalty), so ``w_free+ -
    w_free-`` recovers *exactly* the solution the original mixed-sign-
    constrained L1 problem would have -- not an approximation. Raises
    ``ValueError`` if given together with ``quantile`` set --
    ``QuantileRegressor`` has no ``positive=True`` mode and no equivalent
    reformulation applies cleanly to pinball loss.

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

    if positive_mask is not None and np.any(positive_mask):
        if quantile is not None:
            raise ValueError(
                "positive_mask (strict_shape_constraints) is not supported with loss='quantile' -- "
                "QuantileRegressor has no positive=True mode and no equivalent variable-splitting "
                "reformulation applies cleanly to pinball loss."
            )
        free_idx = np.where(~positive_mask)[0]
        pos_idx = np.where(positive_mask)[0]
        X_expanded = np.hstack([X_std[:, pos_idx], X_std[:, free_idx], -X_std[:, free_idx]])
        model = Lasso(alpha=alpha, fit_intercept=True, positive=True, max_iter=10000)
        model.fit(X_expanded, y_std)
        n_pos, n_free = len(pos_idx), len(free_idx)
        coefs = np.zeros(contributions.shape[1])
        coefs[pos_idx] = model.coef_[:n_pos]
        coefs[free_idx] = model.coef_[n_pos : n_pos + n_free] - model.coef_[n_pos + n_free :]
        intercept_std = float(model.intercept_)
    elif quantile is None:
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
        model.fit(X_std, y_std)
        coefs = model.coef_
        intercept_std = float(model.intercept_)
    else:
        model = QuantileRegressor(quantile=quantile, alpha=alpha, fit_intercept=True, solver="highs")
        model.fit(X_std, y_std)
        coefs = model.coef_
        intercept_std = float(model.intercept_)

    weights = np.where(col_std > 0, coefs * (resid_std / safe_col_std), 0.0)
    intercept = resid_mean + intercept_std * resid_std
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
    quantile: float = None,
    return_fold_std: bool = False,
    m_pair: float = None,
    m_triple: float = None,
):
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

    ``return_fold_std`` (default ``False``): when ``True``, also returns a
    second value, ``{term_key: fold_std_array}`` -- for every term, the
    elementwise standard deviation (same shape as the term's own deviation
    array) across the ``n_folds`` per-fold deviation arrays this function
    already computes internally (no new fold loop -- just retained instead
    of discarded). A reliability signal: a zone/cell whose shrunk value
    swings a lot across which fold happened to estimate it is less
    trustworthy than one that's stable regardless of which rows were held
    out. `NaN` at any position no fold ever populated with a real
    (non-prior) estimate is not specially handled -- ``np.std`` naturally
    reflects however close every fold's prior-shrunk placeholder value was.

    ``m_pair``/``m_triple`` (default ``None``, meaning "use ``m`` for
    this level too" -- bit-identical to every prior release): lets
    ``learn_shrinkage_m`` pass a level-specific shrinkage strength for
    the honest, cross-fitted contributions Lasso stacking learns from,
    so they're shrunk by the *same* strength as whatever the production
    tables for that round actually used (not a plain constant) --
    otherwise stacking would be fit against a systematically different
    representation than what's stored/used at predict time.
    """
    monotonic_constraints = monotonic_constraints or {}
    m_pair = m_pair if m_pair is not None else m
    m_triple = m_triple if m_triple is not None else m
    n = len(residual)
    n_terms = len(main_effect_keys) + len(interaction_keys) + len(triple_keys)
    contributions = np.empty((n, n_terms))
    fold_devs: dict = {}
    if return_fold_std:
        for name in main_effect_keys:
            fold_devs[name] = []
        for key in interaction_keys:
            fold_devs[key] = []
        for key in triple_keys:
            fold_devs[key] = []

    for k in range(n_folds):
        out_mask = fold_ids != k
        in_mask = fold_ids == k
        if not np.any(in_mask):
            continue
        overall_stat_k = _overall_stat(residual[out_mask], quantile)

        col = 0
        main_dev_k = {}
        for name in main_effect_keys:
            dev = _zone_shrunk_deviation(
                zones[name][out_mask],
                residual[out_mask],
                overall_stat_k,
                n_zones[name],
                m,
                monotonic_constraints.get(name, 0),
                quantile=quantile,
            )
            main_dev_k[name] = dev
            if return_fold_std:
                fold_devs[name].append(dev)
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
                zones[a][out_mask],
                zones[b][out_mask],
                partial,
                _overall_stat(partial, quantile),
                n_zones[a],
                n_zones[b],
                m_pair,
                quantile=quantile,
                monotonic_a=monotonic_constraints.get(a, 0),
                monotonic_b=monotonic_constraints.get(b, 0),
            )
            if return_fold_std:
                fold_devs[(a, b)].append(dev)
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
                _overall_stat(partial, quantile),
                n_zones[a],
                n_zones[b],
                n_zones[c],
                m_triple,
                quantile=quantile,
                monotonic_a=monotonic_constraints.get(a, 0),
                monotonic_b=monotonic_constraints.get(b, 0),
                monotonic_c=monotonic_constraints.get(c, 0),
            )
            if return_fold_std:
                fold_devs[(a, b, c)].append(dev)
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

    if not return_fold_std:
        return contributions

    fold_std = {
        key: (np.std(np.stack(devs), axis=0) if len(devs) >= 2 else np.zeros_like(devs[0]))
        for key, devs in fold_devs.items()
    }
    return contributions, fold_std


def _get_pair(interactions: dict, x: str, y: str):
    """Fetch a pair's value regardless of which of the two key orders
    ``interactions`` happened to store it under (pair keys follow
    ``predictor_subset``'s order, not alphabetical)."""
    return interactions[(x, y)] if (x, y) in interactions else interactions[(y, x)]


def _fit_pairs(
    pairs,
    zones: dict,
    n_zones: dict,
    main_effects: dict,
    residual: np.ndarray,
    m: float,
    quantile: float = None,
    monotonic_constraints: dict = None,
    learn_shrinkage_m: bool = False,
) -> tuple:
    """Fully fit (backfit against mains, see module docstring) exactly the
    given ``pairs`` -- shared by ``weak_learner_fit``'s screened and
    unscreened paths so there's a single place that does the expensive
    per-pair work. ``monotonic_constraints`` (default ``None``) is looked
    up per column and forwarded to :func:`_pair_shrunk_deviation` as
    ``monotonic_a``/``monotonic_b`` -- see "inherited monotonicity" there.

    ``learn_shrinkage_m`` (default ``False``): estimate one shrinkage
    strength for this whole level (pooling every pair's own raw joint-cell
    statistic, mains-removed and centered by that pair's own
    ``_overall_stat``) via :func:`zoneboost._shrinkage._estimate_shrinkage_m`,
    instead of using the constant ``m`` for every pair -- see
    ``weak_learner_fit``'s own ``learn_shrinkage_m`` docstring. Recomputes
    each pair's joint raw stat once more for this (the same combined-index
    ``_zone_raw_stat`` call :func:`_pair_shrunk_deviation` makes
    internally), a real but small extra cost, opt-in like every other
    computed-not-hand-set knob in this module.

    Returns
    -------
    interactions : dict
    m_used : float
        The shrinkage strength actually applied to every pair this level
        -- the constant ``m`` unchanged when ``learn_shrinkage_m=False``.
    """
    monotonic_constraints = monotonic_constraints or {}
    partials = {(a, b): residual - main_effects[a][zones[a]] - main_effects[b][zones[b]] for a, b in pairs}

    m_used = m
    if learn_shrinkage_m and pairs:
        deviations = []
        for a, b in partials:
            partial = partials[(a, b)]
            overall = _overall_stat(partial, quantile)
            combined = zones[a] * n_zones[b] + zones[b]
            stat, counts = _zone_raw_stat(combined, partial, n_zones[a] * n_zones[b], quantile)
            deviations.append((stat - overall, counts))
        pooled_residual_var = float(np.concatenate(list(partials.values())).var())
        m_used = _estimate_shrinkage_m(deviations, pooled_residual_var, fallback_m=m)

    interactions = {}
    for a, b in pairs:
        partial = partials[(a, b)]
        interactions[(a, b)] = _pair_shrunk_deviation(
            zones[a],
            zones[b],
            partial,
            _overall_stat(partial, quantile),
            n_zones[a],
            n_zones[b],
            m_used,
            quantile=quantile,
            monotonic_a=monotonic_constraints.get(a, 0),
            monotonic_b=monotonic_constraints.get(b, 0),
        )
    return interactions, m_used


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
    quantile: float = None,
    monotonic_constraints: dict = None,
    forbidden_pairs: frozenset = frozenset(),
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

    The accept/reject ``gain_dev`` test itself always stays mean-based
    (``quantile`` not applied there) -- like the pair-screening statistic,
    it is a cheap diagnostic for "is there signal here," not the literal
    training objective. Only the accepted triple's *stored* value
    (``dev_abc``) uses ``quantile`` when set, since that value becomes part
    of the round's actual output; it also inherits ``monotonic_constraints``
    the same way :func:`_pair_shrunk_deviation`'s own callers do.

    ``forbidden_pairs`` (a ``frozenset`` of 2-element column-name
    ``frozenset``s) skips any candidate triple containing a forbidden pair
    among its three constituent pairs -- a domain-expert-declared "these two
    columns must never interact" also blocks the 3-way term that would
    otherwise still jointly involve both.
    """
    if len(candidate_cols) < 3 or not interactions:
        return {}
    if float(residual.std()) <= 0:
        return {}

    monotonic_constraints = monotonic_constraints or {}
    pair_importance = {pair: _term_importance(interactions[pair]) for pair in interactions}

    scored = []
    for a, b, c in itertools.combinations(candidate_cols, 3):
        if forbidden_pairs and (
            frozenset((a, b)) in forbidden_pairs
            or frozenset((a, c)) in forbidden_pairs
            or frozenset((b, c)) in forbidden_pairs
        ):
            continue
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
                za,
                zb,
                zc,
                mains_removed,
                _overall_stat(mains_removed, quantile),
                n_zones[a],
                n_zones[b],
                n_zones[c],
                m,
                quantile=quantile,
                monotonic_a=monotonic_constraints.get(a, 0),
                monotonic_b=monotonic_constraints.get(b, 0),
                monotonic_c=monotonic_constraints.get(c, 0),
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

    ``info`` may carry an optional 4th element, ``lam`` (the column's own
    adaptive-boundary-continuity mixing weight, see
    :func:`_estimate_boundary_lambda`) -- a plain 3-tuple is treated as
    ``lam=1.0`` (fully smooth, today's behavior) for backward
    compatibility. ``lam`` scales ``weight_hi`` uniformly, so ``lam=0``
    collapses this into the hard, single-zone lookup regardless of
    distance, and intermediate values partially blend the two.

    Returns
    -------
    z_lo, z_hi : ndarray of int
        The value's own zone, and the neighboring zone to blend toward
        (equal to ``z_lo`` when there's nothing to blend into).
    weight_hi : ndarray of float
        0 at ``z_lo``'s own centroid, ``lam`` at ``z_hi``'s centroid,
        linear between, clamped to ``[0, 1]`` past either end
        (leftmost/rightmost zone, or a single-zone column) so it never
        reaches past a non-existent neighbor.
    """
    if info[0] == "categorical":
        z = categorical_zone_index(x_col, info[1])
        zero = np.zeros(len(z))
        return z, z, zero

    boundaries, centers = info[1], info[2]
    lam = info[3] if len(info) > 3 else 1.0
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
    weight_hi = np.clip(weight_hi, 0.0, 1.0) * lam

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
    adaptive_boundary_smoothing: bool = False,
    boundary_shrinkage_m: float = 10.0,
    quantile: float = None,
    convexity_constraints: dict = None,
    bounded_effects: dict = None,
    forbidden_interactions: frozenset = frozenset(),
    track_reliability: bool = False,
    group_col: str = None,
    learn_shrinkage_m: bool = False,
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
    adaptive_boundary_smoothing : bool, default=False
        If ``True``, each continuous column's soft zone lookup (see
        :func:`_column_soft_zone_index`) is scaled by a per-column,
        per-round mixing weight ``lam`` estimated honestly (cross-fitted)
        via :func:`_estimate_boundary_lambda` -- ``lam=1`` reproduces
        today's always-fully-smooth lookup, ``lam=0`` collapses it to the
        hard, single-zone lookup. Shrunk toward ``lam=1`` (full smoothness)
        absent strong out-of-fold evidence a hard lookup fits better (a
        real, local discontinuity), so a column with a genuine sharp
        threshold can represent it instead of always being blurred.
        ``False`` (default) is identical to every prior release -- a real
        approximation/judgment tradeoff, not a free correctness fix, so
        this is opt-in.
    boundary_shrinkage_m : float, default=10.0
        Shrinkage strength for ``adaptive_boundary_smoothing`` -- a
        column's own boundary needs about this many held-out rows near it
        before its cross-fitted hard-vs-smooth evidence is trusted as much
        as the full-smoothness prior. Only used when
        ``adaptive_boundary_smoothing=True``.
    quantile : float, default=None
        ``None`` (default) fits every term's value as an empirical-Bayes
        shrunk *mean* of the residual, identical to every prior release.
        When set (in ``(0, 1)``), every term's value becomes a shrunk
        *quantile* of the residual at this level instead -- the raw
        residual still drives zone-split search, pair screening's cheap
        proxy, and cross-fitting exactly as before (a disclosed
        approximation: those stay squared-error-flavored regardless of
        loss), but the value stored per zone/cell is now loss-optimal for
        pinball loss at ``quantile`` rather than for squared error. See
        :func:`_zone_shrunk_deviation`.
    convexity_constraints : dict, default=None
        ``{column: +1 convex, -1 concave}`` -- forces a continuous column's
        *main effect* onto a convex/concave sequence across its zones (see
        :func:`_project_convexity`), same declaration convention as
        ``monotonic_constraints``. Main effects only -- unlike
        ``monotonic_constraints``, not inherited by interactions.
    bounded_effects : dict, default=None
        ``{column: (lower, upper)}`` -- clips a continuous column's *main
        effect* deviation to this range (applied last, after any
        monotonic/convexity projection), for **this round's own stored
        value only** -- not the cumulative multi-round total, which can
        still exceed the bound once summed across many shrunk rounds. Main
        effects only.
    forbidden_interactions : frozenset, default=frozenset()
        A ``frozenset`` of 2-element column-name ``frozenset``s: these
        pairs are never fit as pairwise interactions (in either the
        exhaustive or screened path), and any 3-way candidate whose three
        constituent pairs include a forbidden one is skipped too -- see
        :func:`_select_triples`.
    track_reliability : bool, default=False
        If ``True``, also computes and returns ``diagnostics`` (see below)
        -- real per-round memory/compute cost (an extra counts array per
        term, plus reusing the cross-fitting fold loop's own per-fold
        deviations instead of discarding them), so opt-in like
        ``adaptive_boundary_smoothing``/``max_pair_interactions``. ``False``
        (default) is bit-identical to every prior release.
    group_col : str, default=None
        Designates a column as a hierarchical grouping key (e.g. hospital,
        region): every ``(feature, group_col)`` pair present in
        ``predictor_subset`` is forced into ``interactions`` regardless of
        ``max_pair_interactions`` screening, reusing
        :func:`_pair_shrunk_deviation`'s existing hierarchical-prior
        shrinkage unchanged -- a (zone, group) cell already shrinks toward
        "what the zone alone predicts + what the group alone predicts,"
        i.e. local <- regional <- global partial pooling, with no new
        math. ``None`` (default) is a no-op, bit-identical to every prior
        release. See ``ZoneBoostRegressor``'s own ``group_col`` parameter
        for the column-subsampling half of this guarantee (this function
        alone can't prevent the group column from being subsampled out of
        ``predictor_subset`` in the first place).
    learn_shrinkage_m : bool, default=False
        Estimate the empirical-Bayes shrinkage strength separately for
        main effects and for pairs (pooling raw zone/cell statistics
        across every column or pair fit at that level this round) via
        :func:`zoneboost._shrinkage._estimate_shrinkage_m` -- a
        DerSimonian-Laird-style method-of-moments estimate of
        ``sigma^2/tau^2`` under the normal-normal hierarchical model this
        shrinkage formula already implies -- instead of using the
        constant ``shrinkage_m`` for both. Falls back to ``shrinkage_m``
        itself whenever there isn't enough evidence to estimate anything
        better (see that function's own docstring). Triples still use
        the plain ``shrinkage_m`` constant (deferred -- ``_select_
        triples``'s own accept/reject gain test already uses ``m`` to
        decide which triples survive, before the accepted set is known,
        making a triple-level estimate circular in a way mains/pairs
        aren't). Real per-round extra cost (each level's raw statistics
        are computed once more for the estimate, on top of what
        ``_zone_shrunk_deviation``/``_pair_shrunk_deviation`` already
        compute internally), so opt-in; ``False`` (default) is
        bit-identical to every prior release.

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
    diagnostics : dict or None
        ``None`` unless ``track_reliability=True`` or ``learn_shrinkage_m=
        True``. When ``track_reliability=True``: ``{"main_effects": {col:
        {"counts": arr, "fold_std": arr_or_None}}, "interactions": {(a,
        b): {...}}, "triples": {(a, b, c): {...}}}`` --
        ``counts`` is each zone/cell's own row count (same shape as that
        term's own deviation array); ``fold_std`` is the elementwise
        standard deviation of that term's shrunk value across the
        cross-fitting folds (``None`` when ``cross_fit_folds`` effectively
        falls back to no cross-fitting for this round) -- see
        :func:`_cross_fitted_contributions`'s ``return_fold_std``. Consumed
        by :func:`zoneboost._reliability.explain_reliability`. When
        ``learn_shrinkage_m=True``, also gains a ``"learned_shrinkage_m"``
        key: ``{"main": m_main, "pair": m_pair}``, the shrinkage strength
        actually applied at each level this round (present alongside the
        ``track_reliability`` keys above if both are set, or alone if only
        ``learn_shrinkage_m`` is).
    """
    monotonic_constraints = monotonic_constraints or {}
    convexity_constraints = convexity_constraints or {}
    bounded_effects = bounded_effects or {}
    zone_info = {
        c: _column_zone_info(X[c], residual, c in categorical_features, max_zones, min_zone_frac)
        for c in predictor_subset
    }
    n_zones = {c: _column_n_zones(zone_info[c]) for c in predictor_subset}
    zones = {c: _column_zone_index(X[c], zone_info[c]) for c in predictor_subset}
    overall_stat = _overall_stat(residual, quantile)

    # Hierarchical/multilevel zones: (feature, group_col) pairs are forced
    # to survive pair screening below, guaranteeing every feature gets a
    # local (zone x group) <- regional (zone marginal) <- global (overall
    # mean) partial-pooling estimate via _pair_shrunk_deviation's existing
    # hierarchical prior -- no new shrinkage math needed. Same iteration
    # order as the exhaustive/screened pair loops below so tuple keys line
    # up with interactions_full exactly.
    forced_group_pairs = set()
    if group_col is not None and group_col in predictor_subset:
        forced_group_pairs = {
            (a, b)
            for a, b in itertools.combinations(predictor_subset, 2)
            if group_col in (a, b) and frozenset((a, b)) not in forbidden_interactions
        }

    m_main = shrinkage_m
    if learn_shrinkage_m and predictor_subset:
        deviations_main = []
        for col in predictor_subset:
            stat, counts = _zone_raw_stat(zones[col], residual, n_zones[col], quantile)
            deviations_main.append((stat - overall_stat, counts))
        m_main = _estimate_shrinkage_m(deviations_main, float(residual.var()), fallback_m=shrinkage_m)

    main_effects = {}
    for col in predictor_subset:
        dev = _zone_shrunk_deviation(
            zones[col],
            residual,
            overall_stat,
            n_zones[col],
            m_main,
            monotonic_constraints.get(col, 0),
            quantile=quantile,
        )
        if col in convexity_constraints:
            counts = np.bincount(zones[col], minlength=n_zones[col]).astype(float)
            centers = zone_info[col][2]
            dev = _project_convexity(dev, counts, centers, convexity_constraints[col])
        if col in bounded_effects:
            lower, upper = bounded_effects[col]
            dev = np.clip(dev, lower, upper)
        main_effects[col] = dev

    n = len(residual)
    effective_folds = min(cross_fit_folds, n)
    if effective_folds < 2:
        fold_ids = None
        soft = None
    else:
        fold_ids = _make_folds(rng, n, effective_folds)
        soft = {c: _column_soft_zone_index(X[c], zone_info[c]) for c in predictor_subset}

    if adaptive_boundary_smoothing and fold_ids is not None:
        # Estimate each continuous column's own smooth-vs-hard mixing
        # weight honestly (cross-fitted), before anything downstream
        # (pair screening, cross-fitting for stacking, etc.) consumes
        # `soft` -- see _estimate_boundary_lambda and the module docstring.
        for c in predictor_subset:
            if zone_info[c][0] != "continuous":
                continue
            z_lo, z_hi, weight_hi = soft[c]
            lam = _estimate_boundary_lambda(
                z_lo, z_hi, weight_hi, residual, fold_ids, effective_folds,
                n_zones[c], shrinkage_m, boundary_shrinkage_m, monotonic_constraints.get(c, 0),
                quantile=quantile,
            )
            kind, boundaries, centers = zone_info[c]
            zone_info[c] = (kind, boundaries, centers, lam)
            soft[c] = _column_soft_zone_index(X[c], zone_info[c])

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
            quantile=quantile,
        ).sum(axis=1)
        screening_residual = residual - oof_main_pred

        pair_scores = {
            (a, b): _pair_interaction_score(zones[a], zones[b], screening_residual, n_zones[a], n_zones[b])
            for a, b in itertools.combinations(predictor_subset, 2)
            if frozenset((a, b)) not in forbidden_interactions
        }
        kept_pairs = set(sorted(pair_scores, key=pair_scores.get, reverse=True)[:max_pair_interactions])
        kept_pairs |= forced_group_pairs

        candidate_cols = (
            _seed_candidate_columns(pair_scores, max_triple_interactions) if max_interaction_order >= 3 else []
        )
        fit_pairs = set(kept_pairs)
        for a, b in itertools.combinations(candidate_cols, 2):
            fit_pairs.add((a, b) if (a, b) in pair_scores else (b, a))
    else:
        kept_pairs = None
        fit_pairs = [
            (a, b)
            for a, b in itertools.combinations(predictor_subset, 2)
            if frozenset((a, b)) not in forbidden_interactions
        ]

    interactions_full, m_pair = _fit_pairs(
        fit_pairs, zones, n_zones, main_effects, residual, shrinkage_m,
        quantile=quantile, monotonic_constraints=monotonic_constraints,
        learn_shrinkage_m=learn_shrinkage_m,
    )

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
            quantile=quantile,
            monotonic_constraints=monotonic_constraints,
            forbidden_pairs=forbidden_interactions,
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
        kept_pairs |= forced_group_pairs

    interactions = (
        {pair: interactions_full[pair] for pair in kept_pairs} if kept_pairs is not None else interactions_full
    )

    fold_std = None
    if effective_folds < 2:
        oof_contributions = weak_learner_contributions(X, zone_info, main_effects, interactions, triples)
    elif track_reliability:
        oof_contributions, fold_std = _cross_fitted_contributions(
            zones,
            soft,
            n_zones,
            residual,
            list(main_effects.keys()),
            list(interactions.keys()),
            list(triples.keys()),
            fold_ids,
            effective_folds,
            m_main,
            monotonic_constraints,
            quantile=quantile,
            return_fold_std=True,
            m_pair=m_pair,
            m_triple=shrinkage_m,
        )
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
            m_main,
            monotonic_constraints,
            quantile=quantile,
            m_pair=m_pair,
            m_triple=shrinkage_m,
        )

    diagnostics = None
    if track_reliability:
        term_counts = {}
        for col in main_effects:
            term_counts[col] = np.bincount(zones[col], minlength=n_zones[col]).astype(float)
        for a, b in interactions:
            combined = zones[a] * n_zones[b] + zones[b]
            term_counts[(a, b)] = (
                np.bincount(combined, minlength=n_zones[a] * n_zones[b])
                .astype(float)
                .reshape(n_zones[a], n_zones[b])
            )
        for a, b, c in triples:
            combined = (zones[a] * n_zones[b] + zones[b]) * n_zones[c] + zones[c]
            term_counts[(a, b, c)] = (
                np.bincount(combined, minlength=n_zones[a] * n_zones[b] * n_zones[c])
                .astype(float)
                .reshape(n_zones[a], n_zones[b], n_zones[c])
            )
        diagnostics = {
            "main_effects": {
                col: {"counts": term_counts[col], "fold_std": fold_std[col] if fold_std else None}
                for col in main_effects
            },
            "interactions": {
                key: {"counts": term_counts[key], "fold_std": fold_std[key] if fold_std else None}
                for key in interactions
            },
            "triples": {
                key: {"counts": term_counts[key], "fold_std": fold_std[key] if fold_std else None}
                for key in triples
            },
        }
    if learn_shrinkage_m:
        diagnostics = diagnostics or {}
        diagnostics["learned_shrinkage_m"] = {"main": m_main, "pair": m_pair}

    return zone_info, main_effects, interactions, triples, oof_contributions, diagnostics


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
