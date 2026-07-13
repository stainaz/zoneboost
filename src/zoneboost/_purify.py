"""Functional-ANOVA purification (Lengerich et al.): a post-fit transform
on ``explain(X)``'s own output, moving any marginal (single-column)
signal left inside a pair's interaction column into its constituent
main effect -- upgrading the decomposition from "exact" to "exact and
canonical" without touching `predict` or the fitted `rounds_` at all.

Cyclic backfitting fits each round's own tables in a single ordered
pass, so a pair's stored deviation can retain a component that's really
just a function of one of its two columns alone. zoneboost's own
per-round tables aren't a stable, shared 2D array the way EBM's are --
zones are re-derived every round, and each round's own Lasso gives every
term a different weight (`round_["weights"]`) -- so purification can't
safely move mass between raw round-level tables before that weighting is
applied (that would only preserve the summed prediction if the main
effect and interaction weights happened to be equal, which Lasso has no
reason to guarantee). Instead this operates on the *already* weighted,
already-summed-across-every-round per-row-per-term contribution table
`explain(X)` produces, where "main effect A", "main effect B", and
"A x B" are just three ordinary columns of the same row -- moving mass
between them trivially preserves that row's own sum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["purify_contributions"]


def _reference_bins(x: pd.Series, is_categorical: bool, n_bins: int) -> np.ndarray:
    """A plain partition of ``x``'s domain for marginalization purposes
    only -- exact groups for categorical columns, quantile bins
    otherwise. Independent of any round's own zone boundaries (which
    vary round to round): this just needs a reasonable partition of the
    empirical distribution, not a residual-driven split search, so a
    quantile binning is the right, simpler tool here --
    :func:`zoneboost._zones.adaptive_zone_boundaries` would conflate two
    different jobs.
    """
    if is_categorical:
        codes, _ = pd.factorize(x, sort=True)
        return codes.astype(int)
    x_arr = np.asarray(x, dtype=float)
    edges = np.unique(np.quantile(x_arr, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return np.zeros(len(x_arr), dtype=int)
    return np.searchsorted(edges[1:-1], x_arr, side="right")


def _marginal_mean(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Each row's own bin-mean of ``values`` -- the same
    ``bincount``-based aggregation pattern used throughout
    ``_weak_learner.py``."""
    n_bins = int(bins.max()) + 1
    sums = np.bincount(bins, weights=values, minlength=n_bins)
    counts = np.bincount(bins, minlength=n_bins).astype(float)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return means[bins]


def purify_contributions(
    contrib: pd.DataFrame,
    X: pd.DataFrame,
    categorical_features: set,
    n_bins: int = 10,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> pd.DataFrame:
    """Purifies every pairwise-interaction column of ``contrib`` (an
    ``explain(X)`` output) against its two constituent main-effect
    columns, on a **copy** -- never mutates ``contrib`` or the model
    that produced it.

    For each ``"A x B"`` column where both ``"A"`` and ``"B"`` are also
    columns of ``contrib``: cyclically (until convergence or
    ``max_iter``) computes each row's own bin-mean of the interaction
    (binning ``X[A]``/``X[B]`` for this purpose only, see
    :func:`_reference_bins`), moves that mean into the corresponding
    main effect, and subtracts it from the interaction -- so every
    row's own ``contrib["A"] + contrib["B"] + contrib["A x B"]`` is
    unchanged (verified directly in tests), while the *split* between
    them becomes canonical relative to ``X``'s own empirical
    distribution: two differently-seeded refits of a similar
    relationship converge toward the same main-effect/interaction split
    instead of shuffling predictively-identical signal between them
    arbitrarily.

    Triples are not purified (deferred -- purifying a triple against
    its constituent pairs *and* mains is a more complex recursive
    extension); main effects are not re-centered into ``"baseline"`` (a
    separate part of the full functional-ANOVA convention, not attempted
    here).

    Purification is defined **relative to the specific `X` passed in**
    -- the empirical measure it marginalizes against. Calling it on two
    different datasets can give different canonical splits; pass a
    representative dataset (e.g. the training data) for a stable result.

    Parameters
    ----------
    contrib : DataFrame
        ``explain(X)``'s own output.
    X : DataFrame
        The same ``X`` passed to ``explain``.
    categorical_features : set
        Column names to bin as exact categories rather than quantiles
        (the model's own ``categorical_features_``).
    n_bins : int, default=10
        Quantile bins for continuous columns.
    max_iter : int, default=50
    tol : float, default=1e-10
        Convergence threshold on the largest remaining per-bin mean.

    Returns
    -------
    DataFrame, same shape as ``contrib``.
    """
    contrib = contrib.copy()
    pair_terms = []
    for col in contrib.columns:
        if col == "baseline":
            continue
        parts = col.split(" x ")
        if len(parts) == 2 and parts[0] in contrib.columns and parts[1] in contrib.columns:
            pair_terms.append((col, parts[0], parts[1]))

    for pair_col, a, b in pair_terms:
        bins_a = _reference_bins(X[a], a in categorical_features, n_bins)
        bins_b = _reference_bins(X[b], b in categorical_features, n_bins)
        values = contrib[pair_col].to_numpy(dtype=float)
        main_a = contrib[a].to_numpy(dtype=float)
        main_b = contrib[b].to_numpy(dtype=float)

        for _ in range(max_iter):
            g = _marginal_mean(values, bins_a)
            main_a = main_a + g
            values = values - g

            h = _marginal_mean(values, bins_b)
            main_b = main_b + h
            values = values - h

            if max(np.max(np.abs(g)), np.max(np.abs(h))) < tol:
                break

        contrib[a] = main_a
        contrib[b] = main_b
        contrib[pair_col] = values

    return contrib
