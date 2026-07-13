"""Learning the empirical-Bayes shrinkage strength from the data itself,
instead of trusting a hand-set constant.

``shrinkage_m`` plays the role of ``sigma^2 / tau^2`` in the standard
normal-normal hierarchical model this shrinkage formula already implies
(``sigma^2`` = within-zone/sampling variance, ``tau^2`` = between-zone
variance of the *true* zone effects) -- a well-known equivalence (Efron
& Morris), the same "compound normal means" problem DerSimonian & Laird
solve for random-effects meta-analysis variance components. Rather than
a full numerical marginal-likelihood maximization (this has to run every
round, so cost matters), :func:`_estimate_shrinkage_m` uses the
DerSimonian-Laird estimator: a standard, closed-form, non-iterative
method-of-moments estimate of ``tau^2`` from exactly this kind of data
(a set of noisy group means, each with its own known sampling
variance) -- then ``m_hat = sigma^2_hat / tau^2_hat``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["_estimate_shrinkage_m"]


def _estimate_shrinkage_m(
    deviations: list,
    residual_var: float,
    fallback_m: float,
) -> float:
    """DerSimonian-Laird method-of-moments estimate of the empirical-Bayes
    shrinkage strength, pooling raw zone/cell statistics across every term
    at one interaction level (main effects or pairs) fit this round.

    Parameters
    ----------
    deviations : list of (ndarray, ndarray)
        One ``(deviation, counts)`` pair per column (or pair) at this
        level. ``deviation`` must already be centered by the caller --
        each zone/cell's raw statistic minus *that term's own* reference
        mean (a pair's own mains-removed ``overall_stat``, not one value
        shared across every pair, since each pair's own prior differs).
    residual_var : float
        The round's own residual variance -- stands in for ``sigma^2``,
        the average within-zone/sampling variance (a disclosed
        simplification: the *average* noise level of what's being fit,
        not a per-zone estimate).
    fallback_m : float
        Returned whenever there isn't enough evidence to estimate
        anything better than the user's own default -- degenerate pooled
        data (``K <= 1``), non-positive ``residual_var``, or no
        detectable between-zone signal beyond sampling noise
        (``tau2_hat <= 0``). Never returns *less* shrinkage than this.

    Returns
    -------
    float
    """
    all_dev = np.concatenate([d for d, _ in deviations])
    all_n = np.concatenate([c for _, c in deviations])
    mask = all_n > 0
    all_dev, all_n = all_dev[mask], all_n[mask]

    k = len(all_dev)
    if k <= 1 or residual_var <= 0:
        return fallback_m

    w = all_n / residual_var
    sum_w = w.sum()
    if sum_w <= 0:
        return fallback_m

    q = float(np.sum(w * all_dev**2))
    denom = sum_w - np.sum(w**2) / sum_w
    if denom <= 0:
        return fallback_m

    tau2 = max(0.0, (q - (k - 1)) / denom)
    if tau2 <= 0:
        return fallback_m

    return residual_var / tau2
