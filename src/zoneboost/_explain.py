"""Exact, non-approximate prediction attribution.

Unlike SHAP or LIME, this isn't a post-hoc approximation of a black-box
model -- it's an algebraic decomposition of what the boosting loop already
computed. Each round's contribution to the running score is

    alpha + beta * raw

where raw is the *mean* of that round's per-term zone lookups (main
effects + interactions + any adaptively-selected 3-way interactions), and
(alpha, beta) is the round's own OLS fit of the residual on raw (see
``_weak_learner._ols_scale``). Because mean is linear, this expands exactly
into

    round_baseline + sum_i( (beta / n_terms) * term_i )

with round_baseline = alpha, a fixed per-round constant. Summing that
across rounds gives, for every row, a set of per-term contributions that
add up EXACTLY to the model's prediction (or, for the classifier, to the
pre-sigmoid log-odds score) -- not an estimate of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._weak_learner import _column_zone_index

__all__ = ["explain_rounds"]


def explain_rounds(X: pd.DataFrame, rounds: list, baseline: float, learning_rate: float) -> pd.DataFrame:
    """Per-row, per-term contribution breakdown across boosting rounds.

    Parameters
    ----------
    X : DataFrame
    rounds : list
        A (possibly truncated) ``rounds_`` list, as stored on
        ZoneBoostRegressor or one of ZoneBoostClassifier's internal
        boosters.
    baseline : float
        The model's starting value before any round is applied.
    learning_rate : float

    Returns
    -------
    DataFrame of shape (len(X), n_terms + 1)
        One column per term that appeared in any round -- each predictor's
        own name for its main effect, ``"A x B"`` for an interaction pair,
        ``"A x B x C"`` for a 3-way interaction -- plus a ``"baseline"``
        column. Row sums equal the model's raw prediction (regression) or
        log-odds score (classification) exactly.
    """
    n = len(X)
    baseline_total = float(baseline)
    term_totals: dict[str, np.ndarray] = {}

    for round_ in rounds:
        zone_info = round_["zone_info"]
        main_effects = round_["main_effects"]
        interactions = round_["interactions"]
        triples = round_["triples"]
        alpha, beta = round_["alpha"], round_["beta"]

        n_terms = len(main_effects) + len(interactions) + len(triples)
        if n_terms == 0:
            continue
        baseline_total += learning_rate * alpha

        for col, (deviation, confidence) in main_effects.items():
            z = _column_zone_index(X[col], zone_info[col])
            share = learning_rate * (beta / n_terms) * (deviation[z] * confidence[z])
            term_totals.setdefault(col, np.zeros(n))
            term_totals[col] += share

        for (a, b), (deviation, confidence) in interactions.items():
            za = _column_zone_index(X[a], zone_info[a])
            zb = _column_zone_index(X[b], zone_info[b])
            share = learning_rate * (beta / n_terms) * (deviation[za, zb] * confidence[za, zb])
            # Canonicalize: a pair's fit order varies round to round (each
            # round samples/orders columns independently), so without
            # sorting, "A x B" and "B x A" would fragment into separate
            # columns instead of accumulating as the same term.
            key = " x ".join(sorted((a, b)))
            term_totals.setdefault(key, np.zeros(n))
            term_totals[key] += share

        for (a, b, c), (deviation, confidence) in triples.items():
            za = _column_zone_index(X[a], zone_info[a])
            zb = _column_zone_index(X[b], zone_info[b])
            zc = _column_zone_index(X[c], zone_info[c])
            share = learning_rate * (beta / n_terms) * (deviation[za, zb, zc] * confidence[za, zb, zc])
            key = " x ".join(sorted((a, b, c)))
            term_totals.setdefault(key, np.zeros(n))
            term_totals[key] += share

    return pd.DataFrame({"baseline": np.full(n, baseline_total), **term_totals}, index=X.index)
