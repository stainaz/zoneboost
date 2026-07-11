"""Exact, non-approximate prediction attribution.

Unlike SHAP or LIME, this isn't a post-hoc approximation of a black-box
model -- it's an algebraic decomposition of what the boosting loop already
computed. Each round's contribution to the running score is

    intercept + contributions @ weights

where ``contributions`` holds one column per term (main effects +
interactions + any adaptively-selected 3-way interactions) and
``(intercept, weights)`` is that round's own Lasso fit of the residual on
those per-term contributions (see ``_weak_learner._fit_lasso_weights``).
Because the dot product is already a per-term sum, this expands exactly
into

    round_baseline + sum_i( weight_i * term_i )

with round_baseline = intercept, a fixed per-round constant. Summing that
across rounds gives, for every row, a set of per-term contributions that
add up EXACTLY to the model's prediction (or, for the classifier, to the
pre-sigmoid log-odds score) -- not an estimate of it.

Each ``term_i`` itself is a soft, interpolated lookup for continuous
columns (see ``_weak_learner._column_soft_zone_index`` and
``_blend_1d``/``_blend_2d``/``_blend_3d``) rather than a hard single-zone
value -- this module must use the exact same blend `predict` does, or the
row-sum-equals-prediction invariant above would break.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._weak_learner import _blend_1d, _blend_2d, _blend_3d, _column_soft_zone_index

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
        weights = round_["weights"]

        n_terms = len(main_effects) + len(interactions) + len(triples)
        if n_terms == 0:
            continue
        baseline_total += learning_rate * round_["intercept"]

        # weights is aligned to main_effects -> interactions -> triples, in
        # each dict's own (Python-guaranteed) insertion order -- the exact
        # same order weak_learner_contributions builds its columns in, and
        # the order the round's weights were fit against.
        i = 0
        for col, deviation in main_effects.items():
            z_lo, z_hi, w = _column_soft_zone_index(X[col], zone_info[col])
            share = learning_rate * weights[i] * _blend_1d(deviation, z_lo, z_hi, w)
            term_totals.setdefault(col, np.zeros(n))
            term_totals[col] += share
            i += 1

        for (a, b), deviation in interactions.items():
            za_lo, za_hi, wa = _column_soft_zone_index(X[a], zone_info[a])
            zb_lo, zb_hi, wb = _column_soft_zone_index(X[b], zone_info[b])
            share = learning_rate * weights[i] * _blend_2d(deviation, za_lo, za_hi, wa, zb_lo, zb_hi, wb)
            # Canonicalize: a pair's fit order varies round to round (each
            # round samples/orders columns independently), so without
            # sorting, "A x B" and "B x A" would fragment into separate
            # columns instead of accumulating as the same term.
            key = " x ".join(sorted((a, b)))
            term_totals.setdefault(key, np.zeros(n))
            term_totals[key] += share
            i += 1

        for (a, b, c), deviation in triples.items():
            za_lo, za_hi, wa = _column_soft_zone_index(X[a], zone_info[a])
            zb_lo, zb_hi, wb = _column_soft_zone_index(X[b], zone_info[b])
            zc_lo, zc_hi, wc = _column_soft_zone_index(X[c], zone_info[c])
            share = learning_rate * weights[i] * _blend_3d(
                deviation, za_lo, za_hi, wa, zb_lo, zb_hi, wb, zc_lo, zc_hi, wc
            )
            key = " x ".join(sorted((a, b, c)))
            term_totals.setdefault(key, np.zeros(n))
            term_totals[key] += share
            i += 1

    return pd.DataFrame({"baseline": np.full(n, baseline_total), **term_totals}, index=X.index)
