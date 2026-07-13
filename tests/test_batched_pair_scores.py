import itertools

import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor
from zoneboost._weak_learner import _batched_pair_scores, _pair_interaction_score


def _reference_scores(zones, n_zones, columns, residual, forbidden_interactions=frozenset()):
    return {
        (a, b): _pair_interaction_score(zones[a], zones[b], residual, n_zones[a], n_zones[b])
        for a, b in itertools.combinations(columns, 2)
        if frozenset((a, b)) not in forbidden_interactions
    }


def _make_zones(rng, columns, n_rows, zone_range=(3, 8)):
    n_zones = {c: int(rng.integers(*zone_range)) for c in columns}
    zones = {c: rng.integers(0, n_zones[c], size=n_rows).astype(np.int64) for c in columns}
    return zones, n_zones


@pytest.mark.parametrize("n_cols,n_rows,seed", [(5, 2000, 0), (3, 500, 1), (10, 4000, 2)])
def test_batched_matches_reference_oracle(n_cols, n_rows, seed):
    rng = np.random.default_rng(seed)
    columns = [f"c{i}" for i in range(n_cols)]
    zones, n_zones = _make_zones(rng, columns, n_rows)
    residual = rng.normal(size=n_rows)

    expected = _reference_scores(zones, n_zones, columns, residual)
    got = _batched_pair_scores(zones, n_zones, columns, residual)

    assert set(got.keys()) == set(expected.keys())
    for k in expected:
        np.testing.assert_allclose(got[k], expected[k], atol=1e-10)


def test_batched_respects_unequal_zone_counts():
    rng = np.random.default_rng(3)
    columns = ["a", "b", "c", "d"]
    n_rows = 3000
    n_zones = {"a": 2, "b": 9, "c": 4, "d": 6}
    zones = {c: rng.integers(0, n_zones[c], size=n_rows).astype(np.int64) for c in columns}
    residual = rng.normal(size=n_rows)

    expected = _reference_scores(zones, n_zones, columns, residual)
    got = _batched_pair_scores(zones, n_zones, columns, residual)

    assert set(got.keys()) == set(expected.keys())
    for k in expected:
        np.testing.assert_allclose(got[k], expected[k], atol=1e-10)


def test_batched_forbidden_interactions_filtered_out():
    rng = np.random.default_rng(4)
    columns = ["a", "b", "c", "d", "e"]
    zones, n_zones = _make_zones(rng, columns, 1500)
    residual = rng.normal(size=1500)
    forbidden = frozenset({frozenset(("a", "b")), frozenset(("c", "e"))})

    got = _batched_pair_scores(zones, n_zones, columns, residual, forbidden)

    assert ("a", "b") not in got
    assert ("c", "e") not in got
    expected = _reference_scores(zones, n_zones, columns, residual, forbidden)
    assert set(got.keys()) == set(expected.keys())
    for k in expected:
        np.testing.assert_allclose(got[k], expected[k], atol=1e-10)


def test_batched_fewer_than_two_columns_returns_empty():
    rng = np.random.default_rng(5)
    zones, n_zones = _make_zones(rng, ["a"], 100)
    residual = rng.normal(size=100)
    assert _batched_pair_scores(zones, n_zones, ["a"], residual) == {}
    assert _batched_pair_scores({}, {}, [], residual) == {}


def test_batched_single_zone_column():
    rng = np.random.default_rng(6)
    columns = ["a", "b"]
    n_rows = 500
    n_zones = {"a": 1, "b": 5}
    zones = {
        "a": np.zeros(n_rows, dtype=np.int64),
        "b": rng.integers(0, n_zones["b"], size=n_rows).astype(np.int64),
    }
    residual = rng.normal(size=n_rows)

    expected = _reference_scores(zones, n_zones, columns, residual)
    got = _batched_pair_scores(zones, n_zones, columns, residual)

    assert set(got.keys()) == set(expected.keys())
    for k in expected:
        np.testing.assert_allclose(got[k], expected[k], atol=1e-10)


def test_batched_default_forbidden_interactions_is_empty():
    rng = np.random.default_rng(7)
    columns = ["a", "b", "c"]
    zones, n_zones = _make_zones(rng, columns, 800)
    residual = rng.normal(size=800)
    got = _batched_pair_scores(zones, n_zones, columns, residual)
    assert len(got) == 3


def test_batched_pair_scores_matches_real_screening_residual():
    # _batched_pair_scores is not wired into the default screening path (see
    # its own docstring for why), but it must still be an exact drop-in
    # replacement wherever a caller does opt into it -- verified here against
    # a *real* screening residual (main-effect signal removed) rather than
    # pure synthetic noise, reusing weak_learner_fit's own internals.
    from zoneboost._weak_learner import (
        _column_n_zones,
        _column_soft_zone_index,
        _column_zone_index,
        _column_zone_info,
        _cross_fitted_contributions,
        _make_folds,
    )

    rng = np.random.default_rng(9)
    n = 1500
    columns = [f"x{i}" for i in range(6)]
    X = pd.DataFrame({c: rng.normal(size=n) for c in columns})
    y = X["x0"] + 0.5 * X["x1"] - 0.3 * X["x2"] + rng.normal(scale=0.4, size=n)
    residual = y.to_numpy()

    zone_info = {c: _column_zone_info(X[c], residual, False, 7, 0.02) for c in columns}
    n_zones = {c: _column_n_zones(zone_info[c]) for c in columns}
    zones = {c: _column_zone_index(X[c], zone_info[c]) for c in columns}
    soft = {c: _column_soft_zone_index(X[c], zone_info[c]) for c in columns}
    fold_ids = _make_folds(rng, n, 5)

    oof_main_pred = _cross_fitted_contributions(
        zones, soft, n_zones, residual, columns, [], [], fold_ids, 5, 10.0,
    ).sum(axis=1)
    screening_residual = residual - oof_main_pred

    expected = _reference_scores(zones, n_zones, columns, screening_residual)
    got = _batched_pair_scores(zones, n_zones, columns, screening_residual)

    assert set(got.keys()) == set(expected.keys())
    for k in expected:
        np.testing.assert_allclose(got[k], expected[k], atol=1e-8)


def test_regressor_screening_path_is_deterministic():
    # end-to-end smoke check that the default (plain per-pair-loop) screening
    # path, exercised whenever max_pair_interactions is set, fits and
    # predicts deterministically.
    rng = np.random.default_rng(8)
    n = 2000
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(10)})
    y = X["x0"] + X["x1"] * X["x2"] - 0.5 * X["x3"] + rng.normal(scale=0.3, size=n)

    model_a = ZoneBoostRegressor(
        n_rounds=15, max_pair_interactions=6, max_interaction_order=2, random_state=1
    ).fit(X, y)
    model_b = ZoneBoostRegressor(
        n_rounds=15, max_pair_interactions=6, max_interaction_order=2, random_state=1
    ).fit(X, y)

    np.testing.assert_array_equal(model_a.predict(X), model_b.predict(X))
