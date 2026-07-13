import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor
from zoneboost._shrinkage import _estimate_shrinkage_m
from zoneboost._weak_learner import (
    _column_n_zones,
    _column_soft_zone_index,
    _column_zone_index,
    _column_zone_info,
    _cross_fitted_contributions,
    _make_folds,
    weak_learner_fit,
)


def test_estimate_shrinkage_m_strong_signal_gives_small_m():
    rng = np.random.default_rng(0)
    true_zone_effects = np.array([5.0, -5.0, 3.0, -3.0, 0.0, 8.0])
    n_per_zone = np.full(6, 200.0)
    residual_var = 1.0
    noisy_means = true_zone_effects + rng.normal(0, np.sqrt(residual_var / n_per_zone))
    deviation = noisy_means  # true grand mean is 0
    m = _estimate_shrinkage_m([(deviation, n_per_zone)], residual_var, fallback_m=10.0)
    assert m < 1.0


def test_estimate_shrinkage_m_pure_noise_gives_heavy_shrinkage():
    rng = np.random.default_rng(1)
    n_per_zone = np.full(6, 50.0)
    residual_var = 1.0
    noisy_means = rng.normal(0, np.sqrt(residual_var / n_per_zone))
    m = _estimate_shrinkage_m([(noisy_means, n_per_zone)], residual_var, fallback_m=10.0)
    assert m >= 10.0


def test_estimate_shrinkage_m_degenerate_inputs_return_fallback():
    # K <= 1 (only one zone with data)
    m1 = _estimate_shrinkage_m([(np.array([1.0]), np.array([10.0]))], 1.0, fallback_m=10.0)
    assert m1 == 10.0
    # residual_var <= 0
    m2 = _estimate_shrinkage_m([(np.array([1.0, 2.0]), np.array([10.0, 10.0]))], 0.0, fallback_m=10.0)
    assert m2 == 10.0
    # all-zero counts
    m3 = _estimate_shrinkage_m([(np.array([1.0, 2.0]), np.array([0.0, 0.0]))], 1.0, fallback_m=10.0)
    assert m3 == 10.0


def _pair_sparse_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    y = (x1**2 + 0.3 * x1 * x2 + rng.normal(0, 1, n)).astype(float)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


def test_pairs_earn_heavier_shrinkage_than_mains_on_average():
    X, y = _pair_sparse_data()
    model = ZoneBoostRegressor(random_state=0, n_rounds=80, learn_shrinkage_m=True).fit(X, y)
    m_mains = [r["diagnostics"]["learned_shrinkage_m"]["main"] for r in model.rounds_]
    m_pairs = [r["diagnostics"]["learned_shrinkage_m"]["pair"] for r in model.rounds_]
    assert np.mean(m_pairs) > np.mean(m_mains)
    assert np.mean(np.array(m_pairs) > np.array(m_mains)) > 0.5


def test_cross_fitted_contributions_m_pair_overrides_interaction_shrinkage():
    X, y = _pair_sparse_data(n=1500)
    zone_info = {c: _column_zone_info(X[c], y, False, 7, 0.02) for c in ["x1", "x2"]}
    n_zones = {c: _column_n_zones(zone_info[c]) for c in ["x1", "x2"]}
    zones = {c: _column_zone_index(X[c], zone_info[c]) for c in ["x1", "x2"]}
    soft = {c: _column_soft_zone_index(X[c], zone_info[c]) for c in ["x1", "x2"]}
    fold_ids = _make_folds(np.random.default_rng(2), len(y), 5)

    contrib_same_m = _cross_fitted_contributions(
        zones, soft, n_zones, y, ["x1", "x2"], [("x1", "x2")], [], fold_ids, 5, 10.0
    )
    contrib_diff_m_pair = _cross_fitted_contributions(
        zones, soft, n_zones, y, ["x1", "x2"], [("x1", "x2")], [], fold_ids, 5, 10.0, m_pair=200.0
    )
    # main effect columns (0, 1) use m=10.0 in both calls -- identical.
    np.testing.assert_array_equal(contrib_same_m[:, :2], contrib_diff_m_pair[:, :2])
    # the interaction column (2) uses a different m in each call -- must differ.
    assert not np.allclose(contrib_same_m[:, 2], contrib_diff_m_pair[:, 2])


def test_weak_learner_fit_diagnostics_present_with_either_flag():
    X, y = _pair_sparse_data(n=800)
    # learn_shrinkage_m alone
    _, _, _, _, _, diag1 = weak_learner_fit(
        X, y, ["x1", "x2"], set(), np.random.default_rng(1), learn_shrinkage_m=True
    )
    assert diag1 is not None
    assert "learned_shrinkage_m" in diag1
    assert "main_effects" not in diag1

    # track_reliability alone
    _, _, _, _, _, diag2 = weak_learner_fit(
        X, y, ["x1", "x2"], set(), np.random.default_rng(1), track_reliability=True
    )
    assert diag2 is not None
    assert "learned_shrinkage_m" not in diag2
    assert "main_effects" in diag2

    # both
    _, _, _, _, _, diag3 = weak_learner_fit(
        X, y, ["x1", "x2"], set(), np.random.default_rng(1), learn_shrinkage_m=True, track_reliability=True
    )
    assert "learned_shrinkage_m" in diag3 and "main_effects" in diag3

    # neither
    _, _, _, _, _, diag4 = weak_learner_fit(X, y, ["x1", "x2"], set(), np.random.default_rng(1))
    assert diag4 is None


def test_learn_shrinkage_m_false_is_bit_identical_default():
    X, y = _pair_sparse_data(n=800)
    model_default = ZoneBoostRegressor(random_state=0, n_rounds=30).fit(X, y)
    model_explicit_false = ZoneBoostRegressor(random_state=0, n_rounds=30, learn_shrinkage_m=False).fit(X, y)
    np.testing.assert_array_equal(model_default.predict(X), model_explicit_false.predict(X))
    assert model_default.rounds_[0]["diagnostics"] is None


def test_learn_shrinkage_m_does_not_change_triples():
    X, y = _pair_sparse_data(n=1500)
    model = ZoneBoostRegressor(
        random_state=0, n_rounds=30, learn_shrinkage_m=True, max_interaction_order=3,
        col_subsample=1.0, track_reliability=True,
    ).fit(X, y)
    # triples (if any were fit) must not appear in learned_shrinkage_m
    for r in model.rounds_:
        assert set(r["diagnostics"]["learned_shrinkage_m"].keys()) == {"main", "pair"}
