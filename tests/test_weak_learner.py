import numpy as np
import pandas as pd

from zoneboost._weak_learner import (
    _make_folds,
    _pair_shrunk_deviation,
    _triple_shrunk_deviation,
    _zone_shrunk_deviation,
    weak_learner_fit,
    weak_learner_score,
)


def _three_way_data(n=600, seed=0):
    # x1, x2, x3 carry real pairwise structure (so the pairwise-importance
    # prefilter has something to latch onto) *plus* a genuine 3-way term
    # that pairwise interactions alone cannot represent -- the realistic
    # case adaptive interaction order targets, not an adversarial one.
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-3, 3, n),
            "x2": rng.uniform(-3, 3, n),
            "x3": rng.uniform(-3, 3, n),
        }
    )
    y = (
        X["x1"] * X["x2"]
        + X["x1"] * X["x3"]
        + X["x2"] * X["x3"]
        + 2.0 * X["x1"] * X["x2"] * X["x3"]
        + rng.normal(0, 0.5, n)
    )
    return X, y.to_numpy()


def test_triple_shrunk_deviation_shape():
    rng = np.random.default_rng(0)
    za = rng.integers(0, 4, 200)
    zb = rng.integers(0, 3, 200)
    zc = rng.integers(0, 5, 200)
    target = rng.normal(size=200)
    deviation = _triple_shrunk_deviation(za, zb, zc, target, float(target.mean()), 4, 3, 5, m=10.0)
    assert deviation.shape == (4, 3, 5)
    assert np.all(np.isfinite(deviation))


def test_zone_shrunk_deviation_matches_hand_computed_m_estimate():
    # 3 zones: zone 0 has 5 rows averaging 10, zone 1 has 1 row at 4,
    # zone 2 has 0 rows. overall_mean fixed at 0 for simplicity.
    zone_values = np.array([0, 0, 0, 0, 0, 1])
    target_values = np.array([9.0, 9.0, 11.0, 11.0, 10.0, 4.0])  # zone 0 mean = 10
    m = 5.0
    deviation = _zone_shrunk_deviation(zone_values, target_values, overall_mean=0.0, n_zones=3, m=m)
    expected_0 = (5 * 10.0 + m * 0.0) / (5 + m)  # = 5.0
    expected_1 = (1 * 4.0 + m * 0.0) / (1 + m)  # = 4/6
    expected_2 = 0.0  # zero count -> falls back exactly to the prior
    np.testing.assert_allclose(deviation, [expected_0, expected_1, expected_2])


def test_pair_shrunk_deviation_sparse_cell_is_pulled_toward_marginal_prior():
    # A sparse joint cell's deviation should sit between 0 (flat global
    # mean) and what its own raw cell mean would say -- pulled toward the
    # additive row+column marginal prior, not the flat global mean.
    rng = np.random.default_rng(0)
    n = 400
    za = rng.integers(0, 3, n)
    zb = rng.integers(0, 3, n)
    target = 2.0 * za + 3.0 * zb + rng.normal(0, 0.1, n)  # real additive marginal structure
    # Force cell (0, 0) to be sparse by removing most of its rows.
    mask = ~((za == 0) & (zb == 0))
    keep = np.where(mask)[0]
    keep = np.concatenate([keep, np.where(~mask)[0][:1]])  # keep exactly 1 row of cell (0,0)
    za, zb, target = za[keep], zb[keep], target[keep]
    overall_mean = float(target.mean())

    deviation = _pair_shrunk_deviation(za, zb, target, overall_mean, 3, 3, m=10.0)
    dev_a = _zone_shrunk_deviation(za, target, overall_mean, 3, m=10.0)
    dev_b = _zone_shrunk_deviation(zb, target, overall_mean, 3, m=10.0)
    marginal_prior_00 = dev_a[0] + dev_b[0]  # relative to overall_mean

    # Sparse cell (0,0)'s shrunk deviation should be much closer to the
    # marginal-based prior than to a naive unshrunk cell mean would be.
    assert abs(deviation[0, 0] - marginal_prior_00) < abs(deviation[0, 0])


def test_max_interaction_order_2_never_produces_triples():
    X, y = _three_way_data()
    residual = y - y.mean()
    rng = np.random.default_rng(0)
    _, _, _, triples, _ = weak_learner_fit(X, residual, list(X.columns), set(), rng, max_interaction_order=2)
    assert triples == {}


def test_max_interaction_order_3_finds_the_genuine_triple():
    X, y = _three_way_data()
    residual = y - y.mean()
    rng = np.random.default_rng(0)
    _, _, _, triples, _ = weak_learner_fit(
        X, residual, list(X.columns), set(), rng, max_interaction_order=3, triple_min_gain=0.01
    )
    assert len(triples) >= 1
    (key,) = triples.keys()
    assert set(key) == {"x1", "x2", "x3"}


def test_max_triple_interactions_caps_count_per_round():
    X, y = _three_way_data(n=800)
    rng = np.random.default_rng(1)
    X = X.copy()
    X["x4"] = rng.uniform(-3, 3, len(X))
    X["x5"] = rng.uniform(-3, 3, len(X))
    residual = y - y.mean()
    _, _, _, triples, _ = weak_learner_fit(
        X,
        residual,
        list(X.columns),
        set(),
        rng,
        max_interaction_order=3,
        max_triple_interactions=1,
        triple_min_gain=0.01,
    )
    assert len(triples) <= 1


def test_high_triple_min_gain_rejects_weak_candidates():
    X, y = _three_way_data()
    residual = y - y.mean()
    rng = np.random.default_rng(0)
    _, _, _, triples, _ = weak_learner_fit(
        X, residual, list(X.columns), set(), rng, max_interaction_order=3, triple_min_gain=1e6
    )
    assert triples == {}


def test_make_folds_covers_every_row_and_is_balanced():
    rng = np.random.default_rng(0)
    n, k = 23, 5
    fold_ids = _make_folds(rng, n, k)
    assert fold_ids.shape == (n,)
    assert set(fold_ids.tolist()) == set(range(k))
    counts = np.bincount(fold_ids, minlength=k)
    assert counts.max() - counts.min() <= 1


def test_oof_raw_cannot_see_a_rows_own_value_when_every_zone_is_a_singleton():
    # Every row is its own category: in-sample, each zone's "mean" is
    # shrunk from that one row's own residual toward the global mean by
    # the m-estimate (never fully reconstructing it, unlike the old
    # confidence-weighted version). Cross-fitted, no other fold ever
    # contains that category, so the honest out-of-fold table has zero
    # support for it everywhere -- deviation falls back to the prior exactly.
    rng = np.random.default_rng(0)
    n = 60
    m = 10.0
    X = pd.DataFrame({"id": [f"c{i}" for i in range(n)]})
    residual = rng.normal(size=n)

    fit_rng = np.random.default_rng(1)
    zone_info, main_effects, interactions, triples, oof_raw = weak_learner_fit(
        X, residual, ["id"], {"id"}, fit_rng, cross_fit_folds=5, shrinkage_m=m
    )
    in_sample_raw = weak_learner_score(X, zone_info, main_effects, interactions, triples)

    expected_in_sample = (residual - residual.mean()) / (1 + m)
    np.testing.assert_allclose(in_sample_raw, expected_in_sample, atol=1e-9)
    np.testing.assert_allclose(oof_raw, 0.0, atol=1e-9)


def test_weak_learner_fit_falls_back_without_crashing_when_rows_fewer_than_folds():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.uniform(-1, 1, 3)})
    residual = rng.normal(size=3)
    fit_rng = np.random.default_rng(1)
    zone_info, main_effects, interactions, triples, oof_raw = weak_learner_fit(
        X, residual, ["x1"], set(), fit_rng, cross_fit_folds=5
    )
    assert oof_raw.shape == (3,)
    assert np.all(np.isfinite(oof_raw))
