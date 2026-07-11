import numpy as np
import pandas as pd

from zoneboost._weak_learner import (
    _column_soft_zone_index,
    _column_zone_info,
    _fit_lasso_weights,
    _make_folds,
    _pair_shrunk_deviation,
    _triple_shrunk_deviation,
    _zone_shrunk_deviation,
    weak_learner_contributions,
    weak_learner_fit,
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


def test_oof_contributions_cannot_see_a_rows_own_value_when_every_zone_is_a_singleton():
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
    zone_info, main_effects, interactions, triples, oof_contributions = weak_learner_fit(
        X, residual, ["id"], {"id"}, fit_rng, cross_fit_folds=5, shrinkage_m=m
    )
    in_sample_contributions = weak_learner_contributions(X, zone_info, main_effects, interactions, triples)

    assert in_sample_contributions.shape == (n, 1)
    assert oof_contributions.shape == (n, 1)
    expected_in_sample = (residual - residual.mean()) / (1 + m)
    np.testing.assert_allclose(in_sample_contributions[:, 0], expected_in_sample, atol=1e-9)
    np.testing.assert_allclose(oof_contributions, 0.0, atol=1e-9)


def test_weak_learner_fit_falls_back_without_crashing_when_rows_fewer_than_folds():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.uniform(-1, 1, 3)})
    residual = rng.normal(size=3)
    fit_rng = np.random.default_rng(1)
    zone_info, main_effects, interactions, triples, oof_contributions = weak_learner_fit(
        X, residual, ["x1"], set(), fit_rng, cross_fit_folds=5
    )
    assert oof_contributions.shape == (3, 1)
    assert np.all(np.isfinite(oof_contributions))


def test_fit_lasso_weights_zeros_out_pure_noise_column():
    rng = np.random.default_rng(0)
    n = 500
    informative = rng.normal(size=n)
    noise = rng.normal(size=n)
    residual = 3.0 * informative + rng.normal(0, 0.1, n)
    contributions = np.column_stack([informative, noise])

    intercept, weights = _fit_lasso_weights(contributions, residual, alpha=0.05)
    assert abs(weights[1]) < abs(weights[0]) * 0.1  # noise column negligible vs informative one
    assert abs(weights[0]) > 0.5  # informative column's weight is substantial


def test_fit_lasso_weights_degenerate_residual_returns_zero_weights():
    contributions = np.random.default_rng(0).normal(size=(50, 3))
    residual = np.full(50, 2.0)  # perfectly constant
    intercept, weights = _fit_lasso_weights(contributions, residual, alpha=0.05)
    assert intercept == 2.0
    np.testing.assert_array_equal(weights, np.zeros(3))


def _step_data(n=200, seed=0):
    # A clean step at x=10 forces exactly one boundary/two real zones, so
    # the centers are known and predictable for testing interpolation.
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.uniform(0, 20, n))
    y = (x > 10).astype(float) * 5.0 + rng.normal(0, 0.1, n)
    return x, y.to_numpy()


def test_soft_zone_index_weight_is_zero_at_its_own_centroid():
    x, y = _step_data()
    info = _column_zone_info(x, y, is_categorical=False, max_zones=2, min_zone_frac=0.02)
    _, boundaries, centers = info
    assert len(centers) == 2

    probe = pd.Series([centers[0], centers[1]])
    z_lo, z_hi, w = _column_soft_zone_index(probe, info)
    np.testing.assert_allclose(w, [0.0, 0.0])  # exactly at its own centroid
    assert z_lo[0] == 0 and z_lo[1] == 1


def test_soft_zone_index_blend_is_continuous_across_the_hard_boundary():
    # The whole point of soft boundaries: the blended VALUE (not just the
    # weight) must be continuous exactly at the hard zone_index switchover
    # -- approaching a boundary from zone 0's side and from zone 1's side
    # must agree, rather than jumping.
    x, y = _step_data()
    info = _column_zone_info(x, y, is_categorical=False, max_zones=2, min_zone_frac=0.02)
    _, boundaries, centers = info
    b = boundaries[0]
    deviation = np.array([-3.0, 7.0])  # arbitrary distinct per-zone values

    eps = 1e-6
    just_below = pd.Series([b - eps])
    just_above = pd.Series([b + eps])
    z_lo_below, z_hi_below, w_below = _column_soft_zone_index(just_below, info)
    z_lo_above, z_hi_above, w_above = _column_soft_zone_index(just_above, info)

    value_below = (1 - w_below[0]) * deviation[z_lo_below[0]] + w_below[0] * deviation[z_hi_below[0]]
    value_above = (1 - w_above[0]) * deviation[z_lo_above[0]] + w_above[0] * deviation[z_hi_above[0]]
    assert abs(value_below - value_above) < 1e-3


def test_soft_zone_index_clamps_past_the_edge_zones():
    x, y = _step_data()
    info = _column_zone_info(x, y, is_categorical=False, max_zones=2, min_zone_frac=0.02)
    _, boundaries, centers = info
    # Further left than zone 0's own centroid, and further right than the
    # last zone's own centroid -- no phantom neighbor to blend toward.
    probe = pd.Series([centers[0] - 100.0, centers[-1] + 100.0])
    _, _, w = _column_soft_zone_index(probe, info)
    np.testing.assert_allclose(w, [0.0, 0.0])


def test_soft_zone_index_categorical_and_missing_always_weight_zero():
    x, y = _step_data()
    info = _column_zone_info(x, y, is_categorical=False, max_zones=2, min_zone_frac=0.02)
    probe = pd.Series([np.nan])
    z_lo, z_hi, w = _column_soft_zone_index(probe, info)
    assert w[0] == 0.0
    assert z_lo[0] == z_hi[0]

    cat_series = pd.Series(["a", "b", "a"])
    cat_info = _column_zone_info(cat_series, np.array([1.0, 2.0, 1.0]), is_categorical=True, max_zones=7, min_zone_frac=0.02)
    z_lo_c, z_hi_c, w_c = _column_soft_zone_index(cat_series, cat_info)
    np.testing.assert_array_equal(z_lo_c, z_hi_c)
    np.testing.assert_allclose(w_c, 0.0)
