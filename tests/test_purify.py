import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor
from zoneboost._purify import _marginal_mean, _reference_bins, purify_contributions


def _correlated_interaction_data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3, 3, n)
    x2 = x1 * 0.7 + rng.normal(0, 1.2, n)
    y = 0.5 * x1 + 0.3 * x2 + 0.4 * x1 * x2 + rng.normal(0, 0.3, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


def _leaky_contrib_table(n=2000, seed=0):
    # a hand-built contribution table with a deliberate main-effect signal
    # (0.4 * x1) hiding inside the interaction column, on top of a genuine
    # interaction (0.2 * x1 * x2) -- purification should recover it.
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    main1 = 0.5 * x1
    main2 = 0.3 * x2
    leaked = 0.4 * x1
    genuine_interaction = 0.2 * x1 * x2
    contrib = pd.DataFrame(
        {
            "baseline": np.zeros(n),
            "x1": main1,
            "x2": main2,
            "x1 x x2": leaked + genuine_interaction,
        }
    )
    return contrib, X


def test_purify_contributions_preserves_row_sums():
    contrib, X = _leaky_contrib_table()
    before = contrib.sum(axis=1).to_numpy()
    purified = purify_contributions(contrib, X, categorical_features=set())
    after = purified.sum(axis=1).to_numpy()
    np.testing.assert_allclose(before, after, atol=1e-8)


def test_purify_contributions_does_not_mutate_input():
    contrib, X = _leaky_contrib_table()
    original = contrib.copy()
    purify_contributions(contrib, X, categorical_features=set())
    pd.testing.assert_frame_equal(contrib, original)


def test_purify_redistributes_leaked_main_effect_signal():
    contrib, X = _leaky_contrib_table()
    purified = purify_contributions(contrib, X, categorical_features=set())
    # x1's main effect should have gained magnitude (absorbed the leak),
    # the interaction should have lost it.
    assert purified["x1"].abs().mean() > contrib["x1"].abs().mean()
    assert purified["x1 x x2"].abs().mean() < contrib["x1 x x2"].abs().mean()


def test_purify_marginal_means_near_zero_after_purification():
    contrib, X = _leaky_contrib_table()
    bins_a = _reference_bins(X["x1"], is_categorical=False, n_bins=10)
    bins_b = _reference_bins(X["x2"], is_categorical=False, n_bins=10)

    g_before = _marginal_mean(contrib["x1 x x2"].to_numpy(), bins_a)
    assert np.max(np.abs(g_before)) > 0.5  # genuinely non-trivial leak before

    purified = purify_contributions(contrib, X, categorical_features=set())
    g_after = _marginal_mean(purified["x1 x x2"].to_numpy(), bins_a)
    h_after = _marginal_mean(purified["x1 x x2"].to_numpy(), bins_b)
    assert np.max(np.abs(g_after)) < 1e-8
    assert np.max(np.abs(h_after)) < 1e-8


def test_purify_only_touches_pairs_with_both_main_effects_present():
    contrib, X = _leaky_contrib_table()
    # drop 'x2' so 'x1 x x2' no longer has both constituents present
    contrib_no_x2 = contrib.drop(columns=["x2"])
    purified = purify_contributions(contrib_no_x2, X, categorical_features=set())
    pd.testing.assert_series_equal(purified["x1 x x2"], contrib_no_x2["x1 x x2"])
    pd.testing.assert_series_equal(purified["x1"], contrib_no_x2["x1"])


def test_purify_is_idempotent():
    contrib, X = _leaky_contrib_table()
    once = purify_contributions(contrib, X, categorical_features=set())
    twice = purify_contributions(once, X, categorical_features=set())
    pd.testing.assert_frame_equal(once, twice, atol=1e-6)


def test_explain_purify_predict_invariance_on_fitted_model():
    X, y = _correlated_interaction_data()
    model = ZoneBoostRegressor(random_state=0, n_rounds=10, col_subsample=1.0, learning_rate=0.3).fit(X, y)
    pred = model.predict(X)

    contrib_unpurified = model.explain(X)
    contrib_purified = model.explain(X, purify=True)

    np.testing.assert_allclose(contrib_unpurified.sum(axis=1).to_numpy(), pred, atol=1e-8)
    np.testing.assert_allclose(contrib_purified.sum(axis=1).to_numpy(), pred, atol=1e-8)


def test_feature_importance_purify_reflects_explain_purify():
    X, y = _correlated_interaction_data()
    model = ZoneBoostRegressor(random_state=0, n_rounds=10, col_subsample=1.0, learning_rate=0.3).fit(X, y)

    fi_before = model.feature_importance(X)
    fi_after = model.feature_importance(X, purify=True)
    # on genuinely correlated x1/x2 with a real interaction, purification
    # should meaningfully shrink the interaction's apparent importance.
    assert fi_after["x1 x x2"] < fi_before["x1 x x2"]

    contrib_purified = model.explain(X, purify=True).drop(columns=["baseline"])
    expected_fi = contrib_purified.abs().mean().sort_values(ascending=False)
    pd.testing.assert_series_equal(fi_after, expected_fi)


def test_purify_reduces_cross_seed_variance_of_interaction_importance():
    X, y = _correlated_interaction_data()
    unpurified_importances = []
    purified_importances = []
    for seed in (0, 1, 2):
        model = ZoneBoostRegressor(
            random_state=seed, n_rounds=10, col_subsample=1.0, learning_rate=0.3
        ).fit(X, y)
        unpurified_importances.append(model.feature_importance(X)["x1 x x2"])
        purified_importances.append(model.feature_importance(X, purify=True)["x1 x x2"])
    assert np.std(purified_importances) <= np.std(unpurified_importances)


def test_categorical_column_uses_exact_groups():
    n = 600
    rng = np.random.default_rng(0)
    cat = rng.choice(["a", "b", "c"], n)
    x1 = rng.uniform(-3, 3, n)
    X = pd.DataFrame({"x1": x1, "cat": cat})
    leaked = np.select([cat == "a", cat == "b", cat == "c"], [1.0, -1.0, 0.5])
    contrib = pd.DataFrame(
        {
            "baseline": np.zeros(n),
            "x1": np.zeros(n),
            "cat": np.zeros(n),
            "x1 x cat": leaked + 0.1 * x1,
        }
    )
    purified = purify_contributions(contrib, X, categorical_features={"cat"})
    # each exact category's marginal mean should now be ~0
    for level in ("a", "b", "c"):
        mask = (cat == level)
        assert abs(purified.loc[mask, "x1 x cat"].mean()) < 1e-6


def test_purify_false_default_is_bit_identical():
    X, y = _correlated_interaction_data(n=500)
    model = ZoneBoostRegressor(random_state=0, n_rounds=20).fit(X, y)
    contrib_default = model.explain(X)
    contrib_explicit_false = model.explain(X, purify=False)
    pd.testing.assert_frame_equal(contrib_default, contrib_explicit_false)
