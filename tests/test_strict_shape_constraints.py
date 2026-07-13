import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso

from zoneboost import ZoneBoostRegressor
from zoneboost._weak_learner import _fit_lasso_weights


def _redundant_monotonic_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = x1 + rng.normal(0, 0.3, n)  # near-duplicate, induces sign ambiguity across rounds
    y = 2 * x1 + rng.normal(0, 1.5, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


def _negative_weight_rounds(model, col):
    count = 0
    for r in model.rounds_:
        if col in r["main_effects"]:
            idx = list(r["main_effects"].keys()).index(col)
            if r["weights"][idx] < 0:
                count += 1
    return count


def _negative_interaction_weight_rounds(model, col):
    count = 0
    for r in model.rounds_:
        keys = list(r["interactions"].keys())
        for i, (a, b) in enumerate(keys):
            if a == col or b == col:
                idx = len(r["main_effects"]) + i
                if r["weights"][idx] < 0:
                    count += 1
    return count


def test_fit_lasso_weights_all_false_mask_matches_plain_lasso():
    rng = np.random.default_rng(0)
    n, p = 500, 5
    X = rng.normal(0, 1, (n, p))
    y = X @ np.array([2.0, -3.0, 0.0, 1.5, -0.5]) + rng.normal(0, 0.5, n)

    intercept_plain, weights_plain = _fit_lasso_weights(X, y, alpha=0.01)
    intercept_masked, weights_masked = _fit_lasso_weights(X, y, alpha=0.01, positive_mask=np.zeros(p, dtype=bool))
    np.testing.assert_allclose(weights_plain, weights_masked)
    assert np.isclose(intercept_plain, intercept_masked)


def test_fit_lasso_weights_all_true_mask_matches_direct_positive_lasso():
    rng = np.random.default_rng(0)
    n, p = 500, 5
    X = rng.normal(0, 1, (n, p))
    y = X @ np.array([2.0, -3.0, 0.0, 1.5, -0.5]) + rng.normal(0, 0.5, n)

    resid_mean, resid_std = float(y.mean()), float(y.std())
    col_std = X.std(axis=0)
    X_std = X / col_std
    y_std = (y - resid_mean) / resid_std
    model = Lasso(alpha=0.01, fit_intercept=True, positive=True, max_iter=10000)
    model.fit(X_std, y_std)
    weights_direct = model.coef_ * (resid_std / col_std)
    intercept_direct = resid_mean + float(model.intercept_) * resid_std

    intercept_masked, weights_masked = _fit_lasso_weights(X, y, alpha=0.01, positive_mask=np.ones(p, dtype=bool))
    np.testing.assert_allclose(weights_masked, weights_direct)
    assert np.isclose(intercept_masked, intercept_direct)


def test_fit_lasso_weights_mixed_mask_clips_only_constrained_column():
    # Column 1's true weight is -3.0 -- forcing it non-negative should clip
    # it to (or toward) 0, while the *unconstrained* fit correctly recovers
    # its true negative sign. Other columns' exact values are free to shift
    # (joint L1 optimization can redistribute explanatory power onto
    # correlated/other columns once one coefficient is pinned -- that's
    # expected, not a bug), so this only checks the one property that must
    # hold: the constrained column's own weight is never negative, and the
    # unconstrained fit's corresponding column is (correctly negative).
    rng = np.random.default_rng(0)
    n, p = 500, 5
    X = rng.normal(0, 1, (n, p))
    true_w = np.array([2.0, -3.0, 0.0, 1.5, -0.5])
    y = X @ true_w + rng.normal(0, 0.5, n)

    mask = np.array([False, True, False, False, False])
    _, weights_mixed = _fit_lasso_weights(X, y, alpha=0.01, positive_mask=mask)
    _, weights_plain = _fit_lasso_weights(X, y, alpha=0.01)

    assert weights_mixed[1] >= 0
    assert weights_plain[1] < 0


def test_fit_lasso_weights_positive_mask_with_quantile_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 3))
    y = X[:, 0] + rng.normal(0, 0.5, 200)
    with pytest.raises(ValueError, match="quantile"):
        _fit_lasso_weights(X, y, alpha=0.01, quantile=0.5, positive_mask=np.array([True, False, False]))


def test_strict_shape_constraints_eliminates_negative_weights_for_monotonic_term():
    X, y = _redundant_monotonic_data()
    model_off = ZoneBoostRegressor(
        random_state=0, n_rounds=100, monotonic_constraints={"x1": 1}, col_subsample=1.0
    ).fit(X, y)
    model_on = ZoneBoostRegressor(
        random_state=0, n_rounds=100, monotonic_constraints={"x1": 1}, col_subsample=1.0,
        strict_shape_constraints=True,
    ).fit(X, y)

    # the unconstrained fit on this engineered (redundant-feature) data must
    # actually produce some negative weights, or this test isn't exercising
    # the gap it claims to fix.
    assert _negative_weight_rounds(model_off, "x1") > 0
    assert _negative_interaction_weight_rounds(model_off, "x1") > 0

    assert _negative_weight_rounds(model_on, "x1") == 0
    assert _negative_interaction_weight_rounds(model_on, "x1") == 0


def test_strict_shape_constraints_convexity_only_constrains_main_effect():
    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.uniform(0, 10, n)
    x2 = x1 + rng.normal(0, 0.3, n)
    y = (x1 - 5) ** 2 + rng.normal(0, 1.5, n)  # convex in x1
    X = pd.DataFrame({"x1": x1, "x2": x2})

    model_on = ZoneBoostRegressor(
        random_state=0, n_rounds=60, convexity_constraints={"x1": 1}, col_subsample=1.0,
        strict_shape_constraints=True,
    ).fit(X, y)
    assert _negative_weight_rounds(model_on, "x1") == 0


def test_strict_shape_constraints_quantile_loss_raises():
    X, y = _redundant_monotonic_data(n=500)
    with pytest.raises(ValueError, match="quantile"):
        ZoneBoostRegressor(
            loss="quantile", monotonic_constraints={"x1": 1}, strict_shape_constraints=True
        ).fit(X, y)


def test_strict_shape_constraints_false_without_constraints_does_not_raise_for_quantile():
    X, y = _redundant_monotonic_data(n=500)
    # strict_shape_constraints=True with loss='quantile' but NO shape
    # constraints declared has nothing to conflict with -- must not raise.
    model = ZoneBoostRegressor(loss="quantile", strict_shape_constraints=True, n_rounds=20).fit(X, y)
    assert np.all(np.isfinite(model.predict(X)))


def test_bounded_effects_unaffected_by_strict_shape_constraints():
    rng = np.random.default_rng(0)
    n = 1500
    x1 = rng.uniform(-10, 10, n)
    y = 3 * np.sin(x1) + rng.normal(0, 0.3, n)
    X = pd.DataFrame({"x1": x1})

    model = ZoneBoostRegressor(
        random_state=0, n_rounds=80, bounded_effects={"x1": (-1.0, 1.0)}, strict_shape_constraints=True,
    ).fit(X, y)
    contrib = model.explain(X)["x1"].to_numpy()
    # bounded_effects only bounds each round's own stored value, not the
    # cumulative total -- strict_shape_constraints must not change this.
    assert contrib.max() - contrib.min() > 2.0  # cumulative range still exceeds the declared width of 2.0


def test_strict_shape_constraints_false_is_bit_identical_default():
    X, y = _redundant_monotonic_data(n=800)
    model_default = ZoneBoostRegressor(
        random_state=0, n_rounds=30, monotonic_constraints={"x1": 1}
    ).fit(X, y)
    model_explicit_false = ZoneBoostRegressor(
        random_state=0, n_rounds=30, monotonic_constraints={"x1": 1}, strict_shape_constraints=False
    ).fit(X, y)
    np.testing.assert_array_equal(model_default.predict(X), model_explicit_false.predict(X))
