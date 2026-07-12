import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor


def _synthetic_regression(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-5, 5, n),
            "x2": rng.uniform(-5, 5, n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    bump = np.where(X["cat"] == "b", 10.0, 0.0)
    y = X["x1"] ** 2 + bump + rng.normal(0, 1, n)
    return X, y.to_numpy()


def test_fit_predict_shape_and_reasonable_fit():
    X, y = _synthetic_regression()
    model = ZoneBoostRegressor(n_rounds=50, categorical_features=["cat"], random_state=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(y),)
    # Should explain most of the (mostly deterministic) structure.
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.7


def test_predict_before_fit_raises():
    model = ZoneBoostRegressor(n_rounds=10)
    with pytest.raises(Exception):
        model.predict(pd.DataFrame({"x": [1, 2, 3]}))


def test_categorical_auto_detection_from_dtype():
    X, y = _synthetic_regression()
    # "cat" is object dtype -- should be auto-detected even without
    # declaring it via categorical_features.
    model = ZoneBoostRegressor(n_rounds=20, random_state=0)
    model.fit(X, y)
    assert "cat" in model.categorical_features_


def test_unseen_category_at_predict_time_does_not_crash():
    X, y = _synthetic_regression()
    model = ZoneBoostRegressor(n_rounds=20, categorical_features=["cat"], random_state=0)
    model.fit(X, y)

    X_new = X.copy()
    X_new.loc[0, "cat"] = "never_seen_before"
    pred = model.predict(X_new)
    assert np.all(np.isfinite(pred))


def test_reproducible_with_same_random_state():
    X, y = _synthetic_regression()
    model_a = ZoneBoostRegressor(n_rounds=20, categorical_features=["cat"], random_state=7).fit(X, y)
    model_b = ZoneBoostRegressor(n_rounds=20, categorical_features=["cat"], random_state=7).fit(X, y)
    np.testing.assert_array_equal(model_a.predict(X), model_b.predict(X))


def test_accepts_numpy_array_input():
    X, y = _synthetic_regression()
    X_arr = X[["x1", "x2"]].to_numpy()  # drop categorical col to keep this purely numeric
    model = ZoneBoostRegressor(n_rounds=20, random_state=0)
    model.fit(X_arr, y)
    pred = model.predict(X_arr)
    assert pred.shape == (len(y),)


def test_validation_fraction_zero_disables_early_stopping():
    X, y = _synthetic_regression(n=100)
    model = ZoneBoostRegressor(n_rounds=15, validation_fraction=0, random_state=0)
    model.fit(X[["x1", "x2"]], y)
    assert model.best_n_rounds_ == 15
    assert model.val_rmse_ == []


def test_n_iter_no_change_can_stop_before_n_rounds():
    X, y = _synthetic_regression(n=300)
    model = ZoneBoostRegressor(
        n_rounds=500, validation_fraction=0.2, n_iter_no_change=5, random_state=0
    )
    model.fit(X[["x1", "x2"]], y)
    assert len(model.rounds_) <= 500


def test_tiny_dataset_does_not_crash():
    X = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    model = ZoneBoostRegressor(n_rounds=5, validation_fraction=0.25, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (8,)


def test_predict_n_rounds_override():
    X, y = _synthetic_regression()
    model = ZoneBoostRegressor(n_rounds=30, categorical_features=["cat"], random_state=0)
    model.fit(X, y)
    pred_5 = model.predict(X, n_rounds=5)
    pred_all = model.predict(X, n_rounds=model.best_n_rounds_)
    assert not np.array_equal(pred_5, pred_all)


def test_missing_values_in_continuous_and_categorical_columns_do_not_crash():
    X, y = _synthetic_regression()
    X_missing = X.copy()
    X_missing.loc[X_missing.sample(20, random_state=1).index, "x1"] = np.nan
    X_missing.loc[X_missing.sample(20, random_state=2).index, "cat"] = np.nan

    model = ZoneBoostRegressor(n_rounds=30, categorical_features=["cat"], random_state=0)
    model.fit(X_missing, y)
    pred = model.predict(X_missing)
    assert np.all(np.isfinite(pred))


def test_explain_sums_exactly_to_predict_with_missing_values_present():
    X, y = _synthetic_regression()
    X_missing = X.copy()
    X_missing.loc[X_missing.sample(20, random_state=1).index, "x1"] = np.nan
    X_missing.loc[X_missing.sample(20, random_state=2).index, "cat"] = np.nan

    model = ZoneBoostRegressor(n_rounds=30, categorical_features=["cat"], random_state=0).fit(X_missing, y)
    pred = model.predict(X_missing)
    contrib = model.explain(X_missing)
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy(), pred, atol=1e-6)


def _three_way_interaction_data(n=600, seed=0):
    # x1/x2/x3 carry real pairwise structure plus a genuine 3-way term
    # pairwise interactions alone cannot represent.
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
    ).to_numpy()
    return X, y


def test_max_interaction_order_2_is_default_and_never_produces_triples():
    X, y = _three_way_interaction_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0).fit(X, y)
    assert all(round_["triples"] == {} for round_ in model.rounds_)


def test_cross_fitting_does_not_overfit_pure_noise_high_cardinality_categorical():
    # Classic leakage-prone shape: a high-cardinality categorical with only
    # a handful of rows per zone, against a target with zero real signal.
    # Without cross-fitted cell means, each sparse zone's mean partly
    # encodes its own training rows' noise, which the boosting loop would
    # chase round after round. With it, the model shouldn't find spurious
    # structure in pure noise.
    rng = np.random.default_rng(0)
    n_train, n_test, n_categories = 600, 300, 150
    X_train = pd.DataFrame({"id": rng.integers(0, n_categories, n_train).astype(str)})
    X_test = pd.DataFrame({"id": rng.integers(0, n_categories, n_test).astype(str)})
    y_train = rng.normal(size=n_train)
    y_test = rng.normal(size=n_test)

    model = ZoneBoostRegressor(n_rounds=100, categorical_features=["id"], random_state=0).fit(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    assert test_r2 > -0.5


def test_ols_rescale_stays_stable_on_pure_noise_without_early_stopping():
    # The decisive case: with early stopping disabled, all rounds are
    # forced through unchecked. A std-ratio rescale explodes here once
    # cross-fitting honestly reveals near-zero signal (raw_std -> 0,
    # dividing by it amplifies noise instead of shrinking to it); OLS
    # naturally produces a small beta instead, keeping the model stable.
    rng = np.random.default_rng(0)
    n_train, n_test, n_categories = 600, 300, 150
    X_train = pd.DataFrame({"id": rng.integers(0, n_categories, n_train).astype(str)})
    X_test = pd.DataFrame({"id": rng.integers(0, n_categories, n_test).astype(str)})
    y_train = rng.normal(size=n_train)
    y_test = rng.normal(size=n_test)

    model = ZoneBoostRegressor(
        n_rounds=100, categorical_features=["id"], random_state=0, validation_fraction=0
    ).fit(X_train, y_train)
    assert model.best_n_rounds_ == 100  # confirms early stopping really was off
    test_r2 = model.score(X_test, y_test)
    assert test_r2 > -0.5


def test_shrinkage_recovers_real_high_cardinality_group_effects():
    # Unlike the pure-noise tests above (which check the model doesn't
    # overfit sparse zones), this checks the complementary property:
    # with genuine group-level signal but few rows per category, the
    # empirical-Bayes shrinkage should still recover it well on held-out
    # data from the same categories.
    rng = np.random.default_rng(0)
    n_categories, rows_per_cat = 150, 4
    true_effects = rng.normal(0, 2.0, n_categories)
    n_train = n_categories * rows_per_cat
    train_ids = rng.integers(0, n_categories, n_train)
    y_train = true_effects[train_ids] + rng.normal(0, 1.0, n_train)
    n_test = n_categories * 20
    test_ids = rng.integers(0, n_categories, n_test)
    y_test = true_effects[test_ids] + rng.normal(0, 1.0, n_test)
    X_train = pd.DataFrame({"id": train_ids.astype(str)})
    X_test = pd.DataFrame({"id": test_ids.astype(str)})

    model = ZoneBoostRegressor(n_rounds=100, categorical_features=["id"], random_state=0).fit(X_train, y_train)
    assert model.score(X_test, y_test) > 0.5


def test_lasso_stacking_separates_real_interaction_from_noise_pairs():
    # Lasso stacking's promised benefit: a genuine interaction should be
    # weighted up and ranked clearly above pairs/mains built from pure
    # noise columns, rather than all terms getting the same diluted
    # 1/n_terms share regardless of relevance.
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame(
        {
            "real1": rng.uniform(-3, 3, n),
            "real2": rng.uniform(-3, 3, n),
            **{f"noise{i}": rng.uniform(-3, 3, n) for i in range(6)},
        }
    )
    y = (3.0 * X["real1"] * X["real2"] + rng.normal(0, 0.3, n)).to_numpy()

    model = ZoneBoostRegressor(n_rounds=100, random_state=0).fit(X, y)
    importance = model.feature_importance(X)

    assert importance.index[0] == "real1 x real2"
    top_noise_importance = max(v for k, v in importance.items() if "noise" in k)
    assert importance.iloc[0] > 5 * top_noise_importance


def test_soft_zone_boundaries_kill_the_cliff_edge_at_a_real_split():
    # A sharp step function forces exactly one real zone boundary. Without
    # soft boundaries, predict() jumps almost the full step size (~5) over
    # an infinitesimal step across it; with them, the jump should be a
    # small fraction of that.
    rng = np.random.default_rng(0)
    n = 800
    X = pd.DataFrame({"x": rng.uniform(0, 20, n)})
    y = (X["x"] > 10).astype(float).to_numpy() * 5.0 + rng.normal(0, 0.2, n)

    model = ZoneBoostRegressor(n_rounds=50, random_state=0, validation_fraction=0).fit(X, y)
    grid = pd.DataFrame({"x": np.linspace(9.5, 10.5, 41)})
    preds = model.predict(grid)
    biggest_jump = np.max(np.abs(np.diff(preds)))
    assert biggest_jump < 1.0  # well under the ~5.0 jump hard boundaries would produce


def test_max_interaction_order_3_improves_fit_on_genuine_triple_interaction():
    # col_subsample=1.0: with only 3 predictors, the default 0.7 subsample
    # rounds down to 2 columns per round, which never gives a 3-way search
    # a chance -- using all 3 columns every round is the realistic setting
    # for a model with this few predictors.
    # n_rounds=100: at 60 (the previous value) the two models' RMSE are close
    # enough that soft zone boundaries' small change to per-round dynamics
    # can flip which one edges out the other by noise; 100 rounds gives the
    # genuine triple signal enough rounds to clearly separate out.
    X, y = _three_way_interaction_data()
    model_pairwise = ZoneBoostRegressor(
        n_rounds=100, random_state=0, col_subsample=1.0, max_interaction_order=2
    ).fit(X, y)
    model_triples = ZoneBoostRegressor(
        n_rounds=100, random_state=0, col_subsample=1.0, max_interaction_order=3
    ).fit(X, y)

    rmse_pairwise = np.sqrt(np.mean((y - model_pairwise.predict(X)) ** 2))
    rmse_triples = np.sqrt(np.mean((y - model_triples.predict(X)) ** 2))
    assert rmse_triples < rmse_pairwise
    assert any(len(round_["triples"]) > 0 for round_ in model_triples.rounds_)


def _non_monotonic_looking_data(n=800, seed=0):
    # An overall increasing trend in x with a real, noisy dip in the
    # middle -- without a constraint the fitted main effect should NOT
    # be monotonic; with monotonic_constraints={"x": 1} it should be.
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    true_effect = x + np.where((x > 4) & (x < 6), -2.0, 0.0)
    y = true_effect + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x": x}), y


def test_monotonic_constraints_forces_non_decreasing_main_effect():
    X, y = _non_monotonic_looking_data()
    model = ZoneBoostRegressor(
        n_rounds=80, random_state=0, validation_fraction=0, monotonic_constraints={"x": 1}
    ).fit(X, y)
    assert model.monotonic_constraints_ == {"x": 1}

    grid = pd.DataFrame({"x": np.linspace(0.1, 9.9, 60)})
    contrib = model.explain(grid)
    effect = contrib["x"].to_numpy()
    assert np.all(np.diff(effect) >= -1e-9)


def test_without_monotonic_constraints_the_same_data_fits_a_non_monotonic_effect():
    X, y = _non_monotonic_looking_data()
    model = ZoneBoostRegressor(n_rounds=80, random_state=0, validation_fraction=0).fit(X, y)
    assert model.monotonic_constraints_ == {}

    grid = pd.DataFrame({"x": np.linspace(0.1, 9.9, 60)})
    contrib = model.explain(grid)
    effect = contrib["x"].to_numpy()
    assert np.any(np.diff(effect) < -1e-9)


def test_monotonic_constraints_explain_still_sums_exactly_to_predict():
    X, y = _non_monotonic_looking_data()
    model = ZoneBoostRegressor(
        n_rounds=40, random_state=0, validation_fraction=0, monotonic_constraints={"x": 1}
    ).fit(X, y)
    pred = model.predict(X)
    contrib = model.explain(X)
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy(), pred, atol=1e-6)


def test_monotonic_constraints_default_none_reproduces_unconstrained_predictions():
    X, y = _non_monotonic_looking_data()
    model_default = ZoneBoostRegressor(n_rounds=40, random_state=0, validation_fraction=0).fit(X, y)
    model_explicit_none = ZoneBoostRegressor(
        n_rounds=40, random_state=0, validation_fraction=0, monotonic_constraints=None
    ).fit(X, y)
    np.testing.assert_array_equal(model_default.predict(X), model_explicit_none.predict(X))


def test_max_pair_interactions_default_none_reproduces_unconstrained_predictions():
    X, y = _synthetic_regression()
    model_default = ZoneBoostRegressor(n_rounds=30, categorical_features=["cat"], random_state=0).fit(X, y)
    model_explicit_none = ZoneBoostRegressor(
        n_rounds=30, categorical_features=["cat"], random_state=0, max_pair_interactions=None
    ).fit(X, y)
    np.testing.assert_array_equal(model_default.predict(X), model_explicit_none.predict(X))


def test_max_pair_interactions_with_many_noise_columns_stays_finite_and_consistent():
    rng = np.random.default_rng(0)
    n = 400
    X, y = _synthetic_regression(n=n)
    for i in range(15):
        X[f"noise{i}"] = rng.uniform(-1, 1, n)

    model = ZoneBoostRegressor(
        n_rounds=30, categorical_features=["cat"], random_state=0, max_pair_interactions=3
    ).fit(X, y)
    pred = model.predict(X)
    assert np.all(np.isfinite(pred))

    contrib = model.explain(X)
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy(), pred, atol=1e-6)


def test_cyclic_backfitting_keeps_pair_importance_small_when_no_real_interaction():
    # A strong main effect in x1 with x2 independent and carrying no real
    # interaction -- end-to-end version of the weak-learner-level backfitting
    # test, confirming feature_importance()/explain() (not just the internal
    # deviation arrays) reflect the fix: the "x1 x x2" term should stay small
    # relative to x1's own main effect, not misleadingly large.
    rng = np.random.default_rng(0)
    n = 600
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    y = x1**2 + rng.normal(0, 0.1, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})

    model = ZoneBoostRegressor(n_rounds=60, random_state=0, validation_fraction=0).fit(X, y)
    importance = model.feature_importance(X)
    assert importance["x1 x x2"] < 0.2 * importance["x1"]


def _noisy_quadratic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = x**2 + rng.normal(0, 1.0, n)
    return pd.DataFrame({"x": x}), y


def test_predict_interval_bounds_contain_point_prediction_and_match_stored_scores():
    X, y = _noisy_quadratic()
    model = ZoneBoostRegressor(n_rounds=60, random_state=0).fit(X, y)
    pred = model.predict(X)
    lower, upper = model.predict_interval(X, alpha=0.1)
    assert np.all(lower <= pred) and np.all(pred <= upper)

    n = len(model.conformal_scores_)
    k = min(int(np.ceil((n + 1) * 0.9)), n)
    expected_margin = model.conformal_scores_[k - 1]
    np.testing.assert_allclose(upper - pred, expected_margin)
    np.testing.assert_allclose(pred - lower, expected_margin)


def test_predict_interval_achieves_target_coverage_on_held_out_data():
    X, y = _noisy_quadratic(n=2500)
    X_train, y_train = X.iloc[:2000], y[:2000]
    X_test, y_test = X.iloc[2000:].reset_index(drop=True), y[2000:]

    model = ZoneBoostRegressor(n_rounds=60, random_state=0).fit(X_train, y_train)
    lower, upper = model.predict_interval(X_test, alpha=0.1)
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    assert 0.80 <= coverage <= 0.98


def test_predict_interval_raises_without_validation_split():
    X, y = _noisy_quadratic()
    model = ZoneBoostRegressor(n_rounds=30, random_state=0, validation_fraction=0).fit(X, y)
    assert model.conformal_scores_ is None
    with pytest.raises(ValueError):
        model.predict_interval(X)


def test_calibration_fraction_default_zero_reproduces_unconstrained_predictions():
    X, y = _noisy_quadratic()
    model_default = ZoneBoostRegressor(n_rounds=40, random_state=0).fit(X, y)
    model_explicit = ZoneBoostRegressor(n_rounds=40, random_state=0, calibration_fraction=0.0).fit(X, y)
    np.testing.assert_array_equal(model_default.predict(X), model_explicit.predict(X))
    np.testing.assert_array_equal(model_default.conformal_scores_, model_explicit.conformal_scores_)


def test_calibration_fraction_uses_a_dedicated_split_disjoint_from_fit_and_val():
    X, y = _noisy_quadratic(n=3000)
    model = ZoneBoostRegressor(
        n_rounds=30, random_state=0, validation_fraction=0.2, calibration_fraction=0.1
    ).fit(X, y)
    assert len(model.conformal_scores_) == int(3000 * 0.1)

    # A dedicated calibration split should not match the size (or, in
    # aggregate scale) of what reusing X_val alone would have produced.
    model_no_cal = ZoneBoostRegressor(
        n_rounds=30, random_state=0, validation_fraction=0.2, calibration_fraction=0.0
    ).fit(X, y)
    assert len(model_no_cal.conformal_scores_) == max(1, int(3000 * 0.2))
    assert len(model.conformal_scores_) != len(model_no_cal.conformal_scores_)


def test_refit_on_full_data_requires_calibration_fraction():
    X, y = _noisy_quadratic()
    with pytest.raises(ValueError):
        ZoneBoostRegressor(n_rounds=30, random_state=0, refit_on_full_data=True, calibration_fraction=0.0).fit(X, y)


def test_refit_on_full_data_trains_on_more_rows_than_fit_split_alone():
    X, y = _noisy_quadratic(n=2000)
    model_no_refit = ZoneBoostRegressor(
        n_rounds=40, random_state=0, validation_fraction=0.3, calibration_fraction=0.1
    ).fit(X, y)
    model_refit = ZoneBoostRegressor(
        n_rounds=40,
        random_state=0,
        validation_fraction=0.3,
        calibration_fraction=0.1,
        refit_on_full_data=True,
    ).fit(X, y)

    # best_n_rounds_ is decided identically either way (same selection
    # phase); only what trains the deployed rounds_ differs.
    assert model_refit.best_n_rounds_ == model_no_refit.best_n_rounds_
    assert not np.array_equal(model_refit.predict(X), model_no_refit.predict(X))

    pred = model_refit.predict(X)
    contrib = model_refit.explain(X)
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy(), pred, atol=1e-6)


def _heteroscedastic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    noise_scale = 0.2 + 0.3 * x
    y = 2 * x + rng.normal(0, 1, n) * noise_scale
    X = pd.DataFrame({"x": x})
    return X, y


def test_loss_default_squared_error_reproduces_unconstrained_predictions():
    X, y = _synthetic_regression()
    model_default = ZoneBoostRegressor(n_rounds=40, random_state=0).fit(X, y)
    model_explicit = ZoneBoostRegressor(n_rounds=40, random_state=0, loss="squared_error").fit(X, y)
    np.testing.assert_array_equal(model_default.predict(X), model_explicit.predict(X))


def test_loss_invalid_value_raises():
    X, y = _synthetic_regression()
    with pytest.raises(ValueError):
        ZoneBoostRegressor(n_rounds=10, loss="bogus").fit(X, y)


def test_quantile_out_of_range_raises():
    X, y = _synthetic_regression()
    with pytest.raises(ValueError):
        ZoneBoostRegressor(n_rounds=10, loss="quantile", quantile=1.5).fit(X, y)
    with pytest.raises(ValueError):
        ZoneBoostRegressor(n_rounds=10, loss="quantile", quantile=0.0).fit(X, y)


def test_loss_quantile_achieves_target_coverage_on_heteroscedastic_data():
    X, y = _heteroscedastic(n=4000)
    model = ZoneBoostRegressor(
        n_rounds=80, loss="quantile", quantile=0.9, random_state=0, validation_fraction=0.2
    ).fit(X, y)
    coverage = np.mean(y < model.predict(X))
    assert 0.83 <= coverage <= 0.96


def test_loss_quantile_predict_interval_raises():
    X, y = _heteroscedastic()
    model = ZoneBoostRegressor(n_rounds=20, loss="quantile", quantile=0.9, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        model.predict_interval(X)


def _non_monotonic_interaction_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    z = rng.uniform(-3, 3, n)
    interaction_effect = np.where((x > 4) & (x < 6), -2.0, 0.0) * z
    y = x + z + interaction_effect + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x": x, "z": z}), y


def test_monotonic_constraints_inherited_by_interactions():
    X, y = _non_monotonic_interaction_data()
    model_unconstrained = ZoneBoostRegressor(
        n_rounds=60, random_state=0, validation_fraction=0, max_zones=4
    ).fit(X, y)
    model_constrained = ZoneBoostRegressor(
        n_rounds=60, random_state=0, validation_fraction=0, monotonic_constraints={"x": 1}, max_zones=4
    ).fit(X, y)

    grid = pd.DataFrame({"x": np.linspace(0.1, 9.9, 40), "z": np.full(40, 2.0)})
    interaction_col = "x x z"

    unconstrained_effect = model_unconstrained.explain(grid)[interaction_col].to_numpy()
    constrained_effect = model_constrained.explain(grid)[interaction_col].to_numpy()
    assert np.any(np.diff(unconstrained_effect) < -1e-9)
    assert np.all(np.diff(constrained_effect) >= -1e-9)


def _wiggly_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, n)
    true_effect = 0.5 * x**2 + np.where((x > -1) & (x < 1), 3.0, 0.0)
    y = true_effect + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x": x}), y


def test_convexity_constraints_forces_convex_main_effect_each_round():
    # convexity_constraints projects each ROUND's own stored main-effect
    # deviation to have non-decreasing divided-difference slopes across its
    # own zone centroids -- checked directly on rounds_ (what the mechanism
    # actually guarantees), since a round's own Lasso-stacked weight for
    # this term can be negative, so the *cumulative*, multi-round curve
    # explain() shows isn't itself guaranteed convex (a real limitation of
    # combining per-round shape constraints with signed Lasso stacking).
    X, y = _wiggly_data()
    model = ZoneBoostRegressor(
        n_rounds=60, random_state=0, validation_fraction=0, convexity_constraints={"x": 1}, max_zones=7
    ).fit(X, y)
    assert model.convexity_constraints_ == {"x": 1}
    checked_any = False
    for round_ in model.rounds_:
        if "x" not in round_["main_effects"]:
            continue
        dev = round_["main_effects"]["x"]
        centers = round_["zone_info"]["x"][2]
        n_real = len(centers)
        if n_real <= 2:
            continue
        gaps = np.diff(centers)
        slopes = np.diff(dev[:n_real]) / gaps
        assert np.all(np.diff(slopes) >= -1e-6)
        checked_any = True
    assert checked_any


def test_convexity_constraints_default_none_reproduces_unconstrained_predictions():
    X, y = _wiggly_data()
    model_default = ZoneBoostRegressor(n_rounds=40, random_state=0).fit(X, y)
    model_explicit = ZoneBoostRegressor(n_rounds=40, random_state=0, convexity_constraints=None).fit(X, y)
    np.testing.assert_array_equal(model_default.predict(X), model_explicit.predict(X))


def test_bounded_effects_clips_each_rounds_own_main_effect_contribution():
    # bounded_effects clips each ROUND's own stored main-effect deviation,
    # not the cumulative multi-round total -- so this checks rounds_
    # directly (what the mechanism actually guarantees), not explain()'s
    # summed-across-all-rounds column.
    X, y = _synthetic_regression()
    model = ZoneBoostRegressor(
        n_rounds=60, random_state=0, validation_fraction=0, bounded_effects={"x1": (-5.0, 5.0)}
    ).fit(X, y)
    assert model.bounded_effects_ == {"x1": (-5.0, 5.0)}
    checked_any = False
    for round_ in model.rounds_:
        if "x1" not in round_["main_effects"]:
            continue
        dev = round_["main_effects"]["x1"]
        assert dev.min() >= -5.0 - 1e-9
        assert dev.max() <= 5.0 + 1e-9
        checked_any = True
    assert checked_any


def test_bounded_effects_default_none_reproduces_unconstrained_predictions():
    X, y = _synthetic_regression()
    model_default = ZoneBoostRegressor(n_rounds=40, random_state=0).fit(X, y)
    model_explicit = ZoneBoostRegressor(n_rounds=40, random_state=0, bounded_effects=None).fit(X, y)
    np.testing.assert_array_equal(model_default.predict(X), model_explicit.predict(X))


def test_bounded_effects_invalid_bounds_raise():
    X, y = _synthetic_regression()
    with pytest.raises(ValueError):
        ZoneBoostRegressor(n_rounds=10, bounded_effects={"x1": (5.0, -5.0)}).fit(X, y)


def _forbidden_interaction_regressor_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(-3, 3, n), "b": rng.uniform(-3, 3, n)})
    y = (X["a"] * X["b"] + rng.normal(0, 0.3, n)).to_numpy()
    return X, y


def test_forbidden_interactions_produces_zero_measured_interaction_importance():
    X, y = _forbidden_interaction_regressor_data()
    model = ZoneBoostRegressor(
        n_rounds=60, random_state=0, validation_fraction=0, forbidden_interactions=[("a", "b")]
    ).fit(X, y)
    assert model.forbidden_interactions_ == {frozenset({"a", "b"})}
    importance = model.feature_importance(X)
    assert "a x b" not in importance.index


def test_forbidden_interactions_default_none_reproduces_unconstrained_predictions():
    X, y = _forbidden_interaction_regressor_data()
    model_default = ZoneBoostRegressor(n_rounds=40, random_state=0).fit(X, y)
    model_explicit = ZoneBoostRegressor(n_rounds=40, random_state=0, forbidden_interactions=None).fit(X, y)
    np.testing.assert_array_equal(model_default.predict(X), model_explicit.predict(X))


def test_forbidden_interactions_invalid_pair_raises():
    X, y = _forbidden_interaction_regressor_data()
    with pytest.raises(ValueError):
        ZoneBoostRegressor(n_rounds=10, forbidden_interactions=[("a", "a")]).fit(X, y)


def _dense_and_sparse_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x_dense = rng.uniform(0, 5, int(n * 0.97))
    x_sparse = rng.uniform(20, 25, int(n * 0.03))
    x = np.concatenate([x_dense, x_sparse])
    y = x + rng.normal(0, 0.5, len(x))
    X = pd.DataFrame({"x": x})
    return X, y


def test_explain_include_reliability_requires_track_reliability_at_fit_time():
    X, y = _dense_and_sparse_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        model.explain(X, include_reliability=True)


def test_explain_include_reliability_default_false_bit_identical():
    X, y = _dense_and_sparse_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0, track_reliability=True).fit(X, y)
    contrib_default = model.explain(X)
    contrib_explicit = model.explain(X, include_reliability=False)
    np.testing.assert_array_equal(contrib_default.to_numpy(), contrib_explicit.to_numpy())


def test_reliability_well_supported_zone_vs_sparse_zone():
    X, y = _dense_and_sparse_data()
    model = ZoneBoostRegressor(n_rounds=40, random_state=0, track_reliability=True, max_zones=5).fit(X, y)
    _, reliability = model.explain(X, include_reliability=True)
    rel = reliability["x"]
    dense_mask = (X["x"] < 5).to_numpy()
    sparse_mask = (X["x"] > 20).to_numpy()

    assert rel.loc[dense_mask, "support"].mean() > rel.loc[sparse_mask, "support"].mean()
    # shrinkage_fraction = weight on the prior -- high for sparse (heavily
    # shrunk), low for well-supported (barely shrunk, trusted from data).
    assert rel.loc[dense_mask, "shrinkage_fraction"].mean() < rel.loc[sparse_mask, "shrinkage_fraction"].mean()
    assert rel.loc[dense_mask, "cross_fold_std"].mean() < rel.loc[sparse_mask, "cross_fold_std"].mean()
    assert (rel["n_rounds_present"] > 0).all()


def test_reliability_extrapolation_frac_flags_far_out_of_range_rows():
    X, y = _dense_and_sparse_data()
    model = ZoneBoostRegressor(n_rounds=40, random_state=0, track_reliability=True, max_zones=5).fit(X, y)
    far_row = pd.DataFrame({"x": [1000.0]})
    typical_row = pd.DataFrame({"x": [2.5]})
    _, rel_far = model.explain(far_row, include_reliability=True)
    _, rel_typical = model.explain(typical_row, include_reliability=True)
    assert rel_far["x"]["extrapolation_frac"].iloc[0] == 1.0
    assert rel_typical["x"]["extrapolation_frac"].iloc[0] == 0.0


def test_reliability_boundary_weight_higher_near_boundary_than_at_centroid():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.uniform(0, 10, n)
    y = x + rng.normal(0, 0.3, n)
    X = pd.DataFrame({"x": x})
    model = ZoneBoostRegressor(n_rounds=30, random_state=0, track_reliability=True, max_zones=4).fit(X, y)

    # find a zone's centroid via the last round's zone_info, then compare a
    # row placed exactly there against one placed far toward a neighbor.
    zone_info = model.rounds_[-1]["zone_info"]["x"]
    centers = zone_info[2]
    at_centroid = pd.DataFrame({"x": [float(centers[0])]})
    boundaries = zone_info[1]
    near_boundary = pd.DataFrame({"x": [float(boundaries[0]) - 1e-6]})

    _, rel_centroid = model.explain(at_centroid, include_reliability=True)
    _, rel_boundary = model.explain(near_boundary, include_reliability=True)
    assert rel_centroid["x"]["boundary_weight"].iloc[0] < rel_boundary["x"]["boundary_weight"].iloc[0]


def test_reliability_pair_term_present():
    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    y = x1 * x2 + rng.normal(0, 0.3, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    model = ZoneBoostRegressor(n_rounds=30, random_state=0, track_reliability=True).fit(X, y)
    _, reliability = model.explain(X, include_reliability=True)
    assert "x1 x x2" in reliability
    assert (reliability["x1 x x2"]["support"] > 0).all()


def test_evidence_report_requires_track_reliability():
    X, y = _dense_and_sparse_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        model.evidence_report(X)


def test_evidence_report_columns_present():
    X, y = _dense_and_sparse_data()
    model = ZoneBoostRegressor(n_rounds=30, random_state=0, track_reliability=True, max_zones=5).fit(X, y)
    report = model.evidence_report(X.iloc[:5])
    for col in ["extrapolating", "unobserved_cell", "pct_contribution_from_sparse_cells", "evidence_score", "evidence_quality"]:
        assert col in report.columns


def test_evidence_report_typical_row_scores_higher_than_out_of_range_row():
    rng = np.random.default_rng(0)
    n_dense = 2900
    n_gap = 20
    x_dense = rng.uniform(-5, 0, n_dense)
    x_gap = rng.uniform(5, 6, n_gap)
    x = np.concatenate([x_dense, x_gap])
    y = x + rng.normal(0, 0.3, len(x))
    X = pd.DataFrame({"x": x})
    model = ZoneBoostRegressor(n_rounds=50, random_state=0, track_reliability=True, max_zones=6).fit(X, y)

    typical_row = pd.DataFrame({"x": [-2.0]})
    gap_row = pd.DataFrame({"x": [5.5]})

    typical_report = model.evidence_report(typical_row)
    gap_report = model.evidence_report(gap_row)
    assert typical_report["evidence_score"].iloc[0] > gap_report["evidence_score"].iloc[0]
    assert gap_report["extrapolating"].iloc[0]
    assert typical_report["evidence_quality"].iloc[0] == "High"
    assert gap_report["evidence_quality"].iloc[0] == "Low"


def _linear_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, n)
    y = x + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"x": x})
    return X, y


def test_edit_effect_overrides_only_affected_rows():
    X, y = _linear_data()
    model = ZoneBoostRegressor(n_rounds=40, random_state=0, validation_fraction=0).fit(X, y)
    contrib_before = model.explain(X)

    model.edit_effect("x", (2.0, 3.0), contribution=10.0)
    contrib_after = model.explain(X)

    mask = ((X["x"] >= 2.0) & (X["x"] <= 3.0)).to_numpy()
    assert (contrib_after.loc[mask, "x"] == 10.0).all()
    np.testing.assert_array_equal(contrib_after.loc[~mask, "x"].to_numpy(), contrib_before.loc[~mask, "x"].to_numpy())


def test_edit_effect_explain_still_sums_exactly_to_predict():
    X, y = _linear_data()
    model = ZoneBoostRegressor(n_rounds=40, random_state=0, validation_fraction=0).fit(X, y)
    model.edit_effect("x", (2.0, 3.0), contribution=10.0)
    contrib = model.explain(X)
    pred = model.predict(X)
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy(), pred, atol=1e-9)


def test_edit_effect_last_applied_wins_for_overlapping_ranges():
    X, y = _linear_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0, validation_fraction=0).fit(X, y)
    model.edit_effect("x", (0.0, 5.0), contribution=100.0)
    model.edit_effect("x", (2.0, 3.0), contribution=50.0)
    contrib = model.explain(X)

    overlap_mask = ((X["x"] >= 2.0) & (X["x"] <= 3.0)).to_numpy()
    first_only_mask = ((X["x"] >= 0.0) & (X["x"] < 2.0)).to_numpy()
    assert (contrib.loc[overlap_mask, "x"] == 50.0).all()
    assert (contrib.loc[first_only_mask, "x"] == 100.0).all()


def test_reset_overrides_restores_original_predictions():
    X, y = _linear_data()
    model = ZoneBoostRegressor(n_rounds=40, random_state=0, validation_fraction=0).fit(X, y)
    pred_before = model.predict(X)
    model.edit_effect("x", (2.0, 3.0), contribution=10.0)
    assert not np.allclose(pred_before, model.predict(X))
    model.reset_overrides()
    np.testing.assert_array_equal(pred_before, model.predict(X))


def test_edit_effect_invalid_feature_raises():
    X, y = _linear_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0, validation_fraction=0).fit(X, y)
    with pytest.raises(ValueError):
        model.edit_effect("nonexistent", (0.0, 1.0), contribution=5.0)


def test_edit_effect_invalid_range_raises():
    X, y = _linear_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0, validation_fraction=0).fit(X, y)
    with pytest.raises(ValueError):
        model.edit_effect("x", (5.0, 1.0), contribution=5.0)


def test_edit_effect_categorical_feature_raises():
    X, y = _synthetic_regression()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0, categorical_features=["cat"]).fit(X, y)
    with pytest.raises(ValueError):
        model.edit_effect("cat", (0.0, 1.0), contribution=5.0)


def test_edit_effect_report_fields_with_eval_data():
    X, y = _linear_data()
    model = ZoneBoostRegressor(n_rounds=40, random_state=0, track_reliability=True, validation_fraction=0).fit(X, y)
    report = model.edit_effect("x", (2.0, 3.0), contribution=10.0, X_eval=X, y_eval=y)

    mask = ((X["x"] >= 2.0) & (X["x"] <= 3.0)).to_numpy()
    assert report["affected_rows"] == int(mask.sum())
    assert report["affected_fraction"] == pytest.approx(mask.sum() / len(X))
    assert report["contribution_change"] == pytest.approx(10.0 - report["original_contribution_mean"])
    assert report["rmse_before"] is not None and report["rmse_after"] is not None
    assert report["rmse_after"] > report["rmse_before"]  # a deliberately bad edit should hurt RMSE
    assert report["prediction_mean_shift"] is not None
    assert report["exceeds_uncertainty"] is True  # a huge, deliberate edit


def test_edit_effect_report_fields_none_without_eval_data():
    X, y = _linear_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0, validation_fraction=0).fit(X, y)
    report = model.edit_effect("x", (2.0, 3.0), contribution=10.0)
    assert report["affected_rows"] is None
    assert report["rmse_before"] is None
    assert report["prediction_mean_shift"] is None


def test_edit_effect_exceeds_uncertainty_none_without_track_reliability():
    X, y = _linear_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0, validation_fraction=0).fit(X, y)
    report = model.edit_effect("x", (2.0, 3.0), contribution=10.0, X_eval=X)
    assert report["exceeds_uncertainty"] is None


def test_edit_effect_constraint_violation_checks_bounded_effects():
    X, y = _linear_data()
    model = ZoneBoostRegressor(
        n_rounds=20, random_state=0, validation_fraction=0, bounded_effects={"x": (-3.0, 3.0)}
    ).fit(X, y)
    violating = model.edit_effect("x", (1.0, 2.0), contribution=10.0)
    assert violating["constraint_violation"] is True

    model2 = ZoneBoostRegressor(
        n_rounds=20, random_state=0, validation_fraction=0, bounded_effects={"x": (-3.0, 3.0)}
    ).fit(X, y)
    compliant = model2.edit_effect("x", (1.0, 2.0), contribution=2.5)
    assert compliant["constraint_violation"] is False


def test_edit_effect_constraint_violation_none_without_bounded_effects():
    X, y = _linear_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0, validation_fraction=0).fit(X, y)
    report = model.edit_effect("x", (1.0, 2.0), contribution=10.0)
    assert report["constraint_violation"] is None


def test_no_edits_reproduces_bit_identical_predictions():
    X, y = _linear_data()
    model_a = ZoneBoostRegressor(n_rounds=40, random_state=0, validation_fraction=0).fit(X, y)
    model_b = ZoneBoostRegressor(n_rounds=40, random_state=0, validation_fraction=0).fit(X, y)
    np.testing.assert_array_equal(model_a.predict(X), model_b.predict(X))
    assert model_a.effect_overrides_ == []


def _additive_two_feature_data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    affordability = rng.uniform(0, 1, n)
    income = rng.uniform(0, 1, n)
    risk = -0.9 * affordability - 0.4 * income + rng.normal(0, 0.1, n)
    X = pd.DataFrame({"affordability": affordability, "income": income})
    return X, risk


def test_counterfactual_already_at_target_returns_no_changes():
    X, y = _additive_two_feature_data()
    model = ZoneBoostRegressor(n_rounds=40, random_state=0, validation_fraction=0).fit(X, y)
    row = X.iloc[[0]]
    pred = model.predict(row)[0]
    result = model.counterfactual(row, target=pred, actionable=["affordability"])
    assert result["feasible"] is True
    assert result["changes"] == {}


def test_counterfactual_prefers_single_feature_solution_when_achievable():
    X, y = _additive_two_feature_data()
    model = ZoneBoostRegressor(n_rounds=60, random_state=0, validation_fraction=0).fit(X, y)
    row = X.iloc[[0]]
    pred = model.predict(row)[0]
    result = model.counterfactual(row, target=pred - 0.15, actionable=["affordability", "income"])
    assert result["feasible"] is True
    assert len(result["changes"]) == 1  # single-feature solution preferred


def test_counterfactual_prediction_change_matches_report():
    X, y = _additive_two_feature_data()
    model = ZoneBoostRegressor(n_rounds=60, random_state=0, validation_fraction=0).fit(X, y)
    row = X.iloc[[0]]
    result = model.counterfactual(row, target=model.predict(row)[0] - 0.15, actionable=["affordability", "income"])
    assert result["prediction_change"] == pytest.approx(
        result["counterfactual_prediction"] - result["original_prediction"]
    )


def test_counterfactual_multi_feature_fallback_engages_for_interaction_target():
    rng = np.random.default_rng(0)
    n = 4000
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    y = x1 * x2 + rng.normal(0, 0.2, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    model = ZoneBoostRegressor(n_rounds=60, random_state=0, validation_fraction=0, max_zones=6).fit(X, y)
    row = pd.DataFrame({"x1": [0.1], "x2": [0.1]})

    result = model.counterfactual(row, target=4.0, actionable=["x1", "x2"], tol=0.3)
    assert result["feasible"] is True
    assert len(result["changes"]) == 2  # neither feature alone can reach target=4.0
    # the interaction term should dominate the consequence, since the true
    # relationship is a pure interaction (y = x1 * x2)
    assert abs(result["interaction_consequences"]["x1 x x2"]) > abs(
        result["interaction_consequences"].get("x1", 0.0)
    )
    assert abs(result["interaction_consequences"]["x1 x x2"]) > abs(
        result["interaction_consequences"].get("x2", 0.0)
    )


def test_counterfactual_infeasible_target_reported_honestly():
    X, y = _additive_two_feature_data()
    model = ZoneBoostRegressor(n_rounds=40, random_state=0, validation_fraction=0).fit(X, y)
    row = X.iloc[[0]]
    result = model.counterfactual(row, target=-100.0, actionable=["affordability"])
    assert result["feasible"] is False


def test_counterfactual_invalid_actionable_feature_raises():
    X, y = _additive_two_feature_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0, validation_fraction=0).fit(X, y)
    row = X.iloc[[0]]
    with pytest.raises(ValueError):
        model.counterfactual(row, target=0.0, actionable=["nonexistent"])


def test_counterfactual_overlapping_actionable_immutable_raises():
    X, y = _additive_two_feature_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0, validation_fraction=0).fit(X, y)
    row = X.iloc[[0]]
    with pytest.raises(ValueError):
        model.counterfactual(row, target=0.0, actionable=["affordability"], immutable=["affordability"])


def test_counterfactual_multi_row_input_raises():
    X, y = _additive_two_feature_data()
    model = ZoneBoostRegressor(n_rounds=20, random_state=0, validation_fraction=0).fit(X, y)
    with pytest.raises(ValueError):
        model.counterfactual(X.iloc[:2], target=0.0, actionable=["affordability"])


def test_counterfactual_zone_transition_frequency_present_for_changed_features():
    X, y = _additive_two_feature_data()
    model = ZoneBoostRegressor(n_rounds=60, random_state=0, validation_fraction=0).fit(X, y)
    row = X.iloc[[0]]
    result = model.counterfactual(row, target=model.predict(row)[0] - 0.15, actionable=["affordability", "income"])
    for feat in result["changes"]:
        assert feat in result["zone_transition_frequency"]
        assert 0.0 <= result["zone_transition_frequency"][feat] <= 1.0


def test_counterfactual_evidence_none_without_track_reliability():
    X, y = _additive_two_feature_data()
    model = ZoneBoostRegressor(n_rounds=40, random_state=0, validation_fraction=0).fit(X, y)
    row = X.iloc[[0]]
    result = model.counterfactual(row, target=model.predict(row)[0] - 0.15, actionable=["affordability"])
    assert result["evidence"] is None


def test_counterfactual_evidence_present_with_track_reliability():
    X, y = _additive_two_feature_data()
    model = ZoneBoostRegressor(n_rounds=40, random_state=0, track_reliability=True, validation_fraction=0).fit(X, y)
    row = X.iloc[[0]]
    result = model.counterfactual(row, target=model.predict(row)[0] - 0.15, actionable=["affordability"])
    assert result["evidence"] is not None
    assert "evidence_score" in result["evidence"].columns
