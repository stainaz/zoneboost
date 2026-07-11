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


def test_max_interaction_order_3_improves_fit_on_genuine_triple_interaction():
    # col_subsample=1.0: with only 3 predictors, the default 0.7 subsample
    # rounds down to 2 columns per round, which never gives a 3-way search
    # a chance -- using all 3 columns every round is the realistic setting
    # for a model with this few predictors.
    X, y = _three_way_interaction_data()
    model_pairwise = ZoneBoostRegressor(
        n_rounds=60, random_state=0, col_subsample=1.0, max_interaction_order=2
    ).fit(X, y)
    model_triples = ZoneBoostRegressor(
        n_rounds=60, random_state=0, col_subsample=1.0, max_interaction_order=3
    ).fit(X, y)

    rmse_pairwise = np.sqrt(np.mean((y - model_pairwise.predict(X)) ** 2))
    rmse_triples = np.sqrt(np.mean((y - model_triples.predict(X)) ** 2))
    assert rmse_triples < rmse_pairwise
    assert any(len(round_["triples"]) > 0 for round_ in model_triples.rounds_)
