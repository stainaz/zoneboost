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
