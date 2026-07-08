import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostClassifier


def _binary_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-5, 5, n),
            "x2": rng.uniform(-5, 5, n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    bump = np.where(X["cat"] == "b", 3.0, 0.0)
    y = ((X["x1"] + bump) > 0).astype(int).to_numpy()
    return X, y


def _multiclass_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-5, 5, n),
            "x2": rng.uniform(-5, 5, n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    y = pd.cut(X["x1"], bins=3, labels=[0, 1, 2]).astype(int).to_numpy()
    return X, y


def test_binary_fit_predict():
    X, y = _binary_data()
    model = ZoneBoostClassifier(n_rounds=30, categorical_features=["cat"], random_state=0)
    model.fit(X, y)
    assert list(model.classes_) == [0, 1]
    assert model.multiclass_ is False

    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    pred = model.predict(X)
    assert pred.shape == (len(y),)
    assert model.score(X, y) > 0.8  # accuracy, via ClassifierMixin


def test_multiclass_fit_predict():
    X, y = _multiclass_data()
    model = ZoneBoostClassifier(n_rounds=30, categorical_features=["cat"], random_state=0)
    model.fit(X, y)
    assert list(model.classes_) == [0, 1, 2]
    assert model.multiclass_ is True

    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    pred = model.predict(X)
    assert set(np.unique(pred)) <= {0, 1, 2}
    assert model.score(X, y) > 0.7


def test_predict_before_fit_raises():
    model = ZoneBoostClassifier(n_rounds=10)
    with pytest.raises(Exception):
        model.predict(pd.DataFrame({"x": [1, 2, 3]}))


def test_single_class_raises():
    X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
    y = [0, 0, 0]
    with pytest.raises(ValueError):
        ZoneBoostClassifier(n_rounds=5).fit(X, y)


def test_categorical_auto_detection_from_dtype():
    X, y = _binary_data()
    model = ZoneBoostClassifier(n_rounds=20, random_state=0)
    model.fit(X, y)
    assert "cat" in model.categorical_features_


def test_unseen_category_at_predict_time_does_not_crash():
    X, y = _binary_data()
    model = ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=0)
    model.fit(X, y)
    X_new = X.copy()
    X_new.loc[0, "cat"] = "never_seen_before"
    proba = model.predict_proba(X_new)
    assert np.all(np.isfinite(proba))


def test_reproducible_with_same_random_state():
    X, y = _binary_data()
    model_a = ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=7).fit(X, y)
    model_b = ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=7).fit(X, y)
    np.testing.assert_array_equal(model_a.predict(X), model_b.predict(X))


def test_accepts_numpy_array_input():
    X, y = _binary_data()
    X_arr = X[["x1", "x2"]].to_numpy()
    model = ZoneBoostClassifier(n_rounds=20, random_state=0)
    model.fit(X_arr, y)
    pred = model.predict(X_arr)
    assert pred.shape == (len(y),)


def test_string_labels_supported():
    X, y = _binary_data()
    y_str = np.where(y == 1, "yes", "no")
    model = ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=0)
    model.fit(X, y_str)
    pred = model.predict(X)
    assert set(np.unique(pred)) <= {"yes", "no"}


def test_missing_values_in_continuous_and_categorical_columns_do_not_crash():
    X, y = _binary_data()
    X_missing = X.copy()
    X_missing.loc[X_missing.sample(20, random_state=1).index, "x1"] = np.nan
    X_missing.loc[X_missing.sample(20, random_state=2).index, "cat"] = np.nan

    model = ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=0)
    model.fit(X_missing, y)
    proba = model.predict_proba(X_missing)
    assert np.all(np.isfinite(proba))
