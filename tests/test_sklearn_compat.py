import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from zoneboost import ZoneBoostClassifier, ZoneBoostRegressor


def _synthetic_regression(n=300, seed=0):
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


def test_get_params_set_params_roundtrip():
    model = ZoneBoostRegressor(n_rounds=30, categorical_features=["cat"], random_state=1)
    params = model.get_params()
    assert params["n_rounds"] == 30
    assert params["categorical_features"] == ["cat"]

    model.set_params(n_rounds=99)
    assert model.n_rounds == 99


def test_clone_does_not_share_state():
    model = ZoneBoostRegressor(n_rounds=30, random_state=1)
    X, y = _synthetic_regression()
    model.fit(X[["x1", "x2"]], y)

    cloned = clone(model)
    assert not hasattr(cloned, "rounds_")  # clone resets to an unfitted estimator
    assert cloned.n_rounds == model.n_rounds


def test_works_inside_pipeline():
    X, y = _synthetic_regression()
    pipe = Pipeline([("model", ZoneBoostRegressor(n_rounds=20, categorical_features=["cat"], random_state=0))])
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(y),)


def test_cross_val_score_runs():
    X, y = _synthetic_regression()
    scores = cross_val_score(
        ZoneBoostRegressor(n_rounds=20, categorical_features=["cat"], random_state=0), X, y, cv=3
    )
    assert len(scores) == 3
    assert all(np.isfinite(scores))


def test_grid_search_cv_runs():
    X, y = _synthetic_regression()
    grid = GridSearchCV(
        ZoneBoostRegressor(categorical_features=["cat"], random_state=0),
        param_grid={"n_rounds": [10, 20]},
        cv=2,
    )
    grid.fit(X, y)
    assert grid.best_params_["n_rounds"] in (10, 20)


def test_score_method_from_regressor_mixin():
    X, y = _synthetic_regression()
    model = ZoneBoostRegressor(n_rounds=30, categorical_features=["cat"], random_state=0).fit(X, y)
    r2 = model.score(X, y)
    assert 0.0 < r2 <= 1.0


def _synthetic_classification(n=300, seed=0):
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


def test_classifier_clone_and_params():
    model = ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=1)
    cloned = clone(model)
    assert not hasattr(cloned, "classes_")
    cloned.set_params(n_rounds=40)
    assert cloned.n_rounds == 40 and model.n_rounds == 20


def test_classifier_works_inside_pipeline():
    X, y = _synthetic_classification()
    pipe = Pipeline([("model", ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=0))])
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(y),)


def test_classifier_cross_val_score_runs():
    X, y = _synthetic_classification()
    scores = cross_val_score(
        ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=0), X, y, cv=3
    )
    assert len(scores) == 3
    assert all(np.isfinite(scores))


def test_classifier_grid_search_cv_runs():
    X, y = _synthetic_classification()
    grid = GridSearchCV(
        ZoneBoostClassifier(categorical_features=["cat"], random_state=0),
        param_grid={"n_rounds": [10, 20]},
        cv=2,
    )
    grid.fit(X, y)
    assert grid.best_params_["n_rounds"] in (10, 20)


def test_classifier_score_method_from_classifier_mixin():
    X, y = _synthetic_classification()
    model = ZoneBoostClassifier(n_rounds=30, categorical_features=["cat"], random_state=0).fit(X, y)
    acc = model.score(X, y)
    assert 0.0 <= acc <= 1.0
