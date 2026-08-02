import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from zoneboost import ZoneBoostRegressor, ZoneForest


def _signal_and_noise_data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3, 3, n)  # genuine signal
    x2 = rng.uniform(-3, 3, n)  # pure noise, unrelated to y
    y = 2.0 * x1 + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


def _grouped_data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    group = rng.choice(["a", "b", "c"], n)
    y = 2.0 * x1 + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "group": group})
    return X, y


def test_fit_stores_n_estimators_models():
    X, y = _signal_and_noise_data()
    model = ZoneForest(ZoneBoostRegressor(n_rounds=15), n_estimators=10, random_state=0).fit(X, y)
    assert len(model.estimators_) == 10
    assert len(model.estimators_samples_) == 10
    assert len(model.estimators_features_) == 10
    assert all(hasattr(m, "rounds_") for m in model.estimators_)


def test_default_base_estimator_is_plain_zoneboost_regressor():
    X, y = _signal_and_noise_data(n=200)
    model = ZoneForest(n_estimators=5, random_state=0).fit(X, y)
    assert isinstance(model.estimators_[0], ZoneBoostRegressor)


def test_predict_shape_and_reasonable_signal_recovery():
    X, y = _signal_and_noise_data()
    model = ZoneForest(ZoneBoostRegressor(n_rounds=20), n_estimators=20, random_state=0).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    # genuine relationship is y ~= 2 * x1 -- correlation with the true signal
    # should be strong.
    assert np.corrcoef(pred, y)[0, 1] > 0.9


def test_explain_sums_to_predict():
    X, y = _signal_and_noise_data(n=500)
    model = ZoneForest(ZoneBoostRegressor(n_rounds=15), n_estimators=15, max_features=0.5, random_state=0).fit(X, y)
    contrib = model.explain(X)
    pred = model.predict(X)
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy(), pred, atol=1e-8)


def test_feature_importance_signal_vs_noise():
    X, y = _signal_and_noise_data()
    model = ZoneForest(ZoneBoostRegressor(n_rounds=20), n_estimators=20, random_state=0).fit(X, y)
    fi = model.feature_importance(X)
    assert fi["x1"] > fi["x2"]


def test_max_features_restricts_column_subset_size():
    X, y = _signal_and_noise_data(n=300)
    X = X.assign(x3=np.random.default_rng(1).uniform(-3, 3, len(X)), x4=np.random.default_rng(2).uniform(-3, 3, len(X)))
    model = ZoneForest(ZoneBoostRegressor(n_rounds=10), n_estimators=10, max_features=0.5, random_state=0).fit(X, y)
    for cols in model.estimators_features_:
        assert len(cols) == 2  # round(4 * 0.5)


def test_group_col_force_included_despite_low_max_features():
    X, y = _grouped_data()
    base = ZoneBoostRegressor(n_rounds=10, group_col="group")
    model = ZoneForest(base, n_estimators=10, max_features=0.34, random_state=0).fit(X, y)
    for cols in model.estimators_features_:
        assert "group" in cols
    # no crash on predict/explain either
    model.predict(X)
    model.explain(X)


def test_bootstrap_row_sampling_is_with_replacement():
    X, y = _signal_and_noise_data(n=100)
    model = ZoneForest(ZoneBoostRegressor(n_rounds=5), n_estimators=5, random_state=0).fit(X, y)
    for idx in model.estimators_samples_:
        assert len(idx) == len(X)
        # with replacement -- vanishingly unlikely to be all-distinct for n=100
        assert len(set(idx.tolist())) < len(idx)


def test_get_params_and_clone_work():
    template = ZoneBoostRegressor(n_rounds=15, max_zones=4)
    model = ZoneForest(base_estimator=template, n_estimators=8, max_samples=0.8, random_state=1)
    params = model.get_params()
    assert params["n_estimators"] == 8
    assert params["base_estimator__n_rounds"] == 15

    cloned = clone(model)
    assert cloned.n_estimators == 8
    assert cloned.base_estimator.n_rounds == 15
    assert cloned is not model


def test_n_jobs_parallel_matches_sequential():
    X, y = _signal_and_noise_data(n=300)
    seq = ZoneForest(ZoneBoostRegressor(n_rounds=10), n_estimators=8, random_state=0, n_jobs=None).fit(X, y)
    par = ZoneForest(ZoneBoostRegressor(n_rounds=10), n_estimators=8, random_state=0, n_jobs=2).fit(X, y)
    np.testing.assert_allclose(seq.predict(X), par.predict(X), atol=1e-10)


def test_n_estimators_too_small_raises():
    X, y = _signal_and_noise_data(n=100)
    with pytest.raises(ValueError):
        ZoneForest(ZoneBoostRegressor(n_rounds=5), n_estimators=1).fit(X, y)


def test_max_samples_out_of_range_raises():
    X, y = _signal_and_noise_data(n=100)
    with pytest.raises(ValueError):
        ZoneForest(ZoneBoostRegressor(n_rounds=5), max_samples=1.5).fit(X, y)


def test_max_features_out_of_range_raises():
    X, y = _signal_and_noise_data(n=100)
    with pytest.raises(ValueError):
        ZoneForest(ZoneBoostRegressor(n_rounds=5), max_features=0.0).fit(X, y)


def test_predict_before_fit_raises():
    model = ZoneForest()
    with pytest.raises(Exception):
        model.predict(pd.DataFrame({"x": [1.0, 2.0]}))
