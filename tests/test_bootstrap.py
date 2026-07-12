import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from zoneboost import BootstrapStability, ZoneBoostClassifier, ZoneBoostRegressor


def _signal_and_noise_data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3, 3, n)  # genuine signal
    x2 = rng.uniform(-3, 3, n)  # pure noise, unrelated to y
    y = 2.0 * x1 + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


def _binary_data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    p = 1 / (1 + np.exp(-x1))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


def _multiclass_data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X, _ = _binary_data(n=n, seed=seed)
    y = rng.choice([0, 1, 2], n)
    return X, y


def test_fit_stores_n_bootstrap_models():
    X, y = _signal_and_noise_data()
    model = BootstrapStability(ZoneBoostRegressor(n_rounds=20), n_bootstrap=12, random_state=0).fit(X, y)
    assert len(model.bootstrap_models_) == 12
    assert all(hasattr(m, "rounds_") for m in model.bootstrap_models_)


def test_contribution_interval_lower_never_exceeds_upper():
    X, y = _signal_and_noise_data()
    model = BootstrapStability(ZoneBoostRegressor(n_rounds=20), n_bootstrap=12, random_state=0).fit(X, y)
    interval = model.contribution_interval(X)
    for term, df in interval.items():
        assert np.all(df["lower"] <= df["upper"] + 1e-9)


def test_feature_importance_interval_signal_vs_noise():
    X, y = _signal_and_noise_data()
    model = BootstrapStability(ZoneBoostRegressor(n_rounds=25), n_bootstrap=15, random_state=0).fit(X, y)
    fi = model.feature_importance_interval(X)
    assert np.all(fi["lower"] <= fi["upper"] + 1e-9)
    # the genuine-signal column's importance interval should sit well above
    # the pure-noise column's -- no overlap.
    assert fi.loc["x1", "lower"] > fi.loc["x2", "upper"]


def test_inclusion_frequency_returns_series_summing_reasonable_range():
    X, y = _signal_and_noise_data()
    model = BootstrapStability(ZoneBoostRegressor(n_rounds=20), n_bootstrap=12, random_state=0).fit(X, y)
    freq = model.inclusion_frequency()
    assert isinstance(freq, pd.Series)
    assert (freq > 0).all() and (freq <= 1.0).all()


def test_predict_confidence_interval_shape_and_ordering():
    X, y = _signal_and_noise_data()
    model = BootstrapStability(ZoneBoostRegressor(n_rounds=20), n_bootstrap=12, random_state=0).fit(X, y)
    lower, upper = model.predict_confidence_interval(X)
    assert lower.shape == (len(X),)
    assert np.all(lower <= upper + 1e-9)


def test_predict_diff_interval_excludes_zero_for_genuinely_different_rows():
    X, y = _signal_and_noise_data()
    model = BootstrapStability(ZoneBoostRegressor(n_rounds=20), n_bootstrap=15, random_state=0).fit(X, y)
    row_a = X.iloc[[0]]
    row_b = row_a.copy()
    row_b["x1"] = row_b["x1"] + 5.0
    lower, upper = model.predict_diff_interval(row_a, row_b)
    assert upper[0] < 0  # b is predicted meaningfully higher than a


def test_predict_diff_interval_includes_zero_for_identical_rows():
    X, y = _signal_and_noise_data()
    model = BootstrapStability(ZoneBoostRegressor(n_rounds=20), n_bootstrap=10, random_state=0).fit(X, y)
    row_a = X.iloc[[0]]
    lower, upper = model.predict_diff_interval(row_a, row_a)
    assert lower[0] == 0.0 and upper[0] == 0.0


def test_predict_diff_interval_raises_on_mismatched_row_counts():
    X, y = _signal_and_noise_data()
    model = BootstrapStability(ZoneBoostRegressor(n_rounds=10), n_bootstrap=5, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        model.predict_diff_interval(X.iloc[:2], X.iloc[:3])


def test_binary_classifier_predict_confidence_interval_and_contribution_interval():
    X, y = _binary_data()
    model = BootstrapStability(ZoneBoostClassifier(n_rounds=15), n_bootstrap=10, random_state=0).fit(X, y)
    lower, upper = model.predict_confidence_interval(X)
    assert np.all(lower >= -1e-9) and np.all(upper <= 1 + 1e-9)
    contrib = model.contribution_interval(X)
    assert "x1" in contrib


def test_multiclass_contribution_interval_nested_per_class():
    X, y = _multiclass_data()
    model = BootstrapStability(ZoneBoostClassifier(n_rounds=12), n_bootstrap=8, random_state=0).fit(X, y)
    contrib = model.contribution_interval(X)
    assert set(contrib.keys()) == {0, 1, 2}
    for k in contrib:
        assert "x1" in contrib[k]


def test_multiclass_feature_importance_interval_is_flat():
    X, y = _multiclass_data()
    model = BootstrapStability(ZoneBoostClassifier(n_rounds=12), n_bootstrap=8, random_state=0).fit(X, y)
    fi = model.feature_importance_interval(X)
    assert isinstance(fi, pd.DataFrame)
    assert "x1" in fi.index


def test_multiclass_inclusion_frequency_is_flat():
    X, y = _multiclass_data()
    model = BootstrapStability(ZoneBoostClassifier(n_rounds=12), n_bootstrap=8, random_state=0).fit(X, y)
    freq = model.inclusion_frequency()
    assert isinstance(freq, pd.Series)


def test_multiclass_predict_confidence_interval_raises():
    X, y = _multiclass_data()
    model = BootstrapStability(ZoneBoostClassifier(n_rounds=10), n_bootstrap=5, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        model.predict_confidence_interval(X)


def test_multiclass_predict_diff_interval_raises():
    X, y = _multiclass_data()
    model = BootstrapStability(ZoneBoostClassifier(n_rounds=10), n_bootstrap=5, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        model.predict_diff_interval(X.iloc[:2], X.iloc[2:4])


def test_alpha_out_of_range_raises():
    X, y = _signal_and_noise_data()
    with pytest.raises(ValueError):
        BootstrapStability(ZoneBoostRegressor(n_rounds=10), alpha=1.5).fit(X, y)


def test_n_bootstrap_too_small_raises():
    X, y = _signal_and_noise_data()
    with pytest.raises(ValueError):
        BootstrapStability(ZoneBoostRegressor(n_rounds=10), n_bootstrap=1).fit(X, y)


def test_get_params_and_clone_work():
    template = ZoneBoostRegressor(n_rounds=25, max_zones=4)
    model = BootstrapStability(estimator=template, n_bootstrap=8, alpha=0.15, random_state=1)
    params = model.get_params()
    assert params["n_bootstrap"] == 8
    assert params["estimator__n_rounds"] == 25

    cloned = clone(model)
    assert cloned.n_bootstrap == 8
    assert cloned.estimator.n_rounds == 25
    assert cloned is not model


def test_predict_confidence_interval_before_fit_raises():
    model = BootstrapStability()
    with pytest.raises(Exception):
        model.predict_confidence_interval(pd.DataFrame({"x": [1.0, 2.0]}))
