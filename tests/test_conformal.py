import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from zoneboost import ConformalizedQuantileRegressor, ZoneBoostRegressor
from zoneboost._conformal import _rearrange


def _heteroscedastic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    noise_scale = 0.2 + 0.4 * x
    y = 2 * x + rng.normal(0, 1, n) * noise_scale
    X = pd.DataFrame({"x": x})
    return X, y


def test_predict_interval_shape_and_lower_never_exceeds_upper():
    X, y = _heteroscedastic()
    model = ConformalizedQuantileRegressor(alpha=0.1, random_state=0, calibration_fraction=0.2).fit(X, y)
    lower, upper = model.predict_interval(X)
    assert lower.shape == (len(X),)
    assert upper.shape == (len(X),)
    assert np.all(lower <= upper)


def test_predict_interval_achieves_roughly_target_coverage():
    X, y = _heteroscedastic(n=4000)
    X_train, y_train = X.iloc[:3000].reset_index(drop=True), y[:3000]
    X_test, y_test = X.iloc[3000:].reset_index(drop=True), y[3000:]
    template = ZoneBoostRegressor(n_rounds=60)
    model = ConformalizedQuantileRegressor(
        estimator=template, alpha=0.1, random_state=0, calibration_fraction=0.2
    ).fit(X_train, y_train)
    lower, upper = model.predict_interval(X_test)
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    assert 0.80 <= coverage <= 0.98


def test_interval_width_is_locally_adaptive_not_constant():
    X, y = _heteroscedastic(n=4000)
    template = ZoneBoostRegressor(n_rounds=60)
    model = ConformalizedQuantileRegressor(
        estimator=template, alpha=0.1, random_state=0, calibration_fraction=0.2
    ).fit(X, y)
    lower, upper = model.predict_interval(X)
    width = upper - lower
    x = X["x"].to_numpy()
    low_x_width = width[x < 2].mean()
    high_x_width = width[x > 8].mean()
    # heteroscedastic noise grows with x -- a locally-adaptive interval must
    # be measurably wider in the high-variance region, unlike a constant-
    # width split-conformal band.
    assert high_x_width > 2 * low_x_width


def test_custom_estimator_template_params_are_respected_by_lo_and_hi():
    X, y = _heteroscedastic()
    template = ZoneBoostRegressor(n_rounds=25, max_zones=4, monotonic_constraints={"x": 1})
    model = ConformalizedQuantileRegressor(estimator=template, alpha=0.2, random_state=0).fit(X, y)
    assert model.lo_.n_rounds == 25
    assert model.hi_.n_rounds == 25
    assert model.lo_.max_zones == 4
    assert model.lo_.monotonic_constraints == {"x": 1}
    # loss/quantile/calibration_fraction must always be managed internally,
    # regardless of what the template itself was constructed with.
    assert model.lo_.loss == "quantile"
    assert model.lo_.quantile == pytest.approx(0.1)
    assert model.hi_.quantile == pytest.approx(0.9)
    assert model.lo_.calibration_fraction == 0.0


def test_get_params_and_clone_work():
    template = ZoneBoostRegressor(n_rounds=30)
    model = ConformalizedQuantileRegressor(estimator=template, alpha=0.15, random_state=1)
    params = model.get_params()
    assert params["alpha"] == 0.15
    assert params["estimator__n_rounds"] == 30

    cloned = clone(model)
    assert cloned.alpha == 0.15
    assert cloned.estimator.n_rounds == 30
    assert cloned is not model


def test_alpha_out_of_range_raises():
    X, y = _heteroscedastic()
    with pytest.raises(ValueError):
        ConformalizedQuantileRegressor(alpha=1.5).fit(X, y)
    with pytest.raises(ValueError):
        ConformalizedQuantileRegressor(alpha=0.0).fit(X, y)


def test_calibration_fraction_out_of_range_raises():
    X, y = _heteroscedastic()
    with pytest.raises(ValueError):
        ConformalizedQuantileRegressor(calibration_fraction=1.5).fit(X, y)


def test_predict_interval_before_fit_raises():
    model = ConformalizedQuantileRegressor()
    with pytest.raises(Exception):
        model.predict_interval(pd.DataFrame({"x": [1.0, 2.0]}))


def test_rearrange_fixes_crossed_rows():
    lo = np.array([1.0, 5.0, 3.0, -2.0])
    hi = np.array([2.0, 4.0, 3.0, -5.0])  # rows 1 and 3 crossed
    lo2, hi2 = _rearrange(lo, hi)
    assert np.all(lo2 <= hi2)
    np.testing.assert_array_equal(lo2, [1.0, 4.0, 3.0, -5.0])
    np.testing.assert_array_equal(hi2, [2.0, 5.0, 3.0, -2.0])


def test_rearrange_is_a_no_op_when_already_ordered():
    lo = np.array([1.0, 2.0, 3.0])
    hi = np.array([4.0, 5.0, 6.0])
    lo2, hi2 = _rearrange(lo, hi)
    np.testing.assert_array_equal(lo, lo2)
    np.testing.assert_array_equal(hi, hi2)


def test_predict_interval_never_crossed_even_with_sparse_extreme_regions():
    # very few calibration rows and a tight alpha (close quantile levels)
    # is the classic setting where two independently-fit quantile models
    # are most likely to cross.
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({"x": rng.uniform(-5, 5, n)})
    y = X["x"].to_numpy() * 2 + rng.normal(0, 3.0, n)
    model = ConformalizedQuantileRegressor(alpha=0.02, random_state=0, calibration_fraction=0.3).fit(X, y)
    lower, upper = model.predict_interval(X)
    assert np.all(lower <= upper)
