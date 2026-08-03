import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from zoneboost import ZoneBoostRegressor, ZoneBoostTimeSeries


def _fast():
    return ZoneBoostRegressor(n_rounds=15)


def _drifting_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    x = rng.uniform(-3, 3, n)
    x_stable = rng.uniform(-3, 3, n)  # genuine relationship never changes
    y = 2.0 * x + 0.5 * x_stable + rng.normal(0, 0.3, n)
    X = pd.DataFrame({"date": dates, "x": x, "x_stable": x_stable})
    return X, y


def _sign_flip_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    x = rng.uniform(-3, 3, n)
    x_stable = rng.uniform(-3, 3, n)  # genuine relationship never changes
    # x's own relationship to y flips sign halfway through -- a drastic,
    # unambiguous regime change -- while x_stable's stays constant
    # throughout, so its own optimal zone boundaries barely move.
    half = n // 2
    coef_x = np.where(np.arange(n) < half, 3.0, -3.0)
    y = coef_x * x + 2.0 * x_stable + rng.normal(0, 0.3, n)
    X = pd.DataFrame({"date": dates, "x": x, "x_stable": x_stable})
    return X, y


def test_fit_requires_time_col():
    X, y = _drifting_data()
    with pytest.raises(ValueError):
        ZoneBoostTimeSeries(base_estimator=_fast()).fit(X, y)


def test_fit_rejects_unknown_time_col():
    X, y = _drifting_data()
    with pytest.raises(ValueError):
        ZoneBoostTimeSeries(base_estimator=_fast(), time_col="not_a_column").fit(X, y)


def test_fit_rejects_invalid_window():
    X, y = _drifting_data()
    with pytest.raises(ValueError):
        ZoneBoostTimeSeries(base_estimator=_fast(), time_col="date", window="bogus").fit(X, y)


def test_fit_raises_when_not_enough_periods():
    X, y = _drifting_data()
    with pytest.raises(ValueError):
        ZoneBoostTimeSeries(base_estimator=_fast(), time_col="date", freq="Y", min_periods=5).fit(X, y)


def test_expanding_window_training_sizes_are_non_decreasing():
    X, y = _drifting_data()
    model = ZoneBoostTimeSeries(
        base_estimator=_fast(), time_col="date", freq="M", min_periods=2, window="expanding", random_state=0
    ).fit(X, y)
    periods = pd.to_datetime(X["date"]).dt.to_period("M")
    sizes = [int((periods <= k).sum()) for k in sorted(model.models_)]
    assert sizes == sorted(sizes)
    assert len(sizes) > 1


def test_rolling_window_uses_fixed_lookback():
    X, y = _drifting_data()
    min_periods = 3
    model = ZoneBoostTimeSeries(
        base_estimator=_fast(), time_col="date", freq="M", min_periods=min_periods, window="rolling", random_state=0
    ).fit(X, y)
    periods = pd.to_datetime(X["date"]).dt.to_period("M")
    sorted_periods = sorted(periods.unique())
    for k in sorted(model.models_):
        idx = sorted_periods.index(k)
        expected_train_periods = sorted_periods[max(0, idx - min_periods + 1) : idx + 1]
        assert len(expected_train_periods) == min_periods


def test_time_col_excluded_from_fitted_feature_space():
    X, y = _drifting_data()
    model = ZoneBoostTimeSeries(base_estimator=_fast(), time_col="date", freq="M", min_periods=2, random_state=0).fit(X, y)
    assert "date" not in model.feature_names_in_
    latest = model.models_[max(model.models_)]
    assert "date" not in latest.predictor_names_


def test_predict_delegates_to_latest_period_model():
    X, y = _drifting_data()
    model = ZoneBoostTimeSeries(base_estimator=_fast(), time_col="date", freq="M", min_periods=2, random_state=0).fit(X, y)
    latest = model.models_[max(model.models_)]
    pred = model.predict(X)
    np.testing.assert_allclose(pred, latest.predict(X.drop(columns=["date"])))


def test_explain_and_feature_importance_delegate_to_latest_period_model():
    X, y = _drifting_data()
    model = ZoneBoostTimeSeries(base_estimator=_fast(), time_col="date", freq="M", min_periods=2, random_state=0).fit(X, y)
    latest = model.models_[max(model.models_)]
    contrib = model.explain(X)
    pd.testing.assert_frame_equal(contrib, latest.explain(X.drop(columns=["date"])))
    fi = model.feature_importance(X)
    pd.testing.assert_series_equal(fi, latest.feature_importance(X.drop(columns=["date"])))


def test_comparisons_has_one_fewer_entry_than_fitted_periods():
    X, y = _drifting_data()
    model = ZoneBoostTimeSeries(base_estimator=_fast(), time_col="date", freq="M", min_periods=2, random_state=0).fit(X, y)
    assert len(model.comparisons_) == len(model.models_) - 1


def test_drift_alerts_omitted_when_conformal_unavailable():
    X, y = _drifting_data()
    base = ZoneBoostRegressor(n_rounds=10, validation_fraction=0.0)
    model = ZoneBoostTimeSeries(
        base_estimator=base, time_col="date", freq="M", min_periods=2, random_state=0
    ).fit(X, y)
    assert len(model.comparisons_) > 0
    assert len(model.drift_alerts_) == 0


def test_drift_alerts_present_by_default():
    X, y = _drifting_data()
    model = ZoneBoostTimeSeries(base_estimator=_fast(), time_col="date", freq="M", min_periods=2, random_state=0).fit(X, y)
    assert len(model.drift_alerts_) == len(model.comparisons_)


def test_stability_report_ranks_drifting_feature_above_stable_one():
    X, y = _sign_flip_data()
    model = ZoneBoostTimeSeries(base_estimator=_fast(), time_col="date", freq="M", min_periods=2, random_state=0).fit(X, y)
    report = model.stability_report()
    assert "x" in report.index and "x_stable" in report.index
    # sorted descending by mean_population_migration -- the genuinely
    # sign-flipping feature should sit above the constant-relationship one.
    assert report.index[0] == "x"
    assert report.loc["x", "mean_population_migration"] > report.loc["x_stable", "mean_population_migration"]


def test_get_params_and_clone_work():
    template = ZoneBoostRegressor(n_rounds=15, max_zones=4)
    model = ZoneBoostTimeSeries(base_estimator=template, time_col="date", freq="M", min_periods=1, random_state=1)
    params = model.get_params()
    assert params["base_estimator__n_rounds"] == 15
    cloned = clone(model)
    assert cloned.base_estimator.n_rounds == 15
    assert cloned is not model


def test_predict_before_fit_raises():
    model = ZoneBoostTimeSeries(base_estimator=_fast(), time_col="date")
    with pytest.raises(Exception):
        model.predict(pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2), "x": [1.0, 2.0]}))


def test_time_col_accepts_integer_index():
    X, y = _drifting_data()
    date_idx = X.columns.get_loc("date")
    model = ZoneBoostTimeSeries(base_estimator=_fast(), time_col=date_idx, freq="M", min_periods=2, random_state=0).fit(X, y)
    assert model.time_col_ == "date"
