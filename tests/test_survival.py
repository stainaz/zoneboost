import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor, ZoneBoostSurvival
from zoneboost._survival import (
    _TIME_COL,
    _concordance_index,
    _default_breakpoints,
    _expand_person_period,
)


def _simulated_survival_data(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 80, n)
    X = pd.DataFrame({"age": age})
    baseline_shape = 0.05
    risk_mult = np.exp(0.03 * (age - 50))
    true_rate = baseline_shape * risk_mult
    event_time = rng.exponential(1.0 / true_rate)
    censor_time = rng.uniform(1, 20, n)
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    return X, duration, event


def test_default_breakpoints_n_intervals_one_degenerates_to_single_interval():
    duration = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.array([1, 1, 0, 1, 1])
    bp = _default_breakpoints(duration, event, 1)
    np.testing.assert_array_equal(bp, [0.0, np.inf])


def test_default_breakpoints_starts_at_zero_ends_at_inf():
    rng = np.random.default_rng(2)
    duration = rng.exponential(5.0, 500)
    event = rng.integers(0, 2, 500)
    bp = _default_breakpoints(duration, event.astype(float), 10)
    assert bp[0] == 0.0
    assert np.isinf(bp[-1])
    assert np.all(np.diff(bp) > 0)


def test_default_breakpoints_all_censored_falls_back_to_all_durations():
    duration = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.zeros(4)
    bp = _default_breakpoints(duration, event, 3)
    assert bp[0] == 0.0
    assert np.isinf(bp[-1])


def test_expand_person_period_matches_hand_computation():
    X = pd.DataFrame({"age": [50.0, 60.0, 70.0]})
    duration = np.array([5.0, 2.0, 12.0])
    event = np.array([1.0, 1.0, 0.0])
    breakpoints = np.array([0.0, 3.0, 6.0, np.inf])

    Xe, ye, oe = _expand_person_period(X, duration, event, breakpoints)

    assert len(Xe) == 6
    exposure = np.exp(oe)
    np.testing.assert_allclose(exposure, [3.0, 2.0, 3.0, 2.0, 3.0, 6.0])
    np.testing.assert_array_equal(ye, [0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(Xe[_TIME_COL].to_numpy(), [0.0, 0.0, 0.0, 3.0, 3.0, 6.0])
    np.testing.assert_array_equal(Xe["age"].to_numpy(), [50.0, 60.0, 70.0, 50.0, 70.0, 70.0])


def test_expand_person_period_subject_surviving_past_all_finite_breakpoints():
    X = pd.DataFrame({"x": [1.0]})
    duration = np.array([100.0])
    event = np.array([1.0])
    breakpoints = np.array([0.0, 1.0, 2.0, np.inf])

    Xe, ye, oe = _expand_person_period(X, duration, event, breakpoints)
    assert len(Xe) == 3
    np.testing.assert_allclose(np.exp(oe), [1.0, 1.0, 98.0])
    np.testing.assert_array_equal(ye, [0.0, 0.0, 1.0])


def test_concordance_index_perfect_ranking_is_one():
    duration = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 1, 1, 1])
    risk = np.array([4.0, 3.0, 2.0, 1.0])  # highest risk -> earliest event
    assert _concordance_index(risk, duration, event) == pytest.approx(1.0)


def test_concordance_index_random_ranking_is_around_half():
    rng = np.random.default_rng(3)
    duration = rng.exponential(1.0, 500)
    event = np.ones(500)
    risk = rng.normal(size=500)  # uncorrelated with outcome
    c = _concordance_index(risk, duration, event)
    assert 0.4 < c < 0.6


def test_ground_truth_recovery_beats_no_covariate_baseline():
    X, duration, event = _simulated_survival_data()
    model = ZoneBoostSurvival(n_intervals=8, random_state=0).fit(X, duration, event)
    cumhaz = model.predict_cumulative_hazard(X, times=[model.max_duration_]).to_numpy().ravel()
    c_with = _concordance_index(cumhaz, duration, event)

    X_flat = pd.DataFrame({"age": np.zeros(len(X))})
    model_flat = ZoneBoostSurvival(n_intervals=8, random_state=0).fit(X_flat, duration, event)
    cumhaz_flat = model_flat.predict_cumulative_hazard(X_flat, times=[model_flat.max_duration_]).to_numpy().ravel()
    c_flat = _concordance_index(cumhaz_flat, duration, event)

    assert c_with > 0.5
    assert c_with > c_flat + 0.05
    assert c_flat == pytest.approx(0.5, abs=1e-9)


def test_survival_function_sanity_invariants():
    X, duration, event = _simulated_survival_data(n=800, seed=5)
    model = ZoneBoostSurvival(n_intervals=6, random_state=0).fit(X, duration, event)

    surv = model.predict_survival_function(X).to_numpy()
    assert np.all(surv <= 1.0 + 1e-9)
    assert np.all(surv >= 0.0)
    assert np.all(np.diff(surv, axis=1) <= 1e-9)

    cumhaz = model.predict_cumulative_hazard(X).to_numpy()
    assert np.all(cumhaz >= -1e-9)
    assert np.all(np.diff(cumhaz, axis=1) >= -1e-9)


def test_n_intervals_one_fits_without_error():
    X, duration, event = _simulated_survival_data(n=500, seed=6)
    model = ZoneBoostSurvival(n_intervals=1, random_state=0).fit(X, duration, event)
    assert len(model.breakpoints_) == 2
    surv = model.predict_survival_function(X)
    assert surv.shape[1] == 1


def test_all_censored_data_fits_and_survival_is_monotonic():
    rng = np.random.default_rng(7)
    n = 400
    X = pd.DataFrame({"x": rng.normal(size=n)})
    duration = rng.uniform(1, 10, n)
    event = np.zeros(n)
    model = ZoneBoostSurvival(n_intervals=5, random_state=0).fit(X, duration, event)
    surv = model.predict_survival_function(X).to_numpy()
    assert np.all(np.diff(surv, axis=1) <= 1e-9)


def test_fit_validation_errors():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="strictly positive"):
        ZoneBoostSurvival().fit(X, [0.0, 1.0, 2.0], [1, 0, 1])
    with pytest.raises(ValueError, match="0/1"):
        ZoneBoostSurvival().fit(X, [1.0, 2.0, 3.0], [1, 2, 0])
    with pytest.raises(ValueError, match="inconsistent lengths"):
        ZoneBoostSurvival().fit(X, [1.0, 2.0], [1, 0, 1])
    with pytest.raises(ValueError, match="breakpoints must start at 0"):
        ZoneBoostSurvival(breakpoints=[1.0, 2.0, 3.0]).fit(X, [1.0, 2.0, 3.0], [1, 0, 1])
    with pytest.raises(ValueError, match="strictly increasing"):
        ZoneBoostSurvival(breakpoints=[0.0, 2.0, 2.0]).fit(X, [1.0, 2.0, 3.0], [1, 0, 1])


def test_custom_breakpoints_used_as_given():
    X, duration, event = _simulated_survival_data(n=500, seed=8)
    model = ZoneBoostSurvival(breakpoints=[0.0, 2.0, 5.0], random_state=0).fit(X, duration, event)
    np.testing.assert_array_equal(model.breakpoints_, [0.0, 2.0, 5.0, np.inf])


def test_template_estimator_params_respected_loss_and_seed_overridden():
    template = ZoneBoostRegressor(n_rounds=7, max_zones=4, random_state=99)
    X, duration, event = _simulated_survival_data(n=500, seed=9)
    model = ZoneBoostSurvival(estimator=template, n_intervals=3, random_state=0).fit(X, duration, event)

    params = model.regressor_.get_params()
    assert params["n_rounds"] == 7
    assert params["max_zones"] == 4
    assert params["loss"] == "poisson"
    assert params["random_state"] == 0


def test_explain_passthrough_sums_to_predict():
    X, duration, event = _simulated_survival_data(n=500, seed=10)
    model = ZoneBoostSurvival(n_intervals=5, random_state=0).fit(X, duration, event)

    Xe, ye, oe = _expand_person_period(X, duration, event.astype(float), model.breakpoints_)
    contrib = model.regressor_.explain(Xe)
    link_sum = contrib.sum(axis=1).to_numpy()
    pred = model.regressor_.predict(Xe, offset=oe)
    mu_from_link = np.exp(link_sum + oe)
    np.testing.assert_allclose(mu_from_link, pred, atol=1e-8)
