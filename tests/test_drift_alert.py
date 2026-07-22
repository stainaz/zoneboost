import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor, compare_models, flag_drift


def _period_data(shift=0.0, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    region = rng.choice(["north", "south"], n)
    y = 2.0 * x + shift + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"x": x, "region": region})
    return X, y


def test_flag_drift_returns_expected_keys():
    X_old, y_old = _period_data(shift=0.0, seed=0)
    X_new, y_new = _period_data(shift=0.0, seed=1)
    model_old = ZoneBoostRegressor(n_rounds=20, random_state=0).fit(X_old, y_old)
    model_new = ZoneBoostRegressor(n_rounds=20, random_state=1).fit(X_new, y_new)

    result = flag_drift(model_old, model_new, X_new, y_new)
    assert set(result.keys()) == {
        "comparison",
        "alpha",
        "global_margin",
        "mean_prediction_shift",
        "drifted",
        "group_alerts",
    }
    assert result["group_alerts"] is None


def test_comparison_matches_compare_models_directly():
    X_old, y_old = _period_data(shift=0.0, seed=0)
    X_new, y_new = _period_data(shift=0.0, seed=1)
    model_old = ZoneBoostRegressor(n_rounds=20, random_state=0).fit(X_old, y_old)
    model_new = ZoneBoostRegressor(n_rounds=20, random_state=1).fit(X_new, y_new)

    result = flag_drift(model_old, model_new, X_new, y_new)
    direct = compare_models(model_old, model_new, X_new, y_new)
    assert result["comparison"]["prediction_shift"] == direct["prediction_shift"]
    assert list(result["comparison"]["feature_importance_change"].index) == list(
        direct["feature_importance_change"].index
    )


def test_no_real_drift_is_not_flagged():
    X_old, y_old = _period_data(shift=0.0, seed=0)
    X_new, y_new = _period_data(shift=0.0, seed=1)
    model_old = ZoneBoostRegressor(n_rounds=20, random_state=0).fit(X_old, y_old)
    model_new = ZoneBoostRegressor(n_rounds=20, random_state=1).fit(X_new, y_new)

    result = flag_drift(model_old, model_new, X_new, y_new)
    assert result["drifted"] is False


def test_genuine_drift_is_flagged():
    X_old, y_old = _period_data(shift=0.0, seed=0)
    X_new, y_new = _period_data(shift=25.0, seed=1)  # large, deliberate shift
    model_old = ZoneBoostRegressor(n_rounds=20, random_state=0).fit(X_old, y_old)
    model_new = ZoneBoostRegressor(n_rounds=20, random_state=1).fit(X_new, y_new)

    result = flag_drift(model_old, model_new, X_new, y_new)
    assert result["drifted"] is True
    assert abs(result["mean_prediction_shift"]) > result["global_margin"]


def test_group_alerts_populated_when_mondrian_col_set():
    X_old, y_old = _period_data(shift=0.0, seed=0)
    X_new, y_new = _period_data(shift=25.0, seed=1)
    model_old = ZoneBoostRegressor(n_rounds=20, random_state=0, mondrian_col="region").fit(X_old, y_old)
    model_new = ZoneBoostRegressor(n_rounds=20, random_state=1, mondrian_col="region").fit(X_new, y_new)

    result = flag_drift(model_old, model_new, X_new, y_new)
    assert set(result["group_alerts"].keys()) == {"north", "south"}
    for group_result in result["group_alerts"].values():
        assert set(group_result.keys()) == {"mean_shift", "margin", "drifted", "used_group_margin"}


def test_unseen_group_falls_back_to_global_margin():
    X_old, y_old = _period_data(shift=0.0, seed=0)
    X_new, y_new = _period_data(shift=0.0, seed=1)
    model_old = ZoneBoostRegressor(n_rounds=20, random_state=0, mondrian_col="region").fit(X_old, y_old)
    model_new = ZoneBoostRegressor(n_rounds=20, random_state=1, mondrian_col="region").fit(X_new, y_new)

    X_eval = X_new.copy()
    X_eval.loc[0, "region"] = "unseen_region"
    result = flag_drift(model_old, model_new, X_eval, y_new)
    assert result["group_alerts"]["unseen_region"]["used_group_margin"] is False
    assert result["group_alerts"]["unseen_region"]["margin"] == result["global_margin"]


def test_alpha_out_of_range_raises():
    X_old, y_old = _period_data(shift=0.0, seed=0)
    X_new, y_new = _period_data(shift=0.0, seed=1)
    model_old = ZoneBoostRegressor(n_rounds=10, random_state=0).fit(X_old, y_old)
    model_new = ZoneBoostRegressor(n_rounds=10, random_state=1).fit(X_new, y_new)
    with pytest.raises(ValueError):
        flag_drift(model_old, model_new, X_new, y_new, alpha=1.5)


def test_missing_conformal_scores_raises():
    X_old, y_old = _period_data(shift=0.0, seed=0)
    X_new, y_new = _period_data(shift=0.0, seed=1)
    model_old = ZoneBoostRegressor(n_rounds=10, random_state=0, validation_fraction=0.0).fit(X_old, y_old)
    model_new = ZoneBoostRegressor(n_rounds=10, random_state=1, validation_fraction=0.0).fit(X_new, y_new)
    with pytest.raises(ValueError):
        flag_drift(model_old, model_new, X_new, y_new)
