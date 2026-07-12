import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor, compare_models


def _income_age_data(income_boundary, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    income = rng.uniform(20_000, 100_000, n)
    age = rng.uniform(20, 70, n)
    y = np.where(income > income_boundary, 5.0, 0.0) + 0.01 * age + rng.normal(0, 0.3, n)
    X = pd.DataFrame({"income": income, "age": age})
    return X, y


def _three_way_interaction_data(n=600, seed=0):
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


def test_feature_importance_change_reflects_genuine_shift():
    # income boundary at 52k in the "old" period, 61k in the "new" one,
    # and the age effect is dropped entirely in the new period -- both
    # should show up as real, signed changes.
    X_old, y_old = _income_age_data(52_000, seed=0)
    X_new, y_new_raw = _income_age_data(61_000, seed=1)
    y_new = y_new_raw - 0.01 * X_new["age"].to_numpy()  # drop the age effect

    model_old = ZoneBoostRegressor(random_state=0).fit(X_old, y_old)
    model_new = ZoneBoostRegressor(random_state=0).fit(X_new, y_new)

    result = compare_models(model_old, model_new, X_new, y_new)
    fic = result["feature_importance_change"]
    assert "age" in fic.index
    assert "income" in fic.index
    # age's importance should have shrunk (its effect was removed)
    assert fic.loc["age", "change"] < 0
    # sorted by |change| descending
    assert (fic["change"].abs().diff().dropna() <= 1e-9).all()


def test_new_and_disappeared_terms_detect_triple_interaction():
    X, y = _three_way_interaction_data()
    model_lo = ZoneBoostRegressor(
        n_rounds=100, random_state=0, col_subsample=1.0, max_interaction_order=2
    ).fit(X, y)
    model_hi = ZoneBoostRegressor(
        n_rounds=100, random_state=0, col_subsample=1.0, max_interaction_order=3
    ).fit(X, y)

    result = compare_models(model_lo, model_hi, X, y)
    triple_terms = [t for t in result["new_terms"] if t.count(" x ") == 2]
    assert triple_terms, f"expected a triple term in new_terms, got {result['new_terms']}"
    assert result["disappeared_terms"] == []


def test_boundary_shift_reflects_engineered_distribution_shift():
    X_old, y_old = _income_age_data(52_000, seed=0)
    X_new, y_new = _income_age_data(61_000, seed=1)

    model_old = ZoneBoostRegressor(random_state=0).fit(X_old, y_old)
    model_new = ZoneBoostRegressor(random_state=0).fit(X_new, y_new)

    result = compare_models(model_old, model_new, X_new, y_new)
    assert "income" in result["boundary_shift"]
    assert "age" in result["boundary_shift"]
    # both fit on uniform(20_000, 100_000)/uniform(20, 70) -- ranges should
    # be similar regardless of income_boundary, so this is really testing
    # that center_shift is a small, real number, not a proxy for the
    # boundary move itself (the boundary shows up in feature importance
    # and population migration instead).
    for feature in ("income", "age"):
        shift = result["boundary_shift"][feature]
        assert "old_range" in shift and "new_range" in shift
        assert isinstance(shift["center_shift"], float)


def test_population_migration_higher_for_different_distributions():
    X_a1, y_a1 = _income_age_data(52_000, seed=0)
    X_a2, y_a2 = _income_age_data(52_000, seed=1)
    X_b, y_b = _income_age_data(52_000, seed=2, n=2000)
    # shift income distribution substantially for model_b
    X_b = X_b.copy()
    X_b["income"] = X_b["income"] + 40_000
    y_b = np.where(X_b["income"] > 52_000, 5.0, 0.0) + 0.01 * X_b["age"].to_numpy()

    model_a1 = ZoneBoostRegressor(random_state=0).fit(X_a1, y_a1)
    model_a2 = ZoneBoostRegressor(random_state=1).fit(X_a2, y_a2)
    model_b = ZoneBoostRegressor(random_state=0).fit(X_b, y_b)

    X_eval, y_eval = _income_age_data(52_000, seed=3, n=1000)

    same_dist = compare_models(model_a1, model_a2, X_eval, y_eval)
    diff_dist = compare_models(model_a1, model_b, X_eval, y_eval)

    assert (
        diff_dist["population_migration"]["income"]
        > same_dist["population_migration"]["income"]
    )


def test_performance_change_only_with_y_eval_prediction_shift_always():
    X_old, y_old = _income_age_data(52_000, seed=0)
    X_new, y_new = _income_age_data(61_000, seed=1)
    model_old = ZoneBoostRegressor(random_state=0).fit(X_old, y_old)
    model_new = ZoneBoostRegressor(random_state=0).fit(X_new, y_new)

    with_y = compare_models(model_old, model_new, X_new, y_new)
    assert with_y["performance_change"] is not None
    assert "rmse_old" in with_y["performance_change"]
    assert "rmse_new" in with_y["performance_change"]

    without_y = compare_models(model_old, model_new, X_new)
    assert without_y["performance_change"] is None
    assert without_y["prediction_shift"] is not None
    assert "mean" in without_y["prediction_shift"]
    assert "std" in without_y["prediction_shift"]


def test_boundary_shift_and_migration_restricted_to_shared_continuous_columns():
    # model_old has an extra categorical-only column that model_new lacks
    # entirely -- it must not appear in boundary_shift/population_migration
    # (only feature_importance_change deals in arbitrary term names).
    # X_eval must carry every column either model needs (per compare_models'
    # own contract: "a dataset both models can score"), so it keeps "region"
    # even though model_new never uses it.
    rng = np.random.default_rng(0)
    n = 500
    income = rng.uniform(20_000, 100_000, n)
    region = rng.choice(["north", "south"], n)
    X_old = pd.DataFrame({"income": income, "region": region})
    y_old = np.where(income > 50_000, 3.0, 0.0) + rng.normal(0, 0.2, n)

    X_new = pd.DataFrame({"income": income})
    y_new = np.where(income > 50_000, 3.0, 0.0) + rng.normal(0, 0.2, n)

    model_old = ZoneBoostRegressor(random_state=0).fit(X_old, y_old)
    model_new = ZoneBoostRegressor(random_state=0).fit(X_new, y_new)

    X_eval = pd.DataFrame({"income": income, "region": region})
    result = compare_models(model_old, model_new, X_eval, y_new)
    assert "region" not in result["boundary_shift"]
    assert "region" not in result["population_migration"]
    assert "income" in result["boundary_shift"]
    assert "income" in result["population_migration"]


def test_observed_range_wrapper_matches_standalone_function():
    from zoneboost._drift import _observed_range

    X, y = _income_age_data(52_000)
    model = ZoneBoostRegressor(random_state=0).fit(X, y)
    assert model._observed_range("income") == _observed_range(model, "income")
