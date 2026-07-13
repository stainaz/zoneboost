import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor


def _minority_segment_data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    region = rng.choice(["majority", "minority"], n, p=[0.9, 0.1])
    x = rng.uniform(-5, 5, n)
    noise_scale = np.where(region == "majority", 1.0, 4.0)
    y = 2 * x + rng.normal(0, noise_scale, n)
    X = pd.DataFrame({"x": x, "region": region})
    return X, y, region


def test_mondrian_improves_minority_segment_coverage():
    X, y, region = _minority_segment_data()
    minority_mask = region == "minority"

    model_marginal = ZoneBoostRegressor(random_state=0, n_rounds=100, validation_fraction=0.3).fit(X, y)
    lo_m, hi_m = model_marginal.predict_interval(X, alpha=0.1)
    coverage_marginal_minority = np.mean(
        (y[minority_mask] >= lo_m[minority_mask]) & (y[minority_mask] <= hi_m[minority_mask])
    )

    model_mondrian = ZoneBoostRegressor(
        random_state=0, n_rounds=100, validation_fraction=0.3, mondrian_col="region"
    ).fit(X, y)
    lo_g, hi_g = model_mondrian.predict_interval(X, alpha=0.1)
    coverage_mondrian_minority = np.mean(
        (y[minority_mask] >= lo_g[minority_mask]) & (y[minority_mask] <= hi_g[minority_mask])
    )

    # marginal coverage undershoots badly on the minority segment; Mondrian
    # brings it back much closer to the 90% target.
    assert coverage_marginal_minority < 0.6
    assert coverage_mondrian_minority > 0.8


def test_conformal_scores_by_group_populated_only_when_mondrian_col_set():
    X, y, _ = _minority_segment_data(n=2000)
    model_off = ZoneBoostRegressor(random_state=0, n_rounds=30, validation_fraction=0.3).fit(X, y)
    assert model_off.conformal_scores_by_group_ is None

    model_on = ZoneBoostRegressor(
        random_state=0, n_rounds=30, validation_fraction=0.3, mondrian_col="region"
    ).fit(X, y)
    assert model_on.conformal_scores_by_group_ is not None
    assert set(model_on.conformal_scores_by_group_.keys()) <= {"majority", "minority"}


def test_small_group_falls_back_to_global_margin():
    X, y, region = _minority_segment_data(n=2000)
    model = ZoneBoostRegressor(
        random_state=0, n_rounds=30, validation_fraction=0.2,
        mondrian_col="region", mondrian_min_group_size=10_000,  # impossibly high
    ).fit(X, y)
    # no group can meet a 10,000-row minimum on a 2000-row dataset
    assert model.conformal_scores_by_group_ == {}

    lo_g, hi_g = model.predict_interval(X, alpha=0.1)
    margin_g = (hi_g - lo_g) / 2

    model_marginal = ZoneBoostRegressor(random_state=0, n_rounds=30, validation_fraction=0.2).fit(X, y)
    lo_m, hi_m = model_marginal.predict_interval(X, alpha=0.1)
    margin_m = (hi_m - lo_m) / 2

    np.testing.assert_allclose(margin_g, margin_m)


def test_unseen_group_at_predict_time_falls_back_to_global_margin():
    X, y, _ = _minority_segment_data(n=2000)
    model = ZoneBoostRegressor(
        random_state=0, n_rounds=30, validation_fraction=0.3, mondrian_col="region"
    ).fit(X, y)
    X_unseen = X.copy()
    X_unseen["region"] = "brand_new_region"
    lo, hi = model.predict_interval(X_unseen, alpha=0.1)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(lo <= hi)


def test_mondrian_col_none_is_bit_identical_default():
    X, y, _ = _minority_segment_data(n=1500)
    model_default = ZoneBoostRegressor(random_state=0, n_rounds=30, validation_fraction=0.3).fit(X, y)
    model_explicit_none = ZoneBoostRegressor(
        random_state=0, n_rounds=30, validation_fraction=0.3, mondrian_col=None
    ).fit(X, y)
    lo1, hi1 = model_default.predict_interval(X)
    lo2, hi2 = model_explicit_none.predict_interval(X)
    np.testing.assert_array_equal(lo1, lo2)
    np.testing.assert_array_equal(hi1, hi2)


def test_invalid_mondrian_col_raises_same_as_group_col():
    X, y, _ = _minority_segment_data(n=500)
    with pytest.raises(ValueError):
        ZoneBoostRegressor(mondrian_col="not_a_real_column").fit(X, y)


def test_mondrian_col_and_group_col_independent():
    X, y, _ = _minority_segment_data(n=1500)
    # both set
    model_both = ZoneBoostRegressor(
        random_state=0, n_rounds=20, validation_fraction=0.3, group_col="region", mondrian_col="region"
    ).fit(X, y)
    assert model_both.group_col_ == "region"
    assert model_both.mondrian_col_ == "region"
    lo, hi = model_both.predict_interval(X, alpha=0.1)
    assert np.all(lo <= hi)

    # only group_col
    model_group_only = ZoneBoostRegressor(
        random_state=0, n_rounds=20, validation_fraction=0.3, group_col="region"
    ).fit(X, y)
    assert model_group_only.group_col_ == "region"
    assert model_group_only.mondrian_col_ is None
    assert model_group_only.conformal_scores_by_group_ is None

    # only mondrian_col
    model_mondrian_only = ZoneBoostRegressor(
        random_state=0, n_rounds=20, validation_fraction=0.3, mondrian_col="region"
    ).fit(X, y)
    assert model_mondrian_only.group_col_ is None
    assert model_mondrian_only.mondrian_col_ == "region"
