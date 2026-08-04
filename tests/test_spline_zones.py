import sqlite3

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from zoneboost import ZoneBoostRegressor, compile_to_sql
from zoneboost._weak_learner import SplineMainEffect, _evaluate_spline_main_effect


def _kinked_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-10, 10, n)
    x2 = rng.uniform(-3, 3, n)
    y = np.where(x1 < 0, -2.0 * x1, 3.0 * x1) + 0.5 * x2 + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


def test_spline_zones_default_none_is_bit_identical():
    X, y = _kinked_data(n=800)
    m_a = ZoneBoostRegressor(n_rounds=20, random_state=1).fit(X, y)
    m_b = ZoneBoostRegressor(n_rounds=20, random_state=1, spline_zones=None).fit(X, y)
    np.testing.assert_array_equal(m_a.predict(X), m_b.predict(X))


def test_spline_zones_recovers_genuine_kink_competitively():
    # A genuine V-shaped kink -- spline zones shouldn't need to be
    # strictly better than the flat mechanism (the flat mechanism can
    # approximate a kink with enough zones too), but it should stay
    # competitive (within ~10%), not measurably worse. Measured, not
    # assumed: at default settings this comes out within ~5%.
    X, y = _kinked_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    m_flat = ZoneBoostRegressor(n_rounds=30, random_state=0, validation_fraction=0.0).fit(X_tr, y_tr)
    m_spline = ZoneBoostRegressor(
        n_rounds=30, random_state=0, validation_fraction=0.0, spline_zones=["x1"]
    ).fit(X_tr, y_tr)
    rmse_flat = np.sqrt(np.mean((m_flat.predict(X_te) - y_te) ** 2))
    rmse_spline = np.sqrt(np.mean((m_spline.predict(X_te) - y_te) ** 2))
    assert rmse_spline < rmse_flat * 1.10


def test_spline_zones_continuous_at_zone_boundary():
    X, y = _kinked_data(n=1500)
    model = ZoneBoostRegressor(n_rounds=10, random_state=0, spline_zones=["x1"]).fit(X, y)
    round_ = model.rounds_[0]
    effect = round_["main_effects"]["x1"]
    assert isinstance(effect, SplineMainEffect)
    boundaries = round_["zone_info"]["x1"][1]
    assert len(boundaries) > 0
    knot = boundaries[0]
    eps = 1e-6
    left = _evaluate_spline_main_effect(np.array([knot - eps]), effect, boundaries)
    right = _evaluate_spline_main_effect(np.array([knot + eps]), effect, boundaries)
    assert abs(left[0] - right[0]) < 1e-4


def test_spline_zones_explain_sums_exactly_to_predict():
    X, y = _kinked_data(n=1000)
    model = ZoneBoostRegressor(n_rounds=15, random_state=0, spline_zones=["x1"]).fit(X, y)
    contrib = model.explain(X)
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy(), model.predict(X), atol=1e-6)


def test_spline_zones_monotonic_constraint_forces_nonnegative_segment_slopes():
    X, y = _kinked_data(n=1500)
    model = ZoneBoostRegressor(
        n_rounds=20, random_state=0, validation_fraction=0, spline_zones=["x1"],
        monotonic_constraints={"x1": 1},
    ).fit(X, y)
    checked_any = False
    for round_ in model.rounds_:
        if "x1" not in round_["main_effects"]:
            continue
        effect = round_["main_effects"]["x1"]
        slopes = np.concatenate(([effect.slope], effect.slope + np.cumsum(effect.bend)))
        assert np.all(slopes >= -1e-6)
        checked_any = True
    assert checked_any


def test_spline_zones_convexity_constraint_forces_nondecreasing_segment_slopes():
    X, y = _kinked_data(n=1500)
    model = ZoneBoostRegressor(
        n_rounds=20, random_state=0, validation_fraction=0, spline_zones=["x1"],
        convexity_constraints={"x1": 1},
    ).fit(X, y)
    checked_any = False
    for round_ in model.rounds_:
        if "x1" not in round_["main_effects"]:
            continue
        effect = round_["main_effects"]["x1"]
        if len(effect.bend) == 0:
            continue
        slopes = np.concatenate(([effect.slope], effect.slope + np.cumsum(effect.bend)))
        assert np.all(np.diff(slopes) >= -1e-6)
        checked_any = True
    assert checked_any


def test_spline_zones_bounded_effects_clips_interior_range_exactly():
    X, y = _kinked_data(n=1500)
    model = ZoneBoostRegressor(
        n_rounds=10, random_state=0, spline_zones=["x1"], bounded_effects={"x1": (-5.0, 5.0)}
    ).fit(X, y)
    round_ = model.rounds_[0]
    effect = round_["main_effects"]["x1"]
    boundaries = round_["zone_info"]["x1"][1]
    grid = np.linspace(boundaries[0], boundaries[-1], 300)
    values = _evaluate_spline_main_effect(grid, effect, boundaries)
    assert values.min() >= -5.0 - 1e-6
    assert values.max() <= 5.0 + 1e-6


def test_spline_zones_categorical_column_raises():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"cat": rng.choice(["a", "b"], n), "x": rng.uniform(-3, 3, n)})
    y = rng.normal(0, 1, n)
    with pytest.raises(ValueError):
        ZoneBoostRegressor(n_rounds=5, spline_zones=["cat"], categorical_features=["cat"]).fit(X, y)


def test_spline_zones_quantile_loss_raises():
    X, y = _kinked_data(n=300)
    with pytest.raises(ValueError):
        ZoneBoostRegressor(n_rounds=5, spline_zones=["x1"], loss="quantile").fit(X, y)


def test_spline_zones_trim_fraction_raises():
    X, y = _kinked_data(n=300)
    with pytest.raises(ValueError):
        ZoneBoostRegressor(n_rounds=5, spline_zones=["x1"], trim_fraction=0.1).fit(X, y)


def test_spline_zones_compile_to_sql_matches_predict():
    X, y = _kinked_data(n=800)
    model = ZoneBoostRegressor(n_rounds=10, random_state=0, spline_zones=["x1"]).fit(X, y)
    sql = compile_to_sql(model, table_name="t")

    conn = sqlite3.connect(":memory:")
    X.to_sql("t", conn, index=False)
    select_stmt = sql.rstrip(";").replace("SELECT", "SELECT rowid,", 1)
    df_sql = pd.read_sql(select_stmt + " ORDER BY rowid", conn)
    pred_sql = df_sql["score"].to_numpy()
    pred_py = model.predict(X)
    np.testing.assert_allclose(pred_sql, pred_py, atol=1e-8)


def test_spline_zones_learn_shrinkage_m_composes_without_error():
    X, y = _kinked_data(n=1000)
    model = ZoneBoostRegressor(
        n_rounds=10, random_state=0, spline_zones=["x1"], learn_shrinkage_m=True
    ).fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_spline_zones_track_reliability_composes_without_error():
    X, y = _kinked_data(n=1000)
    model = ZoneBoostRegressor(
        n_rounds=10, random_state=0, spline_zones=["x1"], track_reliability=True
    ).fit(X, y)
    contrib, reliability = model.explain(X, include_reliability=True)
    assert "x1" in reliability
    assert np.all(np.isfinite(reliability["x1"]["support"]))
    report = model.evidence_report(X)
    assert len(report) == len(X)


def test_spline_zones_cross_fitting_produces_honest_out_of_fold_contributions():
    # weak_learner_fit's own oof_contributions (cross-fitted, no row ever
    # scored by a table that included its own value) should differ
    # meaningfully from the in-sample production table for a spline
    # column -- if cross-fitting silently reused the full-sample fit
    # instead of genuinely refitting per fold, the two would coincide
    # almost exactly instead.
    from zoneboost._weak_learner import weak_learner_contributions, weak_learner_fit

    X, y = _kinked_data(n=1500)
    residual = y - y.mean()
    zone_info, main_effects, interactions, triples, oof_contributions, _ = weak_learner_fit(
        X, residual, ["x1"], set(), np.random.default_rng(1),
        spline_zones=frozenset(["x1"]), spline_shrinkage_m=1.0, cross_fit_folds=5,
    )
    in_sample = weak_learner_contributions(X, zone_info, main_effects, interactions, triples)
    assert np.all(np.isfinite(oof_contributions))
    assert not np.allclose(oof_contributions, in_sample, atol=1e-6)
