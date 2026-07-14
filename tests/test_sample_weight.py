import sqlite3

import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor, compile_to_sql
from zoneboost._weak_learner import _zone_raw_stat
from zoneboost._zones import adaptive_zone_boundaries, zone_centers


def test_uniform_weight_bit_identical_to_none():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
    y = 2 * X["x"] + rng.normal(scale=0.5, size=n)

    m0 = ZoneBoostRegressor(n_rounds=30, random_state=0).fit(X, y)
    m1 = ZoneBoostRegressor(n_rounds=30, random_state=0).fit(X, y, sample_weight=np.ones(n))
    np.testing.assert_array_equal(m0.predict(X), m1.predict(X))


def test_frequency_weight_equivalence_zone_construction():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.uniform(0, 10, n)
    y = 2 * x + rng.normal(scale=0.5, size=n)
    w = rng.integers(1, 5, n).astype(float)

    b_w = adaptive_zone_boundaries(x, y, sample_weight=w)
    x_dup = np.repeat(x, w.astype(int))
    y_dup = np.repeat(y, w.astype(int))
    b_dup = adaptive_zone_boundaries(x_dup, y_dup)
    np.testing.assert_array_equal(b_w, b_dup)

    c_w = zone_centers(x, b_w, sample_weight=w)
    c_dup = zone_centers(x_dup, b_dup)
    np.testing.assert_allclose(c_w, c_dup)


def test_frequency_weight_equivalence_zone_raw_stat():
    zones = np.array([0, 0, 0, 1, 1])
    targets = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
    w = np.array([2.0, 1.0, 3.0, 1.0, 2.0])

    stat_w, counts_w = _zone_raw_stat(zones, targets, 2, sample_weight=w)

    zones_dup = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
    targets_dup = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 5.0, 7.0, 7.0])
    stat_dup, counts_dup = _zone_raw_stat(zones_dup, targets_dup, 2)

    np.testing.assert_allclose(stat_w, stat_dup)
    np.testing.assert_allclose(counts_w, counts_dup)


def test_outlier_down_weighting_measured_directly():
    rng = np.random.default_rng(2)
    n = 2000
    X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
    y = 2 * X["x"] + rng.normal(scale=0.5, size=n)
    outlier_mask = (X["x"] >= 4) & (X["x"] < 5)
    outlier_idx = rng.choice(np.where(outlier_mask)[0], size=5, replace=False)
    y_contam = y.copy()
    y_contam[outlier_idx] += 200
    w = np.ones(n)
    w[outlier_idx] = 0.001

    m_plain = ZoneBoostRegressor(n_rounds=30, random_state=0).fit(X, y_contam)
    m_weighted = ZoneBoostRegressor(n_rounds=30, random_state=0).fit(X, y_contam, sample_weight=w)

    X_test = pd.DataFrame({"x": [4.5]})
    true_val = 2 * 4.5
    pred_plain = m_plain.predict(X_test)[0]
    pred_weighted = m_weighted.predict(X_test)[0]

    assert abs(pred_weighted - true_val) < abs(pred_plain - true_val)
    assert abs(pred_weighted - true_val) < 2.0


def test_pairs_respect_sample_weight():
    rng = np.random.default_rng(4)
    n = 3000
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    y = 0.3 * x1 + 0.3 * x2 + rng.normal(scale=0.3, size=n)
    joint_outlier_mask = (x1 >= 4) & (x1 < 5) & (x2 >= 4) & (x2 < 5)
    outlier_idx = rng.choice(np.where(joint_outlier_mask)[0], size=8, replace=False)
    y_contam = y.copy()
    y_contam[outlier_idx] += 200
    w = np.ones(n)
    w[outlier_idx] = 0.001
    X = pd.DataFrame({"x1": x1, "x2": x2})

    m_plain = ZoneBoostRegressor(n_rounds=40, random_state=0).fit(X, y_contam)
    m_weighted = ZoneBoostRegressor(n_rounds=40, random_state=0).fit(X, y_contam, sample_weight=w)

    X_test = pd.DataFrame({"x1": [4.5], "x2": [4.5]})
    true_val = 0.3 * 4.5 + 0.3 * 4.5
    pred_plain = m_plain.predict(X_test)[0]
    pred_weighted = m_weighted.predict(X_test)[0]

    assert abs(pred_weighted - true_val) < abs(pred_plain - true_val)


@pytest.mark.parametrize("loss", ["poisson", "gamma", "tweedie"])
def test_glm_losses_fit_with_sample_weight(loss):
    rng = np.random.default_rng(5)
    n = 1000
    X = pd.DataFrame({"age": rng.uniform(18, 70, n)})
    w = rng.uniform(0.5, 2.0, n)
    if loss == "gamma":
        y = rng.gamma(shape=2.0, scale=np.exp(1.0 + 0.01 * X["age"]))
    else:
        y = rng.poisson(np.exp(-3.0 + 0.02 * X["age"])).astype(float)

    model = ZoneBoostRegressor(loss=loss, n_rounds=15, random_state=0).fit(X, y, sample_weight=w)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))
    if loss in ("poisson", "tweedie"):
        assert np.all(preds >= 0)


def test_validation_errors():
    rng = np.random.default_rng(6)
    n = 200
    X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
    y = rng.normal(size=n)
    w = np.ones(n)

    with pytest.raises(ValueError, match="loss='quantile'"):
        ZoneBoostRegressor(loss="quantile").fit(X, y, sample_weight=w)
    with pytest.raises(ValueError, match="trim_fraction > 0"):
        ZoneBoostRegressor(trim_fraction=0.1).fit(X, y, sample_weight=w)
    with pytest.raises(ValueError, match="non-negative"):
        ZoneBoostRegressor().fit(X, y, sample_weight=-np.ones(n))
    with pytest.raises(ValueError, match="length"):
        ZoneBoostRegressor().fit(X, y, sample_weight=np.ones(n - 1))


def test_compile_to_sql_unaffected_by_sample_weight_at_fit_time():
    rng = np.random.default_rng(7)
    n = 500
    X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
    y = 2 * X["x"] + rng.normal(scale=0.5, size=n)
    w = rng.uniform(0.5, 2.0, n)

    model = ZoneBoostRegressor(n_rounds=10, random_state=0).fit(X, y, sample_weight=w)
    sql = compile_to_sql(model)
    pred = model.predict(X)

    conn = sqlite3.connect(":memory:")
    try:
        X.to_sql("input_table", conn, index=False)
        scores = np.array([row[0] for row in conn.execute(sql).fetchall()])
    finally:
        conn.close()

    np.testing.assert_allclose(scores, pred, atol=1e-9)
