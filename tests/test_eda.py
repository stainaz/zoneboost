import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor, compare_models
from zoneboost.eda import drift_dashboard, zone_boxplot


def _step_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    region = rng.choice(["north", "south", "east"], n)
    y = np.where(x < 5, 0.0, 10.0) + rng.normal(0, 1, n)
    X = pd.DataFrame({"x": x, "region": region})
    return X, y


def test_zone_boxplot_plot_false_returns_stats_only():
    X, y = _step_data()
    stats = zone_boxplot(X, y, "x", n_zones=3, plot=False)
    assert isinstance(stats, pd.DataFrame)
    assert {"count", "mean", "median", "q1", "q3", "iqr", "outlier_count", "outlier_frac"} <= set(stats.columns)
    assert stats["count"].sum() == len(X)


def test_zone_boxplot_recovers_known_step_signal():
    X, y = _step_data()
    stats = zone_boxplot(X, y, "x", n_zones=2, plot=False)
    # a clean step at x=5: the low-x zone's mean should be far below the
    # high-x zone's.
    means = stats["mean"].to_numpy()
    assert means[-1] - means[0] > 5.0


def test_zone_boxplot_categorical_column():
    X, y = _step_data()
    stats = zone_boxplot(X, y, "region", plot=False)
    assert set(stats.index) == {"north", "south", "east"}
    assert stats["count"].sum() == len(X)


def test_zone_boxplot_valid_range_flags_out_of_range_zone():
    X, y = _step_data()
    # x ranges over [0, 10]; declare only [2, 8] as business-valid.
    stats = zone_boxplot(X, y, "x", n_zones=2, valid_range=(2, 8), plot=False)
    assert "pct_business_invalid" in stats.columns
    assert (stats["pct_business_invalid"] >= 0).all() and (stats["pct_business_invalid"] <= 1).all()


def test_zone_boxplot_valid_range_omitted_by_default():
    X, y = _step_data()
    stats = zone_boxplot(X, y, "x", n_zones=2, plot=False)
    assert "pct_business_invalid" not in stats.columns


def test_zone_boxplot_valid_range_on_categorical_raises():
    X, y = _step_data()
    with pytest.raises(ValueError):
        zone_boxplot(X, y, "region", valid_range=(0, 1), plot=False)


def test_zone_boxplot_plot_true_returns_axes():
    X, y = _step_data()
    stats, ax = zone_boxplot(X, y, "x", n_zones=3)
    assert isinstance(stats, pd.DataFrame)
    assert hasattr(ax, "boxplot")  # matplotlib Axes-like


def test_zone_boxplot_handles_missing_values():
    X, y = _step_data()
    X = X.copy()
    X.loc[:20, "x"] = np.nan
    stats = zone_boxplot(X, y, "x", n_zones=3, plot=False)
    assert "missing" in stats.index
    assert stats.loc["missing", "count"] == 21


def test_zone_boxplot_mismatched_lengths_raise():
    X, y = _step_data()
    with pytest.raises(ValueError):
        zone_boxplot(X, y[:-1], "x", plot=False)


def test_zone_boxplot_unknown_column_raises():
    X, y = _step_data()
    with pytest.raises(ValueError):
        zone_boxplot(X, y, "not_a_column", plot=False)


def _drift_models(seed_old=0, seed_new=1, n=1000):
    rng = np.random.default_rng(0)
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    y_old = 2.0 * x1 + rng.normal(0, 0.5, n)
    y_new = 2.5 * x1 + 0.5 * x2 + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    m_old = ZoneBoostRegressor(n_rounds=15, random_state=seed_old).fit(X, y_old)
    m_new = ZoneBoostRegressor(n_rounds=15, random_state=seed_new).fit(X, y_new)
    return m_old, m_new, X, y_old


def test_drift_dashboard_plot_false_matches_compare_models_exactly():
    m_old, m_new, X, y = _drift_models()
    data = drift_dashboard(m_old, m_new, X, y, plot=False)
    ref = compare_models(m_old, m_new, X, y)
    assert set(data.keys()) == set(ref.keys())
    pd.testing.assert_frame_equal(data["feature_importance_change"], ref["feature_importance_change"])
    assert data["performance_change"] == ref["performance_change"]


def test_drift_dashboard_plot_true_returns_data_and_figure():
    m_old, m_new, X, y = _drift_models()
    data, fig = drift_dashboard(m_old, m_new, X, y)
    assert isinstance(data, dict)
    assert hasattr(fig, "axes")
    assert len(fig.axes) >= 2


def test_drift_dashboard_without_y_eval():
    m_old, m_new, X, _ = _drift_models()
    data, fig = drift_dashboard(m_old, m_new, X)
    assert data["performance_change"] is None


def test_drift_dashboard_no_shared_continuous_columns():
    rng = np.random.default_rng(0)
    n = 800
    region = rng.choice(["a", "b", "c"], n)
    y = rng.normal(0, 1, n)
    X = pd.DataFrame({"region": region})
    m_old = ZoneBoostRegressor(n_rounds=10, categorical_features=["region"], random_state=0).fit(X, y)
    m_new = ZoneBoostRegressor(n_rounds=10, categorical_features=["region"], random_state=1).fit(X, y)
    data, fig = drift_dashboard(m_old, m_new, X)
    assert data["boundary_shift"] == {}
    assert len(fig.axes) == 2  # tornado + prediction shift only
