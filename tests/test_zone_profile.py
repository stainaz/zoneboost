import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from zoneboost import ZoneProfileEncoder


def _zone_signal_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    # a clean step function of x: zone stats should recover it closely.
    p = np.where(x < 5, 0.1, 0.8)
    y = (rng.uniform(0, 1, n) < p).astype(float)
    region = rng.choice(["north", "south", "east"], n)
    X = pd.DataFrame({"x": x, "region": region})
    return X, y


def test_fit_transform_column_names():
    X, y = _zone_signal_data()
    encoder = ZoneProfileEncoder(max_zones=4, min_zone_abs=20).fit(X, y)
    out = encoder.transform(X)
    assert list(out.columns) == [
        "x__zone_mean",
        "x__zone_count",
        "x__zone_var",
        "region__zone_mean",
        "region__zone_count",
        "region__zone_var",
    ]
    assert len(out) == len(X)


def test_get_feature_names_out_matches_transform_columns():
    X, y = _zone_signal_data()
    encoder = ZoneProfileEncoder(max_zones=4, min_zone_abs=20).fit(X, y)
    assert list(encoder.get_feature_names_out()) == list(encoder.transform(X).columns)


def test_continuous_zone_mean_recovers_known_step_signal():
    X, y = _zone_signal_data()
    encoder = ZoneProfileEncoder(max_zones=4, min_zone_abs=20, shrinkage=False).fit(X, y)
    out = encoder.transform(X)
    low = out.loc[X["x"] < 4, "x__zone_mean"]
    high = out.loc[X["x"] > 6, "x__zone_mean"]
    assert low.mean() < 0.3
    assert high.mean() > 0.6


def test_categorical_column_zone_mean_differs_by_category_when_signal_present():
    rng = np.random.default_rng(1)
    n = 2000
    region = rng.choice(["a", "b"], n)
    p = np.where(region == "a", 0.1, 0.9)
    y = (rng.uniform(0, 1, n) < p).astype(float)
    X = pd.DataFrame({"region": region})
    encoder = ZoneProfileEncoder(shrinkage=False).fit(X, y)
    out = encoder.transform(X)
    mean_a = out.loc[X["region"] == "a", "region__zone_mean"].iloc[0]
    mean_b = out.loc[X["region"] == "b", "region__zone_mean"].iloc[0]
    assert mean_b - mean_a > 0.5


def test_shrinkage_pulls_sparse_zone_mean_toward_grand_mean():
    X, y = _zone_signal_data()
    raw = ZoneProfileEncoder(max_zones=4, min_zone_abs=20, shrinkage=False).fit(X, y)
    shrunk = ZoneProfileEncoder(max_zones=4, min_zone_abs=20, shrinkage=True).fit(X, y)
    grand_mean = y.mean()
    # every shrunk zone mean should sit strictly between the raw mean and
    # the grand mean (or equal it, for a zone with essentially full support).
    for raw_m, shrunk_m in zip(raw.zone_stats_["x"]["raw_means"], shrunk.zone_stats_["x"]["shrunk_means"]):
        assert abs(shrunk_m - grand_mean) <= abs(raw_m - grand_mean) + 1e-9


def test_missing_value_falls_back_to_missing_zone_without_error():
    X, y = _zone_signal_data()
    encoder = ZoneProfileEncoder(max_zones=4, min_zone_abs=20).fit(X, y)
    X_missing = X.copy()
    X_missing.loc[0, "x"] = np.nan
    X_missing.loc[1, "region"] = None
    out = encoder.transform(X_missing)
    assert np.isfinite(out.loc[0, "x__zone_mean"])
    assert np.isfinite(out.loc[1, "region__zone_mean"])


def test_unseen_category_at_transform_time_does_not_raise():
    X, y = _zone_signal_data()
    encoder = ZoneProfileEncoder(max_zones=4, min_zone_abs=20).fit(X, y)
    X_new = X.copy()
    X_new.loc[0, "region"] = "unseen_category"
    out = encoder.transform(X_new)
    assert np.isfinite(out.loc[0, "region__zone_mean"])


def test_columns_param_restricts_encoded_columns():
    X, y = _zone_signal_data()
    encoder = ZoneProfileEncoder(columns=["x"], max_zones=4, min_zone_abs=20).fit(X, y)
    out = encoder.transform(X)
    assert list(out.columns) == ["x__zone_mean", "x__zone_count", "x__zone_var"]


def test_unknown_column_in_columns_param_raises():
    X, y = _zone_signal_data()
    with pytest.raises(ValueError):
        ZoneProfileEncoder(columns=["not_a_column"]).fit(X, y)


def test_mismatched_lengths_raise():
    X, y = _zone_signal_data()
    with pytest.raises(ValueError):
        ZoneProfileEncoder().fit(X, y[:-1])


def test_works_inside_pipeline_and_column_transformer_with_linear_model():
    X, y = _zone_signal_data()
    ct = ColumnTransformer([("zones", ZoneProfileEncoder(max_zones=4, min_zone_abs=20), ["x", "region"])])
    pipe = Pipeline([("encode", ct), ("model", LogisticRegression())]).fit(X, y)
    preds = pipe.predict_proba(X)
    assert preds.shape == (len(X), 2)


def test_get_params_and_clone_work():
    model = ZoneProfileEncoder(max_zones=4, min_zone_abs=15, shrinkage=False, random_state=7)
    params = model.get_params()
    assert params["max_zones"] == 4
    assert params["shrinkage"] is False

    cloned = clone(model)
    assert cloned.max_zones == 4
    assert cloned.shrinkage is False
    assert cloned is not model


def test_transform_before_fit_raises():
    encoder = ZoneProfileEncoder()
    with pytest.raises(Exception):
        encoder.transform(pd.DataFrame({"x": [1.0, 2.0]}))
