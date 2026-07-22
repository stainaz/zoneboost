import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from zoneboost import ConditionalZoneGrid


def _segment_flip_data(n_per_segment=800, seed=0):
    rng = np.random.default_rng(seed)
    x_a = rng.uniform(0, 10, n_per_segment)
    y_a = rng.uniform(0, 10, n_per_segment)
    target_a = x_a + y_a + rng.normal(0, 0.1, n_per_segment)  # positive relationship

    x_b = rng.uniform(0, 10, n_per_segment)
    y_b = rng.uniform(0, 10, n_per_segment)
    target_b = -(x_b + y_b) + rng.normal(0, 0.1, n_per_segment)  # negative relationship

    X = pd.DataFrame(
        {
            "x": np.concatenate([x_a, x_b]),
            "y": np.concatenate([y_a, y_b]),
            "region": ["north"] * n_per_segment + ["south"] * n_per_segment,
        }
    )
    target = np.concatenate([target_a, target_b])
    return X, target


def test_fit_stores_global_and_segment_grids():
    X, target = _segment_flip_data()
    grid = ConditionalZoneGrid(columns=["x", "y"], segment_columns=["region"]).fit(X, target)
    assert set(grid.segment_grids_.keys()) == {("north",), ("south",)}
    assert grid.global_grid_ is not None


def test_transform_column_names():
    X, target = _segment_flip_data()
    grid = ConditionalZoneGrid(columns=["x", "y"], segment_columns=["region"]).fit(X, target)
    out = grid.transform(X)
    assert list(out.columns) == [
        "x_y__cell_mean",
        "x_y__cell_count",
        "x_y__cell_var",
        "x_y__used_segment_grid",
    ]
    assert len(out) == len(X)


def test_get_feature_names_out_matches_transform_columns():
    X, target = _segment_flip_data()
    grid = ConditionalZoneGrid(columns=["x", "y"], segment_columns=["region"]).fit(X, target)
    assert list(grid.get_feature_names_out()) == list(grid.transform(X).columns)


def test_segments_recover_opposite_signal_direction():
    X, target = _segment_flip_data()
    grid = ConditionalZoneGrid(columns=["x", "y"], segment_columns=["region"], shrinkage=False).fit(X, target)
    out = grid.transform(X)

    north_low = out.loc[(X["region"] == "north") & (X["x"] < 2) & (X["y"] < 2), "x_y__cell_mean"]
    north_high = out.loc[(X["region"] == "north") & (X["x"] > 8) & (X["y"] > 8), "x_y__cell_mean"]
    south_low = out.loc[(X["region"] == "south") & (X["x"] < 2) & (X["y"] < 2), "x_y__cell_mean"]
    south_high = out.loc[(X["region"] == "south") & (X["x"] > 8) & (X["y"] > 8), "x_y__cell_mean"]

    assert north_high.mean() > north_low.mean()  # positive relationship in "north"
    assert south_high.mean() < south_low.mean()  # negative relationship in "south"


def test_used_segment_grid_flag_is_one_for_qualifying_segments():
    X, target = _segment_flip_data()
    grid = ConditionalZoneGrid(columns=["x", "y"], segment_columns=["region"], min_segment_size=50).fit(X, target)
    out = grid.transform(X)
    assert (out["x_y__used_segment_grid"] == 1).all()


def test_small_segment_falls_back_to_global_grid():
    X, target = _segment_flip_data(n_per_segment=200)
    tiny = pd.DataFrame({"x": [5.0] * 5, "y": [5.0] * 5, "region": ["tiny"] * 5})
    X_with_tiny = pd.concat([X, tiny], ignore_index=True)
    target_with_tiny = np.concatenate([target, np.zeros(5)])

    grid = ConditionalZoneGrid(
        columns=["x", "y"], segment_columns=["region"], min_segment_size=50
    ).fit(X_with_tiny, target_with_tiny)
    assert "tiny" not in [k[0] for k in grid.segment_grids_.keys()]

    out = grid.transform(X_with_tiny)
    tiny_rows = out.iloc[-5:]
    assert (tiny_rows["x_y__used_segment_grid"] == 0).all()


def test_unseen_segment_at_transform_time_falls_back_without_error():
    X, target = _segment_flip_data()
    grid = ConditionalZoneGrid(columns=["x", "y"], segment_columns=["region"]).fit(X, target)
    X_new = X.copy()
    X_new.loc[0, "region"] = "unseen_region"
    out = grid.transform(X_new)
    assert out.loc[0, "x_y__used_segment_grid"] == 0
    assert np.isfinite(out.loc[0, "x_y__cell_mean"])


def test_missing_continuous_value_does_not_raise():
    X, target = _segment_flip_data()
    X_missing = X.copy()
    X_missing.loc[0, "x"] = np.nan
    grid = ConditionalZoneGrid(columns=["x", "y"], segment_columns=["region"]).fit(X, target)
    out = grid.transform(X_missing)
    assert np.isfinite(out.loc[0, "x_y__cell_mean"])


def test_wrong_number_of_columns_raises():
    X, target = _segment_flip_data()
    with pytest.raises(ValueError):
        ConditionalZoneGrid(columns=["x"], segment_columns=["region"]).fit(X, target)


def test_categorical_column_in_columns_raises():
    X, target = _segment_flip_data()
    with pytest.raises(ValueError):
        ConditionalZoneGrid(columns=["x", "region"], segment_columns=["region"]).fit(X, target)


def test_mismatched_lengths_raise():
    X, target = _segment_flip_data()
    with pytest.raises(ValueError):
        ConditionalZoneGrid(columns=["x", "y"], segment_columns=["region"]).fit(X, target[:-1])


def test_unknown_segment_column_raises():
    X, target = _segment_flip_data()
    with pytest.raises(ValueError):
        ConditionalZoneGrid(columns=["x", "y"], segment_columns=["not_a_column"]).fit(X, target)


def test_works_inside_pipeline_and_column_transformer_with_linear_model():
    X, target = _segment_flip_data()
    ct = ColumnTransformer(
        [("grid", ConditionalZoneGrid(columns=["x", "y"], segment_columns=["region"]), ["x", "y", "region"])]
    )
    pipe = Pipeline([("encode", ct), ("model", LinearRegression())]).fit(X, target)
    preds = pipe.predict(X)
    assert preds.shape == (len(X),)


def test_get_params_and_clone_work():
    model = ConditionalZoneGrid(columns=["x", "y"], segment_columns=["region"], min_segment_size=30, shrinkage=False)
    params = model.get_params()
    assert params["min_segment_size"] == 30
    assert params["shrinkage"] is False

    cloned = clone(model)
    assert cloned.min_segment_size == 30
    assert cloned.columns == ["x", "y"]
    assert cloned is not model


def test_transform_before_fit_raises():
    grid = ConditionalZoneGrid(columns=["x", "y"], segment_columns=["region"])
    with pytest.raises(Exception):
        grid.transform(pd.DataFrame({"x": [1.0], "y": [1.0], "region": ["a"]}))
