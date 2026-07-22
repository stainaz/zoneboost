import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from zoneboost import DepthTransformer


def _cluster_with_outliers(n=500, n_outliers=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    X = pd.DataFrame({"x": x, "y": y})
    outliers = pd.DataFrame({"x": [50.0] * n_outliers, "y": [-50.0] * n_outliers})
    return pd.concat([X, outliers], ignore_index=True), n


def test_fit_stores_mean_and_covariance():
    X, _ = _cluster_with_outliers()
    depth = DepthTransformer().fit(X)
    assert depth.columns_ == ["x", "y"]
    assert depth.mean_.shape == (2,)
    assert depth.covariance_.shape == (2, 2)


def test_transform_column_names_default_group_name():
    X, _ = _cluster_with_outliers()
    depth = DepthTransformer().fit(X)
    out = depth.transform(X)
    assert list(out.columns) == ["x_y__depth_distance", "x_y__coreness"]
    assert len(out) == len(X)


def test_custom_group_name():
    X, _ = _cluster_with_outliers()
    depth = DepthTransformer(group_name="cluster").fit(X)
    out = depth.transform(X)
    assert list(out.columns) == ["cluster__depth_distance", "cluster__coreness"]


def test_get_feature_names_out_matches_transform_columns():
    X, _ = _cluster_with_outliers()
    depth = DepthTransformer().fit(X)
    assert list(depth.get_feature_names_out()) == list(depth.transform(X).columns)


def test_outliers_get_lower_coreness_and_higher_distance_than_cluster():
    X, n = _cluster_with_outliers()
    depth = DepthTransformer().fit(X)
    out = depth.transform(X)
    cluster_coreness = out.iloc[:n]["x_y__coreness"]
    outlier_coreness = out.iloc[n:]["x_y__coreness"]
    cluster_distance = out.iloc[:n]["x_y__depth_distance"]
    outlier_distance = out.iloc[n:]["x_y__depth_distance"]
    assert outlier_coreness.max() < cluster_coreness.min()
    assert outlier_distance.min() > cluster_distance.max()


def test_coreness_is_bounded_in_zero_one():
    X, _ = _cluster_with_outliers()
    depth = DepthTransformer().fit(X)
    out = depth.transform(X)
    assert (out["x_y__coreness"] > 0).all() and (out["x_y__coreness"] <= 1.0).all()


def test_missing_value_mean_imputed_without_error():
    X, _ = _cluster_with_outliers()
    X_missing = X.copy()
    X_missing.loc[0, "x"] = np.nan
    depth = DepthTransformer().fit(X)
    out = depth.transform(X_missing)
    assert np.isfinite(out.loc[0, "x_y__depth_distance"])


def test_perfectly_correlated_columns_do_not_raise():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    X = pd.DataFrame({"x": x, "x_copy": x})
    depth = DepthTransformer().fit(X)
    out = depth.transform(X)
    assert np.isfinite(out["x_x_copy__depth_distance"]).all()


def test_declaring_categorical_column_raises():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0], "region": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        DepthTransformer(columns=["x", "region"]).fit(X)


def test_columns_none_excludes_categorical_columns_automatically():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "region": ["a", "b", "a", "b"]})
    depth = DepthTransformer().fit(X)
    assert depth.columns_ == ["x"]


def test_columns_param_restricts_encoded_columns():
    X, _ = _cluster_with_outliers()
    X["z"] = np.arange(len(X))
    depth = DepthTransformer(columns=["x", "y"]).fit(X)
    assert depth.columns_ == ["x", "y"]


def test_unknown_column_in_columns_param_raises():
    X, _ = _cluster_with_outliers()
    with pytest.raises(ValueError):
        DepthTransformer(columns=["not_a_column"]).fit(X)


def test_works_inside_pipeline_and_column_transformer_with_linear_model():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
    labels = rng.integers(0, 2, n)
    ct = ColumnTransformer([("depth", DepthTransformer(), ["x", "y"])])
    pipe = Pipeline([("encode", ct), ("model", LogisticRegression())]).fit(X, labels)
    preds = pipe.predict_proba(X)
    assert preds.shape == (n, 2)


def test_get_params_and_clone_work():
    model = DepthTransformer(columns=["x", "y"], ridge=1e-4, random_state=7)
    params = model.get_params()
    assert params["ridge"] == 1e-4
    cloned = clone(model)
    assert cloned.ridge == 1e-4
    assert cloned.columns == ["x", "y"]
    assert cloned is not model


def test_transform_before_fit_raises():
    depth = DepthTransformer()
    with pytest.raises(Exception):
        depth.transform(pd.DataFrame({"x": [1.0, 2.0]}))


def test_no_numeric_columns_raises():
    X = pd.DataFrame({"region": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        DepthTransformer().fit(X)
