import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from zoneboost import ZoneFeatureSpace


def _mixed_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    price = rng.uniform(10, 500, n)
    sqft = rng.uniform(500, 4000, n)
    debt = rng.uniform(0, 50000, n)
    income = rng.uniform(20000, 200000, n)
    region = rng.choice(["north", "south", "east"], n)
    y = 0.5 * price + 0.001 * sqft + rng.normal(0, 5, n)
    X = pd.DataFrame({"price": price, "sqft": sqft, "debt": debt, "income": income, "region": region})
    return X, y


def _interaction_data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-3, 3, n)
    b = rng.uniform(-3, 3, n)
    c = rng.uniform(-3, 3, n)  # unrelated noise column
    y = 2.0 * a * b + rng.normal(0, 0.5, n)  # genuine interaction, no real main effects
    X = pd.DataFrame({"a": a, "b": b, "c": c})
    return X, y


def test_default_zone_profiles_only():
    X, y = _mixed_data()
    space = ZoneFeatureSpace(random_state=0).fit(X, y)
    out = space.transform(X)
    assert set(out.columns) == {f"{c}__{s}" for c in X.columns for s in ("zone_mean", "zone_count", "zone_var")}


def test_zone_profiles_with_explicit_columns():
    X, y = _mixed_data()
    space = ZoneFeatureSpace(zone_profiles=["price", "sqft"], random_state=0).fit(X, y)
    out = space.transform(X)
    assert set(out.columns) == {f"{c}__{s}" for c in ("price", "sqft") for s in ("zone_mean", "zone_count", "zone_var")}


def test_depth_scores_bool_true_one_joint_group():
    X, y = _mixed_data()
    space = ZoneFeatureSpace(zone_profiles=False, depth_scores=True, random_state=0).fit(X, y)
    out = space.transform(X)
    # one joint group over every numeric column (price, sqft, debt, income)
    assert len(out.columns) == 2
    assert any(c.endswith("__coreness") for c in out.columns)


def test_depth_scores_named_groups():
    X, y = _mixed_data()
    space = ZoneFeatureSpace(
        zone_profiles=False,
        depth_scores=[("size", ["price", "sqft"]), ("financial", ["debt", "income"])],
        random_state=0,
    ).fit(X, y)
    out = space.transform(X)
    assert "size__coreness" in out.columns
    assert "financial__coreness" in out.columns


def test_categorical_depth_scores():
    X, y = _mixed_data()
    space = ZoneFeatureSpace(zone_profiles=False, categorical_depth_scores=True, random_state=0).fit(X, y)
    out = space.transform(X)
    assert "region__coreness" in out.columns


def test_conditional_grids():
    X, y = _mixed_data()
    space = ZoneFeatureSpace(
        zone_profiles=False,
        conditional_grids=[("price", "sqft", ["region"])],
        random_state=0,
    ).fit(X, y)
    out = space.transform(X)
    assert set(out.columns) == {"price_sqft__cell_mean", "price_sqft__cell_count", "price_sqft__cell_var", "price_sqft__used_segment_grid"}


def test_combined_transformers_concatenate():
    X, y = _mixed_data()
    space = ZoneFeatureSpace(
        zone_profiles=["price"],
        depth_scores=[["debt", "income"]],
        categorical_depth_scores=True,
        conditional_grids=[("price", "sqft", ["region"])],
        random_state=0,
    ).fit(X, y)
    out = space.transform(X)
    expected = {"price__zone_mean", "price__zone_count", "price__zone_var"}
    expected |= {"debt_income__depth_distance", "debt_income__coreness"}
    expected |= {"region__count", "region__coreness"}
    expected |= {"price_sqft__cell_mean", "price_sqft__cell_count", "price_sqft__cell_var", "price_sqft__used_segment_grid"}
    assert set(out.columns) == expected
    assert list(out.columns) == space.get_feature_names_out().tolist()


def test_y_required_when_zone_profiles_enabled():
    X, y = _mixed_data()
    with pytest.raises(ValueError):
        ZoneFeatureSpace(zone_profiles=True).fit(X)


def test_y_required_when_conditional_grids_enabled():
    X, y = _mixed_data()
    with pytest.raises(ValueError):
        ZoneFeatureSpace(zone_profiles=False, conditional_grids=[("price", "sqft", ["region"])]).fit(X)


def test_y_not_required_for_unsupervised_only():
    X, y = _mixed_data()
    space = ZoneFeatureSpace(zone_profiles=False, depth_scores=True).fit(X)
    space.transform(X)


def test_nothing_enabled_raises():
    X, y = _mixed_data()
    with pytest.raises(ValueError):
        ZoneFeatureSpace(zone_profiles=False).fit(X, y)


def test_explain_interaction_counts_fully_for_both_source_columns():
    X, y = _interaction_data()
    space = ZoneFeatureSpace(
        zone_profiles=False,
        depth_scores=[["a", "b"]],
        random_state=0,
    ).fit(X, y)
    out = space.transform(X)
    # fake "downstream model" coefficients: one weight per output column
    values = np.ones(len(space.get_feature_names_out()))
    rolled = space.explain(values)
    # both a and b are sources of every output column in the joint depth group
    assert rolled["a"] == rolled["b"]
    assert rolled["a"] == pytest.approx(2.0)  # 2 output columns (depth_distance, coreness), each contributing 1.0


def test_explain_raises_on_wrong_length():
    X, y = _mixed_data()
    space = ZoneFeatureSpace(random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        space.explain([1.0, 2.0])


def test_explain_via_linear_model_pipeline():
    X, y = _mixed_data()
    space = ZoneFeatureSpace(zone_profiles=["price", "sqft"], random_state=0).fit(X, y)
    X_transformed = space.transform(X)
    lr = LinearRegression().fit(X_transformed, y)
    rolled = space.explain(lr.coef_)
    assert set(rolled.index) == {"price", "sqft"}
    # price is the genuine signal (y ~= 0.5 * price)
    assert rolled["price"] > rolled["sqft"]


def test_suggest_interactions_ranks_genuine_interaction_highest():
    X, y = _interaction_data()
    space = ZoneFeatureSpace()
    ranked = space.suggest_interactions(X, y, top_k=3)
    assert ranked[0] in [("a", "b")]


def test_suggest_interactions_requires_at_least_two_columns():
    X, y = _interaction_data()
    space = ZoneFeatureSpace()
    with pytest.raises(ValueError):
        space.suggest_interactions(X[["a"]], y)


def test_split_criterion_correlation_is_forwarded_and_default_unchanged():
    X, y = _mixed_data()
    default = ZoneFeatureSpace(zone_profiles=["price"], random_state=0).fit(X, y)
    explicit = ZoneFeatureSpace(zone_profiles=["price"], split_criterion="variance", random_state=0).fit(X, y)
    pd.testing.assert_frame_equal(default.transform(X), explicit.transform(X))

    corr = ZoneFeatureSpace(zone_profiles=["price"], split_criterion="correlation", random_state=0).fit(X, y)
    corr.transform(X)


def test_get_params_and_clone_work():
    space = ZoneFeatureSpace(zone_profiles=["price"], depth_scores=True, random_state=1)
    params = space.get_params()
    assert params["random_state"] == 1
    cloned = clone(space)
    assert cloned.zone_profiles == ["price"]
    assert cloned is not space


def test_works_inside_sklearn_pipeline():
    X, y = _mixed_data()
    pipe = Pipeline([
        ("space", ZoneFeatureSpace(zone_profiles=["price", "sqft"], random_state=0)),
        ("model", LinearRegression()),
    ])
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(X),)


def test_transform_before_fit_raises():
    space = ZoneFeatureSpace()
    with pytest.raises(Exception):
        space.transform(pd.DataFrame({"x": [1.0, 2.0]}))
