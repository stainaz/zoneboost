import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import FeatureUnion, Pipeline

from zoneboost import DepthCrowd, DepthTransformer


def _expert_scores(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "viral__coreness": rng.uniform(0.5, 1.0, n),
            "immune__coreness": rng.uniform(0.5, 1.0, n),
            "treatment__coreness": rng.uniform(0.5, 1.0, n),
        }
    )
    return X


def test_fit_stores_reference_distribution_per_column():
    X = _expert_scores()
    crowd = DepthCrowd().fit(X)
    assert crowd.columns_ == ["viral__coreness", "immune__coreness", "treatment__coreness"]
    assert set(crowd._reference_.keys()) == set(crowd.columns_)
    assert list(crowd._reference_["viral__coreness"]) == sorted(crowd._reference_["viral__coreness"])


def test_transform_column_names():
    X = _expert_scores()
    crowd = DepthCrowd().fit(X)
    out = crowd.transform(X)
    assert list(out.columns) == [
        "crowd__mean",
        "crowd__median",
        "crowd__std",
        "crowd__min",
        "crowd__vote_count",
        "crowd__vote_share",
        "crowd__most_atypical_expert",
    ]
    assert len(out) == len(X)


def test_get_feature_names_out_matches_transform_columns():
    X = _expert_scores()
    crowd = DepthCrowd().fit(X)
    assert list(crowd.get_feature_names_out()) == list(crowd.transform(X).columns)


def test_custom_group_name():
    X = _expert_scores()
    crowd = DepthCrowd(group_name="panel").fit(X)
    out = crowd.transform(X)
    assert list(out.columns) == [
        "panel__mean",
        "panel__median",
        "panel__std",
        "panel__min",
        "panel__vote_count",
        "panel__vote_share",
        "panel__most_atypical_expert",
    ]


def test_most_atypical_expert_identifies_lowest_scoring_column():
    X = pd.DataFrame(
        {
            "viral__coreness": [0.9, 0.9, 0.1],
            "immune__coreness": [0.8, 0.85, 0.8],
            "treatment__coreness": [0.7, 0.75, 0.7],
        }
    )
    crowd = DepthCrowd(rank_normalize=False).fit(X)
    out = crowd.transform(X)
    assert out.loc[2, "crowd__most_atypical_expert"] == "viral__coreness"
    assert out.loc[2, "crowd__min"] == pytest.approx(0.1)


def test_rank_normalize_true_produces_percentiles_in_zero_one():
    X = _expert_scores()
    crowd = DepthCrowd(rank_normalize=True).fit(X)
    out = crowd.transform(X)
    assert (out["crowd__mean"] >= 0).all() and (out["crowd__mean"] <= 1).all()


def test_rank_normalize_false_uses_raw_values_directly():
    X = pd.DataFrame({"a": [0.2, 0.5, 0.9], "b": [0.3, 0.6, 0.95]})
    crowd_raw = DepthCrowd(rank_normalize=False).fit(X)
    out_raw = crowd_raw.transform(X)
    expected_mean = X.mean(axis=1)
    assert out_raw["crowd__mean"].to_numpy() == pytest.approx(expected_mean.to_numpy())

    crowd_rank = DepthCrowd(rank_normalize=True).fit(X)
    out_rank = crowd_rank.transform(X)
    assert not np.allclose(out_rank["crowd__mean"].to_numpy(), out_raw["crowd__mean"].to_numpy())


def test_vote_count_and_share_use_rank_normalized_threshold():
    n = 100
    X = pd.DataFrame(
        {
            "a": np.linspace(0.0, 1.0, n),
            "b": np.linspace(0.0, 1.0, n),
        }
    )
    crowd = DepthCrowd(rank_normalize=True, vote_threshold=0.05).fit(X)
    out = crowd.transform(X)
    # the lowest-ranked rows on both columns should vote on both experts
    assert out.loc[0, "crowd__vote_count"] == 2
    assert out.loc[0, "crowd__vote_share"] == pytest.approx(1.0)
    # the highest-ranked row should not vote on either
    assert out.loc[n - 1, "crowd__vote_count"] == 0


def test_transform_is_deterministic_across_calls():
    X = _expert_scores()
    crowd = DepthCrowd().fit(X)
    out1 = crowd.transform(X)
    out2 = crowd.transform(X)
    pd.testing.assert_frame_equal(out1, out2)


def test_fewer_than_two_columns_raises():
    X = pd.DataFrame({"only_one__coreness": [0.5, 0.6, 0.7]})
    with pytest.raises(ValueError):
        DepthCrowd().fit(X)


def test_declaring_categorical_column_raises():
    X = pd.DataFrame({"a": [0.5, 0.6], "region": ["north", "south"]})
    with pytest.raises(ValueError):
        DepthCrowd(columns=["a", "region"]).fit(X)


def test_unknown_column_in_columns_param_raises():
    X = _expert_scores()
    with pytest.raises(ValueError):
        DepthCrowd(columns=["viral__coreness", "not_a_column"]).fit(X)


def test_invalid_vote_threshold_raises():
    X = _expert_scores()
    with pytest.raises(ValueError):
        DepthCrowd(vote_threshold=1.5).fit(X)


def test_transform_before_fit_raises():
    crowd = DepthCrowd()
    with pytest.raises(Exception):
        crowd.transform(_expert_scores())


def test_missing_column_at_transform_raises():
    X = _expert_scores()
    crowd = DepthCrowd().fit(X)
    with pytest.raises(ValueError):
        crowd.transform(X.drop(columns=["viral__coreness"]))


def test_get_params_and_clone_work():
    crowd = DepthCrowd(rank_normalize=False, vote_threshold=0.1, group_name="panel")
    params = crowd.get_params()
    assert params["vote_threshold"] == 0.1
    cloned = clone(crowd)
    assert cloned.vote_threshold == 0.1
    assert cloned.group_name == "panel"
    assert cloned is not crowd


def test_works_inside_pipeline_after_feature_union_of_depth_transformers():
    # FeatureUnion prefixes each sub-transformer's own output names with its
    # step name (e.g. "viral__viral__coreness"), so the coreness column
    # names are derived from the fitted union rather than hardcoded here.
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {
            "viral_load": rng.normal(0, 1, n),
            "cd4": rng.normal(0, 1, n),
            "duration": rng.normal(0, 1, n),
        }
    )

    experts = FeatureUnion(
        [
            ("viral", DepthTransformer(columns=["viral_load"], group_name="viral")),
            ("immune", DepthTransformer(columns=["cd4"], group_name="immune")),
        ]
    ).set_output(transform="pandas")
    expert_output = experts.fit_transform(X)
    coreness_cols = [c for c in expert_output.columns if c.endswith("__coreness")]
    assert len(coreness_cols) == 2

    pipe = Pipeline([("experts", experts), ("crowd", DepthCrowd(columns=coreness_cols))])
    out = pipe.fit_transform(X)
    assert out.shape == (n, 7)
    assert list(out.columns) == list(pipe.named_steps["crowd"].get_feature_names_out())
    assert set(out["crowd__most_atypical_expert"].unique()).issubset(set(coreness_cols))
