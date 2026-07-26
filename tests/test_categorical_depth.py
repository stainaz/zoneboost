import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from zoneboost import CategoricalDepthTransformer, DepthCrowd, DepthTransformer


def _treatment_data():
    return pd.DataFrame({"treatment": ["A", "A", "A", "A", "B", "C"]})


def test_fit_stores_category_maps_and_counts():
    X = _treatment_data()
    cat = CategoricalDepthTransformer(columns=["treatment"]).fit(X)
    assert cat.columns_ == ["treatment"]
    assert set(cat._category_maps_["treatment"].keys()) == {"A", "B", "C"}


def test_transform_column_names_default_group_name():
    X = _treatment_data()
    cat = CategoricalDepthTransformer(columns=["treatment"]).fit(X)
    out = cat.transform(X)
    assert list(out.columns) == ["treatment__count", "treatment__coreness"]
    assert len(out) == len(X)


def test_common_category_more_typical_than_rare_one():
    X = _treatment_data()
    cat = CategoricalDepthTransformer(columns=["treatment"]).fit(X)
    out = cat.transform(X)
    assert out.loc[0, "treatment__coreness"] > out.loc[4, "treatment__coreness"]
    assert out.loc[0, "treatment__count"] == 4
    assert out.loc[4, "treatment__count"] == 1
    assert out.loc[0, "treatment__coreness"] == pytest.approx(4 / 6)


def test_get_feature_names_out_matches_transform_columns():
    X = _treatment_data()
    cat = CategoricalDepthTransformer(columns=["treatment"]).fit(X)
    assert list(cat.get_feature_names_out()) == list(cat.transform(X).columns)


def test_custom_group_name():
    X = _treatment_data()
    cat = CategoricalDepthTransformer(columns=["treatment"], group_name="tx").fit(X)
    out = cat.transform(X)
    assert list(out.columns) == ["tx__count", "tx__coreness"]


def test_columns_none_auto_detects_categorical_columns():
    X = pd.DataFrame({"region": ["a", "b", "a"], "amount": [1.0, 2.0, 3.0]})
    cat = CategoricalDepthTransformer().fit(X)
    assert cat.columns_ == ["region"]


def test_bool_column_is_accepted_unlike_depth_transformer():
    X = pd.DataFrame({"flag": [True, True, True, False], "amount": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError):
        DepthTransformer(columns=["flag", "amount"]).fit(X)

    cat = CategoricalDepthTransformer(columns=["flag"]).fit(X)
    out = cat.transform(X)
    assert out.loc[0, "flag__count"] == 3
    assert out.loc[3, "flag__count"] == 1


def test_int_coded_binary_column_is_accepted_explicitly():
    X = pd.DataFrame({"binary": [0, 0, 0, 1]})
    cat = CategoricalDepthTransformer(columns=["binary"]).fit(X)
    out = cat.transform(X)
    assert out.loc[0, "binary__count"] == 3
    assert out.loc[3, "binary__count"] == 1


def test_unseen_combination_at_transform_gets_zero():
    X = _treatment_data()
    cat = CategoricalDepthTransformer(columns=["treatment"]).fit(X)
    out = cat.transform(pd.DataFrame({"treatment": ["D"]}))
    assert out.loc[0, "treatment__count"] == 0
    assert out.loc[0, "treatment__coreness"] == 0.0


def test_missing_value_present_at_fit_gets_real_support():
    X = pd.DataFrame({"treatment": ["A", "A", None, None, None]})
    cat = CategoricalDepthTransformer(columns=["treatment"]).fit(X)
    out = cat.transform(pd.DataFrame({"treatment": [None]}))
    assert out.loc[0, "treatment__count"] == 3


def test_missing_value_absent_at_fit_is_unseen_not_missing():
    X = pd.DataFrame({"treatment": ["A", "A", "B"]})
    cat = CategoricalDepthTransformer(columns=["treatment"]).fit(X)
    out = cat.transform(pd.DataFrame({"treatment": [None]}))
    assert out.loc[0, "treatment__count"] == 0


def test_joint_combination_rarer_than_either_marginal():
    # "a" and "1" are each individually common, but the combination (a, 2)
    # and (b, 1) never co-occur -- the joint index must not just re-derive
    # a marginal count.
    X = pd.DataFrame(
        {
            "letter": ["a", "a", "a", "b", "b", "b"],
            "number": ["1", "1", "1", "2", "2", "2"],
        }
    )
    cat = CategoricalDepthTransformer(columns=["letter", "number"]).fit(X)
    out = cat.transform(pd.DataFrame({"letter": ["a", "b"], "number": ["2", "1"]}))
    assert out.loc[0, "letter_number__count"] == 0
    assert out.loc[1, "letter_number__count"] == 0

    out_common = cat.transform(pd.DataFrame({"letter": ["a"], "number": ["1"]}))
    assert out_common.loc[0, "letter_number__count"] == 3


def test_transform_before_fit_raises():
    cat = CategoricalDepthTransformer(columns=["treatment"])
    with pytest.raises(Exception):
        cat.transform(_treatment_data())


def test_unknown_column_in_columns_param_raises():
    X = _treatment_data()
    with pytest.raises(ValueError):
        CategoricalDepthTransformer(columns=["not_a_column"]).fit(X)


def test_no_categorical_columns_raises():
    X = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        CategoricalDepthTransformer().fit(X)


def test_missing_column_at_transform_raises():
    X = _treatment_data()
    cat = CategoricalDepthTransformer(columns=["treatment"]).fit(X)
    with pytest.raises(ValueError):
        cat.transform(X.drop(columns=["treatment"]))


def test_get_params_and_clone_work():
    cat = CategoricalDepthTransformer(columns=["treatment"], group_name="tx")
    params = cat.get_params()
    assert params["group_name"] == "tx"
    cloned = clone(cat)
    assert cloned.group_name == "tx"
    assert cloned is not cat


def test_feeds_into_depth_crowd_alongside_depth_transformer():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "viral_load": rng.normal(0, 1, n),
            "treatment": rng.choice(["A", "B", "C"], n, p=[0.7, 0.2, 0.1]),
        }
    )

    viral = DepthTransformer(columns=["viral_load"], group_name="viral").fit(X)
    tx = CategoricalDepthTransformer(columns=["treatment"], group_name="tx").fit(X)

    experts = pd.concat([viral.transform(X), tx.transform(X)], axis=1)
    crowd = DepthCrowd(columns=["viral__coreness", "tx__coreness"]).fit(experts)
    out = crowd.transform(experts)
    assert out.shape == (n, 7)
