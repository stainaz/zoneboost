import numpy as np
import pandas as pd

from zoneboost import ZoneBoostClassifier, ZoneBoostRegressor


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-5, 5, n),
            "x2": rng.uniform(-5, 5, n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    return X, rng


def test_regressor_explain_sums_exactly_to_predict():
    X, rng = _data()
    y = X["x1"] ** 2 + (X["cat"] == "b").astype(float) * 10 + rng.normal(0, 1, len(X))
    model = ZoneBoostRegressor(n_rounds=40, categorical_features=["cat"], random_state=0).fit(X, y)

    pred = model.predict(X)
    contrib = model.explain(X)
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy(), pred, atol=1e-6)


def test_regressor_explain_has_no_duplicate_interaction_terms():
    # A pair's fit order can vary round to round (col_subsample reorders
    # columns), so without canonicalizing the key, "A x B" and "B x A"
    # would fragment into separate columns for the same underlying pair.
    X, rng = _data()
    y = X["x1"] * X["x2"] + rng.normal(0, 1, len(X))
    model = ZoneBoostRegressor(n_rounds=40, categorical_features=["cat"], random_state=0).fit(X, y)
    contrib = model.explain(X)
    terms = [c for c in contrib.columns if c != "baseline"]
    assert len(terms) == len(set(terms))
    # exactly one column for the x1/x2 pair, not two
    x1_x2_cols = [t for t in terms if "x1" in t and "x2" in t and "cat" not in t]
    assert len(x1_x2_cols) == 1


def test_regressor_feature_importance_sorted_descending():
    X, rng = _data()
    y = X["x1"] ** 2 + rng.normal(0, 0.1, len(X))  # x1 should dominate
    model = ZoneBoostRegressor(n_rounds=40, categorical_features=["cat"], random_state=0).fit(X, y)
    importance = model.feature_importance(X)
    assert list(importance.values) == sorted(importance.values, reverse=True)
    assert importance.index[0] == "x1"


def test_classifier_binary_explain_sums_exactly_to_log_odds():
    X, rng = _data()
    y = ((X["x1"] + (X["cat"] == "b").astype(float) * 3) > 0).astype(int)
    model = ZoneBoostClassifier(n_rounds=30, categorical_features=["cat"], random_state=0).fit(X, y)

    contrib = model.explain(X)
    log_odds = contrib.sum(axis=1).to_numpy()
    proba_from_explain = 1 / (1 + np.exp(-log_odds))
    proba_from_predict = model.predict_proba(X)[:, 1]
    np.testing.assert_allclose(proba_from_explain, proba_from_predict, atol=1e-6)


def test_classifier_multiclass_explain_returns_dict_per_class():
    # Native multinomial boosting: each class's explain() DataFrame (which
    # includes the "_softmax_centering" identifiability column) sums to that
    # class's own joint-softmax score, so softmax-ing all K classes' sums
    # together reproduces predict_proba(X) exactly (calibrate=False, the
    # default -- see the explain() docstring).
    X, rng = _data()
    y = pd.cut(X["x1"], bins=3, labels=[0, 1, 2]).astype(int)
    model = ZoneBoostClassifier(n_rounds=30, categorical_features=["cat"], random_state=0).fit(X, y)

    explanation = model.explain(X)
    assert set(explanation.keys()) == set(model.classes_)
    scores = np.column_stack([explanation[k].sum(axis=1).to_numpy() for k in model.classes_])
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    proba_from_explain = exp / exp.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(proba_from_explain, model.predict_proba(X), atol=1e-6)


def test_classifier_feature_importance_multiclass_is_averaged():
    X, rng = _data()
    y = pd.cut(X["x1"], bins=3, labels=[0, 1, 2]).astype(int)
    model = ZoneBoostClassifier(n_rounds=30, categorical_features=["cat"], random_state=0).fit(X, y)
    importance = model.feature_importance(X)
    assert "baseline" not in importance.index
    assert (importance >= 0).all()


def _three_way_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-3, 3, n),
            "x2": rng.uniform(-3, 3, n),
            "x3": rng.uniform(-3, 3, n),
        }
    )
    y = (
        X["x1"] * X["x2"]
        + X["x1"] * X["x3"]
        + X["x2"] * X["x3"]
        + 2.0 * X["x1"] * X["x2"] * X["x3"]
        + rng.normal(0, 0.5, n)
    ).to_numpy()
    return X, y


def test_explain_with_triples_sums_exactly_to_predict_and_has_one_triple_column():
    X, y = _three_way_data()
    model = ZoneBoostRegressor(
        n_rounds=60, random_state=0, col_subsample=1.0, max_interaction_order=3
    ).fit(X, y)
    assert any(len(round_["triples"]) > 0 for round_ in model.rounds_)

    pred = model.predict(X)
    contrib = model.explain(X)
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy(), pred, atol=1e-6)

    terms = [c for c in contrib.columns if c != "baseline"]
    assert len(terms) == len(set(terms))
    triple_cols = [t for t in terms if "x1" in t and "x2" in t and "x3" in t]
    assert len(triple_cols) == 1
    assert triple_cols[0] == "x1 x x2 x x3"
