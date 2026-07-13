import json
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor, compile_to_sql


def _run_sql(sql, X):
    conn = sqlite3.connect(":memory:")
    try:
        X.to_sql("input_table", conn, index=False)
        cur = conn.execute(sql)
        return np.array([row[0] for row in cur.fetchall()])
    finally:
        conn.close()


def test_executable_lossless_main_effects_and_pairs():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({"x": rng.uniform(0, 10, n), "cat": rng.choice(["a", "b", "c"], n)})
    y = X["x"] + np.where(X["cat"] == "a", 2.0, 0.0) + rng.normal(scale=0.3, size=n)

    model = ZoneBoostRegressor(n_rounds=10, categorical_features=["cat"], random_state=0).fit(X, y)
    sql = compile_to_sql(model)

    pred = model.predict(X)
    sql_scores = _run_sql(sql, X)

    np.testing.assert_allclose(sql_scores, pred, atol=1e-9)


def test_executable_lossless_missing_and_unseen_category():
    rng = np.random.default_rng(1)
    n = 500
    X = pd.DataFrame({"x": rng.uniform(0, 10, n), "cat": rng.choice(["a", "b", "c"], n)})
    y = X["x"] + np.where(X["cat"] == "a", 2.0, 0.0) + rng.normal(scale=0.3, size=n)
    model = ZoneBoostRegressor(n_rounds=10, categorical_features=["cat"], random_state=0).fit(X, y)

    X_test = pd.DataFrame(
        {
            "x": [1.0, np.nan, 5.0, 9.0],
            "cat": ["a", "b", "zzz_unseen", np.nan],
        }
    )
    pred = model.predict(X_test)
    sql = compile_to_sql(model)
    sql_scores = _run_sql(sql, X_test)

    np.testing.assert_allclose(sql_scores, pred, atol=1e-9)


@pytest.mark.parametrize("loss", ["poisson", "gamma", "tweedie"])
def test_executable_lossless_glm_losses(loss):
    rng = np.random.default_rng(2)
    n = 800
    X = pd.DataFrame({"age": rng.uniform(18, 70, n)})
    exposure = rng.uniform(0.2, 1.0, n)
    if loss == "gamma":
        y = rng.gamma(shape=2.0, scale=np.exp(1.0 + 0.01 * X["age"]))
        offset = np.zeros(n)
        offset_expr = "0"
    else:
        rate = np.exp(-3.0 + 0.02 * X["age"])
        y = rng.poisson(rate * exposure).astype(float)
        offset = np.log(exposure)
        offset_expr = "LN(exposure)"

    model = ZoneBoostRegressor(loss=loss, n_rounds=15, random_state=0).fit(X, y, offset=offset)
    pred = model.predict(X, offset=offset)

    sql = compile_to_sql(model, offset_expr=offset_expr)
    X_sql = X.copy()
    X_sql["exposure"] = exposure
    sql_scores = _run_sql(sql, X_sql)

    np.testing.assert_allclose(sql_scores, pred, rtol=1e-9, atol=1e-9)


def test_executable_lossless_quantile_loss():
    rng = np.random.default_rng(3)
    n = 500
    X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
    y = X["x"] + rng.normal(scale=0.5 + 0.1 * X["x"], size=n)

    model = ZoneBoostRegressor(loss="quantile", quantile=0.9, n_rounds=10, random_state=0).fit(X, y)
    pred = model.predict(X)
    sql = compile_to_sql(model)
    sql_scores = _run_sql(sql, X)

    np.testing.assert_allclose(sql_scores, pred, atol=1e-9)


def test_triples_raises_value_error():
    rng = np.random.default_rng(4)
    n = 300
    X = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n)})
    y = X["x0"] + rng.normal(scale=0.3, size=n)
    model = ZoneBoostRegressor(n_rounds=5, random_state=0).fit(X, y)
    model.rounds_[0]["triples"] = {("x0", "x1", "x0"): np.zeros((2, 2, 2))}
    with pytest.raises(ValueError, match="3-way interactions"):
        compile_to_sql(model)


def test_effect_overrides_raises_value_error():
    rng = np.random.default_rng(5)
    n = 300
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = X["x"] + rng.normal(scale=0.3, size=n)
    model = ZoneBoostRegressor(n_rounds=5, random_state=0).fit(X, y)
    model.effect_overrides_ = [{"term": "x", "values": np.zeros(3)}]
    with pytest.raises(ValueError, match="effect_overrides_"):
        compile_to_sql(model)


def test_bad_dialect_raises_value_error():
    rng = np.random.default_rng(6)
    n = 200
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = X["x"] + rng.normal(scale=0.3, size=n)
    model = ZoneBoostRegressor(n_rounds=5, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="dialect"):
        compile_to_sql(model, dialect="postgres")


def test_include_evidence_card_embeds_parseable_json():
    rng = np.random.default_rng(7)
    n = 300
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = X["x"] + rng.normal(scale=0.3, size=n)
    model = ZoneBoostRegressor(n_rounds=5, random_state=0).fit(X, y)

    sql = compile_to_sql(model, include_evidence_card=True)
    match = re.search(r"/\*\nzoneboost evidence card:\n(.*?)\n\*/", sql, re.DOTALL)
    assert match is not None
    card = json.loads(match.group(1))
    assert "zoneboost_version" in card
    assert "model_class" in card

    # still executes correctly with the comment header present
    pred = model.predict(X)
    sql_scores = _run_sql(sql, X)
    np.testing.assert_allclose(sql_scores, pred, atol=1e-9)
