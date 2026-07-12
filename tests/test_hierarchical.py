import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor
from zoneboost._common import resolve_group_col
from zoneboost._weak_learner import _column_zone_index, _overall_stat, _pair_shrunk_deviation, weak_learner_fit


def _hospital_data(n=3000, n_hospitals=15, seed=0):
    rng = np.random.default_rng(seed)
    hospital = rng.integers(0, n_hospitals, n)
    income = rng.uniform(20_000, 100_000, n)
    hospital_offsets = rng.normal(0, 3.0, n_hospitals)
    y = 0.0001 * income + hospital_offsets[hospital] + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"income": income, "hospital": [f"h{h}" for h in hospital]})
    return X, y, hospital_offsets


def test_group_col_survives_column_subsampling():
    # col_subsample=0.5 with several columns would, left to chance, drop
    # "hospital" from plenty of rounds -- it must never be dropped once
    # group_col is set.
    n = 2000
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.uniform(-1, 1, n),
            "b": rng.uniform(-1, 1, n),
            "c": rng.uniform(-1, 1, n),
            "d": rng.uniform(-1, 1, n),
            "hospital": rng.choice(["h0", "h1", "h2"], n),
        }
    )
    y = X["a"].to_numpy() + rng.normal(0, 0.1, n)

    model = ZoneBoostRegressor(
        random_state=0, group_col="hospital", col_subsample=0.5, n_rounds=50
    ).fit(X, y)

    missing = [i for i, r in enumerate(model.rounds_) if "hospital" not in r["main_effects"]]
    assert missing == [], f"hospital dropped in rounds {missing}"


def test_group_col_pair_survives_screening():
    # hospital is irrelevant to y and would score weakly against the
    # genuinely strong x1*x2/x2*x3/x1*x3 structure -- max_pair_interactions
    # would screen it out for every feature without the forcing fix.
    n = 3000
    rng = np.random.default_rng(0)
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    x3 = rng.uniform(-3, 3, n)
    hospital = rng.integers(0, 10, n)
    y = x1 * x2 + x2 * x3 + x1 * x3 + rng.normal(0, 0.3, n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "hospital": [f"h{h}" for h in hospital]})

    model = ZoneBoostRegressor(
        random_state=0, group_col="hospital", max_pair_interactions=2,
        col_subsample=1.0, n_rounds=30,
    ).fit(X, y)

    for feat in ("x1", "x2", "x3"):
        present = sum(
            1 for r in model.rounds_
            if (feat, "hospital") in r["interactions"] or ("hospital", feat) in r["interactions"]
        )
        assert present == len(model.rounds_), f"{feat} x hospital missing in some rounds"


def test_group_col_forbidden_interaction_still_wins():
    n = 2000
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "income": rng.uniform(20_000, 100_000, n),
            "hospital": rng.choice(["h0", "h1", "h2"], n),
        }
    )
    y = 0.0001 * X["income"].to_numpy() + rng.normal(0, 1.0, n)

    model = ZoneBoostRegressor(
        random_state=0,
        group_col="hospital",
        forbidden_interactions=[("income", "hospital")],
        n_rounds=20,
    ).fit(X, y)

    for r in model.rounds_:
        assert ("income", "hospital") not in r["interactions"]
        assert ("hospital", "income") not in r["interactions"]


def test_group_col_none_is_bit_identical_to_prior_release():
    X, y, _ = _hospital_data()
    model_default = ZoneBoostRegressor(random_state=0, n_rounds=30).fit(X, y)
    model_explicit_none = ZoneBoostRegressor(random_state=0, n_rounds=30, group_col=None).fit(X, y)
    assert np.array_equal(model_default.predict(X), model_explicit_none.predict(X))


def test_group_col_forced_pair_uses_identical_shrinkage_math():
    # A group_col-forced pair must be computed by the *exact same*
    # _pair_shrunk_deviation call (same backfitted residual, same m) any
    # ordinarily-discovered pair would use -- confirming group_col is new
    # wiring around existing math, not a parallel/different code path.
    # The shrinkage-magnitude *property* itself (sparse cells pulled
    # toward the marginal prior) is already unit-tested directly at
    # test_weak_learner.py::test_pair_shrunk_deviation_sparse_cell_is_pulled_toward_marginal_prior.
    rng = np.random.default_rng(0)
    n = 500
    income = rng.uniform(0, 100, n)
    hospital = rng.choice(["big", "small"], n, p=[0.9, 0.1])
    bump = np.where((income > 60) & (hospital == "small"), 8.0, 0.0)
    residual = 0.01 * income + bump + rng.normal(0, 0.2, n)
    X = pd.DataFrame({"income": income, "hospital": hospital})

    zone_info, main_effects, interactions, _, _, _ = weak_learner_fit(
        X, residual, ["income", "hospital"], {"hospital"}, np.random.default_rng(1),
        shrinkage_m=10.0, group_col="hospital",
    )
    pair_key = ("income", "hospital") if ("income", "hospital") in interactions else ("hospital", "income")
    assert pair_key in interactions
    a, b = pair_key

    za = _column_zone_index(X[a], zone_info[a])
    zb = _column_zone_index(X[b], zone_info[b])
    n_a = len(main_effects[a])
    n_b = len(main_effects[b])
    partial = residual - main_effects[a][za] - main_effects[b][zb]
    expected = _pair_shrunk_deviation(za, zb, partial, _overall_stat(partial), n_a, n_b, m=10.0)

    np.testing.assert_allclose(interactions[pair_key], expected)


def test_resolve_group_col_name_and_index():
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert resolve_group_col(X, None) is None
    assert resolve_group_col(X, "b") == "b"
    assert resolve_group_col(X, 1) == "b"


def test_resolve_group_col_unknown_raises():
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        resolve_group_col(X, "nonexistent")
