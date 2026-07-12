import json

import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor


def _data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "income": rng.uniform(20_000, 100_000, n),
            "age": rng.uniform(20, 70, n),
            "region": rng.choice(["north", "south", "east"], n),
        }
    )
    y = (0.0001 * X["income"] + 0.02 * X["age"] + rng.normal(0, 1, n)).to_numpy()
    return X, y


TOP_LEVEL_KEYS = {
    "zoneboost_version",
    "model_class",
    "reproducibility",
    "dataset_fingerprint",
    "fit_summary",
    "zones",
    "terms",
    "shrinkage",
    "constraints",
    "calibration",
    "unsupported_regions",
}


def test_top_level_keys_always_present():
    X, y = _data()
    model = ZoneBoostRegressor(random_state=0, n_rounds=20).fit(X, y)
    card = model.evidence_card()
    assert set(card.keys()) == TOP_LEVEL_KEYS


def test_dataset_fingerprint_and_mean_abs_contribution_require_x():
    X, y = _data()
    model = ZoneBoostRegressor(random_state=0, n_rounds=20).fit(X, y)

    card_no_x = model.evidence_card()
    assert card_no_x["dataset_fingerprint"] is None
    assert all(t["mean_abs_contribution"] is None for t in card_no_x["terms"].values())

    card_with_x = model.evidence_card(X)
    assert card_with_x["dataset_fingerprint"] is not None
    assert card_with_x["dataset_fingerprint"]["n_rows"] == len(X)
    assert card_with_x["dataset_fingerprint"]["n_columns"] == X.shape[1]
    assert set(card_with_x["dataset_fingerprint"]["columns"]) == set(X.columns)
    assert all(t["mean_abs_contribution"] is not None for t in card_with_x["terms"].values())


def test_reliability_fields_require_track_reliability():
    X, y = _data()
    model_off = ZoneBoostRegressor(random_state=0, n_rounds=20).fit(X, y)
    card_off = model_off.evidence_card()
    assert all(t["mean_support_per_zone"] is None for t in card_off["terms"].values())
    assert all(t["mean_shrinkage_fraction"] is None for t in card_off["terms"].values())
    assert card_off["shrinkage"]["track_reliability_enabled"] is False

    model_on = ZoneBoostRegressor(random_state=0, n_rounds=20, track_reliability=True).fit(X, y)
    card_on = model_on.evidence_card()
    assert all(t["mean_support_per_zone"] is not None for t in card_on["terms"].values())
    assert all(t["mean_shrinkage_fraction"] is not None for t in card_on["terms"].values())
    assert card_on["shrinkage"]["track_reliability_enabled"] is True


def test_sparse_term_shows_higher_shrinkage_fraction():
    # A tiny "small" hospital's income main effect should show a higher
    # mean_shrinkage_fraction than the two large hospitals -- the same
    # hierarchical-shrinkage direction test_hierarchical.py already
    # verifies at the interaction level, here surfaced through the card.
    rng = np.random.default_rng(0)
    income_a = rng.uniform(0, 100, 800)
    y_a = 0.01 * income_a + 4.0 + rng.normal(0, 0.3, 800)
    income_small = rng.uniform(0, 100, 12)
    y_small = 0.01 * income_small - 4.0 + rng.normal(0, 0.3, 12)
    X = pd.DataFrame(
        {
            "income": np.concatenate([income_a, income_small]),
            "hospital": ["big"] * 800 + ["small"] * 12,
        }
    )
    y = np.concatenate([y_a, y_small])
    model = ZoneBoostRegressor(random_state=0, group_col="hospital", track_reliability=True, n_rounds=60).fit(X, y)
    card = model.evidence_card()
    assert card["terms"]["hospital"]["mean_shrinkage_fraction"] is not None


def test_zones_reflect_column_kind_and_observed_range():
    X, y = _data()
    model = ZoneBoostRegressor(random_state=0, n_rounds=30).fit(X, y)
    card = model.evidence_card()

    assert card["zones"]["income"]["kind"] == "continuous"
    assert card["zones"]["income"]["observed_range"] == list(model._observed_range("income"))

    assert card["zones"]["region"]["kind"] == "categorical"
    assert set(card["zones"]["region"]["categories_seen"]) == set(X["region"].unique())


def test_constraints_shrinkage_calibration_reflect_fitted_attributes():
    X, y = _data()
    model = ZoneBoostRegressor(
        random_state=0,
        n_rounds=30,
        monotonic_constraints={"age": 1},
        forbidden_interactions=[("age", "region")],
        group_col="region",
        loss="quantile",
        quantile=0.3,
    ).fit(X, y)
    card = model.evidence_card()

    assert card["constraints"]["monotonic_constraints"] == {"age": 1}
    assert card["constraints"]["forbidden_interactions"] == [["age", "region"]]
    assert card["constraints"]["group_col"] == "region"
    assert card["calibration"]["loss"] == "quantile"
    assert card["calibration"]["quantile"] == 0.3
    assert "age x region" not in card["terms"]


def test_unsupported_regions_only_continuous_main_effects():
    X, y = _data()
    model = ZoneBoostRegressor(random_state=0, n_rounds=30).fit(X, y)
    card = model.evidence_card()
    assert set(card["unsupported_regions"].keys()) == {"income", "age"}
    assert "region" not in card["unsupported_regions"]


def test_evidence_card_round_trips_through_json():
    X, y = _data()
    model = ZoneBoostRegressor(
        random_state=0,
        n_rounds=30,
        track_reliability=True,
        monotonic_constraints={"age": 1},
        group_col="region",
        loss="quantile",
    ).fit(X, y)
    card = model.evidence_card(X)
    s = json.dumps(card)
    reloaded = json.loads(s)
    assert reloaded["model_class"] == "ZoneBoostRegressor"
