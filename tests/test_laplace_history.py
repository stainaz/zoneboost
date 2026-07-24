import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from zoneboost import LaplaceHistoryTransformer


def _events():
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 2],
            "claim_date": pd.to_datetime(["2026-01-01", "2026-06-01", "2026-05-01"]),
            "claim_amount": [1000.0, 2000.0, 500.0],
        }
    )


def _rows():
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 3],
            "snapshot_date": pd.to_datetime(["2026-03-01", "2026-07-01", "2026-07-01", "2026-07-01"]),
        }
    )


def _fitted():
    laplace = LaplaceHistoryTransformer(
        entity_col="customer_id",
        time_col="claim_date",
        value_columns=["claim_amount"],
        half_lives=[30, 365],
    )
    return laplace.fit(_rows(), event_history=_events())


def test_fit_indexes_events_per_entity():
    laplace = _fitted()
    assert set(laplace._entity_index.keys()) == {1, 2}
    assert list(laplace._entity_index[1]["times"]) == sorted(laplace._entity_index[1]["times"])


def test_transform_column_names():
    laplace = _fitted()
    out = laplace.transform(_rows(), asof_col="snapshot_date")
    assert list(out.columns) == [
        "claim_date__claim_amount__laplace_30d",
        "claim_date__claim_amount__laplace_365d",
        "claim_date__event_decay_count_30d",
        "claim_date__event_decay_count_365d",
        "claim_date__had_history",
    ]
    assert len(out) == len(_rows())


def test_get_feature_names_out_matches_transform_columns():
    laplace = _fitted()
    out = laplace.transform(_rows(), asof_col="snapshot_date")
    assert list(laplace.get_feature_names_out()) == list(out.columns)


def test_custom_group_name():
    laplace = LaplaceHistoryTransformer(
        entity_col="customer_id",
        time_col="claim_date",
        value_columns=["claim_amount"],
        half_lives=[30],
        group_name="claims",
    ).fit(_rows(), event_history=_events())
    out = laplace.transform(_rows(), asof_col="snapshot_date")
    assert list(out.columns) == [
        "claims__claim_amount__laplace_30d",
        "claims__event_decay_count_30d",
        "claims__had_history",
    ]


def test_leakage_row_before_event_does_not_see_it():
    # customer 1's second claim (June) must not leak into the March-asof row.
    laplace = _fitted()
    out = laplace.transform(_rows(), asof_col="snapshot_date")
    row0 = out.iloc[0]  # customer 1, asof 2026-03-01 -- only the Jan claim is visible
    row1 = out.iloc[1]  # customer 1, asof 2026-07-01 -- both claims are visible
    assert row0["claim_date__had_history"] == 1
    assert row0["claim_date__claim_amount__laplace_365d"] < row1["claim_date__claim_amount__laplace_365d"]
    # the June claim, fresher and larger, must not be reachable from the March row at all:
    # bound row0's value using only the Jan claim's own maximum possible (undecayed) weight.
    assert row0["claim_date__claim_amount__laplace_30d"] < 1000.0


def test_event_exactly_at_asof_is_included():
    events = pd.DataFrame(
        {
            "customer_id": [1],
            "claim_date": pd.to_datetime(["2026-03-01"]),
            "claim_amount": [1000.0],
        }
    )
    rows = pd.DataFrame({"customer_id": [1], "snapshot_date": pd.to_datetime(["2026-03-01"])})
    laplace = LaplaceHistoryTransformer(
        entity_col="customer_id", time_col="claim_date", value_columns=["claim_amount"], half_lives=[30]
    ).fit(rows, event_history=events)
    out = laplace.transform(rows, asof_col="snapshot_date")
    assert out.loc[0, "claim_date__claim_amount__laplace_30d"] == pytest.approx(1000.0)
    assert out.loc[0, "claim_date__had_history"] == 1


def test_unseen_entity_is_cold_start_not_error():
    laplace = _fitted()
    rows = pd.DataFrame({"customer_id": [999], "snapshot_date": pd.to_datetime(["2026-07-01"])})
    out = laplace.transform(rows, asof_col="snapshot_date")
    assert out.loc[0, "claim_date__had_history"] == 0
    assert out.loc[0, "claim_date__claim_amount__laplace_30d"] == 0.0
    assert out.loc[0, "claim_date__claim_amount__laplace_365d"] == 0.0


def test_transform_requires_exactly_one_of_asof_col_or_cutoff_date():
    laplace = _fitted()
    with pytest.raises(ValueError):
        laplace.transform(_rows())
    with pytest.raises(ValueError):
        laplace.transform(_rows(), asof_col="snapshot_date", cutoff_date="2026-07-01")


def test_cutoff_date_broadcasts_to_every_row():
    laplace = _fitted()
    rows = pd.DataFrame({"customer_id": [1, 2]})
    out = laplace.transform(rows, cutoff_date="2026-07-01")
    assert len(out) == 2
    assert (out["claim_date__had_history"] == 1).all()


def test_no_value_columns_still_emits_count_and_had_history():
    laplace = LaplaceHistoryTransformer(
        entity_col="customer_id", time_col="claim_date", half_lives=[30]
    ).fit(_rows(), event_history=_events())
    out = laplace.transform(_rows(), asof_col="snapshot_date")
    assert list(out.columns) == ["claim_date__event_decay_count_30d", "claim_date__had_history"]


def test_non_positive_half_life_raises():
    with pytest.raises(ValueError):
        LaplaceHistoryTransformer(
            entity_col="customer_id", time_col="claim_date", value_columns=["claim_amount"], half_lives=[0]
        ).fit(_rows(), event_history=_events())


def test_missing_event_history_raises():
    with pytest.raises(ValueError):
        LaplaceHistoryTransformer(
            entity_col="customer_id", time_col="claim_date", value_columns=["claim_amount"], half_lives=[30]
        ).fit(_rows())


def test_unknown_column_in_event_history_raises():
    with pytest.raises(ValueError):
        LaplaceHistoryTransformer(
            entity_col="customer_id", time_col="claim_date", value_columns=["not_a_column"], half_lives=[30]
        ).fit(_rows(), event_history=_events())


def test_non_numeric_value_column_raises():
    events = _events().assign(region=["a", "b", "c"])
    with pytest.raises(ValueError):
        LaplaceHistoryTransformer(
            entity_col="customer_id", time_col="claim_date", value_columns=["region"], half_lives=[30]
        ).fit(_rows(), event_history=events)


def test_transform_before_fit_raises():
    laplace = LaplaceHistoryTransformer(
        entity_col="customer_id", time_col="claim_date", value_columns=["claim_amount"], half_lives=[30]
    )
    with pytest.raises(Exception):
        laplace.transform(_rows(), asof_col="snapshot_date")


def test_get_params_and_clone_work():
    laplace = LaplaceHistoryTransformer(
        entity_col="customer_id", time_col="claim_date", value_columns=["claim_amount"], half_lives=[30, 90]
    )
    params = laplace.get_params()
    assert params["half_lives"] == [30, 90]
    cloned = clone(laplace)
    assert cloned.half_lives == [30, 90]
    assert cloned is not laplace


def test_output_feeds_directly_into_a_downstream_model():
    # transform() needs a per-call asof_col/cutoff_date, so it isn't dropped
    # into a Pipeline/ColumnTransformer automatically -- fit/transform are
    # called directly and the resulting columns are used as ordinary
    # features for a downstream model instead.
    laplace = LaplaceHistoryTransformer(
        entity_col="customer_id", time_col="claim_date", value_columns=["claim_amount"], half_lives=[30, 365]
    )
    rows = _rows()
    laplace.fit(rows, event_history=_events())
    features = laplace.transform(rows, asof_col="snapshot_date")
    target = np.arange(len(rows), dtype=float)
    model = LinearRegression().fit(features, target)
    preds = model.predict(features)
    assert preds.shape == (len(rows),)
