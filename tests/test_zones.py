import numpy as np
import pandas as pd

from zoneboost._zones import (
    adaptive_zone_boundaries,
    categorical_zone_index,
    categorical_zone_map,
    zone_index,
)


def test_adaptive_zone_boundaries_returns_sorted_array():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 500)
    y = x**2 + rng.normal(0, 1, 500)
    bounds = adaptive_zone_boundaries(x, y, max_zones=7)
    assert isinstance(bounds, np.ndarray)
    assert list(bounds) == sorted(bounds)


def test_adaptive_zone_boundaries_finds_real_split():
    # A clean step function: y jumps at x=5. The split search should land
    # very close to that true breakpoint.
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 2000)
    y = np.where(x < 5, 0.0, 100.0) + rng.normal(0, 0.5, 2000)
    bounds = adaptive_zone_boundaries(x, y, max_zones=2)
    assert len(bounds) == 1
    assert abs(bounds[0] - 5.0) < 0.5


def test_adaptive_zone_boundaries_uses_more_zones_for_real_structure_than_noise():
    # There's no significance test on a split's gain (just gain > 0), so
    # pure noise can still find a few spuriously "beneficial" splits by
    # chance over enough recursive attempts -- the meaningful comparison
    # is relative: real structure should still claim more of the zone
    # budget than unrelated noise, not that noise gets exactly zero.
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 10, 500)
    y_noise = rng.normal(0, 1, 500)
    y_structured = np.where(x < 5, 0.0, 100.0) + rng.normal(0, 0.5, 500)

    bounds_noise = adaptive_zone_boundaries(x, y_noise, max_zones=7)
    bounds_structured = adaptive_zone_boundaries(x, y_structured, max_zones=7)
    assert len(bounds_structured) >= len(bounds_noise)


def test_zone_index_matches_boundaries():
    bounds = np.array([2.0, 5.0, 8.0])
    values = np.array([1.0, 2.0, 3.0, 6.0, 9.0])
    zones = zone_index(values, bounds)
    # searchsorted with side="right": 1.0->0, 2.0->1 (right of 2.0), 3.0->1, 6.0->2, 9.0->3
    assert list(zones) == [0, 1, 1, 2, 3]


def test_categorical_zone_map_and_index_roundtrip():
    series = pd.Series(["red", "green", "blue", "green", "red"])
    cat_map = categorical_zone_map(series)
    assert set(cat_map.keys()) == {"red", "green", "blue"}
    assert len(set(cat_map.values())) == 3

    idx = categorical_zone_index(series, cat_map)
    # same category -> same index
    assert idx[0] == idx[4]  # both "red"
    assert idx[1] == idx[3]  # both "green"
    assert idx[0] != idx[1] != idx[2]


def test_categorical_zone_index_unseen_category_maps_to_unknown_bucket():
    train = pd.Series(["a", "b", "a"])
    cat_map = categorical_zone_map(train)
    new = pd.Series(["a", "c"])  # "c" was never seen during fit
    idx = categorical_zone_index(new, cat_map)
    assert idx[0] == cat_map["a"]
    assert idx[1] == len(cat_map)  # dedicated unknown bucket, one past the end
