import numpy as np
import pandas as pd

from zoneboost._zones import (
    adaptive_zone_boundaries,
    categorical_zone_index,
    categorical_zone_map,
    zone_centers,
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


def test_adaptive_zone_boundaries_excludes_missing_values_from_split_search():
    # A NaN sorts to an arbitrary position and would otherwise corrupt
    # whichever segment's cumulative sums (and cut *value*) it lands in --
    # missing rows must never enter the search at all.
    rng = np.random.default_rng(3)
    x = rng.uniform(-5, 5, 300)
    y = x**2 + rng.normal(0, 1, 300)
    x_missing = x.copy()
    x_missing[rng.choice(300, size=20, replace=False)] = np.nan

    bounds = adaptive_zone_boundaries(x_missing, y, max_zones=7)
    assert not np.isnan(bounds).any()


def test_zone_index_missing_value_maps_to_dedicated_zone():
    bounds = np.array([2.0, 5.0, 8.0])
    values = np.array([1.0, 6.0, np.nan])
    zones = zone_index(values, bounds)
    assert zones[2] == len(bounds) + 1  # dedicated missing zone, one past the regular ones
    assert zones[0] != zones[2] and zones[1] != zones[2]


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
    # Two distinct fallback buckets past the regular categories: missing
    # (index len(cat_map)) and unseen-but-real-category (index
    # len(cat_map) + 1) -- kept separate rather than merged.
    train = pd.Series(["a", "b", "a"])
    cat_map = categorical_zone_map(train)
    new = pd.Series(["a", "c"])  # "c" was never seen during fit
    idx = categorical_zone_index(new, cat_map)
    assert idx[0] == cat_map["a"]
    assert idx[1] == len(cat_map) + 1  # dedicated "unseen category" bucket


def test_categorical_zone_index_missing_value_maps_to_own_bucket():
    train = pd.Series(["a", "b", "a"])
    cat_map = categorical_zone_map(train)
    new = pd.Series(["a", np.nan, "c"])  # nan (missing) vs "c" (unseen) must differ
    idx = categorical_zone_index(new, cat_map)
    assert idx[0] == cat_map["a"]
    assert idx[1] == len(cat_map)      # dedicated "missing" bucket
    assert idx[2] == len(cat_map) + 1  # dedicated "unseen category" bucket, distinct from missing


def test_categorical_zone_map_excludes_missing_values():
    series = pd.Series(["a", "b", np.nan, "a"])
    cat_map = categorical_zone_map(series)
    assert set(cat_map.keys()) == {"a", "b"}
    assert len(cat_map) == 2


def test_zone_centers_matches_hand_computed_means():
    bounds = np.array([5.0])  # 2 real zones: [<=5], [>5]
    x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0, np.nan])
    centers = zone_centers(x, bounds)
    np.testing.assert_allclose(centers, [2.0, 11.0])


def test_zone_centers_empty_zone_falls_back_to_boundary_midpoint_without_crashing():
    bounds = np.array([5.0, 100.0])  # zone 1 (between 5 and 100) will be empty
    x = np.array([1.0, 2.0, 3.0, 200.0, 201.0])
    centers = zone_centers(x, bounds)
    assert np.all(np.isfinite(centers))
    assert centers[0] < 5.0  # zone 0's real mean
    assert 5.0 < centers[1] < 100.0  # empty zone's fallback midpoint
    assert centers[2] > 100.0  # zone 2's real mean
