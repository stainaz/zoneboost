import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostRegressor
from zoneboost._weak_learner import _overall_stat, _trimmed_mean, _zone_raw_stat


def test_trimmed_mean_drops_correct_count_per_tail():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    # k = floor(0.2 * 6) = 1 -> drop 1 from each tail -> mean([2,3,4,5])
    assert _trimmed_mean(vals, 0.2) == pytest.approx(3.5)


def test_trimmed_mean_zero_fraction_matches_plain_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(size=50)
    assert _trimmed_mean(vals, 0.0) == pytest.approx(vals.mean())


def test_trimmed_mean_degenerate_small_array_falls_back_to_plain_mean():
    vals = np.array([1.0, 2.0])
    # trim_fraction=0.4 -> k=floor(0.8)=0, no trimming happens
    assert _trimmed_mean(vals, 0.4) == pytest.approx(1.5)
    # a fraction large enough that 2*k >= len also falls back
    assert _trimmed_mean(np.array([1.0, 2.0, 3.0]), 0.49) == pytest.approx(np.mean([1.0, 2.0, 3.0]))


def test_zone_raw_stat_trimmed_matches_hand_computation():
    zones = np.array([0, 0, 0, 1, 1, 1])
    targets = np.array([1.0, 2.0, 100.0, 5.0, 6.0, 7.0])
    stat, counts = _zone_raw_stat(zones, targets, 2, trim_fraction=0.34)
    np.testing.assert_allclose(stat, [2.0, 6.0])
    np.testing.assert_allclose(counts, [3.0, 3.0])

    stat_mean, _ = _zone_raw_stat(zones, targets, 2)
    np.testing.assert_allclose(stat_mean, [103.0 / 3, 6.0])


def test_overall_stat_trimmed_matches_hand_computation():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    assert _overall_stat(values, trim_fraction=0.2) == pytest.approx(3.5)
    assert _overall_stat(values) == pytest.approx(values.mean())


def _contaminated_linear_data(n=2000, seed=0, n_outliers=5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    y = 2 * x + rng.normal(scale=0.5, size=n)
    outlier_mask = (x >= 4) & (x < 5)
    outlier_idx = rng.choice(np.where(outlier_mask)[0], size=n_outliers, replace=False)
    y_contam = y.copy()
    y_contam[outlier_idx] += 200
    X = pd.DataFrame({"x": x})
    return X, y_contam


def test_outlier_robustness_measured_directly():
    X, y_contam = _contaminated_linear_data()
    m0 = ZoneBoostRegressor(n_rounds=50, random_state=0, trim_fraction=0.0).fit(X, y_contam)
    m1 = ZoneBoostRegressor(n_rounds=50, random_state=0, trim_fraction=0.2).fit(X, y_contam)

    true_val = 2 * 4.5
    X_test = pd.DataFrame({"x": [4.5]})
    pred0 = m0.predict(X_test)[0]
    pred1 = m1.predict(X_test)[0]

    assert abs(pred1 - true_val) < abs(pred0 - true_val)
    assert abs(pred1 - true_val) < 2.0


def test_loss_quantile_with_trim_fraction_raises():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y = [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="loss='quantile'"):
        ZoneBoostRegressor(loss="quantile", trim_fraction=0.1).fit(X, y)


def test_trim_fraction_out_of_range_raises():
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y = [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match=r"\[0, 0.5\)"):
        ZoneBoostRegressor(trim_fraction=0.5).fit(X, y)
    with pytest.raises(ValueError, match=r"\[0, 0.5\)"):
        ZoneBoostRegressor(trim_fraction=-0.1).fit(X, y)


def test_default_trim_fraction_bit_identical_to_plain_mean():
    rng = np.random.default_rng(1)
    n = 1000
    X = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.normal(size=n)})
    y = X["x0"] + 0.5 * X["x1"] + rng.normal(scale=0.5, size=n)

    m_default = ZoneBoostRegressor(n_rounds=20, random_state=0).fit(X, y)
    m_explicit_zero = ZoneBoostRegressor(n_rounds=20, random_state=0, trim_fraction=0.0).fit(X, y)
    np.testing.assert_array_equal(m_default.predict(X), m_explicit_zero.predict(X))


def test_glm_losses_fit_with_trim_fraction_baseline_unaffected():
    rng = np.random.default_rng(2)
    n = 1000
    X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
    claims = rng.poisson(np.exp(0.1 * X["x"])).astype(float)

    m0 = ZoneBoostRegressor(loss="poisson", n_rounds=20, random_state=0, trim_fraction=0.0).fit(X, claims)
    m1 = ZoneBoostRegressor(loss="poisson", n_rounds=20, random_state=0, trim_fraction=0.1).fit(X, claims)

    assert m0.baseline_ == pytest.approx(m1.baseline_)
    assert np.all(np.isfinite(m1.predict(X)))


def test_pair_and_triple_functions_have_no_trim_fraction_parameter():
    # Disclosed scope, verified structurally rather than via a noisy
    # end-to-end prediction comparison: pairs/triples genuinely cannot
    # receive trim_fraction -- their own shrunk-deviation/fitting functions
    # don't accept it at all, so a pair-level outlier cluster is fit with
    # an ordinary mean regardless of the main-effects-only trim_fraction.
    import inspect

    from zoneboost._weak_learner import _fit_pairs, _pair_shrunk_deviation, _select_triples, _triple_shrunk_deviation

    for fn in (_pair_shrunk_deviation, _triple_shrunk_deviation, _fit_pairs, _select_triples):
        assert "trim_fraction" not in inspect.signature(fn).parameters, fn.__name__


def test_trim_fraction_composes_with_monotonic_and_bounded_effects():
    rng = np.random.default_rng(5)
    n = 500
    X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
    y = 2 * X["x"] + rng.normal(scale=0.5, size=n)

    model = ZoneBoostRegressor(
        n_rounds=20,
        random_state=0,
        trim_fraction=0.1,
        monotonic_constraints={"x": 1},
        bounded_effects={"x": (-50.0, 50.0)},
    ).fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))

    model_strict = ZoneBoostRegressor(
        n_rounds=20,
        random_state=0,
        trim_fraction=0.1,
        monotonic_constraints={"x": 1},
        strict_shape_constraints=True,
    ).fit(X, y)
    assert np.all(np.isfinite(model_strict.predict(X)))
