import numpy as np
import pandas as pd

from zoneboost._weak_learner import _triple_deviation_confidence, weak_learner_fit


def _three_way_data(n=600, seed=0):
    # x1, x2, x3 carry real pairwise structure (so the pairwise-importance
    # prefilter has something to latch onto) *plus* a genuine 3-way term
    # that pairwise interactions alone cannot represent -- the realistic
    # case adaptive interaction order targets, not an adversarial one.
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
    )
    return X, y.to_numpy()


def test_triple_deviation_confidence_shapes_and_ranges():
    rng = np.random.default_rng(0)
    za = rng.integers(0, 4, 200)
    zb = rng.integers(0, 3, 200)
    zc = rng.integers(0, 5, 200)
    target = rng.normal(size=200)
    deviation, confidence = _triple_deviation_confidence(za, zb, zc, target, float(target.mean()), 4, 3, 5)
    assert deviation.shape == (4, 3, 5)
    assert confidence.shape == (4, 3, 5)
    assert (confidence >= 0).all() and (confidence <= 1).all()


def test_max_interaction_order_2_never_produces_triples():
    X, y = _three_way_data()
    residual = y - y.mean()
    _, _, _, triples, _ = weak_learner_fit(X, residual, list(X.columns), set(), max_interaction_order=2)
    assert triples == {}


def test_max_interaction_order_3_finds_the_genuine_triple():
    X, y = _three_way_data()
    residual = y - y.mean()
    _, _, _, triples, _ = weak_learner_fit(
        X, residual, list(X.columns), set(), max_interaction_order=3, triple_min_gain=0.01
    )
    assert len(triples) >= 1
    (key,) = triples.keys()
    assert set(key) == {"x1", "x2", "x3"}


def test_max_triple_interactions_caps_count_per_round():
    X, y = _three_way_data(n=800)
    rng = np.random.default_rng(1)
    X = X.copy()
    X["x4"] = rng.uniform(-3, 3, len(X))
    X["x5"] = rng.uniform(-3, 3, len(X))
    residual = y - y.mean()
    _, _, _, triples, _ = weak_learner_fit(
        X,
        residual,
        list(X.columns),
        set(),
        max_interaction_order=3,
        max_triple_interactions=1,
        triple_min_gain=0.01,
    )
    assert len(triples) <= 1


def test_high_triple_min_gain_rejects_weak_candidates():
    X, y = _three_way_data()
    residual = y - y.mean()
    _, _, _, triples, _ = weak_learner_fit(
        X, residual, list(X.columns), set(), max_interaction_order=3, triple_min_gain=1e6
    )
    assert triples == {}
