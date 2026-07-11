import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostClassifier


def _binary_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-5, 5, n),
            "x2": rng.uniform(-5, 5, n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    bump = np.where(X["cat"] == "b", 3.0, 0.0)
    y = ((X["x1"] + bump) > 0).astype(int).to_numpy()
    return X, y


def _multiclass_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-5, 5, n),
            "x2": rng.uniform(-5, 5, n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    y = pd.cut(X["x1"], bins=3, labels=[0, 1, 2]).astype(int).to_numpy()
    return X, y


def test_binary_fit_predict():
    X, y = _binary_data()
    model = ZoneBoostClassifier(n_rounds=30, categorical_features=["cat"], random_state=0)
    model.fit(X, y)
    assert list(model.classes_) == [0, 1]
    assert model.multiclass_ is False

    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    pred = model.predict(X)
    assert pred.shape == (len(y),)
    assert model.score(X, y) > 0.8  # accuracy, via ClassifierMixin


def test_multiclass_fit_predict():
    X, y = _multiclass_data()
    model = ZoneBoostClassifier(n_rounds=30, categorical_features=["cat"], random_state=0)
    model.fit(X, y)
    assert list(model.classes_) == [0, 1, 2]
    assert model.multiclass_ is True

    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    pred = model.predict(X)
    assert set(np.unique(pred)) <= {0, 1, 2}
    assert model.score(X, y) > 0.7


def test_predict_before_fit_raises():
    model = ZoneBoostClassifier(n_rounds=10)
    with pytest.raises(Exception):
        model.predict(pd.DataFrame({"x": [1, 2, 3]}))


def test_single_class_raises():
    X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
    y = [0, 0, 0]
    with pytest.raises(ValueError):
        ZoneBoostClassifier(n_rounds=5).fit(X, y)


def test_categorical_auto_detection_from_dtype():
    X, y = _binary_data()
    model = ZoneBoostClassifier(n_rounds=20, random_state=0)
    model.fit(X, y)
    assert "cat" in model.categorical_features_


def test_unseen_category_at_predict_time_does_not_crash():
    X, y = _binary_data()
    model = ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=0)
    model.fit(X, y)
    X_new = X.copy()
    X_new.loc[0, "cat"] = "never_seen_before"
    proba = model.predict_proba(X_new)
    assert np.all(np.isfinite(proba))


def test_reproducible_with_same_random_state():
    X, y = _binary_data()
    model_a = ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=7).fit(X, y)
    model_b = ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=7).fit(X, y)
    np.testing.assert_array_equal(model_a.predict(X), model_b.predict(X))


def test_accepts_numpy_array_input():
    X, y = _binary_data()
    X_arr = X[["x1", "x2"]].to_numpy()
    model = ZoneBoostClassifier(n_rounds=20, random_state=0)
    model.fit(X_arr, y)
    pred = model.predict(X_arr)
    assert pred.shape == (len(y),)


def test_string_labels_supported():
    X, y = _binary_data()
    y_str = np.where(y == 1, "yes", "no")
    model = ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=0)
    model.fit(X, y_str)
    pred = model.predict(X)
    assert set(np.unique(pred)) <= {"yes", "no"}


def test_missing_values_in_continuous_and_categorical_columns_do_not_crash():
    X, y = _binary_data()
    X_missing = X.copy()
    X_missing.loc[X_missing.sample(20, random_state=1).index, "x1"] = np.nan
    X_missing.loc[X_missing.sample(20, random_state=2).index, "cat"] = np.nan

    model = ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=0)
    model.fit(X_missing, y)
    proba = model.predict_proba(X_missing)
    assert np.all(np.isfinite(proba))


def _three_way_interaction_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-3, 3, n),
            "x2": rng.uniform(-3, 3, n),
            "x3": rng.uniform(-3, 3, n),
        }
    )
    score = X["x1"] * X["x2"] + X["x1"] * X["x3"] + X["x2"] * X["x3"] + 2.0 * X["x1"] * X["x2"] * X["x3"]
    y = (score > score.median()).astype(int).to_numpy()
    return X, y


def test_max_interaction_order_2_is_default_and_never_produces_triples():
    X, y = _three_way_interaction_data()
    model = ZoneBoostClassifier(n_rounds=20, col_subsample=1.0, random_state=0).fit(X, y)
    assert all(round_["triples"] == {} for round_ in model.booster_.rounds_)


def test_max_interaction_order_3_improves_accuracy_on_genuine_triple_interaction():
    X, y = _three_way_interaction_data()
    model_pairwise = ZoneBoostClassifier(
        n_rounds=60, random_state=0, col_subsample=1.0, max_interaction_order=2
    ).fit(X, y)
    model_triples = ZoneBoostClassifier(
        n_rounds=60, random_state=0, col_subsample=1.0, max_interaction_order=3
    ).fit(X, y)

    acc_pairwise = model_pairwise.score(X, y)
    acc_triples = model_triples.score(X, y)
    assert acc_triples >= acc_pairwise
    assert any(len(round_["triples"]) > 0 for round_ in model_triples.booster_.rounds_)


def _probabilistic_binary_data(n=3000, seed=0):
    # Noisy sigmoid labels (not a deterministic threshold) -- leaves real
    # room for a calibration step to measurably improve reliability, unlike
    # _binary_data's confident near-0/1 labels.
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    p_true = 1 / (1 + np.exp(-x))
    y = (rng.uniform(size=n) < p_true).astype(int)
    return pd.DataFrame({"x": x}), y


def _reliability_error(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    weighted_err, total = 0.0, 0
    for b in range(bins):
        mask = idx == b
        if not np.any(mask):
            continue
        weighted_err += mask.sum() * abs(p[mask].mean() - y[mask].mean())
        total += mask.sum()
    return weighted_err / total


def test_calibrate_raises_without_validation_split():
    X, y = _probabilistic_binary_data()
    with pytest.raises(ValueError):
        ZoneBoostClassifier(n_rounds=30, random_state=0, calibrate=True, validation_fraction=0).fit(X, y)


def test_calibrate_improves_reliability_on_probabilistic_data():
    X, y = _probabilistic_binary_data()
    model_raw = ZoneBoostClassifier(n_rounds=60, random_state=0, calibrate=False).fit(X, y)
    model_cal = ZoneBoostClassifier(n_rounds=60, random_state=0, calibrate=True).fit(X, y)

    p_raw = model_raw.predict_proba(X)[:, 1]
    p_cal = model_cal.predict_proba(X)[:, 1]
    assert _reliability_error(p_cal, y) < _reliability_error(p_raw, y)
    assert model_cal.booster_.calibrator_ is not None
    assert model_raw.booster_.calibrator_ is None


def test_calibrate_multiclass_probabilities_still_sum_to_one():
    X, y = _multiclass_data()
    model = ZoneBoostClassifier(n_rounds=30, categorical_features=["cat"], random_state=0, calibrate=True).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert all(b.calibrator_ is not None for b in model.boosters_.values())


def test_calibrate_default_false_is_unaffected():
    X, y = _binary_data()
    model_default = ZoneBoostClassifier(n_rounds=20, categorical_features=["cat"], random_state=0).fit(X, y)
    model_explicit = ZoneBoostClassifier(
        n_rounds=20, categorical_features=["cat"], random_state=0, calibrate=False
    ).fit(X, y)
    np.testing.assert_array_equal(model_default.predict_proba(X), model_explicit.predict_proba(X))
    assert model_default.booster_.calibrator_ is None


def test_calibrate_does_not_change_explain_or_feature_importance():
    X, y = _probabilistic_binary_data()
    model_raw = ZoneBoostClassifier(n_rounds=40, random_state=0, calibrate=False).fit(X, y)
    model_cal = ZoneBoostClassifier(n_rounds=40, random_state=0, calibrate=True).fit(X, y)

    pd.testing.assert_frame_equal(model_raw.explain(X), model_cal.explain(X))
    pd.testing.assert_series_equal(model_raw.feature_importance(X), model_cal.feature_importance(X))


def test_calibration_fraction_default_zero_reproduces_unconstrained_predictions():
    X, y = _probabilistic_binary_data()
    model_default = ZoneBoostClassifier(n_rounds=30, random_state=0, calibrate=True).fit(X, y)
    model_explicit = ZoneBoostClassifier(n_rounds=30, random_state=0, calibrate=True, calibration_fraction=0.0).fit(
        X, y
    )
    np.testing.assert_array_equal(model_default.predict_proba(X), model_explicit.predict_proba(X))


def test_calibration_fraction_uses_a_dedicated_split():
    X, y = _probabilistic_binary_data(n=3000)
    model = ZoneBoostClassifier(
        n_rounds=30, random_state=0, calibrate=True, validation_fraction=0.2, calibration_fraction=0.1
    ).fit(X, y)
    assert model.booster_.calibrator_ is not None


def test_refit_on_full_data_requires_calibration_fraction():
    X, y = _probabilistic_binary_data()
    with pytest.raises(ValueError):
        ZoneBoostClassifier(n_rounds=20, random_state=0, refit_on_full_data=True, calibration_fraction=0.0).fit(X, y)


def test_refit_on_full_data_trains_on_more_rows_than_fit_split_alone():
    X, y = _probabilistic_binary_data(n=2000)
    model_no_refit = ZoneBoostClassifier(
        n_rounds=30, random_state=0, validation_fraction=0.3, calibration_fraction=0.1
    ).fit(X, y)
    model_refit = ZoneBoostClassifier(
        n_rounds=30,
        random_state=0,
        validation_fraction=0.3,
        calibration_fraction=0.1,
        refit_on_full_data=True,
    ).fit(X, y)

    assert model_refit.booster_.best_n_rounds_ == model_no_refit.booster_.best_n_rounds_
    assert not np.array_equal(model_refit.predict_proba(X), model_no_refit.predict_proba(X))

    proba = model_refit.predict_proba(X)
    contrib = model_refit.explain(X)
    raw_logit = contrib.sum(axis=1).to_numpy()
    reconstructed = 1.0 / (1.0 + np.exp(-raw_logit))  # calibrate=False here, so this matches predict_proba exactly
    np.testing.assert_allclose(reconstructed, proba[:, 1], atol=1e-6)
