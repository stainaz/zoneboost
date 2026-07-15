import numpy as np
import pandas as pd
import pytest

from zoneboost import ZoneBoostClassifier
from zoneboost.classifier import _random_undersample


def _imbalanced_binary_data(n=2000, minority_frac=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n_minority = int(n * minority_frac)
    n_majority = n - n_minority
    x1 = np.concatenate([rng.normal(3, 1, n_minority), rng.normal(-3, 1, n_majority)])
    x2 = rng.uniform(-5, 5, n)
    y = np.concatenate([np.ones(n_minority), np.zeros(n_majority)]).astype(int)
    order = rng.permutation(n)
    X = pd.DataFrame({"x1": x1[order], "x2": x2})
    return X, y[order]


def _imbalanced_multiclass_data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    counts = {0: int(n * 0.75), 1: int(n * 0.20), 2: n - int(n * 0.75) - int(n * 0.20)}
    xs, ys = [], []
    for cls, count in counts.items():
        xs.append(rng.normal(cls * 4, 1, count))
        ys.append(np.full(count, cls))
    x1 = np.concatenate(xs)
    y = np.concatenate(ys).astype(int)
    order = rng.permutation(n)
    X = pd.DataFrame({"x1": x1[order], "x2": rng.uniform(-5, 5, n)})
    return X, y[order], counts


def test_random_undersample_helper_balances_binary():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(180), np.ones(20)]).astype(int)
    X = pd.DataFrame({"x": np.arange(len(y))})

    X_bal, y_bal = _random_undersample(X, y, rng)

    classes, counts = np.unique(y_bal, return_counts=True)
    assert list(classes) == [0, 1]
    assert counts[0] == counts[1] == 20
    assert len(X_bal) == 40
    # every kept minority-class row must be an original minority row
    assert set(X_bal.loc[y_bal == 1, "x"]).issubset(set(np.arange(180, 200)))


def test_random_undersample_helper_balances_multiclass():
    rng = np.random.default_rng(1)
    y = np.concatenate([np.zeros(300), np.ones(80), np.full(20, 2)]).astype(int)
    X = pd.DataFrame({"x": np.arange(len(y))})

    _, y_bal = _random_undersample(X, y, rng)

    classes, counts = np.unique(y_bal, return_counts=True)
    assert list(classes) == [0, 1, 2]
    assert (counts == 20).all()


def test_undersample_false_is_default_and_reproduces_prior_behavior():
    X, y = _imbalanced_binary_data()
    m_explicit_default = ZoneBoostClassifier(n_rounds=20, random_state=0, undersample=False).fit(X, y)
    m_omitted = ZoneBoostClassifier(n_rounds=20, random_state=0).fit(X, y)
    np.testing.assert_array_equal(m_explicit_default.predict_proba(X), m_omitted.predict_proba(X))


def test_undersample_only_touches_fit_split(monkeypatch):
    X, y = _imbalanced_binary_data(n=1000)
    recorded = {}
    import zoneboost.classifier as clf_mod

    real_fn = clf_mod._random_undersample

    def spy(X_fit, y_fit, rng):
        recorded["n"] = len(y_fit)
        return real_fn(X_fit, y_fit, rng)

    monkeypatch.setattr(clf_mod, "_random_undersample", spy)

    model = ZoneBoostClassifier(
        n_rounds=10, validation_fraction=0.2, calibration_fraction=0.1, undersample=True, random_state=0
    ).fit(X, y)

    n_total = len(y)
    expected_val = max(1, int(n_total * 0.2))
    expected_cal = int(n_total * 0.1)
    expected_fit = n_total - expected_val - expected_cal
    assert recorded["n"] == expected_fit
    # sanity: model still fits and predicts fine after being rebalanced
    assert model.predict(X).shape == (n_total,)


def test_undersample_balances_multiclass_fit_split(monkeypatch):
    X, y, counts = _imbalanced_multiclass_data()
    recorded = {}
    import zoneboost.classifier as clf_mod

    real_fn = clf_mod._random_undersample

    def spy(X_fit, y_fit, rng):
        X_bal, y_bal = real_fn(X_fit, y_fit, rng)
        recorded["counts"] = np.unique(y_bal, return_counts=True)[1]
        return X_bal, y_bal

    monkeypatch.setattr(clf_mod, "_random_undersample", spy)

    ZoneBoostClassifier(
        n_rounds=10, validation_fraction=0.0, calibration_fraction=0.0, undersample=True, random_state=0
    ).fit(X, y)

    assert len(set(recorded["counts"])) == 1  # every class equally represented


def test_undersample_improves_minority_recall_on_severe_imbalance():
    X, y = _imbalanced_binary_data(n=3000, minority_frac=0.03, seed=2)

    m_plain = ZoneBoostClassifier(n_rounds=30, validation_fraction=0.0, random_state=0).fit(X, y)
    m_under = ZoneBoostClassifier(
        n_rounds=30, validation_fraction=0.0, undersample=True, random_state=0
    ).fit(X, y)

    minority_mask = y == 1
    recall_plain = (m_plain.predict(X)[minority_mask] == 1).mean()
    recall_under = (m_under.predict(X)[minority_mask] == 1).mean()
    assert recall_under >= recall_plain


def test_get_params_includes_undersample():
    model = ZoneBoostClassifier(undersample=True)
    params = model.get_params()
    assert params["undersample"] is True


def test_validation_fraction_and_calibration_still_evaluate_real_balance():
    X, y = _imbalanced_binary_data(n=2000, minority_frac=0.05)
    model = ZoneBoostClassifier(
        n_rounds=20, validation_fraction=0.25, undersample=True, random_state=0
    ).fit(X, y)
    # val_logloss_ was computed against the real (still-imbalanced) validation
    # split, not a rebalanced one -- just check the booster fit and produced
    # a finite, sane trajectory, since the exact values depend on the split.
    assert len(model.booster_.val_logloss_) > 0
    assert all(np.isfinite(v) for v in model.booster_.val_logloss_)
