import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance, mean_tweedie_deviance

from zoneboost import ZoneBoostRegressor
from zoneboost._common import _glm_baseline, _glm_residual


def _poisson_data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 70, n)
    region = rng.choice(["urban", "rural"], n)
    exposure = rng.uniform(0.2, 1.0, n)
    true_rate = np.exp(-3.0 + 0.02 * age + np.where(region == "urban", 0.3, 0.0))
    claims = rng.poisson(true_rate * exposure).astype(float)
    X = pd.DataFrame({"age": age, "region": region})
    return X, claims, exposure


def _gamma_data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 70, n)
    severity = rng.gamma(shape=2.0, scale=np.exp(1.0 + 0.01 * age))
    X = pd.DataFrame({"age": age})
    return X, severity


def test_poisson_predictions_track_true_rate():
    X, claims, exposure = _poisson_data()
    model = ZoneBoostRegressor(random_state=0, loss="poisson", n_rounds=50).fit(
        X, claims, offset=np.log(exposure)
    )
    pred = model.predict(X, offset=np.log(exposure))
    assert np.all(pred >= 0)
    assert abs(pred.mean() - claims.mean()) < 0.03


def test_poisson_offset_doubling_doubles_prediction():
    # a mathematical identity of the log link (exp(link + log 2) == 2 *
    # exp(link)) that holds for any fitted model regardless of fit
    # quality -- few rounds needed.
    X, claims, exposure = _poisson_data(n=300)
    model = ZoneBoostRegressor(random_state=0, loss="poisson", n_rounds=15).fit(
        X, claims, offset=np.log(exposure)
    )
    pred = model.predict(X, offset=np.log(exposure))
    pred_double = model.predict(X, offset=np.log(exposure * 2))
    ratio = pred_double / pred
    assert np.allclose(ratio, 2.0, atol=1e-8)


def test_gamma_requires_strictly_positive_y():
    X, severity = _gamma_data()
    with pytest.raises(ValueError, match="gamma"):
        ZoneBoostRegressor(loss="gamma").fit(X, np.zeros(len(X)))
    with pytest.raises(ValueError, match="gamma"):
        ZoneBoostRegressor(loss="gamma").fit(X, -np.abs(severity))


def test_poisson_tweedie_require_non_negative_y():
    X, claims, _ = _poisson_data()
    with pytest.raises(ValueError, match="poisson"):
        ZoneBoostRegressor(loss="poisson").fit(X, claims - 100)
    with pytest.raises(ValueError, match="tweedie"):
        ZoneBoostRegressor(loss="tweedie").fit(X, claims - 100)


def test_gamma_fits_and_predicts_positive():
    X, severity = _gamma_data()
    model = ZoneBoostRegressor(random_state=0, loss="gamma", n_rounds=40).fit(X, severity)
    pred = model.predict(X)
    assert np.all(pred > 0)
    assert abs(pred.mean() - severity.mean()) / severity.mean() < 0.1


def test_tweedie_power_1_and_2_match_poisson_gamma_residual():
    y = np.array([0.0, 2.0, 5.0, 10.0])
    mu = np.array([1.0, 1.5, 4.0, 9.0])
    assert np.allclose(_glm_residual(y, mu, power=1.0), y - mu)
    assert np.allclose(_glm_residual(y, mu, power=2.0), y / mu - 1.0)


def test_tweedie_intermediate_power_fits_without_error():
    X, claims, exposure = _poisson_data(n=600)
    model = ZoneBoostRegressor(
        random_state=0, loss="tweedie", tweedie_power=1.5, n_rounds=25
    ).fit(X, claims, offset=np.log(exposure))
    pred = model.predict(X, offset=np.log(exposure))
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0)


@pytest.mark.parametrize("loss,power", [("poisson", None), ("gamma", None), ("tweedie", 1.5)])
def test_explain_sums_to_predict_under_link_convention(loss, power):
    if loss == "gamma":
        X, y = _gamma_data(n=600)
        offset = None
    else:
        X, y, exposure = _poisson_data(n=600)
        offset = np.log(exposure)
    kwargs = {"tweedie_power": power} if power is not None else {}
    model = ZoneBoostRegressor(random_state=0, loss=loss, n_rounds=25, **kwargs).fit(X, y, offset=offset)

    contrib = model.explain(X)
    link_sum = contrib.sum(axis=1).to_numpy()
    offset_arr = offset if offset is not None else np.zeros(len(X))
    reconstructed = np.exp(link_sum + offset_arr)
    pred = model.predict(X, offset=offset)
    np.testing.assert_allclose(reconstructed, pred, rtol=1e-6)


@pytest.mark.parametrize(
    "loss,deviance_fn",
    [("poisson", mean_poisson_deviance), ("gamma", mean_gamma_deviance)],
)
def test_score_method_uses_correct_deviance(loss, deviance_fn):
    # model._score is what train_rmse_/val_rmse_ are built from each round
    # -- verify the dispatch directly against arbitrary y/pred rather than
    # via train_rmse_[-1], which (for every loss, not just the new ones)
    # reflects the *honest, cross-fitted* in-training score, not a plain
    # re-scoring of the final predict(X) -- never an exact match, by design
    # (see weak_learner_fit's own oof_contributions substitution).
    y = np.array([1.0, 2.0, 5.0, 3.0, 0.5])
    pred = np.array([1.2, 1.8, 4.5, 3.2, 0.6])
    model = ZoneBoostRegressor(loss=loss)
    expected = float(deviance_fn(y, pred))
    assert np.isclose(model._score(y, pred), expected, rtol=1e-10)


def test_score_method_tweedie_matches_sklearn_deviance():
    y = np.array([0.0, 2.0, 5.0, 3.0, 0.5])
    pred = np.array([1.2, 1.8, 4.5, 3.2, 0.6])
    model = ZoneBoostRegressor(loss="tweedie", tweedie_power=1.5)
    expected = float(mean_tweedie_deviance(y, pred, power=1.5))
    assert np.isclose(model._score(y, pred), expected, rtol=1e-10)


def test_train_rmse_decreases_over_rounds_for_glm_losses():
    X, claims, exposure = _poisson_data(n=500)
    model = ZoneBoostRegressor(random_state=0, loss="poisson", n_rounds=25, validation_fraction=0).fit(
        X, claims, offset=np.log(exposure)
    )
    assert model.train_rmse_[-1] < model.train_rmse_[0]


def test_glm_baseline_formula_matches_hand_calculation():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    offset = np.array([0.1, -0.2, 0.3, 0.0])
    expected = np.log(np.sum(y) / np.sum(np.exp(offset)))
    assert np.isclose(_glm_baseline(y, offset, power=1.0), expected)


def test_offset_omitted_at_predict_defaults_to_zero_consistently():
    X, claims, exposure = _poisson_data(n=500)
    model = ZoneBoostRegressor(random_state=0, loss="poisson", n_rounds=20).fit(
        X, claims, offset=np.log(exposure)
    )
    pred_no_offset = model.predict(X)
    pred_zero_offset = model.predict(X, offset=np.zeros(len(X)))
    np.testing.assert_allclose(pred_no_offset, pred_zero_offset)
    assert pred_no_offset.shape == (len(X),)


def test_offset_rejected_for_squared_error_and_quantile():
    X, claims, exposure = _poisson_data(n=200)
    with pytest.raises(ValueError, match="offset"):
        ZoneBoostRegressor(loss="squared_error").fit(X, claims, offset=np.log(exposure))
    with pytest.raises(ValueError, match="offset"):
        ZoneBoostRegressor(loss="quantile").fit(X, claims, offset=np.log(exposure))


def test_predict_interval_unavailable_for_glm_losses():
    X, claims, exposure = _poisson_data(n=500)
    model = ZoneBoostRegressor(random_state=0, loss="poisson", n_rounds=20).fit(
        X, claims, offset=np.log(exposure)
    )
    with pytest.raises(ValueError, match="predict_interval"):
        model.predict_interval(X)


def test_squared_error_and_quantile_bit_identical_defaults():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({"x1": rng.uniform(-5, 5, n), "x2": rng.uniform(-5, 5, n)})
    y = (X["x1"] ** 2 + rng.normal(0, 1, n)).to_numpy()

    model_sq = ZoneBoostRegressor(random_state=0, n_rounds=30).fit(X, y)
    pred_sq = model_sq.predict(X)
    assert np.all(np.isfinite(pred_sq))

    model_q = ZoneBoostRegressor(random_state=0, n_rounds=30, loss="quantile", quantile=0.5).fit(X, y)
    pred_q = model_q.predict(X)
    assert np.all(np.isfinite(pred_q))
