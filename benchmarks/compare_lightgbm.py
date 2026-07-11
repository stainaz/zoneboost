"""Honest benchmark: ZoneBoost vs. LightGBM on real data.

Not a leaderboard zoneboost is trying to win -- its actual value proposition is
exact, zero-approximation attribution (`explain()`), not necessarily topping
accuracy on tabular benchmarks the way gradient boosting often does. This
script measures the real gap (or lack of one) rather than assuming it:

- Each model uses its own library's out-of-the-box defaults (only
  ``random_state`` is set for reproducibility) -- no tuning that favors
  either one.
- Cross-validation, not a single train/test split, so results reflect genuine
  variance across splits rather than one split's luck -- the same "never
  grade yourself on your own homework" principle already central to
  zoneboost's own cross-fitting.
- Real datasets (California Housing, Breast Cancer Wisconsin), plus a
  synthetic case with a known interaction to demonstrate
  ZoneBoostRegressor.explain()'s exact interaction attribution -- LightGBM has
  no built-in equivalent (only a post-hoc SHAP-interaction approximation).
  California Housing is randomly subsampled to ``HOUSING_SAMPLE_SIZE`` rows
  (fixed seed) purely so the comparison runs in a couple of minutes -- a
  disclosed speed tradeoff, not a hyperparameter change to any model, and
  still a real, non-trivial sample of real data.

InterpretML's Explainable Boosting Machine (EBM) -- architecturally the
closest existing interpretable model, see docs/how-it-works.html's "How It
Compares" -- was deliberately left out of this script: its default
``outer_bags=8`` spawns a separate joblib/loky worker process per bag per CV
fold, and the fixed process-spawn overhead on Windows dominated wall-clock
time even on small datasets, independent of any real compute cost. That's an
environment/parallelism-backend issue, not a finding about EBM's accuracy, so
it isn't reported here rather than being shipped as a misleading number.

Run: pip install -e ".[benchmark]" && python benchmarks/compare_lightgbm.py
"""

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from zoneboost import ZoneBoostClassifier, ZoneBoostRegressor

RANDOM_STATE = 0
N_FOLDS = 3
HOUSING_SAMPLE_SIZE = 3000


def _time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def benchmark_regression(X, y, name):
    print(f"\n=== Regression: {name} ({len(X)} rows, {X.shape[1]} features) ===")
    models = {
        "ZoneBoostRegressor": lambda: ZoneBoostRegressor(random_state=RANDOM_STATE),
        "LGBMRegressor": lambda: LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
    }
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for model_name, make_model in models.items():
        rmses, r2s, fit_times, pred_times = [], [], [], []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = make_model()
            _, fit_time = _time_call(model.fit, X_train, y_train)
            pred, pred_time = _time_call(model.predict, X_test)
            rmses.append(float(np.sqrt(mean_squared_error(y_test, pred))))
            r2s.append(float(r2_score(y_test, pred)))
            fit_times.append(fit_time)
            pred_times.append(pred_time)
        rows.append(
            {
                "model": model_name,
                "RMSE": f"{np.mean(rmses):.4f} +/- {np.std(rmses):.4f}",
                "R2": f"{np.mean(r2s):.4f} +/- {np.std(r2s):.4f}",
                "fit_time_s": f"{np.mean(fit_times):.2f}",
                "predict_time_s": f"{np.mean(pred_times):.3f}",
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def benchmark_classification(X, y, name):
    print(f"\n=== Classification: {name} ({len(X)} rows, {X.shape[1]} features) ===")
    models = {
        "ZoneBoostClassifier": lambda: ZoneBoostClassifier(random_state=RANDOM_STATE),
        "LGBMClassifier": lambda: LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    }
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for model_name, make_model in models.items():
        accs, aucs, loglosses, fit_times, pred_times = [], [], [], [], []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = make_model()
            _, fit_time = _time_call(model.fit, X_train, y_train)
            proba, pred_time = _time_call(model.predict_proba, X_test)
            p1 = proba[:, 1]
            pred = (p1 >= 0.5).astype(int)
            accs.append(float(accuracy_score(y_test, pred)))
            aucs.append(float(roc_auc_score(y_test, p1)))
            loglosses.append(float(log_loss(y_test, p1)))
            fit_times.append(fit_time)
            pred_times.append(pred_time)
        rows.append(
            {
                "model": model_name,
                "accuracy": f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}",
                "ROC_AUC": f"{np.mean(aucs):.4f} +/- {np.std(aucs):.4f}",
                "log_loss": f"{np.mean(loglosses):.4f} +/- {np.std(loglosses):.4f}",
                "fit_time_s": f"{np.mean(fit_times):.2f}",
                "predict_time_s": f"{np.mean(pred_times):.3f}",
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def _synthetic_interaction_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.uniform(-3, 3, n),
            "x2": rng.uniform(-3, 3, n),
            "x3": rng.uniform(-3, 3, n),
            "x4": rng.uniform(-3, 3, n),  # noise, no relationship to y
        }
    )
    y = X["x1"] * X["x2"] + 2.0 * X["x1"] * X["x2"] * X["x3"] + rng.normal(0, 0.5, n)
    return X, y.to_numpy()


def interaction_detection_demo():
    print("\n=== Interaction attribution (synthetic, known ground truth: x1*x2 + 2*x1*x2*x3) ===")
    X, y = _synthetic_interaction_data()

    zb = ZoneBoostRegressor(
        n_rounds=150, max_interaction_order=3, random_state=RANDOM_STATE, col_subsample=1.0
    ).fit(X, y)
    print("\nZoneBoostRegressor.feature_importance(X):")
    print(zb.feature_importance(X).to_string())
    print(
        "\nLightGBM: no native interaction attribution -- would need SHAP interaction "
        "values, a separate post-hoc approximation, not a built-in decomposition."
    )


if __name__ == "__main__":
    housing = fetch_california_housing(as_frame=True)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(housing.data), size=HOUSING_SAMPLE_SIZE, replace=False)
    housing_X = housing.data.iloc[sample_idx].reset_index(drop=True)
    housing_y = housing.target.to_numpy()[sample_idx]
    benchmark_regression(housing_X, housing_y, f"California Housing (subsampled to {HOUSING_SAMPLE_SIZE} rows)")

    cancer = load_breast_cancer(as_frame=True)
    benchmark_classification(cancer.data, cancer.target.to_numpy(), "Breast Cancer Wisconsin")

    interaction_detection_demo()
