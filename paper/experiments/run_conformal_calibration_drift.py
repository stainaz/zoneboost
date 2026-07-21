"""Three additional real experiments beyond the main benchmark/ablation:

1. Conformal coverage: does ConformalizedQuantileRegressor actually deliver
   its promised marginal coverage, and does its interval width vary with X
   (locally adaptive) rather than being constant?
2. Classifier calibration: Brier score / ECE with calibrate=False vs
   calibrate=True.
3. Drift detection: compare_models on two ZoneBoostRegressor fits from
   genuinely different data slices (a real covariate-shift split, not a
   random split) -- does it correctly flag which features shifted?
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from datasets import load_bike_sharing, load_california_housing, load_diabetes, load_titanic
from zoneboost import ConformalizedQuantileRegressor, ZoneBoostClassifier, ZoneBoostRegressor, compare_models

warnings.filterwarnings("ignore")
RANDOM_STATE = 0
ALPHA = 0.1  # target 90% coverage


def conformal_experiment(loader):
    X, y, task, cat_cols, name = loader()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    template = ZoneBoostRegressor(categorical_features=cat_cols, n_rounds=100)
    cqr = ConformalizedQuantileRegressor(estimator=template, alpha=ALPHA, random_state=RANDOM_STATE).fit(X_train, y_train)
    lo, hi = cqr.predict_interval(X_test)
    covered = np.mean((y_test >= lo) & (y_test <= hi))
    widths = hi - lo
    print(f"[Conformal] {name}: target coverage={1-ALPHA:.2f}  empirical coverage={covered:.4f}  "
          f"mean width={widths.mean():.3f}  width std={widths.std():.3f} (locally adaptive if >0)", flush=True)
    return {
        "dataset": name, "target_coverage": 1 - ALPHA, "empirical_coverage": float(covered),
        "mean_width": float(widths.mean()), "width_std": float(widths.std()),
    }


def calibration_experiment():
    X, y, task, cat_cols, name = load_titanic()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    out = {}
    for calibrate in (False, True):
        model = ZoneBoostClassifier(categorical_features=cat_cols, calibrate=calibrate, random_state=RANDOM_STATE).fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, proba)
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        key = f"calibrate={calibrate}"
        out[key] = {"brier": float(brier), "ece": ece}
        print(f"[Calibration] {name} {key}: Brier={brier:.4f}  ECE={ece:.4f}", flush=True)
    return {"dataset": name, **out}


def drift_experiment():
    X, y, task, cat_cols, name = load_california_housing()
    # Real covariate-shift split (not random): older-vs-newer-looking
    # segment via median house age, a genuine population difference rather
    # than a synthetic perturbation.
    median_age = X["HouseAge"].median()
    old_mask = X["HouseAge"] <= median_age
    X_old, y_old = X[old_mask], y[old_mask]
    X_new, y_new = X[~old_mask], y[~old_mask]

    model_old = ZoneBoostRegressor(n_rounds=100, random_state=RANDOM_STATE).fit(X_old, y_old)
    model_new = ZoneBoostRegressor(n_rounds=100, random_state=RANDOM_STATE).fit(X_new, y_new)
    report = compare_models(model_old, model_new, X_new, y_new)

    fi_change = report["feature_importance_change"]
    top_shifts = fi_change.reindex(fi_change["change"].abs().sort_values(ascending=False).index).head(5)
    print(f"[Drift] {name}: top feature-importance shifts (old HouseAge<={median_age} vs new):", flush=True)
    print(top_shifts.to_string(), flush=True)
    if "performance_change" in report:
        print("performance_change:", report["performance_change"], flush=True)

    return {
        "dataset": name,
        "top_importance_shifts": top_shifts.reset_index().to_dict(orient="records"),
        "performance_change": report.get("performance_change"),
    }


if __name__ == "__main__":
    results = {"conformal": [], "calibration": None, "drift": None}
    for loader in (load_california_housing, load_diabetes, load_bike_sharing):
        results["conformal"].append(conformal_experiment(loader))
    results["calibration"] = calibration_experiment()
    results["drift"] = drift_experiment()

    with open("results/conformal_calibration_drift.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved results/conformal_calibration_drift.json")
