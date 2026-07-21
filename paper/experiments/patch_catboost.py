"""Patch script: re-run just CatBoost on the datasets where the original
run_benchmark.py process failed due to the clone() incompatibility bug
(fixed in model_zoo.py -- factories instead of instances), and merge the
results into results/main_benchmark.json in place."""
import json
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from datasets import load_bike_sharing, load_titanic
from model_zoo import regression_models, classification_models

N_FOLDS = 3
RANDOM_STATE = 0


def _time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def patch_regression(loader, model_name="CatBoost"):
    X, y, task, cat_cols, name = loader()
    make_model, adapt = regression_models(cat_cols)[model_name]
    Xa = adapt(X, cat_cols)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmses, r2s, fit_times, pred_times = [], [], [], []
    for train_idx, test_idx in kf.split(Xa):
        m = make_model()
        X_train, X_test = Xa.iloc[train_idx], Xa.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        _, fit_time = _time_call(m.fit, X_train, y_train)
        pred, pred_time = _time_call(m.predict, X_test)
        rmses.append(float(np.sqrt(mean_squared_error(y_test, pred))))
        r2s.append(float(r2_score(y_test, pred)))
        fit_times.append(fit_time)
        pred_times.append(pred_time)
    row = {
        "dataset": name, "model": model_name,
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        "fit_time_s": float(np.mean(fit_times)), "predict_time_s": float(np.mean(pred_times)),
        "status": "ok",
    }
    print(f"[patched] {name} {model_name}: RMSE={row['rmse_mean']:.4f} R2={row['r2_mean']:.4f}", flush=True)
    return row


def patch_classification(loader, model_name="CatBoost"):
    X, y, task, cat_cols, name = loader()
    n_classes = len(np.unique(y))
    make_model, adapt = classification_models(cat_cols, n_classes)[model_name]
    Xa = adapt(X, cat_cols)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s, aucs, loglosses, fit_times, pred_times = [], [], [], [], [], []
    for train_idx, test_idx in skf.split(Xa, y):
        m = make_model()
        X_train, X_test = Xa.iloc[train_idx], Xa.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        _, fit_time = _time_call(m.fit, X_train, y_train)
        proba, pred_time = _time_call(m.predict_proba, X_test)
        pred = np.argmax(proba, axis=1)
        accs.append(float(accuracy_score(y_test, pred)))
        f1s.append(float(f1_score(y_test, pred, average="macro")))
        loglosses.append(float(log_loss(y_test, proba, labels=list(range(n_classes)))))
        aucs.append(float(roc_auc_score(y_test, proba[:, 1])) if n_classes == 2
                     else float(roc_auc_score(y_test, proba, multi_class="ovr")))
        fit_times.append(fit_time)
        pred_times.append(pred_time)
    row = {
        "dataset": name, "model": model_name,
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)), "f1_macro_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "logloss_mean": float(np.mean(loglosses)), "logloss_std": float(np.std(loglosses)),
        "fit_time_s": float(np.mean(fit_times)), "predict_time_s": float(np.mean(pred_times)),
        "status": "ok",
    }
    print(f"[patched] {name} {model_name}: Acc={row['accuracy_mean']:.4f} AUC={row['auc_mean']:.4f}", flush=True)
    return row


if __name__ == "__main__":
    new_rows = [
        patch_regression(load_bike_sharing),
        patch_classification(load_titanic),
    ]
    with open("results/main_benchmark.json") as f:
        all_rows = json.load(f)
    all_rows = [
        r for r in all_rows
        if not (r["model"] == "CatBoost" and r["dataset"] in ("Bike Sharing Demand", "Titanic"))
    ]
    all_rows += new_rows
    with open("results/main_benchmark.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print("Patched results/main_benchmark.json")
