"""Main cross-validated benchmark: ZoneBoost vs XGBoost/LightGBM/CatBoost/EBM
across 6 public datasets (3 regression, 3 classification).

Methodology mirrors zoneboost/benchmarks/compare_lightgbm.py: every model at
its own library's out-of-the-box defaults (only random_state fixed), scored
via K-fold cross-validation (not a single split), real public datasets.

Results are saved to results/main_benchmark.json after EACH dataset
completes (merging with whatever is already on disk), not only once at the
very end -- a prior run of this script was silently killed mid-way through
and lost every row because the old version only wrote the json at the end.
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from datasets import CLASSIFICATION_DATASETS, REGRESSION_DATASETS
from model_zoo import classification_models, regression_models

warnings.filterwarnings("ignore")

N_FOLDS = 3
RANDOM_STATE = 0
RESULTS_PATH = "results/main_benchmark.json"


def _time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def _load_existing():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def _save_rows(new_rows, dataset_name, model_names):
    """Merge new_rows into the on-disk json, replacing any existing rows for
    the same (dataset, model) pairs so re-runs overwrite cleanly."""
    all_rows = _load_existing()
    all_rows = [
        r for r in all_rows
        if not (r["dataset"] == dataset_name and r["model"] in model_names)
    ]
    all_rows += new_rows
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"  -> saved {len(new_rows)} row(s) for {dataset_name} to {RESULTS_PATH}", flush=True)


def run_regression_dataset(loader, only_models=None):
    X, y, task, cat_cols, name = loader()
    print(f"\n=== Regression: {name} (n={len(X)}, p={X.shape[1]}) ===", flush=True)
    models = regression_models(cat_cols)
    if only_models:
        models = {k: v for k, v in models.items() if k in only_models}
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for model_name, (make_model, adapt) in models.items():
        Xa = adapt(X, cat_cols)
        rmses, r2s, fit_times, pred_times = [], [], [], []
        try:
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
            rows.append({
                "dataset": name, "model": model_name,
                "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
                "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
                "fit_time_s": float(np.mean(fit_times)), "predict_time_s": float(np.mean(pred_times)),
                "status": "ok",
            })
            print(f"  {model_name:10s} RMSE={np.mean(rmses):.4f}+/-{np.std(rmses):.4f}  "
                  f"R2={np.mean(r2s):.4f}+/-{np.std(r2s):.4f}  fit={np.mean(fit_times):.2f}s", flush=True)
        except Exception as e:
            rows.append({"dataset": name, "model": model_name, "status": f"FAIL:{type(e).__name__}:{str(e)[:200]}"})
            print(f"  {model_name:10s} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        _save_rows([rows[-1]], name, [model_name])
    return rows


def run_classification_dataset(loader, only_models=None):
    X, y, task, cat_cols, name = loader()
    n_classes = len(np.unique(y))
    print(f"\n=== Classification: {name} (n={len(X)}, p={X.shape[1]}, classes={n_classes}) ===", flush=True)
    models = classification_models(cat_cols, n_classes)
    if only_models:
        models = {k: v for k, v in models.items() if k in only_models}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for model_name, (make_model, adapt) in models.items():
        Xa = adapt(X, cat_cols)
        accs, f1s, aucs, loglosses, fit_times, pred_times = [], [], [], [], [], []
        try:
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
                if n_classes == 2:
                    aucs.append(float(roc_auc_score(y_test, proba[:, 1])))
                else:
                    aucs.append(float(roc_auc_score(y_test, proba, multi_class="ovr")))
                fit_times.append(fit_time)
                pred_times.append(pred_time)
            rows.append({
                "dataset": name, "model": model_name,
                "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                "f1_macro_mean": float(np.mean(f1s)), "f1_macro_std": float(np.std(f1s)),
                "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                "logloss_mean": float(np.mean(loglosses)), "logloss_std": float(np.std(loglosses)),
                "fit_time_s": float(np.mean(fit_times)), "predict_time_s": float(np.mean(pred_times)),
                "status": "ok",
            })
            print(f"  {model_name:10s} Acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}  "
                  f"AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}  "
                  f"F1={np.mean(f1s):.4f}  fit={np.mean(fit_times):.2f}s", flush=True)
        except Exception as e:
            rows.append({"dataset": name, "model": model_name, "status": f"FAIL:{type(e).__name__}:{str(e)[:200]}"})
            print(f"  {model_name:10s} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        _save_rows([rows[-1]], name, [model_name])
    return rows


if __name__ == "__main__":
    import sys

    # Optional: `python run_benchmark.py Wine` restricts to one dataset by
    # name, so a resumed run can be split into small chunks that each
    # finish well inside whatever background-process time ceiling killed a
    # previous full run. Already-saved (dataset, model) pairs are skipped.
    dataset_filter = sys.argv[1] if len(sys.argv) > 1 else None
    existing = _load_existing()
    done = {(r["dataset"], r["model"]) for r in existing}

    for loader in REGRESSION_DATASETS:
        name = loader()[4]
        if dataset_filter and name != dataset_filter:
            continue
        missing = {m for m in regression_models([])} - {model for (d, model) in done if d == name}
        if not missing:
            print(f"{name}: already fully saved, skipping", flush=True)
            continue
        run_regression_dataset(loader, only_models=missing if missing != set(regression_models([])) else None)

    for loader in CLASSIFICATION_DATASETS:
        name = loader()[4]
        if dataset_filter and name != dataset_filter:
            continue
        all_model_names = set(classification_models([], 2))
        missing = all_model_names - {model for (d, model) in done if d == name}
        if not missing:
            print(f"{name}: already fully saved, skipping", flush=True)
            continue
        run_classification_dataset(loader, only_models=missing if missing != all_model_names else None)

    print(f"\nDone. Results accumulated in {RESULTS_PATH}")
