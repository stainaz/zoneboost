"""Ablation study: which ZoneBoost module actually drives performance?

Six real, publicly-documented knobs (verified against the shipped
constructor signature -- see regressor.py / classifier.py docstrings):

  A1 shrinkage       shrinkage_m=0.01 (near-zero EB shrinkage; exactly 0
                     hits a 0/0 empty-zone edge case, see variants_shrinkage)
                     vs 10 (default) vs learn_shrinkage_m=True (regressor only)
  A2 zone boundary   adaptive per-round split search (default) vs zone
                     boundaries fixed once via quantile binning (continuous
                     columns pre-binned with qcut and passed in as
                     categorical_features, so each round reuses the same
                     bin edges instead of re-deriving a split)
  A3 interaction     main-effects-only (max_pair_interactions=0) vs
     order           pairwise (default, max_interaction_order=2) vs
                     pairwise+triples (max_interaction_order=3)
  A4 categorical     declared categorical, one zone per value (default) vs
     treatment       ordinal-encoded and left continuous, subject to the
                     adaptive cut-point search (only run on datasets that
                     actually have categorical columns)
  A5 subsampling     default (row=0.7, col=0.7) vs none (row=1.0, col=1.0)
  A6 boundary        adaptive_boundary_smoothing=False (default) vs True
     smoothing
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from datasets import CLASSIFICATION_DATASETS, REGRESSION_DATASETS
from zoneboost import ZoneBoostClassifier, ZoneBoostRegressor

warnings.filterwarnings("ignore")

N_FOLDS = 3
RANDOM_STATE = 0
RESULTS_PATH = "results/ablation.json"


def _load_existing():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def _save_rows(new_rows, axis_name, dataset_name):
    """Merge new_rows into the on-disk json, replacing any existing rows for
    the same (axis, dataset) so re-runs overwrite cleanly instead of
    duplicating -- and so a mid-run kill only loses the current axis, not
    every axis run so far."""
    all_rows = _load_existing()
    all_rows = [
        r for r in all_rows
        if not (r["axis"] == axis_name and r["dataset"] == dataset_name)
    ]
    all_rows += new_rows
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"  -> saved {len(new_rows)} row(s) for [{axis_name}] {dataset_name} to {RESULTS_PATH}", flush=True)


def quantile_binned_X(X, cat_cols, n_bins=7):
    """Pre-bin every continuous column into n_bins fixed quantile bins,
    then hand the bin id back as a categorical column -- this reuses
    ZoneBoost's own categorical path (one zone per distinct value) to
    simulate "boundaries fixed once, never re-derived per round" for
    columns that would otherwise get the adaptive split search."""
    X = X.copy()
    new_cat = list(cat_cols)
    for c in X.columns:
        if c in cat_cols:
            continue
        try:
            binned = pd.qcut(X[c], q=n_bins, duplicates="drop")
            X[c] = binned.cat.codes.astype("category")
            new_cat.append(c)
        except (ValueError, TypeError):
            pass  # too few unique values to bin -- leave as-is
    return X, new_cat


def ordinal_encoded_X(X, cat_cols):
    """Cast declared categorical columns to numeric codes and DROP them
    from categorical_features, so ZoneBoost treats them as continuous
    (cut-point search) instead of one-zone-per-category."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("category").cat.codes.astype(float)
    return X


def cv_regression(model, X, y):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmses, r2s = [], []
    for train_idx, test_idx in kf.split(X):
        m = clone(model)
        m.fit(X.iloc[train_idx], y[train_idx])
        pred = m.predict(X.iloc[test_idx])
        rmses.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))
        r2s.append(float(r2_score(y[test_idx], pred)))
    return {"rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s))}


def cv_classification(model, X, y, n_classes):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, aucs, loglosses = [], [], []
    for train_idx, test_idx in skf.split(X, y):
        m = clone(model)
        m.fit(X.iloc[train_idx], y[train_idx])
        proba = m.predict_proba(X.iloc[test_idx])
        pred = np.argmax(proba, axis=1)
        accs.append(float(accuracy_score(y[test_idx], pred)))
        loglosses.append(float(log_loss(y[test_idx], proba, labels=list(range(n_classes)))))
        if n_classes == 2:
            aucs.append(float(roc_auc_score(y[test_idx], proba[:, 1])))
        else:
            aucs.append(float(roc_auc_score(y[test_idx], proba, multi_class="ovr")))
    return {"accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "logloss_mean": float(np.mean(loglosses)), "logloss_std": float(np.std(loglosses))}


def run_axis_regression(name, loader, variants):
    X, y, task, cat_cols, dname = loader()
    out = []
    for vname, X_variant, cat_variant, kwargs in variants(X, cat_cols):
        model = ZoneBoostRegressor(categorical_features=cat_variant, random_state=RANDOM_STATE, **kwargs)
        try:
            metrics = cv_regression(model, X_variant, y)
            out.append({"axis": name, "dataset": dname, "variant": vname, "status": "ok", **metrics})
            print(f"  [{name}] {dname:26s} {vname:20s} RMSE={metrics['rmse_mean']:.4f} R2={metrics['r2_mean']:.4f}", flush=True)
        except Exception as e:
            out.append({"axis": name, "dataset": dname, "variant": vname, "status": f"FAIL:{type(e).__name__}:{str(e)[:150]}"})
            print(f"  [{name}] {dname:26s} {vname:20s} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
    _save_rows(out, name, dname)
    return out


def run_axis_classification(name, loader, variants):
    X, y, task, cat_cols, dname = loader()
    n_classes = len(np.unique(y))
    out = []
    for vname, X_variant, cat_variant, kwargs in variants(X, cat_cols):
        model = ZoneBoostClassifier(categorical_features=cat_variant, random_state=RANDOM_STATE, **kwargs)
        try:
            metrics = cv_classification(model, X_variant, y, n_classes)
            out.append({"axis": name, "dataset": dname, "variant": vname, "status": "ok", **metrics})
            print(f"  [{name}] {dname:26s} {vname:20s} Acc={metrics['accuracy_mean']:.4f} AUC={metrics['auc_mean']:.4f}", flush=True)
        except Exception as e:
            out.append({"axis": name, "dataset": dname, "variant": vname, "status": f"FAIL:{type(e).__name__}:{str(e)[:150]}"})
            print(f"  [{name}] {dname:26s} {vname:20s} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
    _save_rows(out, name, dname)
    return out


# ---- variant generators: each yields (variant_name, X, cat_cols, extra_kwargs) ----

def variants_shrinkage(X, cat_cols):
    # shrinkage_m=0.0 exactly hits a genuine edge case: an empty zone (0
    # supporting rows) with 0 prior weight produces a 0/0 NaN cell mean,
    # which then fails downstream in the Lasso combination step -- not a
    # bug worth patching in the shipped library for this ablation, so we
    # use a small-but-nonzero floor (0.01) as the practical "near-zero
    # shrinkage" arm instead.
    yield "near_zero_shrinkage(m=0.01)", X, cat_cols, {"shrinkage_m": 0.01}
    yield "default(m=10)", X, cat_cols, {"shrinkage_m": 10.0}


def variants_shrinkage_learned(X, cat_cols):
    # regressor-only: learn_shrinkage_m
    yield "near_zero_shrinkage(m=0.01)", X, cat_cols, {"shrinkage_m": 0.01}
    yield "default(m=10)", X, cat_cols, {"shrinkage_m": 10.0}
    yield "learned_shrinkage", X, cat_cols, {"learn_shrinkage_m": True}


def variants_boundary(X, cat_cols):
    yield "adaptive(default)", X, cat_cols, {}
    Xb, catb = quantile_binned_X(X, cat_cols)
    yield "fixed_quantile_bins", Xb, catb, {}


def variants_interaction_order(X, cat_cols):
    yield "main_effects_only", X, cat_cols, {"max_pair_interactions": 0}
    yield "pairwise(default)", X, cat_cols, {"max_interaction_order": 2}
    yield "pairwise+triples", X, cat_cols, {"max_interaction_order": 3}


def variants_categorical(X, cat_cols):
    if not cat_cols:
        return
    yield "own_zone(default)", X, cat_cols, {}
    yield "ordinal_cutpoint", ordinal_encoded_X(X, cat_cols), [], {}


def variants_subsampling(X, cat_cols):
    yield "default(0.7/0.7)", X, cat_cols, {"row_subsample": 0.7, "col_subsample": 0.7}
    yield "no_subsampling(1.0/1.0)", X, cat_cols, {"row_subsample": 1.0, "col_subsample": 1.0}


def variants_smoothing(X, cat_cols):
    yield "hard(default)", X, cat_cols, {"adaptive_boundary_smoothing": False}
    yield "adaptive_smooth", X, cat_cols, {"adaptive_boundary_smoothing": True}


def _already_done(axis_name, dname):
    """Skip an (axis, dataset) pair if it's already fully saved -- lets this
    script be re-run after an interruption without repeating finished work.
    A pair counts as done if any row for it exists (each axis+dataset is
    saved atomically as a whole by _save_rows, so partial runs never leave
    a partial row set on disk)."""
    existing = _load_existing()
    return any(r["axis"] == axis_name and r["dataset"] == dname for r in existing)


if __name__ == "__main__":
    import sys

    all_rows = []
    axes_reg = [
        ("A1_shrinkage", variants_shrinkage_learned),
        ("A2_zone_boundary", variants_boundary),
        ("A3_interaction_order", variants_interaction_order),
        ("A4_categorical_treatment", variants_categorical),
        ("A5_subsampling", variants_subsampling),
        ("A6_boundary_smoothing", variants_smoothing),
    ]
    axes_clf = [
        ("A1_shrinkage", variants_shrinkage),
        ("A2_zone_boundary", variants_boundary),
        ("A3_interaction_order", variants_interaction_order),
        ("A4_categorical_treatment", variants_categorical),
        ("A5_subsampling", variants_subsampling),
        ("A6_boundary_smoothing", variants_smoothing),
    ]

    # Optional: `python run_ablation.py classification Wine` restricts to
    # one phase ("regression"/"classification") and optionally one dataset
    # name, so a resumed run can be split into small chunks that each
    # finish well inside whatever background-process time ceiling killed
    # the previous full run.
    phase = sys.argv[1] if len(sys.argv) > 1 else None
    dataset_filter = sys.argv[2] if len(sys.argv) > 2 else None

    if phase in (None, "regression"):
        print("========== REGRESSION ABLATIONS ==========", flush=True)
        for axis_name, variant_fn in axes_reg:
            for loader in REGRESSION_DATASETS:
                dname = loader()[4]
                if dataset_filter and dname != dataset_filter:
                    continue
                if _already_done(axis_name, dname):
                    print(f"  [{axis_name}] {dname}: already saved, skipping", flush=True)
                    continue
                all_rows += run_axis_regression(axis_name, loader, variant_fn)

    if phase in (None, "classification"):
        print("\n========== CLASSIFICATION ABLATIONS ==========", flush=True)
        for axis_name, variant_fn in axes_clf:
            for loader in CLASSIFICATION_DATASETS:
                dname = loader()[4]
                if dataset_filter and dname != dataset_filter:
                    continue
                if _already_done(axis_name, dname):
                    print(f"  [{axis_name}] {dname}: already saved, skipping", flush=True)
                    continue
                all_rows += run_axis_classification(axis_name, loader, variant_fn)

    print(f"\nDone. Results accumulated in {RESULTS_PATH}")
