"""Interpretability experiment: does ZoneBoost's native, exact explain()
decomposition agree with a post-hoc SHAP approximation of the SAME fitted
model, and how does that compare to SHAP's own approximation cost/variance?

Two things are measured, both real and falsifiable rather than assumed:
  1. Faithfulness: explain(X) contributions sum EXACTLY to
     predict(X) - baseline (checked to float tolerance) -- true by
     construction, but worth verifying end-to-end on real data/real fits,
     not just unit tests on toy inputs.
  2. Agreement with SHAP: fit a SHAP KernelExplainer/TreeExplainer-style
     approximation on top of the same fitted ZoneBoostRegressor and compare
     its per-feature attributions against explain()'s exact ones -- a
     faithfulness check on SHAP's own approximation quality when the ground
     truth is actually known (unlike black-box models, where SHAP is the
     best available answer, not a checkable one).
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from datasets import load_breast_cancer, load_california_housing
from zoneboost import ZoneBoostClassifier, ZoneBoostRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 0


def to_raw_feature_attribution(contrib: pd.DataFrame, feature_names) -> pd.DataFrame:
    """Aggregate ZoneBoost's per-term explain() output (main effects named
    "a", pairwise interactions named "a x b") down to one column per raw
    input feature, so it's directly comparable to SHAP's per-feature
    attribution. An interaction term's contribution is split evenly (50/50)
    between its two constituent features -- the standard, disclosed
    convention for symmetrizing an interaction's credit across the features
    that jointly produced it (mirrors how SHAP interaction values are
    themselves symmetrized off-diagonal)."""
    out = pd.DataFrame(0.0, index=contrib.index, columns=list(feature_names))
    for col in contrib.columns:
        if col == "baseline":
            continue
        if " x " in col:
            a, b = col.split(" x ")
            out[a] += 0.5 * contrib[col]
            out[b] += 0.5 * contrib[col]
        else:
            out[col] += contrib[col]
    return out


def faithfulness_check_regression():
    X, y, task, cat_cols, name = load_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    # explain() returns a fresh 0..N-1 index regardless of X_test's own row
    # labels, so X_test is reset here to keep every downstream .loc lookup
    # (raw_contrib, X_small) aligned by position instead of by label.
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    model = ZoneBoostRegressor(random_state=RANDOM_STATE, n_rounds=100).fit(X_train, y_train)

    pred = model.predict(X_test)
    contrib = model.explain(X_test)  # includes a "baseline" column already
    reconstructed = contrib.sum(axis=1).to_numpy()
    max_abs_err = float(np.max(np.abs(reconstructed - pred)))

    print(f"[Faithfulness] {name}: max|sum(explain)+baseline - predict| = {max_abs_err:.2e}", flush=True)

    # SHAP KernelExplainer on the same fitted model (model-agnostic, since
    # ZoneBoost is not one of shap's built-in tree/linear explainer types).
    # KernelExplainer strips column names before calling the model function
    # (it perturbs a plain numpy array internally), but ZoneBoost's
    # predict() needs named columns to look up each column's own zone
    # table -- so the model function passed to SHAP restores column names
    # from the training frame before delegating to predict().
    def predict_fn(X_arr):
        return model.predict(pd.DataFrame(X_arr, columns=X_train.columns))

    background = shap.sample(X_train, 50, random_state=RANDOM_STATE)
    explainer = shap.KernelExplainer(predict_fn, background)
    X_small = X_test.iloc[:30]
    start = time.perf_counter()
    shap_values = explainer.shap_values(X_small, nsamples=200)
    shap_time = time.perf_counter() - start

    raw_contrib = to_raw_feature_attribution(contrib, X.columns)
    exact_contrib = raw_contrib.loc[X_small.index].to_numpy()
    # Per-row correlation between SHAP's approximate attribution vector and
    # ZoneBoost's exact one -- a global scalar isn't informative since sign
    # and rank matter more than the raw magnitude across very different
    # feature scales.
    row_corrs = [
        float(np.corrcoef(shap_values[i], exact_contrib[i])[0, 1])
        for i in range(len(X_small))
        if np.std(shap_values[i]) > 0 and np.std(exact_contrib[i]) > 0
    ]
    l1_gap = float(np.mean(np.abs(shap_values - exact_contrib).sum(axis=1)))
    print(f"[SHAP agreement] {name}: mean per-row attribution correlation = {np.mean(row_corrs):.4f} "
          f"(n={len(row_corrs)}), mean L1 gap = {l1_gap:.4f}, SHAP wall time for {len(X_small)} rows = {shap_time:.1f}s", flush=True)

    return {
        "dataset": name, "max_abs_reconstruction_error": max_abs_err,
        "shap_mean_row_correlation": float(np.mean(row_corrs)), "shap_mean_l1_gap": l1_gap,
        "shap_wall_time_s": shap_time, "n_rows_explained": len(X_small),
    }


def faithfulness_check_classification():
    X, y, task, cat_cols, name = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    model = ZoneBoostClassifier(random_state=RANDOM_STATE, n_rounds=100).fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    contrib = model.explain(X_test)  # log-odds space, includes "baseline" column
    logit = np.log(proba / (1 - proba))
    reconstructed_logit = contrib.sum(axis=1).to_numpy()
    max_abs_err = float(np.max(np.abs(reconstructed_logit - logit)))
    print(f"[Faithfulness] {name} (log-odds space): max abs error = {max_abs_err:.2e}", flush=True)
    return {"dataset": name, "max_abs_reconstruction_error_logodds": max_abs_err}


if __name__ == "__main__":
    results = []
    results.append(faithfulness_check_regression())
    results.append(faithfulness_check_classification())
    with open("results/interpretability.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/interpretability.json")
