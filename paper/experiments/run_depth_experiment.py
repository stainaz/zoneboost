"""Depth-based typicality scoring (DepthTransformer): does a row's
Mahalanobis-distance-based "coreness" actually separate genuinely atypical
rows from the bulk of a real dataset?

Real datasets have no ground-truth "this row is an outlier" label, so this
follows the standard contaminated-data protocol for evaluating unsupervised
outlier/depth scores: inject a small, disclosed fraction of synthetic
extreme rows into each real dataset, fit DepthTransformer on the combined
(numeric-only) feature set, and check whether its score actually separates
injected rows from the real ones via ROC-AUC -- not merely a plausibility
argument, a checkable number against a known label.

Injection: each injected row starts as a real row (typical values in every
column), then a random subset of 1-3 numeric columns is overwritten with an
extreme value (mean +/- U(2.5, 4) standard deviations, sign at random) --
a partial anomaly (a few fields go haywire, the rest stay normal), which is
both more realistic than perturbing every column at once and a harder
detection task. Two earlier, easier versions of this experiment (all
columns perturbed simultaneously, at 6-10 sigma and then at 2.5-4 sigma)
both produced AUC >= 0.998 on every dataset regardless of magnitude --
perturbing every dimension at once makes joint Mahalanobis distance blow up
by construction (a curse-of-dimensionality effect, not a property of
DepthTransformer), which is too easy to be an informative result. Reported
here instead: the harder, more realistic partial-anomaly protocol.
"""
import json
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from datasets import load_bike_sharing, load_breast_cancer, load_california_housing, load_diabetes
from zoneboost import DepthTransformer

warnings.filterwarnings("ignore")
RANDOM_STATE = 0
CONTAMINATION_FRACTION = 0.02


def _inject_outliers(X, seed):
    rng = np.random.default_rng(seed)
    numeric_cols = [c for c in X.columns if not str(X[c].dtype) in ("category", "object", "bool")]
    n_outliers = max(5, round(CONTAMINATION_FRACTION * len(X)))
    means = X[numeric_cols].mean()
    stds = X[numeric_cols].std()

    # Each injected row is a real row (typical everywhere) with 1-3 randomly
    # chosen numeric columns overwritten by an extreme value -- a partial
    # anomaly, not every dimension perturbed at once (see module docstring).
    sample_idx = rng.choice(len(X), size=n_outliers, replace=len(X) < n_outliers)
    outlier_df = X.iloc[sample_idx].copy().reset_index(drop=True)
    outlier_df[numeric_cols] = outlier_df[numeric_cols].astype(float)
    max_perturbed = min(3, len(numeric_cols))
    for i in range(n_outliers):
        n_perturb = rng.integers(1, max_perturbed + 1)
        cols_to_perturb = rng.choice(numeric_cols, size=n_perturb, replace=False)
        for c in cols_to_perturb:
            magnitude = rng.uniform(2.5, 4)
            sign = rng.choice([-1.0, 1.0])
            outlier_df.loc[i, c] = means[c] + sign * magnitude * stds[c]

    combined = pd_concat_reset(X, outlier_df)
    labels = np.concatenate([np.zeros(len(X)), np.ones(n_outliers)])
    return combined, labels, numeric_cols


def pd_concat_reset(a, b):
    import pandas as pd

    return pd.concat([a, b], ignore_index=True)


def depth_experiment(loader):
    X, y, task, cat_cols, name = loader()
    X_combined, labels, numeric_cols = _inject_outliers(X, seed=RANDOM_STATE)

    depth = DepthTransformer(random_state=RANDOM_STATE).fit(X_combined)
    out = depth.transform(X_combined)
    distance_col = [c for c in out.columns if c.endswith("__depth_distance")][0]
    coreness_col = [c for c in out.columns if c.endswith("__coreness")][0]

    auc = roc_auc_score(labels, out[distance_col])
    mean_coreness_normal = float(out.loc[labels == 0, coreness_col].mean())
    mean_coreness_injected = float(out.loc[labels == 1, coreness_col].mean())

    print(
        f"[Depth] {name}: n_numeric_cols={len(numeric_cols)}  n_injected={int(labels.sum())}  "
        f"AUC={auc:.4f}  mean_coreness(normal)={mean_coreness_normal:.4f}  "
        f"mean_coreness(injected)={mean_coreness_injected:.4f}",
        flush=True,
    )
    return {
        "dataset": name,
        "n_numeric_cols": len(numeric_cols),
        "n_rows": len(X),
        "n_injected": int(labels.sum()),
        "auc": float(auc),
        "mean_coreness_normal": mean_coreness_normal,
        "mean_coreness_injected": mean_coreness_injected,
    }


if __name__ == "__main__":
    results = []
    for loader in (load_california_housing, load_diabetes, load_bike_sharing, load_breast_cancer):
        results.append(depth_experiment(loader))

    with open("results/depth_transformer.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved results/depth_transformer.json")
