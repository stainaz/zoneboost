"""Dataset loaders for the ZoneBoost paper experiment suite.

Every loader returns (X, y, task, cat_cols, name):
    X        : pandas DataFrame, categorical columns cast to 'category' dtype
    y        : numpy array (float for regression, int-coded for classification)
    task     : 'regression' | 'binary' | 'multiclass'
    cat_cols : list of column names to be treated as nominal categorical
    name     : short display name

Large datasets are subsampled (fixed seed) purely for CV wall-clock time across
~5 models x multiple ablation settings x 3 folds -- a disclosed speed
tradeoff, matching the existing benchmarks/compare_lightgbm.py methodology,
not a hyperparameter change favoring any model.
"""
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 0
SUBSAMPLE_CAP = 4000


def _subsample(X, y, cap=SUBSAMPLE_CAP, seed=RANDOM_STATE):
    if len(X) <= cap:
        return X.reset_index(drop=True), y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=cap, replace=False)
    return X.iloc[idx].reset_index(drop=True), np.asarray(y)[idx]


def load_california_housing():
    from sklearn.datasets import fetch_california_housing

    d = fetch_california_housing(as_frame=True)
    X, y = _subsample(d.data, d.target.to_numpy())
    return X, y, "regression", [], "California Housing"


def load_diabetes():
    from sklearn.datasets import load_diabetes

    d = load_diabetes(as_frame=True)
    return d.data.reset_index(drop=True), d.target.to_numpy(), "regression", [], "Diabetes"


def load_bike_sharing():
    from sklearn.datasets import fetch_openml

    d = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True, parser="auto")
    X = d.data.copy()
    cat_cols = ["season", "holiday", "workingday", "weather", "year"]
    for c in cat_cols:
        X[c] = X[c].astype("category")
    y = d.target.to_numpy(dtype=float)
    X, y = _subsample(X, y)
    return X, y, "regression", cat_cols, "Bike Sharing Demand"


def load_breast_cancer():
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer(as_frame=True)
    return d.data.reset_index(drop=True), d.target.to_numpy(), "binary", [], "Breast Cancer Wisconsin"


def load_wine():
    from sklearn.datasets import load_wine

    d = load_wine(as_frame=True)
    return d.data.reset_index(drop=True), d.target.to_numpy(), "multiclass", [], "Wine"


def load_titanic():
    from sklearn.datasets import fetch_openml

    d = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
    df = d.data.copy()
    df["survived"] = d.target
    keep = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]
    df = df[keep].dropna(subset=["survived"]).reset_index(drop=True)
    y = df["survived"].astype(int).to_numpy()
    X = df.drop(columns=["survived"])
    cat_cols = ["pclass", "sex", "embarked"]
    for c in cat_cols:
        X[c] = X[c].astype("category")
    X["age"] = X["age"].astype(float)
    X["fare"] = X["fare"].astype(float)
    return X, y, "binary", cat_cols, "Titanic"


ALL_DATASETS = [
    load_california_housing,
    load_diabetes,
    load_bike_sharing,
    load_breast_cancer,
    load_wine,
    load_titanic,
]

REGRESSION_DATASETS = [load_california_housing, load_diabetes, load_bike_sharing]
CLASSIFICATION_DATASETS = [load_breast_cancer, load_wine, load_titanic]


if __name__ == "__main__":
    for loader in ALL_DATASETS:
        X, y, task, cat_cols, name = loader()
        n_missing = int(X.isna().sum().sum())
        print(f"{name:28s} task={task:11s} n={len(X):5d} p={X.shape[1]:2d} cat={cat_cols} missing_cells={n_missing}")
