"""Turn results/*.json into LaTeX table snippets, ready to paste over the
% RESULTS-PLACEHOLDER-* markers in paper.tex."""
import json

import pandas as pd


def main_benchmark_tables():
    with open("results/main_benchmark.json") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    reg = df[df["status"] == "ok"][df.columns.intersection(
        ["dataset", "model", "rmse_mean", "rmse_std", "r2_mean", "r2_std", "fit_time_s"]
    )].dropna(subset=["rmse_mean"]) if "rmse_mean" in df.columns else pd.DataFrame()
    clf = df[df["status"] == "ok"][df.columns.intersection(
        ["dataset", "model", "accuracy_mean", "accuracy_std", "auc_mean", "auc_std",
         "f1_macro_mean", "logloss_mean", "fit_time_s"]
    )].dropna(subset=["accuracy_mean"]) if "accuracy_mean" in df.columns else pd.DataFrame()

    failed = df[df["status"] != "ok"]
    print("=== REGRESSION (main benchmark) ===")
    if not reg.empty:
        print(reg.to_string(index=False))
    print("\n=== CLASSIFICATION (main benchmark) ===")
    if not clf.empty:
        print(clf.to_string(index=False))
    if not failed.empty:
        print("\n=== FAILURES ===")
        print(failed[["dataset", "model", "status"]].to_string(index=False))


def ablation_table():
    with open("results/ablation.json") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"]
    failed = df[df["status"] != "ok"]
    print("\n=== ABLATION (ok rows) ===")
    print(ok.to_string(index=False))
    if not failed.empty:
        print("\n=== ABLATION FAILURES ===")
        print(failed[["axis", "dataset", "variant", "status"]].to_string(index=False))


def aux_tables():
    with open("results/interpretability.json") as f:
        interp = json.load(f)
    print("\n=== INTERPRETABILITY ===")
    for r in interp:
        print(r)

    with open("results/conformal_calibration_drift.json") as f:
        ccd = json.load(f)
    print("\n=== CONFORMAL ===")
    for r in ccd["conformal"]:
        print(r)
    print("\n=== CALIBRATION ===")
    print(ccd["calibration"])
    print("\n=== DRIFT ===")
    print(ccd["drift"])

    with open("results/depth_transformer.json") as f:
        depth = json.load(f)
    print("\n=== DEPTH TRANSFORMER ===")
    for r in depth:
        print(r)


if __name__ == "__main__":
    main_benchmark_tables()
    ablation_table()
    aux_tables()
