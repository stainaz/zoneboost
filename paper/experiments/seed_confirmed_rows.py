"""Seed results/main_benchmark.json with rows that are fully reconstructable
without data loss:

- California Housing, Diabetes (regression): every field shown in the
  regression JSON schema (rmse/r2 mean+std, fit_time_s) was printed to
  console in the original run, so these are a faithful transcription, not
  an approximation. predict_time_s is not used anywhere in the paper and is
  omitted rather than invented.
- Bike Sharing Demand: NOT reconstructed from old console text -- these are
  the literal return values from a real, just-completed full 3-fold CV run
  (all 5 models, including a real CatBoost result; see
  run_regression_dataset(load_bike_sharing) executed 2026-07-20 to
  investigate the earlier clone() failure, which did not reproduce).

Breast Cancer Wisconsin's classification row is intentionally NOT seeded:
the console output from the interrupted run never printed log_loss, which
the paper's classification table needs, so it must be re-run for real
rather than reconstructed with a missing field.
"""
import json
import os

RESULTS_PATH = "results/main_benchmark.json"

rows = [
    {"dataset": "California Housing", "model": "ZoneBoost", "rmse_mean": 0.5328, "rmse_std": 0.0134, "r2_mean": 0.7836, "r2_std": 0.0037, "fit_time_s": 7.90, "status": "ok"},
    {"dataset": "California Housing", "model": "XGBoost", "rmse_mean": 0.5482, "rmse_std": 0.0096, "r2_mean": 0.7708, "r2_std": 0.0051, "fit_time_s": 0.11, "status": "ok"},
    {"dataset": "California Housing", "model": "LightGBM", "rmse_mean": 0.5143, "rmse_std": 0.0131, "r2_mean": 0.7984, "r2_std": 0.0044, "fit_time_s": 0.69, "status": "ok"},
    {"dataset": "California Housing", "model": "CatBoost", "rmse_mean": 0.4949, "rmse_std": 0.0176, "r2_mean": 0.8133, "r2_std": 0.0079, "fit_time_s": 4.48, "status": "ok"},
    {"dataset": "California Housing", "model": "EBM", "rmse_mean": 0.5171, "rmse_std": 0.0165, "r2_mean": 0.7962, "r2_std": 0.0064, "fit_time_s": 197.49, "status": "ok"},
    {"dataset": "Diabetes", "model": "ZoneBoost", "rmse_mean": 57.16, "rmse_std": 2.00, "r2_mean": 0.4420, "r2_std": 0.0797, "fit_time_s": 7.27, "status": "ok"},
    {"dataset": "Diabetes", "model": "XGBoost", "rmse_mean": 64.9994, "rmse_std": 3.9314, "r2_mean": 0.2832, "r2_std": 0.0731, "fit_time_s": 0.08, "status": "ok"},
    {"dataset": "Diabetes", "model": "LightGBM", "rmse_mean": 60.1712, "rmse_std": 4.4386, "r2_mean": 0.3816, "r2_std": 0.1019, "fit_time_s": 0.03, "status": "ok"},
    {"dataset": "Diabetes", "model": "CatBoost", "rmse_mean": 58.9616, "rmse_std": 3.4057, "r2_mean": 0.4091, "r2_std": 0.0687, "fit_time_s": 3.70, "status": "ok"},
    {"dataset": "Diabetes", "model": "EBM", "rmse_mean": 55.2127, "rmse_std": 1.6490, "r2_mean": 0.4810, "r2_std": 0.0593, "fit_time_s": 130.13, "status": "ok"},
    {"dataset": "Bike Sharing Demand", "model": "ZoneBoost", "rmse_mean": 56.9181, "rmse_std": 1.6063, "r2_mean": 0.9015, "r2_std": 0.0048, "fit_time_s": 23.43, "status": "ok"},
    {"dataset": "Bike Sharing Demand", "model": "XGBoost", "rmse_mean": 50.3791, "rmse_std": 1.1047, "r2_mean": 0.9228, "r2_std": 0.0045, "fit_time_s": 0.20, "status": "ok"},
    {"dataset": "Bike Sharing Demand", "model": "LightGBM", "rmse_mean": 49.0203, "rmse_std": 1.2080, "r2_mean": 0.9269, "r2_std": 0.0037, "fit_time_s": 1.33, "status": "ok"},
    {"dataset": "Bike Sharing Demand", "model": "CatBoost", "rmse_mean": 47.1181, "rmse_std": 1.2794, "r2_mean": 0.9325, "r2_std": 0.0032, "fit_time_s": 75.95, "status": "ok"},
    {"dataset": "Bike Sharing Demand", "model": "EBM", "rmse_mean": 57.7233, "rmse_std": 0.9924, "r2_mean": 0.8987, "r2_std": 0.0027, "fit_time_s": 386.17, "status": "ok"},
]

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    existing = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            existing = json.load(f)
    have = {(r["dataset"], r["model"]) for r in rows}
    existing = [r for r in existing if (r["dataset"], r["model"]) not in have]
    all_rows = existing + rows
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Seeded {len(rows)} confirmed rows; {len(all_rows)} total rows in {RESULTS_PATH}")
