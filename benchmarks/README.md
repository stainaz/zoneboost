# Benchmarks

A reproducible script comparing zoneboost against LightGBM -- not part of the
installable `zoneboost` package, so this dependency is never required for
normal use.

InterpretML's Explainable Boosting Machine (EBM) -- architecturally the
closest existing interpretable model -- was deliberately left out of the
script itself; see the module docstring in `compare_lightgbm.py` for why
(its default parallel outer-bagging had environment-specific process-spawn
overhead that swamped wall-clock time independent of real compute cost, so
it isn't reported as a misleading number).

## Setup

```bash
pip install -e ".[benchmark]"
```

## Running

```bash
python benchmarks/compare_lightgbm.py
```

Prints cross-validated accuracy/timing tables for a regression dataset
(California Housing, subsampled for speed) and a classification dataset
(Breast Cancer Wisconsin), plus a demonstration of
`ZoneBoostRegressor.explain()`'s exact interaction attribution on synthetic
data with a known interaction (LightGBM has no built-in equivalent).

Real results and commentary are written up in the main
[README](../README.md#benchmarks) and [docs site](https://stainaz.github.io/zoneboost/benchmarks.html).
