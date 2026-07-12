"""Model evidence cards: a compact, JSON-serializable snapshot of a fitted
`ZoneBoostRegressor` -- zones/boundaries, per-term support/shrinkage,
constraint declarations, calibration/conformal coverage, unsupported
regions, and reproducibility info -- assembled entirely from attributes
the model already exposes after `fit`. Pure aggregation, no new modeling
math: every number here is read off `rounds_`, `get_params()`, or a
method (`feature_importance`, `_observed_range`) this session's earlier
items already added.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ._drift import _observed_range
from ._reliability import _find_term_key
from ._version import __version__

__all__ = ["evidence_card"]


def _jsonable(obj):
    """Recursively coerce numpy/pandas scalars and non-JSON containers
    (``set``/``frozenset``, tuple dict keys) into plain Python types --
    the whole point of a *machine-readable* card is that ``json.dumps``
    never trips over it."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    return obj


def _dataset_fingerprint(X: pd.DataFrame) -> dict:
    row_hash = pd.util.hash_pandas_object(X, index=False)
    digest = hashlib.sha256(row_hash.to_numpy().tobytes()).hexdigest()
    return {
        "n_rows": int(len(X)),
        "n_columns": int(X.shape[1]),
        "columns": list(X.columns),
        "dtypes": {col: str(dt) for col, dt in X.dtypes.items()},
        "hash": digest,
    }


def _term_diagnostics_summary(rounds: list, kind: str, key, shrinkage_m: float):
    """Aggregates a term's own per-zone/cell ``counts`` array (already
    stored per round when ``track_reliability=True``) across every round
    it appeared in -- ``None`` if the model wasn't fit with
    ``track_reliability=True`` (``diagnostics`` is ``None`` on every
    round in that case). No ``X`` needed: unlike
    :func:`zoneboost._reliability.explain_reliability`, this doesn't look
    up any specific row's own zone, just the raw stored counts."""
    term_id = frozenset(key) if isinstance(key, tuple) else frozenset([key])
    all_counts = []
    n_present = 0
    for round_ in rounds:
        if round_["diagnostics"] is None:
            return None
        term_dict = round_[kind]
        dkey = key if key in term_dict else _find_term_key(term_dict, term_id)
        if dkey is None:
            continue
        n_present += 1
        all_counts.append(round_["diagnostics"][kind][dkey]["counts"].ravel())
    if not all_counts:
        return None
    counts = np.concatenate(all_counts)
    return {
        "mean_support_per_zone": float(counts.mean()),
        "mean_shrinkage_fraction": float((shrinkage_m / (counts + shrinkage_m)).mean()),
    }


def _term_name(key) -> str:
    return key if isinstance(key, str) else " x ".join(sorted(key))


def _conformal_summary(scores) -> dict:
    return {
        "n": int(len(scores)),
        "median": float(np.median(scores)),
        "p90": float(np.quantile(scores, 0.9)),
    }


def evidence_card(model, X: pd.DataFrame = None) -> dict:
    """See :meth:`zoneboost.ZoneBoostRegressor.evidence_card` -- the
    method is a thin wrapper around this standalone function."""
    rounds = model.rounds_
    shrinkage_m = model.shrinkage_m

    # zones: each column's own *last* round as a main effect -- a
    # representative snapshot, not a full per-round boundary history
    # (same disclosed precedent as items 6/7's own drift/hierarchical
    # reporting).
    zones = {}
    for col in model.predictor_names_:
        last_round = None
        n_present = 0
        for round_ in rounds:
            if col in round_["main_effects"]:
                n_present += 1
                last_round = round_
        if last_round is None:
            zones[col] = {"kind": None, "n_rounds_present": 0}
            continue
        zone_info_col = last_round["zone_info"][col]
        kind = zone_info_col[0]
        entry = {"kind": kind, "n_rounds_present": n_present}
        if kind == "continuous":
            entry["observed_range"] = _observed_range(model, col)
        else:
            entry["categories_seen"] = sorted(zone_info_col[1].keys())
        zones[col] = entry

    # terms: union of every main effect / interaction / triple across all
    # rounds, keyed by the same "A x B" name explain()/feature_importance()
    # already use.
    term_keys = {"main_effects": {}, "interactions": {}, "triples": {}}
    for round_ in rounds:
        for col in round_["main_effects"]:
            term_keys["main_effects"].setdefault(col, col)
        for key in round_["interactions"]:
            term_keys["interactions"].setdefault(frozenset(key), key)
        for key in round_["triples"]:
            term_keys["triples"].setdefault(frozenset(key), key)

    importance = model.feature_importance(X) if X is not None else None

    terms = {}
    n_rounds_total = len(rounds)
    for kind, keys in term_keys.items():
        for term_id, key in keys.items():
            name = _term_name(key)
            if kind == "main_effects":
                n_present = sum(1 for round_ in rounds if key in round_[kind])
            else:
                n_present = sum(1 for round_ in rounds if _find_term_key(round_[kind], term_id) is not None)
            diag = _term_diagnostics_summary(rounds, kind, key, shrinkage_m)
            terms[name] = {
                "kind": kind.rstrip("s") if kind != "main_effects" else "main_effect",
                "n_rounds_present": n_present,
                "n_rounds_total": n_rounds_total,
                "mean_abs_contribution": float(importance[name]) if importance is not None and name in importance.index else None,
                "mean_support_per_zone": diag["mean_support_per_zone"] if diag else None,
                "mean_shrinkage_fraction": diag["mean_shrinkage_fraction"] if diag else None,
            }

    continuous_cols = model._continuous_main_effect_columns()
    unsupported_regions = {col: _observed_range(model, col) for col in sorted(continuous_cols)}

    conformal_summary = _conformal_summary(model.conformal_scores_) if model.conformal_scores_ is not None else None

    card = {
        "zoneboost_version": __version__,
        "model_class": type(model).__name__,
        "reproducibility": {
            "params": model.get_params(),
            "random_state": model.random_state,
        },
        "dataset_fingerprint": _dataset_fingerprint(X) if X is not None else None,
        "fit_summary": {
            "n_rounds_fit": len(rounds),
            "best_n_rounds": model.best_n_rounds_,
            "baseline": model.baseline_,
            "final_train_rmse": model.train_rmse_[model.best_n_rounds_ - 1] if model.train_rmse_ else None,
            "final_val_rmse": model.val_rmse_[model.best_n_rounds_ - 1] if model.val_rmse_ else None,
        },
        "zones": zones,
        "terms": terms,
        "shrinkage": {
            "shrinkage_m": model.shrinkage_m,
            "boundary_shrinkage_m": model.boundary_shrinkage_m if model.adaptive_boundary_smoothing else None,
            "track_reliability_enabled": model.track_reliability,
        },
        "constraints": {
            "monotonic_constraints": model.monotonic_constraints_,
            "convexity_constraints": model.convexity_constraints_,
            "bounded_effects": model.bounded_effects_,
            "forbidden_interactions": sorted(sorted(pair) for pair in model.forbidden_interactions_),
            "group_col": model.group_col_,
            "n_effect_overrides": len(model.effect_overrides_),
        },
        "calibration": {
            "loss": model.loss,
            "quantile": model.quantile if model.loss == "quantile" else None,
            "conformal_scores_available": model.conformal_scores_ is not None,
            "conformal_scores_summary": conformal_summary,
        },
        "unsupported_regions": unsupported_regions,
    }
    return _jsonable(card)
