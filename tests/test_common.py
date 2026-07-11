import numpy as np
import pandas as pd
import pytest

from zoneboost._common import resolve_categorical_features, resolve_monotonic_constraints


def _df():
    return pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
            "cat": ["a", "b", "a"],
        }
    )


def test_resolve_monotonic_constraints_empty_when_not_declared():
    X = _df()
    cats = resolve_categorical_features(X, None)
    assert resolve_monotonic_constraints(X, None, cats) == {}
    assert resolve_monotonic_constraints(X, {}, cats) == {}


def test_resolve_monotonic_constraints_resolves_names_and_indices():
    X = _df()
    cats = resolve_categorical_features(X, None)
    resolved = resolve_monotonic_constraints(X, {"x1": 1, 1: -1}, cats)
    assert resolved == {"x1": 1, "x2": -1}


def test_resolve_monotonic_constraints_invalid_direction_raises():
    X = _df()
    cats = resolve_categorical_features(X, None)
    with pytest.raises(ValueError):
        resolve_monotonic_constraints(X, {"x1": 2}, cats)


def test_resolve_monotonic_constraints_drops_categorical_column_silently():
    X = _df()
    cats = resolve_categorical_features(X, None)
    assert "cat" in cats
    resolved = resolve_monotonic_constraints(X, {"cat": 1, "x1": 1}, cats)
    assert resolved == {"x1": 1}
