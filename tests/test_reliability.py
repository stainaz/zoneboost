import pandas as pd
import pytest

from zoneboost._reliability import evidence_report


def _make_reliability(support_values, extrapolation_frac=None):
    n = len(support_values)
    data = {
        "support": support_values,
        "shrinkage_fraction": [0.0] * n,
        "cross_fold_std": [0.0] * n,
        "n_rounds_present": [10] * n,
    }
    if extrapolation_frac is not None:
        data["extrapolation_frac"] = extrapolation_frac
        data["boundary_weight"] = [0.0] * n
    return pd.DataFrame(data)


def test_evidence_report_high_support_gives_high_score():
    contrib = pd.DataFrame({"baseline": [1.0, 1.0], "x": [2.0, 3.0]})
    reliability = {"x": _make_reliability([500.0, 500.0], extrapolation_frac=[0.0, 0.0])}
    report = evidence_report(contrib, reliability, shrinkage_m=10.0)
    assert (report["evidence_score"] == 1.0).all()
    assert (report["evidence_quality"] == "High").all()
    assert not report["extrapolating"].any()
    assert not report["unobserved_cell"].any()
    assert (report["pct_contribution_from_sparse_cells"] == 0.0).all()


def test_evidence_report_low_support_term_flagged_sparse():
    contrib = pd.DataFrame({"baseline": [1.0], "x": [4.0], "z": [1.0]})
    reliability = {
        "x": _make_reliability([2.0], extrapolation_frac=[0.0]),  # well below shrinkage_m=10
        "z": _make_reliability([500.0], extrapolation_frac=[0.0]),
    }
    report = evidence_report(contrib, reliability, shrinkage_m=10.0)
    # x contributes 4.0, z contributes 1.0 -- 4/5 = 0.8 of total |contribution| is sparse
    assert report["pct_contribution_from_sparse_cells"].iloc[0] == pytest.approx(0.8)
    assert not report["extrapolating"].iloc[0]
    assert report["evidence_score"].iloc[0] == pytest.approx(0.2)
    assert report["evidence_quality"].iloc[0] == "Low"


def test_evidence_report_extrapolating_halves_score():
    contrib = pd.DataFrame({"baseline": [1.0], "x": [2.0]})
    reliability = {"x": _make_reliability([500.0], extrapolation_frac=[1.0])}
    report = evidence_report(contrib, reliability, shrinkage_m=10.0)
    assert report["extrapolating"].iloc[0]
    assert report["evidence_score"].iloc[0] == pytest.approx(0.5)
    assert report["evidence_quality"].iloc[0] == "Low"


def test_evidence_report_unobserved_cell_flagged_below_one():
    contrib = pd.DataFrame({"baseline": [1.0], "x": [2.0]})
    reliability = {"x": _make_reliability([0.0], extrapolation_frac=[0.0])}
    report = evidence_report(contrib, reliability, shrinkage_m=10.0)
    assert report["unobserved_cell"].iloc[0]


def test_evidence_report_custom_sparse_threshold():
    contrib = pd.DataFrame({"baseline": [1.0], "x": [2.0]})
    reliability = {"x": _make_reliability([20.0], extrapolation_frac=[0.0])}
    default_report = evidence_report(contrib, reliability, shrinkage_m=10.0)
    strict_report = evidence_report(contrib, reliability, shrinkage_m=10.0, sparse_threshold=50.0)
    assert default_report["pct_contribution_from_sparse_cells"].iloc[0] == 0.0  # 20 >= 10
    assert strict_report["pct_contribution_from_sparse_cells"].iloc[0] == 1.0  # 20 < 50


def test_evidence_report_ignores_bookkeeping_columns():
    contrib = pd.DataFrame({"baseline": [1.0], "_softmax_centering": [5.0], "x": [2.0]})
    reliability = {"x": _make_reliability([500.0], extrapolation_frac=[0.0])}
    report = evidence_report(contrib, reliability, shrinkage_m=10.0)
    assert report["pct_contribution_from_sparse_cells"].iloc[0] == 0.0
