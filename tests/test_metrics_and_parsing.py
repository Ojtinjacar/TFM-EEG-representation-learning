import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from downstream import subject_level_metrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_run_downstream():
    spec = importlib.util.spec_from_file_location(
        "run_downstream", os.path.join(ROOT, "run_downstream.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# subject_level_metrics
# ---------------------------------------------------------------------------

def test_degenerate_subject_yields_nan_not_rmse():
    df = pd.DataFrame({
        "subject": ["A"] * 3,
        "y_true": [36.0, 36.0, 36.0],
        "y_pred": [30.0, 31.0, 29.0],
    })
    metrics = subject_level_metrics(df)
    assert np.isnan(metrics.loc[0, "nrmse"])
    assert np.isnan(metrics.loc[0, "r2"])
    assert metrics.loc[0, "rmse"] > 0


def test_mixed_subjects_average_skips_degenerates():
    df = pd.DataFrame({
        "subject": ["A", "A", "B", "B"],
        "y_true": [6.0, 36.0, 9.0, 9.0],
        "y_pred": [7.0, 30.0, 10.0, 8.0],
    })
    metrics = subject_level_metrics(df)
    # Subject A has variation (finite nRMSE); subject B is degenerate (NaN).
    assert np.isfinite(metrics["nrmse"]).sum() == 1
    # pandas mean() skips NaN: the aggregate equals subject A's value.
    assert np.isclose(metrics["nrmse"].mean(), metrics["nrmse"].dropna().iloc[0])


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------

def test_parse_output_accepts_nan_metrics():
    rd = _load_run_downstream()
    output = (
        "[tag] Test loss=1.0, Test nRMSE (Subject-Avg)=nan, "
        "Test RMSE (Subject-Avg)=5.12, Test R2 (Subject-Avg)=nan\n"
        "SUBJECT_AVG_PRED: 36.000000 30.500000\n"
        "SESSION_AVG_PRED: B010 36.000000 30.500000\n"
    )
    nrmse, r2, rmse, subject_avgs, session_avgs = rd.parse_output(output)
    assert np.isnan(nrmse) and np.isnan(r2)
    assert rmse == 5.12
    assert subject_avgs == [(36.0, 30.5)]
    assert session_avgs == [("B010", 36.0, 30.5)]


def test_parse_output_regular_metrics_and_negative_r2():
    rd = _load_run_downstream()
    output = (
        "[tag] Test loss=1.0, Test nRMSE (Subject-Avg)=1.8321, "
        "Test RMSE (Subject-Avg)=13.37, Test R2 (Subject-Avg)=-2.5501\n"
        "SUBJECT_AVG_PRED: 6.000000 9.100000\n"
    )
    nrmse, r2, rmse, subject_avgs, session_avgs = rd.parse_output(output)
    assert nrmse == 1.8321
    assert r2 == -2.5501
    assert rmse == 13.37
    assert subject_avgs == [(6.0, 9.1)]
    assert session_avgs == []


def test_parse_output_unparseable_returns_none():
    rd = _load_run_downstream()
    nrmse, r2, rmse, subject_avgs, session_avgs = rd.parse_output("garbage output")
    assert nrmse is None and r2 is None and rmse is None
    assert subject_avgs == [] and session_avgs == []
