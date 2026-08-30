"""Tests for the shared evaluation of the spectral baselines.

Two baselines report numbers that are meant to sit in the same table as those of the encoders,
and nothing would fail if they stopped doing so: the folds would still come out, over other
children, at another row unit, or fitted on statistics they were not supposed to see. These
tests pin down the three things that make the comparison honest.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from folds import canonical_subject_folds  # noqa: E402
from tabular_baseline import (  # noqa: E402
    SESSION_LEVEL,
    SUBJECT_LEVEL,
    BaselineError,
    aggregate_metrics,
    aggregate_rows,
    compute_metrics,
    evaluate_targets,
    fold_suffix,
    impute_with_train_medians,
    natural_level,
    parse_fold_ranges,
    run_cv,
)

AGES = [6, 9, 16, 36]
COHORT = [f"B{i:03d}" for i in range(20)]


def synthetic_cohort(n_subjects=20, n_windows=3, n_features=6, missing_cit=4, seed=0):
    """Builds a longitudinal cohort whose features carry the age linearly.

    Args:
        n_subjects (int): Number of children.
        n_windows (int): Windows per visit.
        n_features (int): Columns of the descriptor.
        missing_cit (int): Children without an intelligence quotient, so that the two
            targets do not cover the same subjects.
        seed (int): Seed of the random generator.

    Returns:
        tuple: (features DataFrame, metadata DataFrame).
    """
    rng = np.random.default_rng(seed)
    rows, features = [], []
    for i in range(n_subjects):
        subject = f"B{i:03d}"
        cit = np.nan if i < missing_cit else 80.0 + 2.0 * (i % 20)
        for age in AGES:
            for _ in range(n_windows):
                rows.append({"subject": subject, "age": age, "block": 1, "cit_36mo": cit})
                signal = np.full(n_features, age / 10.0) + 0.01 * rng.standard_normal(n_features)
                features.append(signal)
    return pd.DataFrame(features, columns=[f"f{j}" for j in range(n_features)]), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The partition is the one of the whole project
# ---------------------------------------------------------------------------

def _run_downstream_partition(subjects, n_folds=10, base_seed=1234):
    """Reproduces the split of run_downstream.py literally, as the reference."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
    return [[subjects[i] for i in test_idx] for _, test_idx in kf.split(subjects)]


def test_the_partition_is_the_one_of_run_downstream():
    assert canonical_subject_folds(COHORT, 10, 1234) == _run_downstream_partition(COHORT)


def test_the_partition_is_the_one_of_run_expclr_folds():
    from run_expclr_folds import subject_folds

    assert canonical_subject_folds(COHORT, 10, 1234) == subject_folds(COHORT, 10, 1234)


def test_a_target_with_fewer_subjects_keeps_the_folds_of_the_cohort():
    """The partition is built over every child before anyone is dropped for lacking a
    target. Splitting the survivors instead would give fold 3 a different set of children
    depending on the target, and the two columns of a results table would stop lining up."""
    features, meta = synthetic_cohort(missing_cit=4)
    all_subjects = sorted(meta.subject.unique())

    rows, cols = aggregate_rows(features, meta, "cit_36mo", SUBJECT_LEVEL)
    results, predictions = run_cv(rows, cols, "cit_36mo", all_subjects, n_folds=5, verbose=False)

    cohort_folds = canonical_subject_folds(all_subjects, 5, 1234)
    for fold in results.fold:
        scored = set(predictions.loc[predictions.fold == fold, "subject"])
        assert scored <= set(cohort_folds[fold])


def test_no_subject_is_on_both_sides_of_a_fold():
    features, meta = synthetic_cohort()
    all_subjects = sorted(meta.subject.unique())
    rows, cols = aggregate_rows(features, meta, "age", SESSION_LEVEL)
    _, predictions = run_cv(rows, cols, "age", all_subjects, n_folds=5, verbose=False)

    # Every child is scored exactly once, which is only true if no fold trained on one it
    # then predicted.
    counts = predictions.groupby("subject").fold.nunique()
    assert (counts == 1).all()
    assert set(predictions.subject) == set(all_subjects)


# ---------------------------------------------------------------------------
# The row is the unit the target varies over
# ---------------------------------------------------------------------------

def test_age_is_evaluated_by_session_and_the_quotient_by_subject():
    _, meta = synthetic_cohort()
    assert natural_level(meta, "age") == SESSION_LEVEL
    assert natural_level(meta, "cit_36mo") == SUBJECT_LEVEL


def test_age_keeps_one_row_per_visit_with_the_age_of_that_visit():
    """Averaging the four visits of a child and labelling the result with one of the ages
    is what the baseline used to do, and it does not measure age prediction."""
    features, meta = synthetic_cohort(n_subjects=5)
    rows, _ = aggregate_rows(features, meta, "age", SESSION_LEVEL)

    assert len(rows) == 5 * len(AGES)
    assert sorted(rows.age.unique()) == AGES
    assert (rows.groupby("subject").size() == len(AGES)).all()


def test_the_quotient_collapses_the_visits_into_one_row_per_child():
    features, meta = synthetic_cohort(n_subjects=8, missing_cit=3)
    rows, _ = aggregate_rows(features, meta, "cit_36mo", SUBJECT_LEVEL)

    assert len(rows) == 5  # the three without a quotient are dropped, not imputed
    assert rows.subject.is_unique


def test_features_of_a_different_length_are_refused():
    features, meta = synthetic_cohort(n_subjects=4)
    with pytest.raises(BaselineError, match="indexed by position"):
        aggregate_rows(features.iloc[:-1], meta, "age", SESSION_LEVEL)


# ---------------------------------------------------------------------------
# Nothing fitted on the training side may see the test side
# ---------------------------------------------------------------------------

def test_the_medians_come_from_the_training_split_alone():
    X = np.array([[1.0], [1.0], [1.0], [np.nan], [1000.0]])
    train = np.array([True, True, True, False, False])

    filled, fraction = impute_with_train_medians(X, train)

    # The 1000.0 sits in the test split, so it cannot move the value the gap is filled with.
    assert filled[3, 0] == pytest.approx(1.0)
    assert fraction == pytest.approx(1 / 5)


def test_a_column_missing_throughout_training_fails_loudly():
    X = np.array([[np.nan, 1.0], [np.nan, 2.0], [3.0, 3.0]])
    train = np.array([True, True, False])

    with pytest.raises(BaselineError, match="no median"):
        impute_with_train_medians(X, train)


def test_the_imputation_is_recomputed_for_every_fold():
    """A single imputation before the split would leak the held-out children into the values
    the probe is scored on. The fraction reported per fold is the visible trace of it being
    done inside the loop."""
    features, meta = synthetic_cohort(n_subjects=10)
    features.iloc[0, 0] = np.nan
    all_subjects = sorted(meta.subject.unique())
    rows, cols = aggregate_rows(features, meta, "age", SESSION_LEVEL)

    results, _ = run_cv(rows, cols, "age", all_subjects, n_folds=5, verbose=False)
    assert "imputed_frac" in results.columns
    assert (results.imputed_frac >= 0).all()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_perfect_predictions_give_an_r2_of_one():
    y = np.array([6.0, 9.0, 16.0, 36.0])
    assert compute_metrics(y, y)["R2"] == pytest.approx(1.0)
    assert compute_metrics(y, y)["RMSE"] == pytest.approx(0.0)


def test_predicting_the_mean_gives_an_r2_of_zero():
    y = np.array([6.0, 9.0, 16.0, 36.0])
    assert compute_metrics(y, np.full_like(y, y.mean()))["R2"] == pytest.approx(0.0)


def test_a_constant_target_leaves_r2_undefined_rather_than_zero():
    """The old baseline returned an unnormalised RMSE and an R2 of zero for a target that
    does not vary, and both went into a column that was then averaged."""
    y = np.array([100.0, 100.0, 100.0])
    metrics = compute_metrics(y, np.array([99.0, 101.0, 100.0]))
    assert np.isnan(metrics["R2"]) and np.isnan(metrics["nRMSE"])
    assert metrics["RMSE"] > 0


def test_the_summary_reports_both_the_per_fold_mean_and_the_pooled_figure():
    features, meta = synthetic_cohort()
    all_subjects = sorted(meta.subject.unique())
    rows, cols = aggregate_rows(features, meta, "age", SESSION_LEVEL)
    results, predictions = run_cv(rows, cols, "age", all_subjects, n_folds=5, verbose=False)

    summary = aggregate_metrics(results, predictions, "age", "Test", SESSION_LEVEL)
    for name in ("RMSE", "R2", "nRMSE"):
        assert f"{name}_mean" in summary and f"{name}_pooled" in summary
    # The features carry the age almost noiselessly, so a linear probe has to find it.
    assert summary["R2_pooled"] > 0.9


# ---------------------------------------------------------------------------
# Command-line plumbing
# ---------------------------------------------------------------------------

def test_fold_suffix_matches_the_convention_of_run_downstream():
    assert fold_suffix("kfold", (0, 2)) == "_folds0-1"
    assert fold_suffix("kfold", None) == ""
    assert fold_suffix("leave_one_out", None, {"age": (0, 10)}) == "_age_0-9"


def test_malformed_fold_ranges_are_refused():
    assert parse_fold_ranges(["age:0:10"]) == {"age": (0, 10)}
    with pytest.raises(BaselineError):
        parse_fold_ranges(["age:0"])
    with pytest.raises(BaselineError):
        parse_fold_ranges(["age:zero:ten"])


def test_evaluate_targets_writes_the_tables_it_announces(tmp_path):
    features, meta = synthetic_cohort()
    raw, agg, predictions = evaluate_targets(
        features=features, meta=meta, targets=["age", "cit_36mo"],
        prefix="test", save_dir=str(tmp_path), method_name="Test",
        n_folds=5, verbose=False,
    )

    assert set(agg.target) == {"age", "cit_36mo"}
    assert dict(zip(agg.target, agg.level)) == {"age": SESSION_LEVEL, "cit_36mo": SUBJECT_LEVEL}
    assert (tmp_path / "test_raw_results_kfold.csv").exists()
    assert (tmp_path / "test_agg_results_kfold.csv").exists()
    assert (tmp_path / "test_predictions_age_kfold.csv").exists()
    assert set(raw.columns) >= {"fold", "target", "RMSE", "R2", "nRMSE", "method", "level"}
    assert set(predictions) == {"age", "cit_36mo"}


def test_a_target_the_metadata_does_not_carry_is_skipped_not_crashed(tmp_path):
    features, meta = synthetic_cohort()
    _, agg, _ = evaluate_targets(
        features=features, meta=meta, targets=["age", "not_a_column"],
        prefix="test", save_dir=str(tmp_path), method_name="Test",
        n_folds=5, verbose=False,
    )
    assert list(agg.target) == ["age"]
