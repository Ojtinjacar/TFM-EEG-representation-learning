import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_expclr import (
    aggregate_to_sessions,
    bootstrap_ci,
    extract_embeddings,
    fit_probe,
    metrics_by_visit,
    paired_bootstrap_difference,
    session_metrics,
)
from models import EnhancedAttentionLSTM


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def windows():
    rows = []
    for s in range(10):
        for age in (6, 9, 16, 36)[: 2 + s % 3]:
            for block in (1, 2):
                for _ in range(5):
                    rows.append({"subject": f"S{s:02d}", "age": age, "block": block})
    return pd.DataFrame(rows)


def test_extraction_is_deterministic_and_leaves_encoder_frozen():
    model = EnhancedAttentionLSTM(input_size=250, hidden_size=32, n_channels=4, sfreq=250,
                                  lstm_hidden_size=16)
    X = np.random.randn(24, 4, 250).astype(np.float32)
    device = torch.device("cpu")

    before = [b.clone() for b in model.buffers()]
    first = extract_embeddings(model, X, device, batch_size=8)
    second = extract_embeddings(model, X, device, batch_size=8)
    after = list(model.buffers())

    assert np.array_equal(first, second), "the probe is not deterministic"
    for b0, b1 in zip(before, after):
        assert torch.equal(b0, b1), "the BatchNorm buffers were updated, so it is not frozen"


def test_extraction_is_invariant_to_batch_size():
    model = EnhancedAttentionLSTM(input_size=250, hidden_size=32, n_channels=4, sfreq=250,
                                  lstm_hidden_size=16)
    X = np.random.randn(20, 4, 250).astype(np.float32)
    a = extract_embeddings(model, X, torch.device("cpu"), batch_size=4)
    b = extract_embeddings(model, X, torch.device("cpu"), batch_size=20)
    assert np.allclose(a, b, atol=1e-5)


def test_probe_recovers_a_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    groups = np.repeat(np.arange(20), 10)
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(0, 0.1, 200)
    scaler, model = fit_probe(X, y, groups)
    pred = model.predict(scaler.transform(X))
    assert np.corrcoef(pred, y)[0, 1] > 0.95


def test_probe_is_deterministic():
    rng = np.random.default_rng(1)
    X, groups = rng.normal(size=(120, 5)), np.repeat(np.arange(12), 10)
    y = X[:, 0] + rng.normal(0, 0.2, 120)
    a = fit_probe(X, y, groups)[1].predict(StandardScalerFit(X))
    b = fit_probe(X, y, groups)[1].predict(StandardScalerFit(X))
    assert np.array_equal(a, b)


def StandardScalerFit(X):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X)


def test_probe_survives_a_single_training_group():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 4))
    scaler, model = fit_probe(X, rng.normal(size=30), np.zeros(30))
    assert model.predict(scaler.transform(X)).shape == (30,)


def test_aggregation_lands_on_sessions_not_subjects(windows):
    y_true = windows.age.values.astype(float)
    y_pred = y_true + np.random.normal(0, 1, len(windows))
    sessions = aggregate_to_sessions(windows, y_true, y_pred)
    expected = windows.groupby(["subject", "age", "block"]).ngroups
    assert len(sessions) == expected
    assert sessions.groupby(["subject", "age", "block"]).size().max() == 1


def test_aggregation_preserves_the_true_age_of_each_visit(windows):
    y_true = windows.age.values.astype(float)
    sessions = aggregate_to_sessions(windows, y_true, y_true.copy())
    assert set(sessions.y_true.unique()) <= {6.0, 9.0, 16.0, 36.0}


def test_aggregation_uses_the_median_and_resists_outliers(windows):
    y_true = windows.age.values.astype(float)
    y_pred = y_true.copy()
    y_pred[0] = 1e6
    sessions = aggregate_to_sessions(windows, y_true, y_pred)
    assert sessions.y_pred.max() < 100, "the median should absorb the extreme value"


def test_metrics_are_perfect_on_perfect_predictions(windows):
    y = windows.age.values.astype(float)
    m = session_metrics(aggregate_to_sessions(windows, y, y))
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_r2_is_zero_when_predicting_the_mean(windows):
    y = windows.age.values.astype(float)
    m = session_metrics(aggregate_to_sessions(windows, y, np.full_like(y, y.mean())))
    assert abs(m["r2"]) < 0.05


def test_shuffling_labels_across_subjects_destroys_r2(windows):
    rng = np.random.default_rng(3)
    y = windows.age.values.astype(float)
    subj = windows.subject.values
    mapping = dict(zip(np.unique(subj), rng.permutation(np.unique(y).repeat(4)[:len(np.unique(subj))])))
    y_shuffled = np.array([mapping[s] for s in subj], dtype=float)
    m = session_metrics(aggregate_to_sessions(windows, y_shuffled, y))
    assert m["r2"] < 0.3, "predicting the real age from shuffled labels should not work"


def test_metrics_by_visit_covers_every_visit(windows):
    y = windows.age.values.astype(float)
    tab = metrics_by_visit(aggregate_to_sessions(windows, y, y + 1))
    assert set(tab.age) == set(windows.age.unique())
    assert (tab.mae > 0).all()


def test_bootstrap_interval_brackets_the_observed_value(windows):
    rng = np.random.default_rng(4)
    y = windows.age.values.astype(float)
    sessions = aggregate_to_sessions(windows, y, y + rng.normal(0, 2, len(y)))
    obs = session_metrics(sessions)["mae"]
    low, high = bootstrap_ci(sessions, n_boot=200)
    assert low <= obs <= high


def test_paired_bootstrap_detects_a_clear_difference(windows):
    rng = np.random.default_rng(5)
    y = windows.age.values.astype(float)
    good = aggregate_to_sessions(windows, y, y + rng.normal(0, 0.5, len(y)))
    bad = aggregate_to_sessions(windows, y, y + rng.normal(0, 6, len(y)))
    res = paired_bootstrap_difference(good, bad, n_boot=300)
    assert res["diff"] < 0, "the good method must have the lower MAE"
    assert res["ci_high"] < 0, "the interval must exclude zero"


def test_paired_bootstrap_finds_no_difference_between_identical_methods(windows):
    rng = np.random.default_rng(6)
    y = windows.age.values.astype(float)
    same = aggregate_to_sessions(windows, y, y + rng.normal(0, 1, len(y)))
    res = paired_bootstrap_difference(same, same.copy(), n_boot=200)
    assert res["diff"] == pytest.approx(0.0)
    assert res["ci_low"] <= 0 <= res["ci_high"]
