import importlib.util
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_run_downstream():
    spec = importlib.util.spec_from_file_location(
        "run_downstream_for_test", os.path.join(ROOT, "run_downstream.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_experiment_does_not_crash_the_fold(monkeypatch, capsys):
    """Regression test for the fold_failures NameError (audit finding, ALTO).

    A downstream experiment returning no predictions must be recorded as a
    failure and must NOT abort the fold: with the bug, execute_fold raised
    NameError and every already-collected result of the run was lost.
    """
    rd = _load_run_downstream()

    monkeypatch.setattr(
        rd, "run_downstream_experiment",
        lambda **kwargs: (None, None, None, [], []),
    )

    args = SimpleNamespace(
        zone="all", frequency="all", no_skip=False, allow_legacy=False,
        base_seed=1234, vae_beta=1.0, vae_free_bits=0.0,
        methods=["PCA"],
    )
    results = rd.execute_fold(
        fold_idx=0,
        train_subjects=["A", "B"],
        test_subjects=["C"],
        args=args,
        targets=["age"],
        eval_modes=["linear_probe"],
        target_subject_dict={"age": ["A", "B", "C"]},
    )

    assert results == []
    out = capsys.readouterr().out
    assert "FAILURES" in out
    assert "NameError" not in out
