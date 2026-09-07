"""Tests for the fold partition of the ExpCLR evaluation.

The point of this script sharing a partition with run_downstream.py is that its figures can
be read next to those of the other experiments. That only holds if both really do split the
cohort the same way, which is what these tests pin down: nothing else in the code would fail
if the two drifted apart, the numbers would simply stop being comparable without saying so.
"""

import os
import sys

import pytest
from sklearn.model_selection import KFold

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from run_expclr_folds import subject_folds  # noqa: E402


COHORT = [f"B{i:03d}" for i in range(45)]


def _run_downstream_partition(subjects, n_folds=10, base_seed=1234):
    """Reproduces run_downstream.py:1281 literally, as the reference to compare against."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
    return [[subjects[i] for i in test_idx] for _, test_idx in kf.split(subjects)]


def test_the_partition_is_the_one_of_the_rest_of_the_pipeline():
    assert subject_folds(COHORT, 10, 1234) == _run_downstream_partition(COHORT)


def test_the_defaults_match_the_ones_of_run_downstream():
    """A different default here would silently produce an incomparable partition."""
    import run_expclr_folds as rf

    parser_defaults = {}
    for line in open(os.path.join(ROOT, "run_expclr_folds.py"), encoding="utf-8"):
        for flag, name in (("--n_folds", "n_folds"), ("--base_seed", "base_seed")):
            if f'"{flag}"' in line:
                parser_defaults[name] = int(line.split("default=")[1].split(",")[0])
    assert parser_defaults == {"n_folds": 10, "base_seed": 1234}
    assert rf.subject_folds(COHORT, **parser_defaults) == _run_downstream_partition(COHORT)


def test_every_subject_is_held_out_exactly_once():
    folds = subject_folds(COHORT, 10, 1234)
    held_out = [s for fold in folds for s in fold]
    assert sorted(held_out) == sorted(COHORT)
    assert len(held_out) == len(set(held_out))


def test_the_partition_does_not_depend_on_the_tuning_subjects():
    """Withdrawing a subject from scoring must not move anybody else between folds.

    This is why the folds are built before the tuning subjects are removed. Building them
    over the remaining subjects instead would reshuffle the whole cohort, and the fold a
    subject lands in would depend on how many were tuned on.
    """
    tuning = {"B000", "B017", "B032"}
    folds = subject_folds(COHORT, 10, 1234)
    without_tuning = [[s for s in fold if s not in tuning] for fold in folds]

    reference = _run_downstream_partition(COHORT)
    for fold, expected in zip(without_tuning, reference):
        assert fold == [s for s in expected if s not in tuning]


def test_a_different_seed_gives_a_different_partition():
    """Guards the comparison itself: if the seed did not matter, the first test would pass
    for the wrong reason."""
    assert subject_folds(COHORT, 10, 1234) != subject_folds(COHORT, 10, 42)


@pytest.mark.parametrize("n_folds", [5, 10, 15])
def test_the_number_of_folds_is_honoured(n_folds):
    assert len(subject_folds(COHORT, n_folds, 1234)) == n_folds


def test_run_downstream_still_partitions_the_way_this_mirrors_it():
    """The reference above is a copy, so it cannot notice run_downstream.py drifting.

    Reading its source is what does. If that call or its defaults change, the two scripts
    stop sharing a partition, and the failure should surface here rather than as two sets
    of numbers that are quietly no longer comparable.
    """
    source = open(os.path.join(ROOT, "run_downstream.py"), encoding="utf-8").read()
    assert ("KFold(n_splits=args.n_folds, shuffle=True, random_state=args.base_seed)"
            in source), "run_downstream.py no longer builds its folds the way this mirrors"
    assert "kf.split(unique_subjects)" in source, "it no longer splits over the subjects"
    assert "unique_subjects = sorted(list(all_subjects_set))" in source, \
        "the subjects are no longer sorted before splitting, so the order differs"

    defaults = {}
    for flag, name in (('"--n_folds"', "n_folds"), ('"--base_seed"', "base_seed")):
        head = source.index(flag)
        defaults[name] = int(source[head:head + 400].split("default=")[1].split(",")[0])
    assert defaults == {"n_folds": 10, "base_seed": 1234}
