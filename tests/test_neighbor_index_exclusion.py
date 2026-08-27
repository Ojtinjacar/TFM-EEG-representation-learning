"""Tests that held-out subjects cannot supply a positive.

Under the 'crosssubj' strategy the positive comes from a different subject of the
same age, so an index built over the whole cohort lets a training anchor be
pulled towards a window of a subject its fold is meant to have never seen. The
other strategies take the positive from the anchor's own subject and cannot leak
this way, which is why only this one is filtered.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from build_neighbor_index_ablation import build_ablation_table  # noqa: E402

SUBJECTS = ["S1", "S1", "S2", "S2", "S3", "S3"]


@pytest.fixture
def cohort():
    """Six windows of three subjects, all at the same age and block.

    Returns:
        tuple[dict, pd.DataFrame]: Representations keyed as the builder expects,
            and the window metadata.
    """
    rng = np.random.default_rng(7)
    meta = pd.DataFrame({
        "subject": SUBJECTS,
        "age": [6] * len(SUBJECTS),
        "block": [1] * len(SUBJECTS),
        "epoch_index": list(range(len(SUBJECTS))),
    })
    reprs = {"feat_z": rng.normal(size=(len(SUBJECTS), 8))}
    return reprs, meta


def _neighbour_subjects(table, meta):
    """Returns the subjects that appear as somebody's positive.

    Args:
        table (pd.DataFrame): Output of ``build_ablation_table``.
        meta (pd.DataFrame): Window metadata.

    Returns:
        set[str]: Subjects supplying at least one positive.
    """
    return set(meta["subject"].to_numpy()[table["neigh_gpos"].to_numpy()])


def test_without_exclusion_every_subject_can_be_a_positive(cohort):
    """The baseline: nothing is filtered, so the leak is there to be closed."""
    reprs, meta = cohort
    table = build_ablation_table(reprs, meta, "cosine", "crosssubj", k=2)
    assert _neighbour_subjects(table, meta) == set(SUBJECTS)


def test_a_held_out_subject_supplies_no_positive(cohort):
    """The point of the fix: no anchor trains against a held-out window."""
    reprs, meta = cohort
    table = build_ablation_table(reprs, meta, "cosine", "crosssubj", k=2,
                                 exclude_subjects=["S3"])
    assert "S3" not in _neighbour_subjects(table, meta)


def test_the_positive_still_comes_from_another_subject(cohort):
    """Filtering candidates must not quietly turn this into a same-subject index."""
    reprs, meta = cohort
    table = build_ablation_table(reprs, meta, "cosine", "crosssubj", k=2,
                                 exclude_subjects=["S3"])
    subj = meta["subject"].to_numpy()
    assert not (subj[table["anchor_gpos"].to_numpy()]
                == subj[table["neigh_gpos"].to_numpy()]).any()


def test_anchors_of_the_only_remaining_subject_lose_their_positive(cohort):
    """With every other subject held out there is nobody left to pair them with.

    The anchors of the held-out subjects still get one, and that is harmless: the
    trainer drops their rows before the first epoch. What must not happen is a
    training anchor reaching for a held-out window.
    """
    reprs, meta = cohort
    table = build_ablation_table(reprs, meta, "cosine", "crosssubj", k=2,
                                 exclude_subjects=["S2", "S3"])
    subj = meta["subject"].to_numpy()
    anchors = set(subj[table["anchor_gpos"].to_numpy()])
    assert "S1" not in anchors
    assert _neighbour_subjects(table, meta) == {"S1"}
