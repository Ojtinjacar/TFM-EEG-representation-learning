"""Tests for the quantitative characterisation of the latent space.

The ones that matter are controls. A geometry metric that returns a plausible number on any input
is worse than none, because it looks like evidence: so the Shepard correlation is checked against an
embedding that reproduces the descriptor exactly (must saturate) and against noise (must vanish),
and subject identifiability is checked against an embedding that encodes identity and one that
cannot.

Run with:
    conda run -n dasci-cimcyc python -m pytest tests/test_latent_geometry.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from latent_geometry import (  # noqa: E402
    cluster_agreement,
    shepard_correlation,
    shepard_pairs,
    subject_identifiability,
)


@pytest.fixture
def descriptor():
    """A descriptor with structure: 12 subjects, 4 sessions each, 10 windows per session."""
    rng = np.random.default_rng(0)
    rows, subjects, sessions, ages = [], [], [], []
    for s in range(12):
        centre = rng.normal(size=8) * 3
        for age in (6, 9, 16, 36):
            for _ in range(10):
                rows.append(centre + age * 0.1 + rng.normal(scale=0.3, size=8))
                subjects.append(f"S{s:02d}")
                sessions.append(f"S{s:02d}_{age}")
                ages.append(age)
    return (np.array(rows), np.array(subjects), np.array(sessions), np.array(ages))


# --- Shepard: does the embedding reproduce the descriptor geometry? -----------------------

def test_perfect_embedding_saturates_the_shepard_correlation(descriptor):
    """An embedding that *is* the descriptor must saturate. Otherwise the metric is broken.

    Saturation happens against the un-normalised distance. Against the row-normalised D_ij it
    cannot: the target is not row-normalised, so pairs at equal descriptor distance get different
    D_ij depending on their row's mean. That gap is a property of the objective, not of the metric,
    and the assertion below pins the ceiling so a regression in either direction shows up.
    """
    features, subjects, _, _ = descriptor
    result = shepard_correlation(features, features, subjects, n_pairs=20_000)
    assert result["spearman_raw_all"] > 0.99
    assert 0.7 < result["spearman_all"] < 0.95, (
        "row normalisation prevents saturation, so a change here means the loss changed")


def test_scaling_the_embedding_does_not_change_the_correlation(descriptor):
    """The loss is invariant to global rescaling, so this diagnostic must be too."""
    features, subjects, _, _ = descriptor
    base = shepard_correlation(features, features, subjects, n_pairs=20_000)["spearman_all"]
    scaled = shepard_correlation(features * 1000.0, features, subjects,
                                 n_pairs=20_000)["spearman_all"]
    assert scaled == pytest.approx(base, abs=1e-6)


def test_random_embedding_has_no_shepard_correlation(descriptor):
    """The control: without it, the test above could pass on a metric that always returns ~1."""
    features, subjects, _, _ = descriptor
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(len(features), 16))
    result = shepard_correlation(noise, features, subjects, n_pairs=20_000)
    assert abs(result["spearman_all"]) < 0.1


def test_shepard_separates_within_and_between_subject_pairs(descriptor):
    """The split is what distinguishes reproducing the geometry from memorising identity."""
    features, subjects, _, _ = descriptor
    pairs = shepard_pairs(features, features, subjects, n_pairs=20_000)
    assert pairs.same_subject.sum() > 0, "there should be within-subject pairs"
    assert (~pairs.same_subject).sum() > 0, "there should be between-subject pairs"
    # Every pair lands in exactly one of the two halves.
    assert pairs.same_subject.sum() + (~pairs.same_subject).sum() == len(pairs)


def test_shepard_never_pairs_a_window_with_itself(descriptor):
    """A self-pair has zero distance on both sides and would inflate the correlation for free."""
    features, subjects, _, _ = descriptor
    pairs = shepard_pairs(features, features, subjects, n_pairs=5_000)
    assert (pairs.distance > 0).all()


# --- Subject identifiability -------------------------------------------------------------

def test_identity_encoding_embedding_is_almost_perfectly_identifiable(descriptor):
    """An embedding built from the subject centroid must be nearly always attributable."""
    _, subjects, sessions, _ = descriptor
    rng = np.random.default_rng(2)
    centres = {s: rng.normal(size=6) * 10 for s in np.unique(subjects)}
    embeddings = np.array([centres[s] + rng.normal(scale=0.1, size=6) for s in subjects])
    result = subject_identifiability(embeddings, subjects, sessions)
    assert result["subject_accuracy"] > 0.9
    assert result["times_chance"] > 10


def test_noise_embedding_is_not_identifiable(descriptor):
    """The control: pure noise must sit near chance, not above it."""
    _, subjects, sessions, _ = descriptor
    rng = np.random.default_rng(3)
    result = subject_identifiability(rng.normal(size=(len(subjects), 6)), subjects, sessions)
    assert result["subject_accuracy"] < 5 * result["chance"]


def test_chance_level_is_one_over_the_number_of_subjects(descriptor):
    _, subjects, sessions, _ = descriptor
    result = subject_identifiability(np.zeros((len(subjects), 4)), subjects, sessions)
    assert result["chance"] == pytest.approx(1 / 12)


# --- Cluster agreement -------------------------------------------------------------------

def test_agreement_detects_which_factor_a_clustering_tracks(descriptor):
    """Clustering by subject must score high against subject and low against age, and vice versa."""
    _, subjects, _, ages = descriptor

    by_subject = cluster_agreement(subjects, subjects, ages)
    assert by_subject["ari_subject"] == pytest.approx(1.0)
    assert by_subject["ari_age"] < 0.1

    by_age = cluster_agreement(ages, subjects, ages)
    assert by_age["ari_age"] == pytest.approx(1.0)
    assert by_age["ari_subject"] < 0.1
