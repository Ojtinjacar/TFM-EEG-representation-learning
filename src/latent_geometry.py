"""Quantitative characterisation of a latent space.

A two-dimensional projection cannot settle what it is asked to settle. It preserves
neighbourhoods and not distances, so a tidy scatter can sit over a geometry that failed and
an untidy one over a geometry that worked. Whatever is claimed from a figure has to be
measured in the space the encoder actually produced.

This holds the measure that decides whether such a figure is readable at all: how much of
the arrangement is the identity of the child rather than anything about their EEG.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def subject_identifiability(embeddings, subjects, sessions, k=1):
    """Measures how much of a representation is the identity of the subject.

    A nearest-neighbour classifier is asked which child a window came from, holding out one
    whole recording session at a time. The session is the unit held out rather than the
    window because neighbouring windows of one recording share impedance, noise and the
    state of the child that day, so a window-level split lets the answer leak in through
    the window next to it and measures memorisation instead of a persistent trait.

    A high value means an age-coloured projection of this space is not evidence about
    maturation: the arrangement is who the child is.

    Args:
        embeddings (np.ndarray): Representation matrix, shape (n_windows, n_dims).
        subjects (array-like): Subject identifier per window.
        sessions (array-like): Session identifier per window, held out as a group.
        k (int): Number of neighbours.

    Returns:
        dict: Accuracy, the chance level, their ratio, and how many windows were scored.
    """
    subjects, sessions = np.asarray(subjects), np.asarray(sessions)
    embeddings = np.asarray(embeddings)
    correct = total = 0
    for train_idx, test_idx in LeaveOneGroupOut().split(embeddings, subjects, sessions):
        # A subject with a single session is absent from its own training split, so it
        # could only be missed. Skipping keeps the score honest instead of counting a
        # guaranteed failure.
        if subjects[test_idx][0] not in set(subjects[train_idx]):
            continue
        # Fitted on the training side alone: standardising the whole matrix first would
        # let the held-out session inform the scale it is then measured on.
        scaler = StandardScaler().fit(embeddings[train_idx])
        model = KNeighborsClassifier(n_neighbors=k).fit(
            scaler.transform(embeddings[train_idx]), subjects[train_idx])
        correct += int((model.predict(scaler.transform(embeddings[test_idx]))
                        == subjects[test_idx]).sum())
        total += len(test_idx)

    chance = 1.0 / len(np.unique(subjects))
    accuracy = correct / total if total else np.nan
    return {"subject_accuracy": accuracy, "chance": chance,
            "times_chance": accuracy / chance if total else np.nan, "n_scored": total}
