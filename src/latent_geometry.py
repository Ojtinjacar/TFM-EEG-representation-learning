"""Quantitative characterisation of an ExpCLR latent space.

Exists because a 2D projection cannot answer the question ExpCLR poses. The loss does not build
clusters: it asks that the pairwise distances of the embedding reproduce those of the expert
descriptor. t-SNE preserves neighbourhoods and explicitly not distances, so a scatter plot can look
tidy over a geometry that failed, and vice versa. The Shepard correlation here measures the
objective directly.

The second half addresses a confound documented, but never measured, in the previous TFM: its
latent space grouped by *subject* ("practically all windows of the same patient concentrate in a
single cluster", ch. 6), which was reported as a virtue -- "persistent individual fingerprints" --
rather than as the thing that makes an age-coloured projection unreadable. With 45 subjects and
~58 windows each, that has to be quantified before any figure is interpreted, so
:func:`subject_identifiability` and the silhouette contrast are the gatekeepers of the plots, not
an afterthought.

Every metric is computed in the representation space. The projection is only ever used for drawing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from loss import ExpCLRLoss


def shepard_pairs(embeddings, features, subjects, delta=1.0, feat_max_dist=None,
                  n_pairs=200_000, seed=0):
    """Samples pairs and returns the distance each side of the objective assigns them.

    Args:
        embeddings: Representation matrix (n_windows, e).
        features: Prepared expert descriptor (n_windows, d), standardised as in training.
        subjects: Subject identifier per window.
        delta: Margin used during training, so the target is on the trained scale.
        feat_max_dist: ``max_kl ||f_k - f_l||`` from training. None takes it over the sample.
        n_pairs: Pairs to draw.
        seed: Seed for the sampling.

    Returns:
        DataFrame with ``target``, ``distance``, ``raw_distance`` and ``same_subject`` per pair.

        Two distances on purpose. ``distance`` is ``D_ij``, row-normalised by ``mu_i`` exactly as
        the loss does, so it is the quantity being optimised. ``raw_distance`` is the plain
        Euclidean one. They differ because the target ``(1 - s_ij) * Delta`` is *not* row-normalised
        while ``D_ij`` is: two pairs at the same descriptor distance get different ``D_ij`` if their
        rows have different mean distance. As a result even an embedding identical to the descriptor
        cannot reach a correlation of 1 against ``distance``, only against ``raw_distance``. The gap
        between the two correlations measures how much the row normalisation distorts the objective.
    """
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    criterion = ExpCLRLoss(torch.device("cpu"), delta=delta, feat_max_dist=feat_max_dist,
                           temperature=None)
    f = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    z = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)

    with torch.no_grad():
        emb_dist = torch.cdist(z, z, p=2)
        distance = criterion.normalized_distance(z)[i, j].numpy()
        raw_distance = emb_dist[i, j].numpy()
        feat_dist = torch.cdist(f, f, p=2)
        max_dist = criterion.feat_max_dist or float(feat_dist.max())
        sim = (1.0 - feat_dist[i, j] / max_dist).clamp_min(0.0)
        target = ((1.0 - (sim ** 2 if criterion.squared_similarity else sim)) * delta).numpy()

    return pd.DataFrame({"target": target, "distance": distance, "raw_distance": raw_distance,
                         "same_subject": np.asarray(subjects)[i] == np.asarray(subjects)[j]})


def shepard_correlation(embeddings, features, subjects, **kwargs):
    """Measures how well the embedding geometry reproduces the descriptor geometry.

    This is the objective of ExpCLR expressed as a single number. Reported split by whether the
    pair comes from the same subject, because a model that merely memorised subject identity scores
    well on the within-subject half and poorly on the between-subject half, which is precisely the
    failure a projection cannot show.

    Args:
        embeddings: Representation matrix.
        features: Prepared expert descriptor.
        subjects: Subject identifier per window.
        **kwargs: Forwarded to :func:`shepard_pairs`.

    Returns:
        Dict of Spearman and Pearson coefficients overall, within subject and between subjects,
        plus ``spearman_raw_all``, the same correlation against the un-normalised distance. The
        latter is the ceiling: it reaches 1 for a faithful embedding, while the normalised one
        cannot (see :func:`shepard_pairs`).
    """
    pairs = shepard_pairs(embeddings, features, subjects, **kwargs)
    out = {"n_pairs": int(len(pairs))}
    for label, subset in (("all", pairs),
                          ("within_subject", pairs[pairs.same_subject]),
                          ("between_subject", pairs[~pairs.same_subject])):
        if len(subset) < 3:
            out[f"spearman_{label}"] = out[f"pearson_{label}"] = np.nan
            continue
        out[f"spearman_{label}"] = float(spearmanr(subset.target, subset.distance).statistic)
        out[f"pearson_{label}"] = float(pearsonr(subset.target, subset.distance)[0])
    out["spearman_raw_all"] = float(spearmanr(pairs.target, pairs.raw_distance).statistic)
    return out


def subject_identifiability(embeddings, subjects, sessions, k=1):
    """Measures how much of the embedding is subject identity rather than anything else.

    A nearest-neighbour classifier predicts which child a window came from, held out one whole
    session at a time so that neighbouring windows of the same recording cannot leak the answer.
    Compared against the 1/n_subjects a coin flip would give.

    If this comes out high, an age-coloured 2D projection is not evidence about maturation: the
    space is organised by who the child is. The previous TFM described this qualitatively across
    four figures and never measured it.

    Args:
        embeddings: Representation matrix.
        subjects: Subject identifier per window.
        sessions: Session identifier per window, used as the held-out group.
        k: Neighbours.

    Returns:
        Dict with the accuracy, the chance level and the ratio between them.
    """
    subjects, sessions = np.asarray(subjects), np.asarray(sessions)
    X = StandardScaler().fit_transform(np.asarray(embeddings))
    correct = total = 0
    for train_idx, test_idx in LeaveOneGroupOut().split(X, subjects, sessions):
        # A subject with a single session cannot be predicted: it is absent from its own training
        # split. Skipping keeps the score honest rather than counting a guaranteed miss.
        if subjects[test_idx][0] not in set(subjects[train_idx]):
            continue
        model = KNeighborsClassifier(n_neighbors=k).fit(X[train_idx], subjects[train_idx])
        correct += int((model.predict(X[test_idx]) == subjects[test_idx]).sum())
        total += len(test_idx)
    chance = 1.0 / len(np.unique(subjects))
    accuracy = correct / total if total else np.nan
    return {"subject_accuracy": accuracy, "chance": chance,
            "times_chance": accuracy / chance if total else np.nan, "n_scored": total}


def cluster_agreement(cluster_labels, subjects, ages):
    """Compares a clustering against subject identity and against age.

    Reported together on purpose: the interesting quantity is which of the two the clustering
    tracks. The previous TFM read its clusters as developmental without ever computing this.

    Args:
        cluster_labels: Cluster assignment per window.
        subjects: Subject identifier per window.
        ages: Age per window.

    Returns:
        Dict with adjusted Rand index and normalised mutual information against each.
    """
    return {
        "ari_subject": float(adjusted_rand_score(subjects, cluster_labels)),
        "ari_age": float(adjusted_rand_score(ages, cluster_labels)),
        "nmi_subject": float(normalized_mutual_info_score(subjects, cluster_labels)),
        "nmi_age": float(normalized_mutual_info_score(ages, cluster_labels)),
    }
