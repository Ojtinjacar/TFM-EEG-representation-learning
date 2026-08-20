import os
import torch 
import random
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity

import itertools
import math
import scipy.stats as stats
from scipy.stats import kruskal
# import scikit_posthocs as sp_post

# -------------------------------------------------------------------------
# GLOBAL CLUSTER COLOURS
# -------------------------------------------------------------------------
CLUSTER_COLORS_HEX = {
    0: "#1f77b4",  # blue
    1: "#d62728",  # red
    2: "#2ca02c",  # green
    3: "#ff7f0e",  # orange
    4: "#9467bd",  # purple
    5: "#8c564b",  # brown
    6: "#e377c2",  # pink
    7: "#7f7f7f",  # grey
    8: "#bcbd22",  # olive yellow
    9: "#17becf",  # cyan
}

CLUSTER_COLORS_RGBA = {
    0: "rgba(31,119,180,0.5)",   # #1f77b4
    1: "rgba(214,39,40,0.5)",    # #d62728
    2: "rgba(44,160,44,0.5)",    # #2ca02c
    3: "rgba(255,127,14,0.5)",   # #ff7f0e
    4: "rgba(148,103,189,0.5)",  # #9467bd
    5: "rgba(140,86,75,0.5)",    # #8c564b
    6: "rgba(227,119,194,0.5)",  # #e377c2
    7: "rgba(127,127,127,0.5)",  # #7f7f7f
    8: "rgba(188,189,34,0.5)",   # #bcbd22
    9: "rgba(23,190,207,0.5)",   # #17becf
}

def visualization_emb(kmeans, embeddings, name_method='PCA', save_fig_dir=None):
    """
    t-SNE + scatter plot of KMeans clusters.
    """
    graf = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    df_vis = pd.DataFrame(graf, columns=['Dim 1', 'Dim 2'])
    df_vis['Cluster'] = kmeans.labels_

    plt.figure(figsize=(12, 9))
    sns.scatterplot(x='Dim 1', y='Dim 2', hue='Cluster', data=df_vis, palette='coolwarm')
    plt.title(f'KMeans Cluster Visualisation — {name_method}')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend(title='Cluster')
    plt.tight_layout()

    if save_fig_dir is not None:
        os.makedirs(save_fig_dir, exist_ok=True)
        safe_name = name_method.replace(" ", "_").replace("-", "_")
        fig_path = os.path.join(save_fig_dir, f"tsne_clusters_{safe_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] t-SNE figure ({name_method}) saved to: {fig_path}")
        plt.close()
    else:
        plt.show()
        plt.close()

    return graf


def visualization_emb_by_label(embeddings, labels, label_name='Age', name_method='PCA',
                               save_fig_dir=None, cmap='viridis', tsne_coords=None):
    """
    t-SNE + scatter plot coloured by a continuous or categorical label.

    Args:
        embeddings: representation array (N, D)
        labels: label array (N,) used for colouring
        label_name: label name for the title and legend
        name_method: representation method name
        save_fig_dir: directory to save the figure
        cmap: colourmap for the scatter plot
        tsne_coords: pre-computed t-SNE coordinates (optional, to reuse)

    Returns:
        t-SNE coordinates (N, 2)
    """
    if tsne_coords is None:
        graf = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        graf = tsne_coords

    df_vis = pd.DataFrame(graf, columns=['Dim 1', 'Dim 2'])
    df_vis[label_name] = labels

    plt.figure(figsize=(12, 9))

    # Determine whether the label is categorical or continuous
    unique_labels = np.unique(labels)
    is_categorical = len(unique_labels) <= 10

    if is_categorical:
        # Use a discrete palette for few categories
        palette = sns.color_palette(cmap, n_colors=len(unique_labels))
        scatter = sns.scatterplot(
            x='Dim 1', y='Dim 2', hue=label_name, data=df_vis,
            palette=palette, alpha=0.7
        )
        plt.legend(title=label_name, bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        # Use a continuous colourmap
        scatter = plt.scatter(
            df_vis['Dim 1'], df_vis['Dim 2'],
            c=labels, cmap=cmap, alpha=0.7, s=20
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label(label_name)

    plt.title(f't-SNE of {name_method} coloured by {label_name}')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.tight_layout()

    if save_fig_dir is not None:
        os.makedirs(save_fig_dir, exist_ok=True)
        safe_name = name_method.replace(" ", "_").replace("-", "_")
        safe_label = label_name.replace(" ", "_").replace("-", "_")
        fig_path = os.path.join(save_fig_dir, f"tsne_{safe_label}_{safe_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] t-SNE figure by {label_name} ({name_method}) saved to: {fig_path}")
        plt.close()
    else:
        plt.show()
        plt.close()

    return graf


def compute_distance_bounds(representations):
    """
    Computes the lower and upper distance bounds in the latent space.

    Args:
        representations: representation array (N, D)

    Returns:
        (min_dist, max_dist): tuple with minimum and maximum pairwise distances
    """
    from sklearn.metrics.pairwise import euclidean_distances

    # For large datasets, subsample for efficiency
    n_samples = representations.shape[0]
    if n_samples > 5000:
        # Subsample to compute approximate bounds
        idx = np.random.choice(n_samples, size=5000, replace=False)
        sample = representations[idx]
    else:
        sample = representations

    distances = euclidean_distances(sample)

    # Ignore the diagonal (self-distance = 0)
    np.fill_diagonal(distances, np.inf)
    min_dist = np.min(distances)

    np.fill_diagonal(distances, -np.inf)
    max_dist = np.max(distances)

    return min_dist, max_dist


def compute_subject_compactness(representations, subject_labels, min_dist, max_dist):
    """
    Computes the compactness of each subject's windows in the latent space.

    Measures the mean pairwise distance between all windows of the same subject,
    normalised by the space bounds so it is comparable across methods.

    Args:
        representations: representation array (N, D)
        subject_labels: subject ID array (N,)
        min_dist: lower distance bound of the space
        max_dist: upper distance bound of the space

    Returns:
        Mean normalised compactness (0 = very compact, 1 = very dispersed)
    """
    from sklearn.metrics.pairwise import euclidean_distances

    unique_subjects = np.unique(subject_labels)
    subject_compactness = []

    for subject in unique_subjects:
        subject_mask = subject_labels == subject
        subject_reps = representations[subject_mask]

        # At least 2 windows are needed to compute distances
        if subject_reps.shape[0] < 2:
            continue

        # Compute pairwise distances between all windows of this subject
        pairwise_dist = euclidean_distances(subject_reps)

        # Upper triangle only (excluding diagonal)
        upper_tri_idx = np.triu_indices(pairwise_dist.shape[0], k=1)
        distances = pairwise_dist[upper_tri_idx]

        if len(distances) == 0:
            continue

        # Normalise by bounds
        if max_dist > min_dist:
            normalized = (distances - min_dist) / (max_dist - min_dist)
        else:
            normalized = np.zeros_like(distances)

        # Mean normalised distance for this subject
        subject_compactness.append(normalized.mean())

    if len(subject_compactness) == 0:
        return np.nan

    return np.mean(subject_compactness)


def compute_session_compactness(representations, subject_labels, session_labels, min_dist, max_dist):
    """
    Computes the compactness of each subject's windows within each session (age).

    Measures the mean pairwise distance between windows of the same subject AND
    same session, normalised by the space bounds so it is comparable across methods.

    Args:
        representations: representation array (N, D)
        subject_labels: subject ID array (N,)
        session_labels: session (age) label array (N,)
        min_dist: lower distance bound of the space
        max_dist: upper distance bound of the space

    Returns:
        Mean normalised compactness (0 = very compact, 1 = very dispersed)
    """
    from sklearn.metrics.pairwise import euclidean_distances

    unique_subjects = np.unique(subject_labels)
    unique_sessions = np.unique(session_labels)
    session_compactness = []

    for subject in unique_subjects:
        for session in unique_sessions:
            # Windows of the same subject AND same session
            mask = (subject_labels == subject) & (session_labels == session)
            group_reps = representations[mask]

            # At least 2 windows needed
            if group_reps.shape[0] < 2:
                continue

            # Compute pairwise distances within this subject-session group
            pairwise_dist = euclidean_distances(group_reps)

            # Upper triangle only (excluding diagonal)
            upper_tri_idx = np.triu_indices(pairwise_dist.shape[0], k=1)
            distances = pairwise_dist[upper_tri_idx]

            if len(distances) == 0:
                continue

            # Normalise by bounds
            if max_dist > min_dist:
                normalized = (distances - min_dist) / (max_dist - min_dist)
            else:
                normalized = np.zeros_like(distances)

            # Mean normalised distance for this subject-session
            session_compactness.append(normalized.mean())

    if len(session_compactness) == 0:
        return np.nan

    return np.mean(session_compactness)


def compute_subject_silhouette(representations, subject_labels):
    """
    Computes the Silhouette Score treating subjects as clusters.

    Measures how well separated subjects are in the latent space,
    using the standard sklearn metric.

    Args:
        representations: representation array (N, D)
        subject_labels: subject ID array (N,)

    Returns:
        Silhouette Score [-1, 1] (higher is better, >0 indicates separation)
    """
    unique_subjects = np.unique(subject_labels)

    # Silhouette requires at least 2 clusters
    if len(unique_subjects) < 2:
        return np.nan

    return silhouette_score(representations, subject_labels)


def compute_session_silhouette(representations, subject_labels, session_labels):
    """
    Computes the Silhouette Score treating subject+session combinations as clusters.

    Measures how well separated each subject's sessions are.

    Args:
        representations: representation array (N, D)
        subject_labels: subject ID array (N,)
        session_labels: session (age) label array (N,)

    Returns:
        Silhouette Score [-1, 1] (higher is better, >0 indicates separation)
    """
    # Create combined subject+session labels
    combined_labels = np.array([
        f"{subj}_{sess}" for subj, sess in zip(subject_labels, session_labels)
    ])

    unique_combined = np.unique(combined_labels)

    # Silhouette requires at least 2 clusters
    if len(unique_combined) < 2:
        return np.nan

    return silhouette_score(representations, combined_labels)


def compute_age_silhouette(representations, age_labels):
    """
    Computes the Silhouette Score treating age groups as clusters.

    Measures how well separated the different ages are in the latent space.

    Args:
        representations: representation array (N, D)
        age_labels: age label array (N,)

    Returns:
        Silhouette Score [-1, 1] (higher is better, >0 indicates separation)
    """
    unique_ages = np.unique(age_labels)

    # Silhouette requires at least 2 clusters
    if len(unique_ages) < 2:
        return np.nan

    return silhouette_score(representations, age_labels)


def compute_hopkins_statistic(representations, sample_size=None, random_state=42):
    """
    Computes the Hopkins Statistic to measure clustering tendency.

    The Hopkins statistic measures whether the data has cluster structure by
    comparing distances of real points vs. uniformly random points.

    Args:
        representations: representation array (N, D)
        sample_size: number of points to sample (default: min(N*0.1, 100))
        random_state: random seed for reproducibility

    Returns:
        Hopkins statistic H ∈ [0, 1]:
        - H ≈ 0.5: random/uniform data (no structure)
        - H > 0.7: significant clustering tendency
        - H < 0.3: regular/uniform data
    """
    np.random.seed(random_state)

    N, D = representations.shape

    if sample_size is None:
        sample_size = min(int(N * 0.1), 100)
    sample_size = min(sample_size, N - 1)

    if sample_size < 2:
        return np.nan

    # Per-dimension min-max range for generating uniform points
    mins = representations.min(axis=0)
    maxs = representations.max(axis=0)

    # Sample indices of real points
    sample_indices = np.random.choice(N, size=sample_size, replace=False)
    sample_points = representations[sample_indices]

    # Generate uniform random points in the hyperrectangle
    random_points = np.random.uniform(mins, maxs, size=(sample_size, D))

    # For each sampled point, compute distance to nearest neighbour
    # excluding the point itself
    from sklearn.neighbors import NearestNeighbors

    # Fit NN on all data
    nn = NearestNeighbors(n_neighbors=2)  # 2 because nearest will be itself if it's in the dataset
    nn.fit(representations)

    # Distances from real sampled points to their nearest neighbour (excluding self)
    distances_real, _ = nn.kneighbors(sample_points)
    u_distances = distances_real[:, 1]  # Second column (first is self with dist 0)

    # Distances from random points to their nearest neighbour in real data
    distances_random, _ = nn.kneighbors(random_points)
    w_distances = distances_random[:, 0]  # First column (not in the dataset)

    # Hopkins statistic
    sum_u = np.sum(u_distances)
    sum_w = np.sum(w_distances)

    if sum_u + sum_w == 0:
        return np.nan

    H = sum_w / (sum_u + sum_w)

    return H


def compute_hopkins_by_subject(representations, subject_labels, sample_size=None, random_state=42):
    """
    Computes the Hopkins Statistic averaged per subject.

    For each subject, computes Hopkins using only their samples, then averages.
    Measures whether there is cluster structure within each subject.

    Args:
        representations: representation array (N, D)
        subject_labels: subject ID array (N,)
        sample_size: number of points to sample per subject
        random_state: base random seed

    Returns:
        Mean Hopkins statistic per subject
    """
    unique_subjects = np.unique(subject_labels)
    hopkins_scores = []

    for i, subj in enumerate(unique_subjects):
        mask = subject_labels == subj
        subj_reps = representations[mask]

        # Need enough samples
        if len(subj_reps) < 10:
            continue

        H = compute_hopkins_statistic(
            subj_reps,
            sample_size=sample_size,
            random_state=random_state + i
        )

        if not np.isnan(H):
            hopkins_scores.append(H)

    if len(hopkins_scores) == 0:
        return np.nan

    return np.mean(hopkins_scores)


def compute_hopkins_by_age(representations, age_labels, sample_size=None, random_state=42):
    """
    Computes the Hopkins Statistic averaged per age group.

    For each age, computes Hopkins using only samples of that age, then averages.
    Measures whether there is cluster structure within each age group.

    Args:
        representations: representation array (N, D)
        age_labels: age label array (N,)
        sample_size: number of points to sample per age group
        random_state: base random seed

    Returns:
        Mean Hopkins statistic per age group
    """
    unique_ages = np.unique(age_labels)
    hopkins_scores = []

    for i, age in enumerate(unique_ages):
        mask = age_labels == age
        age_reps = representations[mask]

        # Need enough samples
        if len(age_reps) < 10:
            continue

        H = compute_hopkins_statistic(
            age_reps,
            sample_size=sample_size,
            random_state=random_state + i
        )

        if not np.isnan(H):
            hopkins_scores.append(H)

    if len(hopkins_scores) == 0:
        return np.nan

    return np.mean(hopkins_scores)


def compute_session_silhouette_by_subject(representations, subject_labels, session_labels):
    """
    Computes the session Silhouette Score decomposed per subject.

    For each subject, computes the session silhouette using only that subject's
    samples, then averages. This prevents between-subject variability from
    contaminating the session-separation metric.

    Args:
        representations: representation array (N, D)
        subject_labels: subject ID array (N,)
        session_labels: session (age) label array (N,)

    Returns:
        Mean Silhouette Score per subject [-1, 1]
    """
    unique_subjects = np.unique(subject_labels)
    silhouettes = []

    for subj in unique_subjects:
        # Mask for this subject
        mask = subject_labels == subj
        subj_reps = representations[mask]
        subj_sessions = session_labels[mask]

        # At least 2 distinct sessions and enough samples needed
        unique_sessions = np.unique(subj_sessions)
        if len(unique_sessions) < 2:
            continue

        # Compute session silhouette within this subject
        try:
            sil = silhouette_score(subj_reps, subj_sessions)
            silhouettes.append(sil)
        except ValueError:
            # Can fail if any cluster has only one sample
            continue

    if len(silhouettes) == 0:
        return np.nan

    return np.mean(silhouettes)


def compute_subject_silhouette_by_age(representations, subject_labels, age_labels):
    """
    Computes the subject Silhouette Score decomposed per age group.

    For each age, computes the subject silhouette using only that age's samples,
    then averages. This prevents between-age variability from contaminating the
    subject-separation metric.

    Args:
        representations: representation array (N, D)
        subject_labels: subject ID array (N,)
        age_labels: age label array (N,)

    Returns:
        Mean Silhouette Score per age group [-1, 1]
    """
    unique_ages = np.unique(age_labels)
    silhouettes = []

    for age in unique_ages:
        # Mask for this age group
        mask = age_labels == age
        age_reps = representations[mask]
        age_subjects = subject_labels[mask]

        # At least 2 distinct subjects needed
        unique_subjects = np.unique(age_subjects)
        if len(unique_subjects) < 2:
            continue

        # Compute subject silhouette within this age group
        try:
            sil = silhouette_score(age_reps, age_subjects)
            silhouettes.append(sil)
        except ValueError:
            # Can fail if any cluster has only one sample
            continue

    if len(silhouettes) == 0:
        return np.nan

    return np.mean(silhouettes)


def compute_davies_bouldin_by_subject(representations, subject_labels, session_labels):
    """
    Computes the session Davies-Bouldin Score decomposed per subject.

    For each subject, computes the session DB using only that subject's samples,
    then averages. This prevents between-subject variability from contaminating
    the metric.

    Args:
        representations: representation array (N, D)
        subject_labels: subject ID array (N,)
        session_labels: session/age label array (N,)

    Returns:
        Mean Davies-Bouldin Score per subject (lower is better)
    """
    unique_subjects = np.unique(subject_labels)
    db_scores = []

    for subject in unique_subjects:
        # Mask for this subject
        mask = subject_labels == subject
        subj_reps = representations[mask]
        subj_sessions = session_labels[mask]

        # At least 2 distinct sessions needed
        unique_sessions = np.unique(subj_sessions)
        if len(unique_sessions) < 2:
            continue

        # At least 2 samples per session required for DB
        valid = True
        for sess in unique_sessions:
            if np.sum(subj_sessions == sess) < 2:
                valid = False
                break
        if not valid:
            continue

        # Compute session Davies-Bouldin within this subject
        try:
            db = davies_bouldin_score(subj_reps, subj_sessions)
            db_scores.append(db)
        except ValueError:
            continue

    if len(db_scores) == 0:
        return np.nan

    return np.mean(db_scores)


def compute_davies_bouldin_by_age(representations, subject_labels, age_labels):
    """
    Computes the subject Davies-Bouldin Score decomposed per age group.

    For each age, computes the subject DB using only that age's samples, then
    averages. This prevents between-age variability from contaminating the
    subject-separation metric.

    Args:
        representations: representation array (N, D)
        subject_labels: subject ID array (N,)
        age_labels: age label array (N,)

    Returns:
        Mean Davies-Bouldin Score per age group (lower is better)
    """
    unique_ages = np.unique(age_labels)
    db_scores = []

    for age in unique_ages:
        # Mask for this age group
        mask = age_labels == age
        age_reps = representations[mask]
        age_subjects = subject_labels[mask]

        # At least 2 distinct subjects needed
        unique_subjects = np.unique(age_subjects)
        if len(unique_subjects) < 2:
            continue

        # At least 2 samples per subject required for DB
        valid = True
        for subj in unique_subjects:
            if np.sum(age_subjects == subj) < 2:
                valid = False
                break
        if not valid:
            continue

        # Compute subject Davies-Bouldin within this age group
        try:
            db = davies_bouldin_score(age_reps, age_subjects)
            db_scores.append(db)
        except ValueError:
            continue

    if len(db_scores) == 0:
        return np.nan

    return np.mean(db_scores)


def compute_clustering_metrics(methods_dict, meta=None, subject_col='subject', session_col='age'):
    """
    Computes clustering metrics for multiple methods.

    Args:
        methods_dict: {name: (representations, kmeans_model)}
        meta: metadata DataFrame (optional, for subject/session compactness)
        subject_col: column name for subject ID
        session_col: column name for session (age)

    Returns:
        DataFrame with metrics per method
    """
    results = []

    for method_name, (representations, kmeans_model) in methods_dict.items():
        labels = kmeans_model.labels_
        silhouette = silhouette_score(representations, labels)
        davies = davies_bouldin_score(representations, labels)
        calinski = calinski_harabasz_score(representations, labels)

        result = {
            "Method": method_name,
            "Silhouette": silhouette,
            "Davies-Bouldin": davies,
            "Calinski-Harabasz": calinski
        }

        # Compute compactness metrics if metadata is provided
        if meta is not None:
            min_dist, max_dist = compute_distance_bounds(representations)

            # Subject Compactness: mean distance between windows of the same subject
            if subject_col in meta.columns:
                subject_labels = meta[subject_col].values
                subject_comp = compute_subject_compactness(
                    representations, subject_labels, min_dist, max_dist
                )
                result["Subject-Compactness"] = subject_comp

                # Subject Silhouette: how well separated subjects are
                subject_sil = compute_subject_silhouette(
                    representations, subject_labels
                )
                result["Subject-Silhouette"] = subject_sil

            # Session Compactness: mean distance between windows of the same subject AND session
            if subject_col in meta.columns and session_col in meta.columns:
                session_labels = meta[session_col].values
                session_comp = compute_session_compactness(
                    representations, subject_labels, session_labels, min_dist, max_dist
                )
                result["Session-Compactness"] = session_comp

                # Session Silhouette: how well separated sessions are
                session_sil = compute_session_silhouette(
                    representations, subject_labels, session_labels
                )
                result["Session-Silhouette"] = session_sil

            # Age Silhouette: how well separated age groups are
            if session_col in meta.columns:
                age_labels = meta[session_col].values
                age_sil = compute_age_silhouette(representations, age_labels)
                result["Age-Silhouette"] = age_sil

            # Session Silhouette by Subject: average session silhouette per subject
            # Avoids between-subject variability contaminating the metric
            if subject_col in meta.columns and session_col in meta.columns:
                session_sil_by_subj = compute_session_silhouette_by_subject(
                    representations, subject_labels, session_labels
                )
                result["Session-Sil-by-Subject"] = session_sil_by_subj

            # Subject Silhouette by Age: average subject silhouette per age group
            # Avoids between-age variability contaminating the metric
            if subject_col in meta.columns and session_col in meta.columns:
                subj_sil_by_age = compute_subject_silhouette_by_age(
                    representations, subject_labels, age_labels
                )
                result["Subject-Sil-by-Age"] = subj_sil_by_age

            # Session Davies-Bouldin by Subject: average session DB per subject
            if subject_col in meta.columns and session_col in meta.columns:
                session_db_by_subj = compute_davies_bouldin_by_subject(
                    representations, subject_labels, session_labels
                )
                result["Session-DB-by-Subject"] = session_db_by_subj

            # Subject Davies-Bouldin by Age: average subject DB per age group
            if subject_col in meta.columns and session_col in meta.columns:
                subj_db_by_age = compute_davies_bouldin_by_age(
                    representations, subject_labels, age_labels
                )
                result["Subject-DB-by-Age"] = subj_db_by_age

            # Global Hopkins Statistic: clustering tendency across the whole space
            hopkins_global = compute_hopkins_statistic(representations)
            result["Hopkins-Global"] = hopkins_global

            # Hopkins averaged per subject: is there structure within each subject?
            if subject_col in meta.columns:
                hopkins_by_subj = compute_hopkins_by_subject(
                    representations, subject_labels
                )
                result["Hopkins-by-Subject"] = hopkins_by_subj

            # Hopkins averaged per age: is there structure within each age group?
            if session_col in meta.columns:
                hopkins_by_age = compute_hopkins_by_age(
                    representations, age_labels
                )
                result["Hopkins-by-Age"] = hopkins_by_age

        results.append(result)

    return pd.DataFrame(results)


def compute_similarity_matrix(methods_dict):
    """
    Computes the mean cosine similarity matrix between the
    representations of different methods.
    """
    method_names = list(methods_dict.keys())
    sim_matrix = pd.DataFrame(index=method_names, columns=method_names, dtype=float)

    for name1 in method_names:
        rep1 = methods_dict[name1][0]
        for name2 in method_names:
            rep2 = methods_dict[name2][0]
            sim = cosine_similarity(rep1, rep2)
            sim_mean = np.mean(sim)
            sim_matrix.loc[name1, name2] = sim_mean

    return sim_matrix.astype(float)

def set_seed(seed):
    """Seeds the python, numpy and torch RNGs for reproducible training.

    Args:
        seed (int): Seed applied to all three generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_dataset(X, train_percentaje=0.8, seed=None):
    """
    Converts a NumPy array to a Torch tensor and splits it into train/val.
    """
    data = torch.tensor(X, dtype=torch.float32)  # (N, C, T)

    train_size = int(train_percentaje * X.shape[0])
    val_size = X.shape[0] - train_size
    dataset = TensorDataset(data, data)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return random_split(dataset, [train_size, val_size], generator=generator)


def create_dataloader(train_data, val_data, batch_size=128, seed=None):
    """
    Creates train and validation dataloaders.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              generator=generator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def found_k_clusters(embeddings_np, filename="cluster_metrics.png"):
    sec = range(2, 31)
    inertias, sl_scores, db_scores, ch_scores = [], [], [], []

    for k in sec:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(embeddings_np)
        inertias.append(kmeans.inertia_)
        sl_scores.append(silhouette_score(embeddings_np, kmeans.labels_))
        db_scores.append(davies_bouldin_score(embeddings_np, kmeans.labels_))
        ch_scores.append(calinski_harabasz_score(embeddings_np, kmeans.labels_))

    elbow_data = pd.DataFrame({
        'Clusters (k)': list(sec),
        'Inertia': inertias
    })

    silhouette_data = pd.DataFrame({
        'Clusters (k)': list(sec),
        'Silhouette Score': sl_scores
    })

    db_data = pd.DataFrame({
        'Clusters (k)': list(sec),
        'Davies-Bouldin Index': db_scores
    })

    ch_data = pd.DataFrame({
        'Clusters (k)': list(sec),
        'Calinski-Harabasz Score': ch_scores
    })

    # Four plots in a single figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.lineplot(data=elbow_data, x='Clusters (k)', y='Inertia', marker='o', color='blue', ax=axes[0,0])
    axes[0,0].set_title('Elbow Method — Inertia')
    axes[0,0].set_xlabel('Number of Clusters (k)')
    axes[0,0].set_ylabel('Inertia')
    axes[0,0].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=silhouette_data, x='Clusters (k)', y='Silhouette Score', marker='o', color='green', ax=axes[0,1])
    axes[0,1].set_title('Silhouette')
    axes[0,1].set_xlabel('Number of Clusters (k)')
    axes[0,1].set_ylabel('Silhouette')
    axes[0,1].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=db_data, x='Clusters (k)', y='Davies-Bouldin Index', marker='o', color='red', ax=axes[1,0])
    axes[1,0].set_title('Davies-Bouldin')
    axes[1,0].set_xlabel('Number of Clusters (k)')
    axes[1,0].set_ylabel('Davies-Bouldin')
    axes[1,0].grid(True, linestyle='--', alpha=0.7)

    sns.lineplot(data=ch_data, x='Clusters (k)', y='Calinski-Harabasz Score', marker='o', color='purple', ax=axes[1,1])
    axes[1,1].set_title('Calinski-Harabasz')
    axes[1,1].set_xlabel('Number of Clusters (k)')
    axes[1,1].set_ylabel('Calinski-Harabasz')
    axes[1,1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig) 

def infer_embeddings_in_batches(model, X_data, batch_size=512, device=None):

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(X_data), batch_size):
            batch = X_data[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            batch_embeddings = model.get_embedding(batch_tensor)
            all_embeddings.append(batch_embeddings.cpu())

    # Concatenate all embeddings into a numpy array
    return torch.cat(all_embeddings, dim=0).numpy()

def plot_random_samples_by_age(meta, X, age, subject_ids, patient_id_column="subject"):

    """
    Selects and plots random EEG windows for subjects of a specified age.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata with columns 'age' and patient_id_column.
    X : np.ndarray
        EEG data in format (n_samples, n_channels, n_timepoints).
    age : int
        Age of interest.
    subject_ids : list of str
        List of patient IDs to plot.
    patient_id_column : str, optional
        Name of the column in meta containing the patient ID.
    """
    # Filter indices and IDs for the specified age
    mask_age = meta['age'] == age
    indices_age = meta[mask_age].index.to_numpy()
    patient_ids_age = meta.loc[mask_age, patient_id_column].to_numpy()

    # Collect indices and samples
    idx_list = []
    samples = []
    for sid in subject_ids:
        indices = indices_age[patient_ids_age == sid]
        if len(indices) == 0:
            raise ValueError(f"Subject {sid} with age {age} not found")
        idx = random.choice(indices)
        idx_list.append(idx)
        samples.append(X[idx])

    n_channels = samples[0].shape[0]
    fig, axs = plt.subplots(1, len(samples), figsize=(8 * len(samples), 8), sharey=True)
    if len(samples) == 1:
        axs = [axs]  # Ensure iterable

    for i, (sample, sid, idx) in enumerate(zip(samples, subject_ids, idx_list)):
        for ch in range(n_channels):
            axs[i].plot(sample[ch] + ch * 10, label=f'Channel {ch + 1}' if i == 0 else "")
        axs[i].set_title(f"{sid} (index {idx})")
        axs[i].set_xlabel("Time (samples)")
        axs[i].set_yticks([])
        axs[i].grid(True)

    axs[0].set_ylabel("Amplitude + offset (different channels)")
    plt.tight_layout()
    plt.show()

    return idx_list, samples

def visual_clusters(
    embeddings,
    edad,
    n_clusters=2,
    random_state=42,
):
    """
    - Applies KMeans and t-SNE on embeddings.
    - Renames clusters so that 0 is the one with the MOST patients,
      1 the next, etc.
    - Plots the t-SNE with global CLUSTER_COLORS_HEX colours.
    - Returns kmeans (with already re-labelled labels) and tsne_result.
    """

    # -----------------------
    # 1) KMeans
    # -----------------------
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init='auto'
    ).fit(embeddings)

    labels_original = kmeans.labels_

    # -----------------------
    # 2) Re-label clusters by number of samples (approx. number of windows)
    #    0 = largest cluster, 1 = next, etc.
    # -----------------------
    unique, counts = np.unique(labels_original, return_counts=True)
    # sort from largest to smallest
    order = unique[np.argsort(-counts)]
    remap = {old: new for new, old in enumerate(order)}

    labels_std = np.array([remap[l] for l in labels_original])
    kmeans.labels_ = labels_std  # IMPORTANT: the entire pipeline will use this

    # -----------------------
    # 3) TSNE
    # -----------------------
    tsne = TSNE(n_components=2, random_state=random_state)
    tsne_result = tsne.fit_transform(embeddings)

    df_visualizacion = pd.DataFrame(tsne_result, columns=['Dim 1', 'Dim 2'])
    df_visualizacion['Cluster'] = labels_std

    # -----------------------
    # 4) Colores consistentes
    # -----------------------
    unique_clusters = sorted(df_visualizacion['Cluster'].unique())
    palette = {c: CLUSTER_COLORS_HEX.get(c, "#000000") for c in unique_clusters}

    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        x='Dim 1',
        y='Dim 2',
        hue='Cluster',
        data=df_visualizacion,
        palette=palette
    )
    plt.title(f'SimCLR cluster visualisation (age = {edad})')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend(title='Cluster')
    plt.tight_layout()

    return kmeans, tsne_result

def plot_patient_highlight(df_visualizacion, paciente_destacado, height=9, width=12):
    """
    Generates a static scatter plot highlighting one patient and leaving the rest in the background.

    df_visualizacion must have columns: ['Dim 1', 'Dim 2', 'Paciente']
    paciente_destacado: ID of the patient to highlight
    """
    plt.figure(figsize=(width, height))

    # Colours
    colors = plt.get_cmap('tab20').colors
    pacientes = df_visualizacion['Paciente'].unique()
    color_map = {p: colors[i % len(colors)] for i, p in enumerate(pacientes)}

    # Draw all patients with low opacity
    for paciente in pacientes:
        df_p = df_visualizacion[df_visualizacion['Paciente'] == paciente]
        if paciente == paciente_destacado:
            plt.scatter(df_p['Dim 1'], df_p['Dim 2'],
                        label=str(paciente),
                        color=color_map[paciente],
                        alpha=0.9, s=50, edgecolor='black')
        else:
            plt.scatter(df_p['Dim 1'], df_p['Dim 2'],
                        label=str(paciente),
                        color=color_map[paciente],
                        alpha=0.1, s=30)
    
    plt.title(f"t-SNE Visualisation — Patient {paciente_destacado} highlighted")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_patient_cluster_distribution(
    df_visualizacion,
    filename="patient.png",
    figsize=(16, 8),
    cluster_color_map=None,
):
    """
    Generates a stacked bar chart with the distribution of windows
    per patient and cluster, using global colours by cluster ID.
    """

    # Group and count
    conteo = (
        df_visualizacion
        .groupby(['Paciente', 'Cluster'])
        .size()
        .unstack(fill_value=0)
    )
    # keep the original patient order
    conteo = conteo.reindex(df_visualizacion['Paciente'].unique(), axis=0)

    unique_clusters = sorted(conteo.columns)

    # Colours consistent with t-SNE and Sankey
    if cluster_color_map is None:
        cluster_color_map = CLUSTER_COLORS_HEX

    colors = [cluster_color_map.get(c, "#000000") for c in unique_clusters]
    color_map = {c: cluster_color_map.get(c, "#000000") for c in unique_clusters}

    ax = conteo.plot(
        kind='bar',
        stacked=True,
        figsize=figsize,
        color=colors
    )

    plt.title('Window distribution per patient and cluster')
    plt.xlabel('Patient ID')
    plt.ylabel('Number of windows')
    plt.xticks(rotation=90)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    return color_map

def plot_cluster_metadata_distributions(meta, kmeans_labels, color_map,
                                        age_filter=6,
                                        categorical_cols=None, continuous_cols=None):
    """
    Distribution plots per cluster for categorical and continuous variables.
    Scaled to percentage with median line in histograms.
    """
    if categorical_cols is None:
        categorical_cols = []
    if continuous_cols is None:
        continuous_cols = []

    meta_cluster = meta[meta['age'] == age_filter].copy()
    meta_cluster = meta_cluster.assign(cluster=kmeans_labels)

    def rice_bins(data):
        data = np.asarray(data)
        data = data[~np.isnan(data)]
        return int(np.ceil(2 * (len(data) ** (1/3)))) if len(data) >= 2 else 1

    # === CATEGORICAL ===
    for col in categorical_cols:
        fig, axes = plt.subplots(
            1, meta_cluster['cluster'].nunique(),
            figsize=(5 * meta_cluster['cluster'].nunique(), 4),
            sharey=True
        )
        if meta_cluster['cluster'].nunique() == 1:
            axes = [axes]
        for i, clust in enumerate(sorted(meta_cluster['cluster'].unique())):
            ax = axes[i]
            cluster_data = meta_cluster.loc[meta_cluster['cluster'] == clust, col]
            counts = cluster_data.value_counts(dropna=False)
            total = counts.sum()
            percentages = (counts / total * 100).sort_index()

            percentages.plot(kind='bar', ax=ax, color=color_map.get(clust, 'gray'))
            ax.set_title(f'{col} - Cluster {clust}')
            ax.set_xlabel(col)
            ax.set_ylabel('Percentage (%)')
        plt.tight_layout()
        plt.show()

    # === CONTINUOUS ===
    for col in continuous_cols:
        fig, axes = plt.subplots(
            1, meta_cluster['cluster'].nunique(),
            figsize=(5 * meta_cluster['cluster'].nunique(), 4),
            sharey=True
        )
        if meta_cluster['cluster'].nunique() == 1:
            axes = [axes]

        total_age = meta_cluster.shape[0]  # total windows at that age
        total_valid_age = meta_cluster[col].dropna().shape[0]  # total valid windows at that age for that attribute

        for i, clust in enumerate(sorted(meta_cluster['cluster'].unique())):
            ax = axes[i]
            cluster_data = meta_cluster.loc[meta_cluster['cluster'] == clust, col].dropna()
            n = len(cluster_data)

            # % representativeness within valid windows (sums to 100)
            perc_valid = (n / total_valid_age * 100) if total_valid_age > 0 else 0
            # % of the total windows at that age
            perc_age = (n / total_age * 100) if total_age > 0 else 0

            if n > 0:
                bins = rice_bins(cluster_data)
                ax.hist(cluster_data, bins=bins, alpha=0.7,
                        color=color_map.get(clust, 'gray'),
                        weights=np.ones(len(cluster_data)) / len(cluster_data) * 100)  # Normalise to %
                median_val = np.median(cluster_data)
                ax.axvline(median_val, color='red', linestyle='--',
                           label=f'Median: {median_val:.2f}')

                ax.set_title(
                    f'{col} - Cluster {clust} '
                    f'({n}/{total_valid_age} {perc_valid:.1f}% T: {perc_age:.1f}% age)'
                )
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                ax.set_title(
                    f'{col} - Cluster {clust} '
                    f'(n=0/{total_valid_age}, 0% valid, 0% age)'
                )

            ax.set_xlabel(col)
            ax.set_ylabel('Percentage (%)')
        plt.tight_layout()
        plt.show()

def plot_continuous_cluster_boxplots(meta, kmeans_labels, color_map=None, age_filter=6, continuous_cols=None, n_cols=4):
    """
    Boxplots of continuous variables per cluster in a stacked grid.
    - Each variable with an independent Y axis.
    - Boxplots coloured according to color_map.
    - Labels with n/total valid, % valid and % age.
    - Mann-Whitney significance between cluster pairs (p-values in console).
    - Optionally save the entire figure as a single image.
    """
    if continuous_cols is None:
        continuous_cols = []
    if color_map is None:
        color_map = {}

    meta_cluster = meta[meta['age'] == age_filter].copy()
    meta_cluster = meta_cluster.assign(cluster=kmeans_labels)

    n_vars = len(continuous_cols)
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()  # for easy indexing even with an incomplete row

    cluster_order = sorted(meta_cluster['cluster'].unique())
    n_clusters = len(cluster_order)
    total_age = meta_cluster.shape[0]

    for idx, col in enumerate(continuous_cols):
        ax = axes[idx]

        # Prepare data per cluster
        cluster_data_dict = {clust: meta_cluster.loc[meta_cluster['cluster'] == clust, col].dropna()
                             for clust in cluster_order}

        total_valid_age = meta_cluster[col].dropna().shape[0]
        cluster_counts = {clust: len(vals) for clust, vals in cluster_data_dict.items()}
        perc_valid = {clust: (len(vals)/total_valid_age*100 if total_valid_age>0 else 0)
                      for clust, vals in cluster_data_dict.items()}
        perc_age = {clust: (len(vals)/total_age*100 if total_age>0 else 0)
                    for clust, vals in cluster_data_dict.items()}

        # Create boxplots
        boxprops = dict(linestyle='-', linewidth=1.5, facecolor='lightgray')
        medianprops = dict(color='red', linewidth=2)
        bplot = ax.boxplot([cluster_data_dict[clust] if len(cluster_data_dict[clust])>0 else [np.nan] 
                            for clust in cluster_order],
                           labels=[f"C{cl}" for cl in cluster_order],
                           patch_artist=True,
                           boxprops=boxprops,
                           medianprops=medianprops)

        # Colour patches
        for patch, clust in zip(bplot['boxes'], cluster_order):
            patch.set_facecolor(color_map.get(clust, 'gray'))
            patch.set_alpha(0.7)

        # Labels with n and percentages
        new_labels = [f"C{cl}\n{cluster_counts[cl]}/{total_valid_age}\n{perc_valid[cl]:.1f}% val\n{perc_age[cl]:.1f}% age"
                      for cl in cluster_order]
        ax.set_xticklabels(new_labels)
        ax.set_title(col)
        ax.set_ylabel("Value")

    # Remove empty axes if there are leftover subplots
    for j in range(n_vars, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_cluster_significance(meta, kmeans_labels, continuous_cols=None, age_filter=6):
    """
    Bar chart of p-values using the Kruskal-Wallis H-test for continuous variables per cluster.
    - Horizontal red bar at 0.05 indicates significance threshold.
    """
    if continuous_cols is None:
        continuous_cols = []

    meta_cluster = meta[meta['age'] == age_filter].copy()
    meta_cluster = meta_cluster.assign(cluster=kmeans_labels)
    cluster_order = sorted(meta_cluster['cluster'].unique())

    # Compute p-values
    results = []
    for col in continuous_cols:
        cluster_data_dict = {clust: meta_cluster.loc[meta_cluster['cluster']==clust, col].dropna()
                             for clust in cluster_order}
        data_list = [vals for vals in cluster_data_dict.values() if len(vals) > 0]
        if len(data_list) >= 2:
            stat, pval = kruskal(*data_list)
        else:
            pval = np.nan
        results.append({'variable': col, 'p_value': pval})

    df_results = pd.DataFrame(results)

    # Plot
    plt.figure(figsize=(max(6, len(continuous_cols)*1.2),5))
    bars = plt.bar(df_results['variable'], df_results['p_value'], color='skyblue', edgecolor='k', alpha=0.8)
    plt.axhline(0.05, color='red', linestyle='--', linewidth=1.5, label='α = 0.05')
    plt.ylabel('p-value (Kruskal-Wallis)')
    plt.xlabel('Continuous variable')
    plt.title(f"Kruskal-Wallis significance — age={age_filter}")

    # p-value labels above each bar
    for bar, pval in zip(bars, df_results['p_value']):
        height = bar.get_height()
        if not np.isnan(pval):
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, f"{pval:.3f}",
                     ha='center', va='bottom', fontsize=9)

    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_dunn_posthoc_bars(meta, kmeans_labels, continuous_cols=None, age_filter=6, alpha=0.05):
    """
    Applies Kruskal-Wallis to each continuous variable.
    For significant ones (p<alpha), applies Dunn post-hoc.
    Displays EVERYTHING in a single chart with bars grouped by variable.
    - X axis = variables
    - Y axis = p-values
    - Colours = cluster comparison (C1-C2, C1-C3, ...)
    """
    if continuous_cols is None:
        continuous_cols = []

    meta_cluster = meta[meta['age'] == age_filter].copy()
    meta_cluster = meta_cluster.assign(cluster=kmeans_labels)
    cluster_order = sorted(meta_cluster['cluster'].unique())

    all_results = []

    for col in continuous_cols:
        # --- Kruskal-Wallis test ---
        groups = [meta_cluster.loc[meta_cluster['cluster'] == c, col].dropna() for c in cluster_order]
        if any(len(g) == 0 for g in groups):  # skip if any group is empty
            continue
        stat, p_kw = kruskal(*groups)

        if p_kw < alpha:  # Only continue if globally significant
            # --- Dunn post-hoc ---
            dunn_res = sp_post.posthoc_dunn(
                meta_cluster,
                val_col=col,
                group_col="cluster",
                p_adjust="bonferroni"
            )

            for c1, c2 in itertools.combinations(cluster_order, 2):
                all_results.append({
                    "Variable": col,
                    "Pair": f"C{c1}-C{c2}",
                    "p-value": dunn_res.loc[c1, c2]
                })

    if not all_results:
        print("⚠️ No variable was significant under Kruskal-Wallis.")
        return

    df_plot = pd.DataFrame(all_results)

    # --- Single combined plot ---
    plt.figure(figsize=(max(8, len(continuous_cols)*1.5), 6))
    sns.barplot(
        x="Variable", y="p-value", hue="Pair",
        data=df_plot, dodge=True
    )
    plt.axhline(alpha, color="red", linestyle="--", label=f"α={alpha}")
    plt.title(f"Dunn post-hoc for significant continuous variables (age={age_filter})")
    plt.ylabel("p-value")
    plt.xlabel("Variables")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Cluster comparisons", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def save_model(model, save_dir, filename="contrastive_model.pth"):
    """
    Saves the model state to the specified path.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to save.
    save_dir : str
        Directory where the model will be saved.
    filename : str, optional
        File name (default "contrastive_model.pth").
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")