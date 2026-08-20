"""
EEG representation clustering analysis.

Loads general pre-trained models (no fold_id) and analyses their latent spaces
via clustering:
- Optimal K search (Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz)
- t-SNE visualisation with KMeans clusters
- Comparative clustering metrics
- Cosine similarity matrix across methods

Supported methods: PCA, SimCLR, AE, MAE, TripletLoss
"""

import os
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from models import (
    EnhancedAttentionLSTM,
    CNNAutoencoder,
    AttentionLSTMAutoencoder,
    MaskedAttentionLSTMAutoencoder
)
from utils import (
    found_k_clusters,
    infer_embeddings_in_batches,
    visualization_emb,
    visualization_emb_by_label,
    compute_clustering_metrics,
    compute_similarity_matrix
)

# Available methods
AVAILABLE_METHODS = ["Raw", "PCA", "SimCLR", "AE", "MAE", "TripletLoss"]


def get_model_path(method, zone, frequency, target=None, save_model_dir="save/models"):
    """
    Builds the path to the general pre-trained model (no fold_id).

    Args:
        method: representation method (SimCLR, AE, MAE, TripletLoss)
        zone: brain region (all, frontal, etc.)
        frequency: frequency band (all, alpha, etc.)
        target: target for TripletLoss (age, cit_36mo). Ignored for other methods.
        save_model_dir: directory containing the models

    Returns:
        Model path, or None if not applicable (PCA)
    """
    if method == "Raw" or method == "PCA":
        return None

    if method == "SimCLR":
        filename = f"SimCLR_{zone}_{frequency}_batch_512_lr_0.001_wd_0.0001_temperature_0.05.pth"
    elif method == "AE":
        filename = f"AE_{zone}_{frequency}_hidden128_e100.pth"
    elif method == "MAE":
        filename = f"MAE_{zone}_{frequency}_hidden128_mask30_block25_e100.pth"
    elif method == "TripletLoss":
        if target is None:
            raise ValueError("TripletLoss requires a target to be specified")
        filename = f"Triplet_{target}_{zone}_{frequency}_emb128_m0.4.pth"
    else:
        raise ValueError(f"Unknown method: {method}")

    return os.path.join(save_model_dir, filename)


def load_model(method, model_path, input_size, n_channels, hidden_size=128,
               sfreq=250, device=None):
    """
    Loads a pre-trained model.

    Args:
        method: model type (SimCLR, AE, MAE, TripletLoss)
        model_path: path to the .pth file
        input_size: temporal length of the signals
        n_channels: number of EEG channels
        hidden_size: latent space dimensionality
        sfreq: sampling frequency
        device: device (cuda/cpu)

    Returns:
        Model loaded in evaluation mode
    """
    if method == "SimCLR" or method == "TripletLoss":
        model = EnhancedAttentionLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            n_channels=n_channels,
            sfreq=sfreq,
            lstm_hidden_size=hidden_size // 2
        )
    elif method == "AE":
        model = AttentionLSTMAutoencoder(
            input_size=input_size,
            hidden_size=hidden_size,
            n_channels=n_channels,
            sfreq=sfreq,
            lstm_hidden_size=hidden_size // 2
        )
    elif method == "MAE":
        model = MaskedAttentionLSTMAutoencoder(
            input_size=input_size,
            hidden_size=hidden_size,
            n_channels=n_channels,
            sfreq=sfreq,
            lstm_hidden_size=hidden_size // 2
        )
    else:
        raise ValueError(f"Unknown method for model loading: {method}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def extract_representations(X, method, model_path=None, hidden_size=128,
                           sfreq=250, batch_size=128, device=None):
    """
    Extracts representations using the specified method.

    Args:
        X: EEG data (N, C, T)
        method: representation method
        model_path: model path (None for PCA)
        hidden_size: latent space dimensionality
        sfreq: sampling frequency
        batch_size: batch size for inference
        device: target device

    Returns:
        Representations (N, hidden_size)
    """
    if method == "Raw":
        # Flattened signals with no transformation
        return X.reshape(X.shape[0], -1)

    if method == "PCA":
        X_flat = X.reshape(X.shape[0], -1)
        pca = PCA(n_components=hidden_size)
        return pca.fit_transform(X_flat)

    # Load model and extract embeddings
    model = load_model(
        method=method,
        model_path=model_path,
        input_size=X.shape[2],
        n_channels=X.shape[1],
        hidden_size=hidden_size,
        sfreq=sfreq,
        device=device
    )
    emb = infer_embeddings_in_batches(model, X, batch_size=batch_size, device=device)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norm + 1e-12)


def find_optimal_k(representations, k_range=range(2, 31)):
    """
    Finds the optimal K using the Davies-Bouldin Index.

    Args:
        representations: representation array (N, D)
        k_range: range of K values to evaluate

    Returns:
        Optimal K based on Davies-Bouldin Index (lower is better)
    """
    from sklearn.metrics import davies_bouldin_score

    best_k, best_score = 2, float('inf')
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(representations)
        score = davies_bouldin_score(representations, labels)
        if score < best_score:  # Lower is better for DB
            best_k, best_score = k, score
    return best_k, best_score


def analyze_single_method(representations, method_name, k_clusters, save_fig_dir):
    """
    Analyses one method: K search, KMeans, visualisation.

    Args:
        representations: representation array (N, D)
        method_name: method name used for labels
        k_clusters: number of clusters for the final KMeans (None for auto)
        save_fig_dir: directory to save figures

    Returns:
        Tuple (representations, kmeans_fitted, k_used)
    """
    # Optimal K search (generates plots with all metrics)
    k_search_path = os.path.join(save_fig_dir, f"{method_name}_k_search.png")
    found_k_clusters(representations, filename=k_search_path)
    print(f"    K search plot saved: {k_search_path}", flush=True)

    # If K is not specified, find it automatically with Davies-Bouldin
    if k_clusters is None:
        k_clusters, best_score = find_optimal_k(representations)
        print(f"    Optimal K (Davies-Bouldin): {k_clusters} (score: {best_score:.3f})", flush=True)

    # Fit KMeans with the chosen K
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init="auto")
    kmeans.fit(representations)

    # t-SNE visualisation
    visualization_emb(kmeans, representations, name_method=method_name, save_fig_dir=save_fig_dir)

    return representations, kmeans, k_clusters


def plot_metrics_comparison(metrics_df, save_fig_dir):
    """
    Generates a comparative figure of clustering metrics.
    Includes Subject-Compactness, Session-Compactness and Silhouettes when available.
    Also includes Age-Silhouette and metrics decomposed by subject/age.
    """
    # Determine number of columns based on available metrics
    has_compactness = "Subject-Compactness" in metrics_df.columns
    has_silhouette = "Subject-Silhouette" in metrics_df.columns
    has_age_silhouette = "Age-Silhouette" in metrics_df.columns
    has_decomposed_sil = "Session-Sil-by-Subject" in metrics_df.columns
    has_decomposed_db = "Session-DB-by-Subject" in metrics_df.columns
    has_hopkins = "Hopkins-Global" in metrics_df.columns

    # Column count: 3 base + 2 compactness + 2 silhouette + 1 age + 2 decomposed sil + 2 decomposed db + 3 hopkins
    n_cols = 3
    if has_compactness:
        n_cols += 2
    if has_silhouette:
        n_cols += 2
    if has_age_silhouette:
        n_cols += 1
    if has_decomposed_sil:
        n_cols += 2
    if has_decomposed_db:
        n_cols += 2
    if has_hopkins:
        n_cols += 3

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))
    col_idx = 0

    # Silhouette (higher is better)
    bars1 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Silhouette"], color="skyblue")
    axes[col_idx].set_title("Silhouette Score\n(higher is better)")
    axes[col_idx].set_ylabel("Score")
    axes[col_idx].tick_params(axis='x', rotation=45)
    for bar in bars1:
        height = bar.get_height()
        axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                     ha='center', va='bottom', fontsize=8)
    col_idx += 1

    # Davies-Bouldin (lower is better)
    bars2 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Davies-Bouldin"], color="orange")
    axes[col_idx].set_title("Davies-Bouldin Index\n(lower is better)")
    axes[col_idx].set_ylabel("Score")
    axes[col_idx].tick_params(axis='x', rotation=45)
    for bar in bars2:
        height = bar.get_height()
        axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                     ha='center', va='bottom', fontsize=8)
    col_idx += 1

    # Calinski-Harabasz (higher is better)
    bars3 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Calinski-Harabasz"], color="seagreen")
    axes[col_idx].set_title("Calinski-Harabasz Index\n(higher is better)")
    axes[col_idx].set_ylabel("Score")
    axes[col_idx].tick_params(axis='x', rotation=45)
    for bar in bars3:
        height = bar.get_height()
        axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.1f}",
                     ha='center', va='bottom', fontsize=8)
    col_idx += 1

    # Subject-Compactness (lower is better)
    if has_compactness:
        bars4 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Subject-Compactness"], color="coral")
        axes[col_idx].set_title("Subject-Compactness\n(lower is better)")
        axes[col_idx].set_ylabel("Score")
        axes[col_idx].tick_params(axis='x', rotation=45)
        for bar in bars4:
            height = bar.get_height()
            axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                         ha='center', va='bottom', fontsize=8)
        col_idx += 1

        # Session-Compactness (lower is better)
        bars5 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Session-Compactness"], color="mediumpurple")
        axes[col_idx].set_title("Session-Compactness\n(lower is better)")
        axes[col_idx].set_ylabel("Score")
        axes[col_idx].tick_params(axis='x', rotation=45)
        for bar in bars5:
            height = bar.get_height()
            axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                         ha='center', va='bottom', fontsize=8)
        col_idx += 1

    # Subject-Silhouette (higher is better, >0 indicates separation)
    if has_silhouette:
        bars6 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Subject-Silhouette"], color="steelblue")
        axes[col_idx].set_title("Subject Silhouette\n(higher is better)")
        axes[col_idx].set_ylabel("Score")
        axes[col_idx].axhline(y=0.0, color='red', linestyle='--', linewidth=1)
        axes[col_idx].set_ylim(-1, 1)
        axes[col_idx].tick_params(axis='x', rotation=45)
        for bar in bars6:
            height = bar.get_height()
            axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                         ha='center', va='bottom', fontsize=8)
        col_idx += 1

        # Session-Silhouette (higher is better, >0 indicates separation)
        bars7 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Session-Silhouette"], color="darkorchid")
        axes[col_idx].set_title("Session Silhouette\n(higher is better)")
        axes[col_idx].set_ylabel("Score")
        axes[col_idx].axhline(y=0.0, color='red', linestyle='--', linewidth=1)
        axes[col_idx].set_ylim(-1, 1)
        axes[col_idx].tick_params(axis='x', rotation=45)
        for bar in bars7:
            height = bar.get_height()
            axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                         ha='center', va='bottom', fontsize=8)
        col_idx += 1

    # Age-Silhouette (higher is better, >0 indicates separation)
    if has_age_silhouette:
        bars8 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Age-Silhouette"], color="teal")
        axes[col_idx].set_title("Age Silhouette\n(higher is better)")
        axes[col_idx].set_ylabel("Score")
        axes[col_idx].axhline(y=0.0, color='red', linestyle='--', linewidth=1)
        axes[col_idx].set_ylim(-1, 1)
        axes[col_idx].tick_params(axis='x', rotation=45)
        for bar in bars8:
            height = bar.get_height()
            axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                         ha='center', va='bottom', fontsize=8)
        col_idx += 1

    # Session-Silhouette-by-Subject (average session silhouette per subject)
    if has_decomposed_sil:
        bars9 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Session-Sil-by-Subject"], color="goldenrod")
        axes[col_idx].set_title("Session Sil by Subject\n(mayor mejor)")
        axes[col_idx].set_ylabel("Score")
        axes[col_idx].axhline(y=0.0, color='red', linestyle='--', linewidth=1)
        axes[col_idx].set_ylim(-1, 1)
        axes[col_idx].tick_params(axis='x', rotation=45)
        for bar in bars9:
            height = bar.get_height()
            axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                         ha='center', va='bottom', fontsize=8)
        col_idx += 1

        # Subject-Silhouette-by-Age (average subject silhouette per age)
        bars10 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Subject-Sil-by-Age"], color="indianred")
        axes[col_idx].set_title("Subject Sil by Age\n(mayor mejor)")
        axes[col_idx].set_ylabel("Score")
        axes[col_idx].axhline(y=0.0, color='red', linestyle='--', linewidth=1)
        axes[col_idx].set_ylim(-1, 1)
        axes[col_idx].tick_params(axis='x', rotation=45)
        for bar in bars10:
            height = bar.get_height()
            axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                         ha='center', va='bottom', fontsize=8)
        col_idx += 1

    # Session-DB-by-Subject (average session Davies-Bouldin per subject)
    if has_decomposed_db:
        bars_db1 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Session-DB-by-Subject"], color="darkorange")
        axes[col_idx].set_title("Session DB by Subject\n(menor mejor)")
        axes[col_idx].set_ylabel("Score")
        axes[col_idx].tick_params(axis='x', rotation=45)
        for bar in bars_db1:
            height = bar.get_height()
            axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                         ha='center', va='bottom', fontsize=8)
        col_idx += 1

        # Subject-DB-by-Age (average subject Davies-Bouldin per age)
        bars_db2 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Subject-DB-by-Age"], color="chocolate")
        axes[col_idx].set_title("Subject DB by Age\n(menor mejor)")
        axes[col_idx].set_ylabel("Score")
        axes[col_idx].tick_params(axis='x', rotation=45)
        for bar in bars_db2:
            height = bar.get_height()
            axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                         ha='center', va='bottom', fontsize=8)
        col_idx += 1

    # Hopkins Statistics (>0.7 indicates clustering tendency, ~0.5 is random)
    if has_hopkins:
        # Hopkins Global
        bars11 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Hopkins-Global"], color="darkgreen")
        axes[col_idx].set_title("Hopkins Global\n(>0.7 clustering)")
        axes[col_idx].set_ylabel("Score")
        axes[col_idx].axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Random')
        axes[col_idx].axhline(y=0.7, color='green', linestyle='--', linewidth=1, label='Clustering')
        axes[col_idx].set_ylim(0, 1)
        axes[col_idx].tick_params(axis='x', rotation=45)
        for bar in bars11:
            height = bar.get_height()
            axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                         ha='center', va='bottom', fontsize=8)
        col_idx += 1

        # Hopkins by Subject
        bars12 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Hopkins-by-Subject"], color="forestgreen")
        axes[col_idx].set_title("Hopkins by Subject\n(>0.7 clustering)")
        axes[col_idx].set_ylabel("Score")
        axes[col_idx].axhline(y=0.5, color='red', linestyle='--', linewidth=1)
        axes[col_idx].axhline(y=0.7, color='green', linestyle='--', linewidth=1)
        axes[col_idx].set_ylim(0, 1)
        axes[col_idx].tick_params(axis='x', rotation=45)
        for bar in bars12:
            height = bar.get_height()
            axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                         ha='center', va='bottom', fontsize=8)
        col_idx += 1

        # Hopkins by Age
        bars13 = axes[col_idx].bar(metrics_df["Method"], metrics_df["Hopkins-by-Age"], color="limegreen")
        axes[col_idx].set_title("Hopkins by Age\n(>0.7 clustering)")
        axes[col_idx].set_ylabel("Score")
        axes[col_idx].axhline(y=0.5, color='red', linestyle='--', linewidth=1)
        axes[col_idx].axhline(y=0.7, color='green', linestyle='--', linewidth=1)
        axes[col_idx].set_ylim(0, 1)
        axes[col_idx].tick_params(axis='x', rotation=45)
        for bar in bars13:
            height = bar.get_height()
            axes[col_idx].text(bar.get_x() + bar.get_width()/2, height, f"{height:.3f}",
                         ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(save_fig_dir, "clustering_metrics_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Metrics figure saved to: {fig_path}", flush=True)


def plot_similarity_heatmap(sim_matrix, save_fig_dir):
    """
    Generates a cosine similarity heatmap across methods.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Mean Cosine Similarity"}
    )
    plt.title("Cosine Similarity between Representations", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    fig_path = os.path.join(save_fig_dir, "similarity_cosine_heatmap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Similarity figure saved to: {fig_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="EEG representation clustering analysis"
    )

    # Data paths
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/5_s/processed_windows.npy",
        help="Path to the .npy file with EEG windows.",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="data/processed/5_s/processed_metadata.csv",
        help="Path to the metadata CSV.",
    )

    # Zone and frequency
    parser.add_argument(
        "--zone",
        type=str,
        default="all",
        help="Brain region (all, frontal, central, etc.)",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="all",
        help="Frequency band (all, alpha, beta, etc.)",
    )

    # Output paths
    parser.add_argument(
        "--save-model-dir",
        type=str,
        default="save/models",
        help="Directory containing pre-trained models.",
    )
    parser.add_argument(
        "--save-fig-dir",
        type=str,
        default="save/figures/clustering",
        help="Directory to save figures.",
    )

    # Methods to analyse
    parser.add_argument(
        "--methods",
        nargs="+",
        default=AVAILABLE_METHODS,
        choices=AVAILABLE_METHODS,
        help="Methods to analyse.",
    )
    parser.add_argument(
        "--triplet-targets",
        nargs="+",
        type=str,
        default=["age", "cit_36mo"],
        choices=["age", "cit_36mo"],
        help="Targets for TripletLoss models (all are analysed).",
    )

    # Model parameters
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Latent space dimensionality.",
    )
    parser.add_argument(
        "--sampling-freq",
        type=int,
        default=250,
        help="Sampling frequency.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference.",
    )

    # KMeans parameters
    parser.add_argument(
        "--k-clusters",
        type=int,
        default=None,
        help="Number of clusters for KMeans. If not specified, uses Silhouette auto-detection.",
    )
    parser.add_argument(
        "--k-per-method",
        nargs="+",
        type=str,
        default=None,
        metavar="METHOD:K",
        help="Method-specific K. Format: 'PCA:5 SimCLR:3 AE:4'. "
             "Falls back to --k-clusters or auto-detection if a method is not listed.",
    )

    args = parser.parse_args()

    # Parse per-method K if specified
    k_per_method = {}
    if args.k_per_method:
        for item in args.k_per_method:
            method, k = item.split(":")
            k_per_method[method] = int(k)

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    # Create output directories
    os.makedirs(args.save_fig_dir, exist_ok=True)

    # Load data
    print("[INFO] Loading data...", flush=True)
    X = np.load(args.data_path)
    meta = pd.read_csv(args.meta_path)
    print(f"  EEG shape: {X.shape}", flush=True)
    print(f"  Number of subjects: {meta['subject'].nunique()}", flush=True)

    # Results dictionary
    methods_results = {}  # {method_name: (representations, kmeans)}
    k_used_per_method = {}  # To report K used

    print(f"\n{'='*80}", flush=True)
    print("REPRESENTATION EXTRACTION", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Methods: {args.methods}", flush=True)
    print(f"Zone: {args.zone}, Frequency: {args.frequency}", flush=True)

    # Process each method
    for method in args.methods:
        # TripletLoss: process each specified target
        if method == "TripletLoss":
            targets_to_process = args.triplet_targets
        else:
            targets_to_process = [None]

        for target in targets_to_process:
            method_label = f"{method}_{target}" if target else method
            print(f"\n[{method_label}] Processing...", flush=True)

            # Get model path
            model_path = get_model_path(
                method=method,
                zone=args.zone,
                frequency=args.frequency,
                target=target,
                save_model_dir=args.save_model_dir
            )

            # Verify model exists
            if model_path is not None and not os.path.exists(model_path):
                print(f"  [WARNING] Model not found: {model_path}. Skipping...", flush=True)
                continue

            # Extract representations
            try:
                representations = extract_representations(
                    X=X,
                    method=method,
                    model_path=model_path,
                    hidden_size=args.hidden_size,
                    sfreq=args.sampling_freq,
                    batch_size=args.batch_size,
                    device=device
                )
                print(f"  Representations shape: {representations.shape}", flush=True)
            except Exception as e:
                print(f"  [ERROR] Error extracting representations: {e}", flush=True)
                continue

            # Determine K for this method
            if method_label in k_per_method:
                k = k_per_method[method_label]
            elif method in k_per_method:
                k = k_per_method[method]
            elif args.k_clusters is not None:
                k = args.k_clusters
            else:
                k = None  # Auto-detection inside analyze_single_method

            # Analyse method
            print(f"  Analysing clustering...", flush=True)
            reps, kmeans, k_final = analyze_single_method(
                representations=representations,
                method_name=method_label,
                k_clusters=k,
                save_fig_dir=args.save_fig_dir
            )

            methods_results[method_label] = (reps, kmeans)
            k_used_per_method[method_label] = k_final
            print(f"  [OK] Analysis completed (K={k_final})", flush=True)

    if not methods_results:
        print("\n[ERROR] No method was processed. Check that the models exist.", flush=True)
        return

    # t-SNE visualisation coloured by age for all methods
    print(f"\n{'='*80}", flush=True)
    print("t-SNE VISUALISATION BY AGE", flush=True)
    print(f"{'='*80}", flush=True)

    age_labels = meta['age'].values
    for method_label, (representations, _) in methods_results.items():
        print(f"  Generating t-SNE by age for {method_label}...", flush=True)
        visualization_emb_by_label(
            embeddings=representations,
            labels=age_labels,
            label_name='Age',
            name_method=method_label,
            save_fig_dir=args.save_fig_dir,
            cmap='viridis'
        )

    # Comparative clustering metrics
    print(f"\n{'='*80}", flush=True)
    print("CLUSTERING METRICS", flush=True)
    print(f"{'='*80}", flush=True)

    # Add K-used column; pass metadata to compute Subject/Session variance
    metrics_df = compute_clustering_metrics(
        methods_results,
        meta=meta,
        subject_col='subject',
        session_col='age'
    )
    metrics_df["K"] = metrics_df["Method"].map(k_used_per_method)
    print(metrics_df.to_string(index=False), flush=True)

    # Save metrics
    metrics_csv_path = os.path.join(args.save_fig_dir, "clustering_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n[INFO] Metrics saved to: {metrics_csv_path}", flush=True)

    # Metrics figure
    plot_metrics_comparison(metrics_df, args.save_fig_dir)

    # Cosine similarity matrix
    print(f"\n{'='*80}", flush=True)
    print("SIMILARITY BETWEEN REPRESENTATIONS", flush=True)
    print(f"{'='*80}", flush=True)

    # Exclude Raw from the similarity matrix (incompatible dimensions: 87500 vs 128)
    methods_for_similarity = {k: v for k, v in methods_results.items() if k != "Raw"}
    sim_matrix = compute_similarity_matrix(methods_for_similarity)
    print(sim_matrix.to_string(), flush=True)

    # Save similarity matrix
    sim_csv_path = os.path.join(args.save_fig_dir, "similarity_matrix.csv")
    sim_matrix.to_csv(sim_csv_path)
    print(f"\n[INFO] Similarity matrix saved to: {sim_csv_path}", flush=True)

    # Similarity figure
    plot_similarity_heatmap(sim_matrix, args.save_fig_dir)

    print(f"\n{'='*80}", flush=True)
    print("ANALYSIS COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Results saved to: {args.save_fig_dir}", flush=True)


if __name__ == "__main__":
    main()