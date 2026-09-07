"""Maps of the latent spaces, and the clustering table that accompanies them.

A projection to two dimensions preserves neighbourhoods and not distances, so it can show a
tidy arrangement over a geometry that failed and the reverse. It is a way of looking, never
evidence on its own. Everything quantitative here is therefore computed in the space the
encoder produced, and the projection is only ever used for drawing.

The format is the one of the earlier work on this dataset, so its figures can be compared
against these on equal terms: the same t-SNE, the same clusters, the same clustering
metrics. What this adds is the layout, every method side by side in one grid rather than
across pages.

Whatever is chosen, the hyperparameters of the projection travel with the figure. A
projection whose settings are not written down cannot be reproduced, and its defaults move
between library versions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import compute_clustering_metrics, infer_embeddings_in_batches

# Declared rather than defaulted: scikit-learn has changed these between versions, so a
# figure whose settings live only in a default cannot be reproduced from the text.
TSNE_PARAMS = {"perplexity": 30, "learning_rate": "auto", "max_iter": 1000,
               "init": "pca", "random_state": 42}
PCA_PARAMS = {"random_state": 42}

COLOURINGS = ("cluster", "age", "subject", "band_power")


class ProjectionError(RuntimeError):
    """Raised when a projection cannot be produced as requested."""


def project(embeddings, technique, n_clusters=None):
    """Projects a representation to two dimensions.

    Args:
        embeddings (np.ndarray): Representation matrix, shape (n_windows, n_dims).
        technique (str): ``tsne`` or ``pca``.
        n_clusters (int): Unused; kept out of the projection on purpose.

    Returns:
        tuple: (coordinates of shape (n_windows, 2), caption describing the settings).

    Raises:
        ProjectionError: If the technique is unknown.
    """
    if technique == "tsne":
        coords = TSNE(n_components=2, **TSNE_PARAMS).fit_transform(embeddings)
        settings = TSNE_PARAMS
    elif technique == "pca":
        coords = PCA(n_components=2, **PCA_PARAMS).fit_transform(embeddings)
        settings = PCA_PARAMS
    else:
        raise ProjectionError(f"Unknown projection {technique!r}")

    caption = f"{technique}: " + ", ".join(f"{k}={v}" for k, v in settings.items())
    return np.asarray(coords), caption


def fit_clusters(embeddings, n_clusters):
    """Fits the clustering the earlier work both coloured by and measured against.

    Args:
        embeddings (np.ndarray): Representation matrix.
        n_clusters (int): Number of clusters.

    Returns:
        KMeans: The fitted model, so the colouring and the metrics use the same one.
    """
    return KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(embeddings)


def colour_values(kind, meta, embeddings, features=None, n_clusters=4, clusters=None):
    """Returns the values a panel is coloured by, and how to read them.

    Args:
        kind (str): One of :data:`COLOURINGS`.
        meta (pd.DataFrame): Window metadata.
        embeddings (np.ndarray): Representation, used only by ``cluster``.
        features (pd.DataFrame): Expert features, used only by ``band_power``.
        n_clusters (int): Clusters for ``cluster`` when none is supplied.
        clusters (KMeans): Clustering already fitted on this representation.

    Returns:
        tuple: (values per window, label, whether the scale is categorical).

    Raises:
        ValueError: If the colouring is unknown, or its source is missing.
    """
    if kind == "cluster":
        # The clusters the earlier work coloured by; they come from the representation
        # itself, so they describe the arrangement rather than anything known about it.
        model = clusters if clusters is not None else fit_clusters(embeddings, n_clusters)
        return model.labels_, f"KMeans (k={model.n_clusters})", True
    if kind == "age":
        return meta["age"].to_numpy(), "Age (months)", True
    if kind == "subject":
        return meta["subject"].to_numpy(), "Subject", False
    if kind == "band_power":
        if features is None:
            raise ValueError(
                "band_power needs the expert features; build them with "
                "src/build_expert_features.py or drop it from --color_by"
            )
        columns = [c for c in features.columns if c.startswith("total_power_")]
        if not columns:
            raise ValueError("The expert features carry no total_power column")
        return features[columns].mean(axis=1).to_numpy(), "Total power (1-20 Hz)", False
    raise ValueError(f"Unknown colouring {kind!r}; expected one of {COLOURINGS}")


def draw_grid(panels, colourings, caption, save_path, dpi=200):
    """Draws every method against every colouring in one figure.

    One row per method and one column per colouring, which is what allows the methods to be
    compared at a glance rather than across pages.

    Args:
        panels (dict): Method name to (coordinates, {colouring: (values, label, categorical)}).
        colourings (list): Colourings, in column order.
        caption (str): Projection settings, written under the figure.
        save_path (str): Where the figure is written.
        dpi (int): Resolution; the grid holds many panels, so it does not need print dpi.
    """
    methods = list(panels)
    fig, axes = plt.subplots(len(methods), len(colourings),
                             figsize=(4.2 * len(colourings), 3.8 * len(methods)),
                             squeeze=False)
    for row, method in enumerate(methods):
        coords, colours = panels[method]
        for col, kind in enumerate(colourings):
            ax = axes[row][col]
            values, label, categorical = colours[kind]
            if categorical:
                for level in np.unique(values):
                    mask = values == level
                    ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.7,
                               label=str(level))
                if row == 0 and col == 0:
                    ax.legend(fontsize=6, markerscale=2, loc="best")
            else:
                numeric = pd.factorize(values)[0] if values.dtype == object else values
                ax.scatter(coords[:, 0], coords[:, 1], c=numeric, cmap="viridis",
                           s=6, alpha=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(method, fontsize=10)

    fig.suptitle("Latent spaces", fontsize=13)
    fig.text(0.5, 0.005, caption, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Grid written to: {save_path}", flush=True)


def draw_per_method(panels, colourings, caption, save_dir):
    """Draws one figure per method and colouring, in the format of the earlier work.

    Args:
        panels (dict): As in :func:`draw_grid`.
        colourings (list): Colourings to draw.
        caption (str): Projection settings, written under each figure.
        save_dir (str): Directory the figures are written to.
    """
    os.makedirs(save_dir, exist_ok=True)
    for method, (coords, colours) in panels.items():
        for kind in colourings:
            values, label, categorical = colours[kind]
            fig, ax = plt.subplots(figsize=(8, 6))
            if categorical:
                for level in np.unique(values):
                    mask = values == level
                    ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.7,
                               label=str(level))
                ax.legend(title=label, bbox_to_anchor=(1.02, 1), loc="upper left",
                          fontsize=7)
            else:
                numeric = pd.factorize(values)[0] if values.dtype == object else values
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=numeric, cmap="viridis",
                                     s=10, alpha=0.7)
                fig.colorbar(scatter, ax=ax, label=label)
            ax.set_title(f"{method} coloured by {label}")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            fig.text(0.5, 0.005, caption, ha="center", fontsize=8)
            fig.tight_layout(rect=(0, 0.03, 1, 1))
            safe = f"{method}_{kind}".replace(" ", "_").replace("-", "_")
            path = os.path.join(save_dir, f"latent_{safe}.png")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"[INFO] Per-method figures written to: {save_dir}", flush=True)


def extract_embeddings(method, windows, args, device):
    """Loads the encoder a method left behind and returns its representation.

    ``PCA`` and ``raw`` need no encoder: they are the baselines the earlier work compared
    against, and they are computed from the windows themselves.

    Args:
        method (str): Method name, including variants.
        windows (np.ndarray): Windows, shape (n_windows, n_channels, n_samples).
        args (argparse.Namespace): Parsed arguments.
        device (torch.device): Device to run inference on.

    Returns:
        np.ndarray: L2-normalised representation, shape (n_windows, n_dims).

    Raises:
        CheckpointNotFound: If the method has no checkpoint for this fold and zone.
    """
    from subject_identity_probe import find_checkpoint, load_backbone

    flat = windows.reshape(len(windows), -1)
    if method == "raw":
        embeddings = flat
    elif method == "PCA":
        embeddings = PCA(n_components=args.embedding_size,
                         **PCA_PARAMS).fit_transform(flat)
    else:
        path = find_checkpoint(method, args.fold_id, args.model_dirs, args.zone,
                               args.frequency)
        backbone = load_backbone(method, path, windows.shape[1], windows.shape[2],
                                 args.embedding_size, args.sampling_frequency, device)
        embeddings = infer_embeddings_in_batches(backbone, windows,
                                                 batch_size=args.batch_size, device=device)
        print(f"[INFO] {method}: {os.path.basename(path)}", flush=True)

    # As the earlier work did, so the projections are comparable with its figures.
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.asarray(embeddings) / (norm + 1e-12)


def main(args):
    """Builds the requested figures and the table that backs them.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Raises:
        SystemExit: If no method could be represented.
    """
    import run_downstream as rd

    windows_path, meta_path = rd.config_data_paths(args.zone, args.frequency)
    windows = np.load(windows_path)
    meta = pd.read_csv(meta_path)
    features = None
    if "band_power" in args.color_by:
        parquet = os.path.join("data", "processed", "expert_features",
                               "window_features.parquet")
        features = pd.read_parquet(parquet) if os.path.exists(parquet) else None

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] {len(windows)} windows, zone {args.zone!r}, device={device}", flush=True)

    panels, representations = {}, {}
    for method in args.methods:
        try:
            embeddings = extract_embeddings(method, windows, args, device)
        except Exception as exc:                                    # noqa: BLE001
            # A method without weights is left out of the figure and said so, rather than
            # leaving a gap the reader would have to guess at.
            print(f"[WARNING] {method} left out: {exc}", flush=True)
            continue

        coords, caption = project(embeddings, args.projection)
        clusters = fit_clusters(embeddings, args.n_clusters)
        colours = {kind: colour_values(kind, meta, embeddings, features, args.n_clusters,
                                       clusters)
                   for kind in args.color_by}
        panels[method] = (coords, colours)
        # The metrics of the earlier work read the representation and the clustering
        # together, so both travel as a pair.
        representations[method] = (embeddings, clusters)

    if not panels:
        raise SystemExit("[ERROR] No method could be represented.")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.layout in ("grid", "both"):
        draw_grid(panels, args.color_by, caption,
                  os.path.join(args.save_dir, f"latent_grid_{args.projection}.png"),
                  dpi=args.dpi)
    if args.layout in ("per_method", "both"):
        draw_per_method(panels, args.color_by, caption, args.save_dir)

    # The same table the earlier work reported, so its rows can be compared with these.
    legacy = compute_clustering_metrics(representations, meta=meta)
    path = os.path.join(args.save_dir, "clustering_metrics.csv")
    legacy.to_csv(path, index=False)
    print(f"[INFO] Clustering metrics written to: {path}", flush=True)

    with open(os.path.join(args.save_dir, "projection.json"), "w") as fh:
        json.dump({"projection": args.projection, "settings": caption,
                   "methods": list(panels), "zone": args.zone,
                   "frequency": args.frequency, "fold_id": args.fold_id}, fh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw the latent spaces and measure whether they can be read."
    )
    parser.add_argument("--methods", nargs="+",
                        default=["PCA", "AE", "MAE", "SimCLR", "VAE", "InterFusion", "ExpCLR"],
                        help="Methods to represent. Those without weights are left out.")
    parser.add_argument("--projection", choices=["tsne", "pca"], default="tsne",
                        help="How the representation is taken down to two dimensions.")
    parser.add_argument("--layout", choices=["grid", "per_method", "both"], default="grid",
                        help="One figure comparing every method, one figure per method, "
                             "or both.")
    parser.add_argument("--color_by", nargs="+", default=["age", "subject", "band_power"],
                        choices=list(COLOURINGS),
                        help="What each panel is coloured by.")
    parser.add_argument("--n_clusters", type=int, default=4,
                        help="Clusters for the 'cluster' colouring.")
    parser.add_argument("--fold_id", type=str, default="fold0", help="Fold to represent.")
    parser.add_argument("--zone", type=str, default="all", help="Head zone.")
    parser.add_argument("--frequency", type=str, default="all", help="Frequency band.")
    parser.add_argument("--model_dirs", nargs="+", default=["save/models"],
                        help="Checkpoint directories, in priority order.")
    parser.add_argument("--save_dir", type=str, default="save/figures/latent",
                        help="Output directory.")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Resolution of the grid, which holds many panels.")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--sampling_frequency", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=512)
    main(parser.parse_args())
