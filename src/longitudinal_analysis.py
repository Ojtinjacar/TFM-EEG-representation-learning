import os
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns 

from scipy import signal

import plotly.graph_objects as go

from loss import NTXentLoss
from models import EnhancedAttentionLSTM
from train_simclr import CIMCYCDataset

from utils import (
    infer_embeddings_in_batches,
    found_k_clusters,
    plot_random_samples_by_age,
    save_model,
    visual_clusters,
    plot_patient_highlight,
    plot_patient_cluster_distribution,
    CLUSTER_COLORS_RGBA
)

def plot_sankey_patient_flow_from_dfvis(
    df_vis_by_age: Dict[int, pd.DataFrame],
    ages: List[int],
    output_path: str,
    cluster_color_map: Optional[Dict[int, str]] = None,
    include_patient_names_in_labels: bool = True,
):
    """
    Draws and saves a multi-age Sankey diagram showing patient flow between
    clusters across several ages (e.g. [6, 9, 16, 36]).

    df_vis_by_age: dict age -> DataFrame with columns:
                   ['Dim 1', 'Dim 2', 'Cluster', 'Paciente']

    ages          : list of ages in months in the desired display order,
                    e.g. [6, 9, 16, 36].

    output_path   : full path to the output file (png).

    cluster_color_map: optional dict {local_cluster_id: rgba_color}
                       If an entry exists for the local cluster ID (0, 1, 2, ...),
                       that colour is used directly (to match the t-SNE plot).
                       Otherwise the reference-cluster mapping is used.

    include_patient_names_in_labels: if True, node labels include
                                     the list of patients (one per line).
    """
    # Ensure ages are sorted
    ages_sorted = sorted(ages)
    if len(ages_sorted) < 2:
        print("[WARN] At least 2 ages are needed to draw a Sankey diagram.")
        return

    # Check that df_vis exists for all ages
    for age in ages_sorted:
        if age not in df_vis_by_age:
            print(f"[WARN] No df_vis for age {age}, skipping Sankey.")
            return

    # ---------------------------------------------------
    # 1) Majority vote: 1 cluster per patient per age
    # ---------------------------------------------------
    pat_cluster_age: Dict[int, Dict[str, int]] = {}
    clusters_by_age: Dict[int, List[int]] = {}

    for age in ages_sorted:
        df = df_vis_by_age[age].copy()

        counts = (
            df.groupby(["Paciente", "Cluster"])
              .size()
              .reset_index(name="n")
        )
        majority = (
            counts.sort_values(["Paciente", "n"], ascending=[True, False])
                  .drop_duplicates("Paciente")
                  .reset_index(drop=True)
        )

        pat_cluster_age[age] = dict(zip(majority["Paciente"], majority["Cluster"]))
        clusters_by_age[age] = sorted(majority["Cluster"].unique())

    # Patients present in ALL ages
    common_patients = set(pat_cluster_age[ages_sorted[0]].keys())
    for age in ages_sorted[1:]:
        common_patients &= set(pat_cluster_age[age].keys())
    common_patients = sorted(common_patients)

    if not common_patients:
        print("[WARN] No patients are common across all specified ages.")
        return

    # ---------------------------------------------------
    # 2) Align clusters to reference age (for BM/EI/XF equivalences)
    # ---------------------------------------------------
    ref_age = ages_sorted[0]
    ref_clusters = clusters_by_age[ref_age]

    to_ref: Dict[int, Dict[int, int]] = {}
    to_ref[ref_age] = {c: c for c in ref_clusters}

    for age in ages_sorted[1:]:
        mapping = {}
        for c in clusters_by_age[age]:
            pats_c = [p for p in common_patients if pat_cluster_age[age][p] == c]
            if pats_c:
                ref_labels = [pat_cluster_age[ref_age][p] for p in pats_c]
                values, counts = np.unique(ref_labels, return_counts=True)
                ref_c = values[np.argmax(counts)]
            else:
                ref_c = ref_clusters[0]
            mapping[c] = ref_c
        to_ref[age] = mapping

    # ---------------------------------------------------
    # 3) Colours: try local cluster ID first (t-SNE),
    #    fall back to reference cluster colour
    # ---------------------------------------------------
    if cluster_color_map is None:
        default_colors = [
            "rgba(72,61,139,1)",   # dark blue
            "rgba(178,34,34,1)",   # dark red
            "rgba(0,128,0,1)",     # green
            "rgba(255,165,0,1)",   # orange
            "rgba(0,139,139,1)",   # teal
            "rgba(199,21,133,1)",  # dark magenta
        ]
        cluster_color_map = {}
        for i, c in enumerate(ref_clusters):
            cluster_color_map[c] = default_colors[i % len(default_colors)]

    # ---------------------------------------------------
    # 4) Define nodes: one per (age, local_cluster)
    # ---------------------------------------------------
    node_labels = []
    node_colors = []
    node_customdata = []
    node_x = []

    node_index: Dict[int, Dict[int, int]] = {}
    nA = len(ages_sorted)

    for j, age in enumerate(ages_sorted):
        node_index[age] = {}
        x_pos = 0.0 if nA == 1 else j / (nA - 1)

        for c in clusters_by_age[age]:
            idx = len(node_labels)
            node_index[age][c] = idx

            pats_c = [p for p in common_patients if pat_cluster_age[age][p] == c]
            pats_c_sorted = sorted(pats_c)
            patients_str = ", ".join(pats_c_sorted)

            if include_patient_names_in_labels:
                if pats_c_sorted:
                    label = "<br>" + "<br>".join(pats_c_sorted)
                else:
                    label = ""
            else:
                label = ""

            node_labels.append(label)

            # Colour: try local ID first, fall back to reference
            if c in cluster_color_map:
                color = cluster_color_map[c]
            else:
                ref_c = to_ref[age].get(c, ref_clusters[0])
                color = cluster_color_map.get(ref_c, "rgba(128,128,128,1)")

            node_colors.append(color)
            node_customdata.append(patients_str)
            node_x.append(x_pos)

    # ---------------------------------------------------
    # 5) Flows between consecutive ages
    # ---------------------------------------------------
    link_map: Dict[tuple, List[str]] = {}

    for i in range(len(ages_sorted) - 1):
        a_from = ages_sorted[i]
        a_to = ages_sorted[i + 1]
        for p in common_patients:
            c_from = pat_cluster_age[a_from][p]
            c_to = pat_cluster_age[a_to][p]
            key = (a_from, c_from, a_to, c_to)
            if key not in link_map:
                link_map[key] = []
            link_map[key].append(p)

    sources = []
    targets = []
    values = []
    link_colors = []
    link_customdata = []

    for (a_from, c_from, a_to, c_to), pats in link_map.items():
        src_idx = node_index[a_from][c_from]
        tgt_idx = node_index[a_to][c_to]
        count = len(pats)

        sources.append(src_idx)
        targets.append(tgt_idx)
        values.append(count)

        pats_sorted = sorted(pats)
        link_customdata.append(", ".join(pats_sorted))

        # Link colour: try source local cluster, fall back to reference
        if c_from in cluster_color_map:
            base_col = cluster_color_map[c_from]
        else:
            ref_c = to_ref[a_from].get(c_from, ref_clusters[0])
            base_col = cluster_color_map.get(ref_c, "rgba(128,128,128,1)")

        if base_col.startswith("rgba") and base_col.endswith(")"):
            link_col = base_col.replace("1)", "0.3)")
        else:
            link_col = base_col
        link_colors.append(link_col)

    # ---------------------------------------------------
    # 6) Build Sankey figure
    # ---------------------------------------------------
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=30,
            line=dict(width=0.5),
            label=node_labels,
            color=node_colors,
            customdata=node_customdata,
            hovertemplate=(
                "Patients in cluster:<br>%{customdata}<extra></extra>"
            ),
            x=node_x,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=link_customdata,
            hovertemplate=(
                "Flow %{source.label} → %{target.label}<br>" +
                "N patients: %{value}<br>" +
                "Patients: %{customdata}<extra></extra>"
            ),
        )
    )])

    # ---------------------------------------------------
    # 7) Age labels above each “pillar”
    # ---------------------------------------------------
    annotations = []
    unique_x = {}
    for age in ages_sorted:
        # All nodes at this age share x_pos — take it from the first cluster

        c0 = clusters_by_age[age][0]
        x_pos = node_x[node_index[age][c0]]
        unique_x[age] = x_pos

    for age in ages_sorted:
        annotations.append(
            dict(
                x=unique_x[age],
                y=1.05,
                xref="paper",
                yref="paper",
                text=f"{age} months",
                showarrow=False,
                font=dict(size=12, color="black"),
            )
        )

    title = "Patient flow between clusters"
    fig.update_layout(
        title_text=title,
        font=dict(size=10),
        height=500,
        width=300 + 200 * (len(ages_sorted) - 1),
        margin=dict(l=20, r=20, t=50, b=20),
        annotations=annotations,
    )

    fig.write_image(output_path, scale=2)
    print(f"[INFO] Sankey saved to: {output_path}")

# -------------------------------------------------------------------------
# SPECTRAL FEATURES
# -------------------------------------------------------------------------
def features_window_multicanal(window, fs,
                               band=(6.0, 12.0),
                               welch_nperseg=None):
    """
    Computes features for a multi-channel window in a specific frequency band.

    Parameters
    ----------
    window : np.ndarray
        2D array (n_channels, n_samples).
    fs : float
        Sampling frequency in Hz.
    band : tuple(float, float)
        Frequency band of interest (low_hz, high_hz).
    welch_nperseg : int or None
        nperseg for Welch's method.

    Returns
    -------
    dict with:
        - 'per_channel': per-channel metrics
        - 'agg': mean metrics across channels
    """
    n_channels, n_samples = window.shape
    if welch_nperseg is None:
        welch_nperseg = min(256, n_samples)

    var = np.zeros(n_channels)
    rms = np.zeros(n_channels)
    ptp = np.zeros(n_channels)
    zcr = np.zeros(n_channels)
    hjorth_mobility = np.zeros(n_channels)
    hjorth_complexity = np.zeros(n_channels)
    band_power = np.zeros(n_channels)
    spec_entropy = np.zeros(n_channels)

    for ch in range(n_channels):
        x = window[ch].astype(float)
        x -= np.mean(x)

        # Temporal metrics
        var[ch] = np.var(x)
        rms[ch] = np.sqrt(np.mean(x ** 2))
        ptp[ch] = np.ptp(x)

        # Zero-crossing rate
        zc = np.sum(np.abs(np.diff(np.sign(x)))) / 2.0
        zcr[ch] = zc / float(max(1, n_samples - 1))

        # Hjorth
        dx = np.diff(x)
        ddx = np.diff(dx)
        mobility = np.sqrt(np.var(dx) / (var[ch] + 1e-16))
        complexity = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-16)) / (mobility + 1e-16)
        hjorth_mobility[ch] = mobility
        hjorth_complexity[ch] = complexity

        # PSD and band power
        freqs, Pxx = signal.welch(x, fs=fs, nperseg=welch_nperseg)

        low_hz, high_hz = band
        band_mask = (freqs >= low_hz) & (freqs <= high_hz)
        band_power[ch] = np.sum(Pxx[band_mask]) if np.any(band_mask) else 0.0

        # Spectral entropy
        pnorm = Pxx / (np.sum(Pxx) + 1e-16)
        spec_entropy[ch] = -np.sum(pnorm * np.log2(pnorm + 1e-16))

    per_channel = {
        'var': var,
        'rms': rms,
        'ptp': ptp,
        'zcr': zcr,
        'hjorth_mobility': hjorth_mobility,
        'hjorth_complexity': hjorth_complexity,
        'band_power': band_power,
        'spectral_entropy': spec_entropy
    }
    agg = {k + '_mean': np.nanmean(v) for k, v in per_channel.items()}
    return {'per_channel': per_channel, 'agg': agg}


# -------------------------------------------------------------------------
# PER-AGE SIMCLR TRAINING
# -------------------------------------------------------------------------
def train_models_per_age(
    X: np.ndarray,
    X_by_age: Dict[int, np.ndarray],
    ages: List[int],
    device: torch.device,
    save_model_dir: str,
    save_fig_dir: str,
    num_epochs: int,
    batch_size: int,
    hidden_size: int,
    sampling_frequency: int,
    temperature: float,
    seed: int = 42,
):
    """
    Trains one SimCLR model per age (simclr_age_{age}.pth) and saves:
      - the model
      - the loss curve figure
    """
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_fig_dir, exist_ok=True)

    for edad in ages:

        if edad not in X_by_age:
            print(f"[WARN] No data for age {edad}, skipping training.")
            continue

        print(f"\n=== Training model for age {edad} ===")
        train_dataset = CIMCYCDataset(X)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed)
        )

        model = EnhancedAttentionLSTM(
            input_size=X_by_age[edad].shape[2],
            hidden_size=hidden_size,
            n_channels=X_by_age[edad].shape[1],
            sfreq=sampling_frequency,
            lstm_hidden_size=hidden_size // 2
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = NTXentLoss(device, batch_size=batch_size, temperature=temperature)

        train_losses = []
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0
            for anchor, aug1, aug2 in train_loader:
                anchor, aug1, aug2 = anchor.to(device), aug1.to(device), aug2.to(device)
                optimizer.zero_grad()
                z1 = model(aug1)
                z2 = model(aug2)
                loss = criterion(z1, z2, anchor)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            epoch_train_loss = total_train_loss / len(train_loader)
            train_losses.append(epoch_train_loss)
            print(f"[Age {edad}] Epoch {epoch+1}/{num_epochs} - Loss: {epoch_train_loss:.4f}")

        # Save model
        filename = f"simclr_age_{edad}.pth"
        save_model(model, save_model_dir, filename=filename)
        print(f"[INFO] Model saved: {os.path.join(save_model_dir, filename)}")

        # Loss curve figure
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Learning Curve — Age {edad}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(save_fig_dir, f"loss_curve_SimCLR_age_{edad}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Loss figure saved to: {fig_path}")


# -------------------------------------------------------------------------
# PER-AGE ANALYSIS HELPERS
# -------------------------------------------------------------------------
def load_model_for_age(
    edad: int,
    model_input_shape,
    device: torch.device,
    hidden_size: int,
    sampling_frequency: int,
    save_model_dir: str,
) -> Optional[torch.nn.Module]:
    """
    Loads simclr_age_{age}.pth from disk.
    Returns None if the file does not exist.
    """
    model_path = os.path.join(save_model_dir, f"simclr_age_{edad}.pth")
    if not os.path.exists(model_path):
        print(f"[WARN] Model not found for age {edad}: {model_path}")
        return None

    n_channels, input_size = model_input_shape
    model = EnhancedAttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        n_channels=n_channels,
        sfreq=sampling_frequency,
        lstm_hidden_size=hidden_size // 2
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[INFO] Model loaded for age {edad} from: {model_path}")
    return model


def compute_and_save_embeddings_for_age(
    edad: int,
    X_age: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    save_results_dir: str,
) -> np.ndarray:
    """
    Computes embeddings for windows of a specific age and saves them to disk.
    """
    emb = infer_embeddings_in_batches(model, X_age)
    emb_path = os.path.join(save_results_dir, f"embeddings_age_{edad}.npy")
    np.save(emb_path, emb)
    print(f"[INFO] Embeddings for age {edad} saved to: {emb_path}")
    return emb


def clustering_and_patient_plots_for_age(
    edad: int,
    emb: np.ndarray,
    meta: pd.DataFrame,
    k_clusters: int,
    paciente_destacado: Optional[str],
    save_fig_dir: str,
):
    """
    Runs:
      - K search (found_k_clusters)
      - t-SNE + KMeans visualisation (visual_clusters)
    """
    # K heuristic
    found_k_clusters(emb, filename=os.path.join(save_fig_dir, f"SimCLR_{edad}_k_search_.png"))

    # t-SNE + KMeans
    kmeans, graf = visual_clusters(emb, edad, n_clusters=k_clusters)

    # Save t-SNE figure (assumes visual_clusters has drawn it)
    fig_path = os.path.join(save_fig_dir, f"SimCLR_tsne_clusters_age_{edad}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] t-SNE cluster figure for age {edad} saved to: {fig_path}")

    # Visualisation DataFrame
    indices_edad = meta[meta['age'] == edad].index.to_numpy()
    patient_ids = meta.loc[indices_edad, 'subject'].to_numpy()

    df_visualizacion = pd.DataFrame(graf, columns=['Dim 1', 'Dim 2'])
    df_visualizacion['Cluster'] = kmeans.labels_
    df_visualizacion['Paciente'] = patient_ids

    return df_visualizacion, graf, kmeans


def specific_analysis_age36(
    X: np.ndarray,
    X_36: np.ndarray,
    graf_36: np.ndarray,
    meta: pd.DataFrame,
    sampling_frequency: int,
    save_fig_dir: str,
    sampling_idx1: int,
    sampling_idx2: int,
):
    """
    Specific analysis for 36 months:
      - random sample plots per subject
      - feature comparison for two specific windows
      - t-SNE coloured by power band (6–12 Hz)
    """
    # 1) Random samples for subjects of interest
    print("[INFO] Plotting random samples for subjects of interest (36 months)...")

    idxs_a, sps_a = plot_random_samples_by_age(
        meta=meta,
        X=X,
        age=36,
        subject_ids=["B042", "B048", "B260"]
    )
    fig_path = os.path.join(save_fig_dir, "random_samples_age36_group1.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] random_samples group 1 (36 months) figure saved to: {fig_path}")

    idxs_b, sps_b = plot_random_samples_by_age(
        meta=meta,
        X=X,
        age=36,
        subject_ids=["B048", "B093", "B074", "B063", "B010"]
    )
    fig_path = os.path.join(save_fig_dir, "random_samples_age36_group2.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] random_samples group 2 (36 months) figure saved to: {fig_path}")

    # 2) Feature comparison between two specific windows
    print("[INFO] Computing aggregated features for two specific windows...")
    feats1 = features_window_multicanal(X[sampling_idx1], sampling_frequency)['agg']
    feats2 = features_window_multicanal(X[sampling_idx2], sampling_frequency)['agg']

    keys = list(feats1.keys())
    values1 = [feats1[k] for k in keys]
    values2 = [feats2[k] for k in keys]

    n_features = len(keys)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        constrained_layout=True
    )

    color1 = 'tab:blue'
    color2 = 'tab:orange'
    axs = axs.flatten()

    for i, key in enumerate(keys):
        ax = axs[i]
        vals = [values1[i], values2[i]]
        ax.bar(
            [f"Window {sampling_idx1}", f"Window {sampling_idx2}"],
            vals,
            color=[color1, color2]
        )
        ax.set_title(key)
        ax.grid(axis='y')

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.suptitle('Aggregated feature comparison per window', fontsize=16)
    fig_path = os.path.join(save_fig_dir, "features_comparison_two_windows.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Feature comparison figure saved to: {fig_path}")

    # 3) Power band 6–12 Hz in t-SNE space (36 months)
    print("[INFO] Computing power_band_mean for all 36-month windows...")
    power_band_values = []
    for i in range(X_36.shape[0]):
        feats = features_window_multicanal(X_36[i], sampling_frequency)
        power_band_values.append(feats['agg']['band_power_mean'])
    power_band_values = np.array(power_band_values)

    df_visualizacion = pd.DataFrame(graf_36, columns=['Dim 1', 'Dim 2'])
    df_visualizacion['Power Band'] = power_band_values

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(
        df_visualizacion['Dim 1'], df_visualizacion['Dim 2'],
        c=df_visualizacion['Power Band'],
        cmap='coolwarm',
        s=50,
        edgecolor='k',
        alpha=0.8
    )
    plt.title('Windows in t-SNE space coloured by Power Band (6–12 Hz)')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Power Band (6-12 Hz)')
    plt.tight_layout()
    fig_path = os.path.join(save_fig_dir, "tsne_age36_colored_by_power_band.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] t-SNE figure coloured by power band (36 months) saved to: {fig_path}")


# -------------------------------------------------------------------------
# ORCHESTRATOR: PER-AGE ANALYSIS
# -------------------------------------------------------------------------
def analyse_per_age(
    X: np.ndarray,
    X_by_age: Dict[int, np.ndarray],
    meta: pd.DataFrame,
    ages: List[int],
    device: torch.device,
    save_model_dir: str,
    save_fig_dir: str,
    save_results_dir: str,
    hidden_size: int,
    sampling_frequency: int,
    k_map: Dict[int, int],
    highlight_map: Dict[int, str],
    sampling_idx1: int,
    sampling_idx2: int,
):
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_fig_dir, exist_ok=True)
    os.makedirs(save_results_dir, exist_ok=True)

    tsne_by_age = {}
    emb_by_age = {}
    df_vis_by_age = {}

    # Basic shape: (C, T)
    n_channels = X.shape[1]
    input_size = X.shape[2]
    model_input_shape = (n_channels, input_size)

    for edad in ages:
        if edad not in X_by_age:
            print(f"[WARN] No data for age {edad}, skipping analysis.")
            continue

        print(f"\n=== Analysis for age {edad} ===")
        X_age = X_by_age[edad]

        # Load model
        model = load_model_for_age(
            edad=edad,
            model_input_shape=model_input_shape,
            device=device,
            hidden_size=hidden_size,
            sampling_frequency=sampling_frequency,
            save_model_dir=save_model_dir,
        )
        if model is None:
            continue

        # Embeddings
        emb = compute_and_save_embeddings_for_age(
            edad=edad,
            X_age=X_age,
            model=model,
            device=device,
            save_results_dir=save_results_dir,
        )
        emb_by_age[edad] = emb

        # Clustering + visualisation
        df_vis, graf, kmeans = clustering_and_patient_plots_for_age(
            edad=edad,
            emb=emb,
            meta=meta,
            k_clusters=k_map.get(edad, 2),
            paciente_destacado=highlight_map.get(edad),
            save_fig_dir=save_fig_dir,
        )
        tsne_by_age[edad] = graf
        df_vis_by_age[edad] = df_vis

        # Window distribution per patient
        _ = plot_patient_cluster_distribution(
            df_vis,
            filename=os.path.join(
                save_fig_dir,
                f"SimCLR_{edad}_pacients_cluster_distribution.png"
            )
        )

    # Special analysis for age 36
    if 36 not in ages or 36 not in X_by_age or 36 not in tsne_by_age:
        print("\n[INFO] Age 36 not included or missing data/t-SNE. Skipping 36-month specific analysis.")
        return

    print("\n=== Specific analysis for age 36 months ===")
    X_36 = X_by_age[36]
    graf_36 = tsne_by_age[36]

    specific_analysis_age36(
        X=X,
        X_36=X_36,
        graf_36=graf_36,
        meta=meta,
        sampling_frequency=sampling_frequency,
        save_fig_dir=save_fig_dir,
        sampling_idx1=sampling_idx1,
        sampling_idx2=sampling_idx2,
    )

    # ---------------------------------------------------
    # Multi-age Sankey diagram (example: 6 → 9 → 16 → 36 months)
    # ---------------------------------------------------
    try:
        sankey_path = os.path.join(save_fig_dir, "sankey_diagram_pacient_cluster.png")

        plot_sankey_patient_flow_from_dfvis(
            df_vis_by_age=df_vis_by_age,
            ages=ages,
            output_path=sankey_path,
            cluster_color_map=CLUSTER_COLORS_RGBA,
            include_patient_names_in_labels=True,
        )
    except Exception as e:
        print(f"[WARN] Error drawing Sankey: {e}")

# -------------------------------------------------------------------------
# ARGUMENT PARSER
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-age SimCLR pipeline with clustering and feature analysis."
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/5_s",
        help="Base directory containing processed_windows.npy and processed_metadata.csv.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="processed_windows.npy",
        help="Name of the .npy file with EEG windows.",
    )
    parser.add_argument(
        "--meta-file",
        type=str,
        default="processed_metadata.csv",
        help="Name of the metadata CSV file.",
    )

    # Outputs
    parser.add_argument(
        "--save-model-dir",
        type=str,
        default="save/models",
        help="Directory to save trained SimCLR models.",
    )
    parser.add_argument(
        "--save-fig-dir",
        type=str,
        default="save/figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--save-results-dir",
        type=str,
        default="save/results",
        help="Directory to save result files.",
    )

    # Hyperparameters
    parser.add_argument(
        "--sampling-frequency",
        type=int,
        default=250,
        help="Sampling frequency of the EEG signals.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Latent space dimensionality.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs for SimCLR.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for SimCLR training.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Temperature for NTXentLoss.",
    )
    parser.add_argument(
        "--ages",
        type=int,
        nargs="+",
        default=[6, 9, 16, 36],
        help="List of ages (in months) to process.",
    )
    parser.add_argument(
        "--k-6",
        type=int,
        default=2,
        help="Number of clusters for age 6.",
    )
    parser.add_argument(
        "--k-9",
        type=int,
        default=2,
        help="Number of clusters for age 9.",
    )
    parser.add_argument(
        "--k-16",
        type=int,
        default=2,
        help="Number of clusters for age 16.",
    )
    parser.add_argument(
        "--k-36",
        type=int,
        default=3,
        help="Number of clusters for age 36.",
    )

    # Highlighted patients
    parser.add_argument(
        "--highlight-6",
        type=str,
        default="B063",
        help="Patient ID to highlight at age 6.",
    )
    parser.add_argument(
        "--highlight-9",
        type=str,
        default="B216",
        help="Patient ID to highlight at age 9.",
    )
    parser.add_argument(
        "--highlight-16",
        type=str,
        default="B260",
        help="Patient ID to highlight at age 16.",
    )
    parser.add_argument(
        "--highlight-36",
        type=str,
        default="B048",
        help="Patient ID to highlight at age 36.",
    )

    # Window indices for feature comparison
    parser.add_argument(
        "--feat-window-idx1",
        type=int,
        default=785,
        help="Index of the first window for feature comparison.",
    )
    parser.add_argument(
        "--feat-window-idx2",
        type=int,
        default=2290,
        help="Index of the second window for feature comparison.",
    )

    # Training flag
    parser.add_argument(
        "--train",
        action="store_true",
        help="If passed, trains the per-age SimCLR models.",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    args = parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data_path = os.path.join(args.data_dir, args.data_file)
    meta_path = os.path.join(args.data_dir, args.meta_file)

    print("[INFO] Loading data...")
    X = np.load(data_path)
    meta = pd.read_csv(meta_path)
    print("EEG shape: ", X.shape)

    # Age -> windows dictionary
    edades_unicas = sorted(meta['age'].unique())
    print("Unique ages in metadata:", edades_unicas)
    X_by_age: Dict[int, np.ndarray] = {}
    for edad in edades_unicas:
        indices = meta[meta['age'] == edad].index.to_numpy()
        X_by_age[edad] = X[indices]
        print(f"Age {edad}: {X_by_age[edad].shape[0]} windows")

    # Maps for cluster count and highlighted patient
    k_map = {
        6: args.k_6,
        9: args.k_9,
        16: args.k_16,
        36: args.k_36,
    }
    highlight_map = {
        6: args.highlight_6,
        9: args.highlight_9,
        16: args.highlight_16,
        36: args.highlight_36,
    }

    # Training
    if args.train:
        train_models_per_age(
            X=X,
            X_by_age=X_by_age,
            ages=args.ages,
            device=device,
            save_model_dir=args.save_model_dir,
            save_fig_dir=args.save_fig_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            sampling_frequency=args.sampling_frequency,
            temperature=args.temperature,
        )
    else:
        print("[INFO] --train flag not set; assuming models are already trained and saved to disk.")

    # Analysis / clustering
    analyse_per_age(
        X=X,
        X_by_age=X_by_age,
        meta=meta,
        ages=args.ages,
        device=device,
        save_model_dir=args.save_model_dir,
        save_fig_dir=args.save_fig_dir,
        save_results_dir=args.save_results_dir,
        hidden_size=args.hidden_size,
        sampling_frequency=args.sampling_frequency,
        k_map=k_map,
        highlight_map=highlight_map,
        sampling_idx1=args.feat_window_idx1,
        sampling_idx2=args.feat_window_idx2,
    )
    print("\n[INFO] Per-age SimCLR pipeline complete.")

if __name__ == "__main__":
    main()