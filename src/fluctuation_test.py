"""
This script is a convergence or fluctuation test to measure 
how clusters are found in different clustered SimCLR's trainings representation.

Context: SimCLR is a very sensible method. It changes behaviour with:
  - Batch size
  - Quality and amount of augmentations
  - Temperature parameter

In this test we aim to guarantee that our parameter set allows SimCLR to converge 
to reproducible results across multiple trainings.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from train_simclr import CIMCYCDataset
from models import EnhancedAttentionLSTM
from loss import NTXentLoss
from utils import infer_embeddings_in_batches

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ------------------------------------------------------------
# 1) COMPUTE k-METRICS (NO PLOTS)
# ------------------------------------------------------------
def compute_k_metrics(embeddings_np, k_range=range(2, 10)):
    inertias, sl_scores, db_scores, ch_scores = [], [], [], []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(embeddings_np)
        inertias.append(kmeans.inertia_)
        sl_scores.append(silhouette_score(embeddings_np, kmeans.labels_))
        db_scores.append(davies_bouldin_score(embeddings_np, kmeans.labels_))
        ch_scores.append(calinski_harabasz_score(embeddings_np, kmeans.labels_))

    return {
        "k": list(k_range),
        "inertia": inertias,
        "silhouette": sl_scores,
        "db": db_scores,
        "ch": ch_scores
    }

# ------------------------------------------------------------
# 2) PLOT SUPERPOSED METRICS USING THE SAME STYLE AS found_k_clusters
# ------------------------------------------------------------
def plot_superposed_k_metrics(metrics_list, age, save_path):
    """
    metrics_list = list of dicts returned by compute_k_metrics (one per repetition)
    Produces the SAME 2x2 layout as found_k_clusters BUT overlaying all repetitions.
    """

    k_vals = metrics_list[0]["k"]

    # Expand into long-format rows for sns.lineplot
    rep_rows_inertia = []
    rep_rows_sil = []
    rep_rows_db = []
    rep_rows_ch = []

    for rep_i, m in enumerate(metrics_list, start=1):
        for k, v in zip(k_vals, m["inertia"]):
            rep_rows_inertia.append({"k": k, "value": v, "rep": f"Rep {rep_i}"})
        for k, v in zip(k_vals, m["silhouette"]):
            rep_rows_sil.append({"k": k, "value": v, "rep": f"Rep {rep_i}"})
        for k, v in zip(k_vals, m["db"]):
            rep_rows_db.append({"k": k, "value": v, "rep": f"Rep {rep_i}"})
        for k, v in zip(k_vals, m["ch"]):
            rep_rows_ch.append({"k": k, "value": v, "rep": f"Rep {rep_i}"})

    elbow_df      = pd.DataFrame(rep_rows_inertia)
    silhouette_df = pd.DataFrame(rep_rows_sil)
    db_df         = pd.DataFrame(rep_rows_db)
    ch_df         = pd.DataFrame(rep_rows_ch)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Inertia
    sns.lineplot(data=elbow_df, x="k", y="value", hue="rep",
                 marker="o", ax=axes[0,0])
    axes[0,0].set_title(f"Elbow Method — Inertia (Age {age})")
    axes[0,0].set_xlabel("Number of Clusters (k)")
    axes[0,0].set_ylabel("Inertia")
    axes[0,0].grid(True, linestyle="--", alpha=0.7)

    # Silhouette
    sns.lineplot(data=silhouette_df, x="k", y="value", hue="rep",
                 marker="o", ax=axes[0,1])
    axes[0,1].set_title(f"Silhouette (Age {age})")
    axes[0,1].set_xlabel("Number of Clusters (k)")
    axes[0,1].set_ylabel("Silhouette")
    axes[0,1].grid(True, linestyle="--", alpha=0.7)

    # Davies-Bouldin
    sns.lineplot(data=db_df, x="k", y="value", hue="rep",
                 marker="o", ax=axes[1,0])
    axes[1,0].set_title(f"Davies-Bouldin (Age {age})")
    axes[1,0].set_xlabel("Number of Clusters (k)")
    axes[1,0].set_ylabel("Davies-Bouldin")
    axes[1,0].grid(True, linestyle="--", alpha=0.7)

    # Calinski-Harabasz
    sns.lineplot(data=ch_df, x="k", y="value", hue="rep",
                 marker="o", ax=axes[1,1])
    axes[1,1].set_title(f"Calinski-Harabasz (Age {age})")
    axes[1,1].set_xlabel("Number of Clusters (k)")
    axes[1,1].set_ylabel("Calinski-Harabasz")
    axes[1,1].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Superposed plot saved → {save_path}")

# ------------------------------------------------------------
# 3) TRAIN ONE MODEL
# ------------------------------------------------------------
def train_one_model(X, X_age, age, device, hidden_size, batch_size, num_epochs, sampling_frequency, temperature, seed=42):

    dataset = CIMCYCDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))

    model = EnhancedAttentionLSTM(
        input_size=X_age.shape[2],
        hidden_size=hidden_size,
        n_channels=X_age.shape[1],
        sfreq=sampling_frequency,
        lstm_hidden_size=hidden_size // 2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = NTXentLoss(device, batch_size=batch_size, temperature=temperature)

    for epoch in range(num_epochs):
        model.train()
        for anchor, aug1, aug2 in loader:
            anchor, aug1, aug2 = anchor.to(device), aug1.to(device), aug2.to(device)

            optimizer.zero_grad()
            z1 = model(aug1)
            z2 = model(aug2)
            loss = criterion(z1, z2, anchor)
            loss.backward()
            optimizer.step()

    return model

# ------------------------------------------------------------
# 4) REPEAT TRAINING & GENERATE SUPERPOSED METRICS FIGURE
# ------------------------------------------------------------
def repeat_train_and_cluster(
    X, X_by_age, ages,
    device,
    save_fig_dir,
    hidden_size=128,
    batch_size=512,
    num_epochs=50,
    sampling_frequency=250,
    temperature=0.05,
    repetitions=5
):

    os.makedirs(save_fig_dir, exist_ok=True)
    metrics_by_age = {age: [] for age in ages}

    for age in ages:
        if age not in X_by_age:
            print(f"[WARN] No data for age {age}.")
            continue

        print(f"\n===== AGE {age} =====")
        X_age = X_by_age[age]

        for r in range(1, repetitions + 1):
            print(f"\n--- Repetition {r}/{repetitions} for age {age} ---")

            # Train
            model = train_one_model(
                X, X_age, age, device,
                hidden_size, batch_size, num_epochs,
                sampling_frequency, temperature
            )

            # Embeddings
            emb = infer_embeddings_in_batches(model, X_age)

            # Metrics
            metrics = compute_k_metrics(emb)
            metrics_by_age[age].append(metrics)

        # Plot superposed figure
        out_path = os.path.join(save_fig_dir, f"SimCLR_metrics_age_{age}.png")
        plot_superposed_k_metrics(metrics_by_age[age], age, out_path)

# ------------------------------------------------------------
# 5) MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":

    data_dir = "data/processed/5_s"
    X = np.load(os.path.join(data_dir, "processed_windows.npy"))
    meta = pd.read_csv(os.path.join(data_dir, "processed_metadata.csv"))

    X_by_age = {}
    for age in sorted(meta["age"].unique()):
        idx = meta[meta["age"] == age].index.to_numpy()
        X_by_age[age] = X[idx]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    repeat_train_and_cluster(
        X=X,
        X_by_age=X_by_age,
        ages=[6, 9, 16, 36],
        device=device,
        save_fig_dir="save/figures/clustering_convergence_test",
        repetitions=5,
        batch_size=512,
        num_epochs=50
    )