"""

 Visualisation of embeddings generated via Contrastive Learning

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.express as px
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

def explained_variance(embeddings):

    # Calcular PCA
    print(embeddings.shape)
    n_components = embeddings.shape[1]  # Todas las dimensiones
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(embeddings)
    explained_var = pca.explained_variance_ratio_

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, n_components + 1), explained_var * 100, marker='o')
    plt.title("Explained variance by principal components")
    plt.xlabel("Component")
    plt.ylabel("Explained variance (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Total explained variance:", round(np.sum(explained_var) * 100, 2), "%")
    for i, val in enumerate(explained_var[:5]):
        print(f"Component {i+1}: {round(val * 100, 2)}%")

def collapse_check(embeddings):

    stds = np.std(embeddings, axis=0)        # Standard deviation per dimension
    mean_std = np.mean(stds)                 # Mean standard deviation
    print("STD per dimension:", stds)
    print("Mean STD:", mean_std)

    # embeddings: [n_samples, embedding_dim]
    pairwise_dists = np.linalg.norm(embeddings[:, None] - embeddings[None, :], axis=-1)
    mean_dist = np.mean(pairwise_dists)
    print(f"Mean pairwise distance between embeddings: {mean_dist:.6f}")

    sim_matrix = cosine_similarity(embeddings)
    mean_sim = (sim_matrix - np.eye(sim_matrix.shape[0])).mean()
    print(f"Mean cosine similarity (off-diagonal): {mean_sim:.4f}")

    N, d = embeddings.shape

    # Step 1: Compute the mean of the embeddings
    z_mean = np.mean(embeddings, axis=0)

    # Step 2: Compute the covariance matrix C ∈ ℝ^{d x d}
    centered_embeddings = embeddings - z_mean  # [N, d]
    C = (centered_embeddings.T @ centered_embeddings) / N  # [d, d]

    # Step 3: SVD decomposition: C = U S V^T, where S = diag(σ_k)
    U, S, Vt = np.linalg.svd(C)

    # Step 4: Plot singular values on a logarithmic scale
    plt.figure(figsize=(8, 5))
    plt.plot(np.log(S + 1e-10), marker='o')  # +1e-10 to avoid log(0)
    plt.title("Log of singular values (effective dimensionality)")
    plt.xlabel("Dimension index")
    plt.ylabel("log(σ_k)")
    plt.grid(True)
    plt.show()

    # Step 5: Count how many values are "collapsed" (below a small threshold)
    collapse_threshold = 0.05
    num_collapsed = np.sum(S < collapse_threshold)
    print(f"Number of collapsed dimensions: {num_collapsed}/{d}")

def plot_3d_interactive(embeddings_3d, metadata, color_by="Age"):
    df = pd.DataFrame({
        "x": embeddings_3d[:, 0],
        "y": embeddings_3d[:, 1],
        "z": embeddings_3d[:, 2],
        color_by: metadata
    })

    fig = px.scatter_3d(
        df, x="x", y="y", z="z", color=color_by,
        title=f"3D Embeddings colored by {color_by}",
        opacity=0.7
    )
    fig.show()

def main():
    # save_dir = "save_6_12/save_1000_005_256"
    save_dir = "save"
    embeddings_path = os.path.join(save_dir, "embeddings.npy")
    metadata_path = os.path.join(save_dir, "metadata.npy")

    embeddings = np.load(embeddings_path)
    metadata = np.load(metadata_path, allow_pickle=True)

    # Standarized 
    # scaler = StandardScaler()
    # embeddings= scaler.fit_transform(embeddings)

    explained_variance(embeddings)

    # ----------------------------- #
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    print(tsne_results.shape)

    df_tsne = pd.DataFrame({
        "Dim-1": tsne_results[:, 0],
        "Dim-2": tsne_results[:, 1],
        "subject": metadata[:, 0],
        "age": metadata[:, 1].astype(int),
        "block": metadata[:, 3].astype(int),
        "window_quality": metadata[:, 5].astype(float)
    })

    # ----------------------------- #
    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(embeddings)

    df_pca = pd.DataFrame({
        "Dim-1": pca_results[:, 0],
        "Dim-2": pca_results[:, 1],
        "subject": metadata[:, 0],
        "age": metadata[:, 1].astype(int),
        "block": metadata[:, 3].astype(int),
        "window_quality": metadata[:, 5].astype(float)
    })

    # ----------------------------- #
    # Plot function
    def plot_by(df, feature, title, method_name):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x="Dim-1", y="Dim-2",
            hue=feature,
            palette="tab10",
            data=df,
            alpha=0.7
        )
        plt.title(f"{method_name} of Embeddings Colored by {title}")
        plt.legend(loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.tight_layout()
        plt.show()

    # ----------------------------- #
    # Plot all
    for feature, title in [
        ("subject", "Subject ID"),
        ("age", "Age (months)"),
        ("block", "Block Task (1=Pompas, 2=Video)"),
        ("window_quality", "Window Quality")
    ]:
        plot_by(df_tsne, feature, title, method_name="t-SNE")
        plot_by(df_pca, feature, title, method_name="PCA")
    
    # === 3D Visualisation with t-SNE ===
    tsne_3d = TSNE(n_components=3, perplexity=30, learning_rate='auto', n_iter=1000, random_state=42)
    tsne_3d_results = tsne_3d.fit_transform(embeddings)

    print("Visualising 3D t-SNE by age")
    plot_3d_interactive(tsne_3d_results, metadata[:, 1].astype(int))

    collapse_check(embeddings)

    pca = PCA(n_components=3, random_state=42)
    pca_3d = pca.fit_transform(embeddings)

    print("Visualizando PCA 3D por edad")
    plot_3d_interactive(pca_3d, metadata[:, 1].astype(int))

if __name__ == "__main__":
    main()