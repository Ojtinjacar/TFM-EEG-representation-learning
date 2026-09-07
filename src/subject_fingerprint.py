import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils import infer_embeddings_in_batches
from models import EnhancedAttentionLSTM, CNNAutoencoder, AttentionLSTMAutoencoder, MaskedAttentionLSTMAutoencoder

"""
 In this script, we run all experiments associated with fingerprint characterization per subject.


"""

# ---------------------------------------------------------------------
# Visualization function
# ---------------------------------------------------------------------
def plot_patient_temporal_highlight(df_vis, paciente_destacado,
                                    height=9, width=12, save_path=None):
    """
    Highlights a subject in the latent space, colouring its windows by recording session.
    df_vis must have columns: ['Dim 1', 'Dim 2', 'subject', 'age']
    """
    plt.figure(figsize=(width, height))
    
    # Highlighted subject
    df_p = df_vis[df_vis['subject'] == paciente_destacado]
    recordings = df_vis['age'].unique()  # use all ages to assign consistent colours
    cmap = plt.get_cmap('tab10')
    color_map = {rec: cmap(i % 10) for i, rec in enumerate(recordings)}
    
    # Background: other subjects, coloured by age but with low opacity
    df_rest = df_vis[df_vis['subject'] != paciente_destacado]
    for rec in recordings:
        df_rec_rest = df_rest[df_rest['age'] == rec]
        if not df_rec_rest.empty:
            plt.scatter(df_rec_rest['Dim 1'], df_rec_rest['Dim 2'],
                        color=color_map[rec], alpha=0.15, s=20,
                        label=f"Other patients - {rec}")
    
    # Highlighted subject: coloured by recording session (higher opacity)
    for rec in df_p['age'].unique():
        df_rec = df_p[df_p['age'] == rec]
        plt.scatter(df_rec['Dim 1'], df_rec['Dim 2'],
                    label=f"{paciente_destacado} - {rec}",
                    color=color_map[rec], alpha=0.9, s=50, edgecolor='black')
    
    plt.title(f"Latent space — Patient {paciente_destacado}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Subject fingerprint analysis pipeline"
    )

    # Data paths
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/5_s/processed_windows.npy",
        help="Path to the .npy file with EEG windows (X).",
    )

    parser.add_argument(
        "--meta-path",
        type=str,
        default="data/processed/5_s/processed_metadata.csv",
        help="Path to the metadata CSV.",
    )


    parser.add_argument(
        "--save-fig-dir",
        type=str,
        default="save/figures/subject_fingerprint/AE",
        help="Directory to save figures.",
    )

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
        help="Sampling frequency (for time-based models).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training/inference.",
    )

    parser.add_argument(
        "--subjects", 
        nargs="+", 
        default=None, 
        help="List of subject IDs to highlight"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model file to use.",
    )

    args = parser.parse_args()


    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------
    print("[INFO] Loading data...")
    X = np.load(args.data_path)
    meta = pd.read_csv(args.meta_path)
    print("EEG dimensions: ", X.shape)

    # -----------------------------------------------------------------
    # Load model and extract representations
    # -----------------------------------------------------------------
    print(f"[INFO] Loading model {args.model_path} and extracting representations...")
    
    if "MAE" in args.model_path:
        model = MaskedAttentionLSTMAutoencoder(
            input_size=X.shape[2],
            hidden_size=args.hidden_size,
            n_channels=X.shape[1],
            sfreq=args.sampling_freq,
            lstm_hidden_size=args.hidden_size // 2
        ).to(device)
    elif "AE" in args.model_path:
        model = AttentionLSTMAutoencoder(
            input_size=X.shape[2],
            hidden_size=args.hidden_size,
            n_channels=X.shape[1],
            sfreq=args.sampling_freq,
            lstm_hidden_size=args.hidden_size // 2
        ).to(device)
    elif "SimCLR" or "Triplet" in args.model_path:
        model = EnhancedAttentionLSTM(
            input_size=X.shape[2],
            hidden_size=args.hidden_size,
            n_channels=X.shape[1],
            sfreq=args.sampling_freq,
            lstm_hidden_size=args.hidden_size // 2
        ).to(device)
    else:
        raise ValueError("Unrecognised model. Use: SimCLR, AE, MAE or Triplet.")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Helper function to infer embeddings
    representation = infer_embeddings_in_batches(model, X, batch_size=args.batch_size, device=device)
    norm = np.linalg.norm(representation, axis=1, keepdims=True)
    representation = representation / (norm + 1e-12)
    print('Dimensiones de las representaciones: ', representation.shape)

    # -----------------------------------------------------------------
    # t-SNE projection
    # -----------------------------------------------------------------
    print("[INFO] Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(representation)

    df_vis = pd.DataFrame({
        "Dim 1": emb_2d[:,0],
        "Dim 2": emb_2d[:,1],
        "subject": meta["subject"].values,
        "age": meta["age"].values
    })

    # -----------------------------------------------------------------
    # Generate plots for each subject in the list
    # -----------------------------------------------------------------

    if args.subjects is None:
        args.subjects = meta["subject"].unique().tolist()
        print(f"[INFO] No subjects specified, using all: {args.subjects}")

    os.makedirs(args.save_fig_dir, exist_ok=True)
    for subject in args.subjects:
        print(f"[INFO] Generating plot for subject {subject}...")
        save_path = os.path.join(args.save_fig_dir, f"subject_windows_{subject}.png")
        plot_patient_temporal_highlight(df_vis, subject, save_path=save_path)

    print("[INFO] Process completed.")

if __name__ == "__main__":
    main()