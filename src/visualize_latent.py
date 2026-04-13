"""
Script to visualise the latent space coloured by age or IQ quartiles.

Processes fine-tuned models with target 'age' for all available methods,
including Triplet Loss models (age and cit_36mo).
Generates t-SNE visualisations of the latent space coloured by the corresponding label.
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models import EnhancedAttentionLSTM, AttentionLSTMAutoencoder, MaskedAttentionLSTMAutoencoder
from utils import infer_embeddings_in_batches, visualization_emb_by_label


# Available methods for analysis
AVAILABLE_METHODS = ["supervised", "SimCLR", "AE", "MAE", "Triplet_age", "Triplet_cit_36mo"]

# Modelos pre-entrenados (solo backbone, sin head)
PRETRAINED_METHODS = ["Triplet_age", "Triplet_cit_36mo"]

# Modelos fine-tuned (backbone + head)
FINETUNED_METHODS = ["supervised", "SimCLR", "AE", "MAE", "Triplet_age", "Triplet_cit_36mo"]


class Head(nn.Module):
    """Regression head for the fine-tuned model."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Linear(in_dim // 2, out_dim)
        )

    def forward(self, x):
        return self.fc(x)


class FullModel(nn.Module):
    """Backbone + head para cargar modelos fine-tuned."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        emb = self.backbone.get_embedding(x)
        out = self.head(emb)
        return out

    def get_embedding(self, x):
        """Extracts the backbone embedding."""
        return self.backbone.get_embedding(x)


def get_model_path(method, zone, frequency, save_model_dir="save/models",
                   pretrained=False, embedding_size=128, margin=0.4):
    """
    Builds the path to the model.

    Args:
        method: Representation method (SimCLR, AE, MAE, Triplet_age, Triplet_cit_36mo)
        zone: Brain zone (all, frontal, etc.)
        frequency: Frequency band (all, alpha, etc.)
        save_model_dir: Directory where models are stored
        pretrained: If True, loads pre-trained model; if False, loads fine-tuned
        embedding_size: Embedding size (only for pre-trained Triplet)
        margin: Triplet Loss margin (only for pre-trained Triplet)

    Returns:
        Path to the model
    """
    if pretrained and method in PRETRAINED_METHODS:
        # Pre-trained model (Triplet only): Triplet_age_all_all_emb128_m0.4.pth
        target = method.replace("Triplet_", "")
        filename = f"Triplet_{target}_{zone}_{frequency}_emb{embedding_size}_m{margin}.pth"
    elif method in PRETRAINED_METHODS:
        # Triplet fine-tuned model: TripletLoss_all_all_age_fine_tuning.pth
        target = method.replace("Triplet_", "")
        filename = f"TripletLoss_{zone}_{frequency}_{target}_fine_tuning.pth"
    elif method == "supervised":
        filename = f"{method}_{zone}_{frequency}_age.pth"
    else:
        # SimCLR, AE, MAE fine-tuned
        filename = f"{method}_{zone}_{frequency}_age_fine_tuning.pth"
    return os.path.join(save_model_dir, filename)


def load_model(method, model_path, input_size, n_channels, hidden_size=128,
               sfreq=250, device='cuda', pretrained=False):
    """
    Loads a pre-trained model (backbone only) or fine-tuned model (backbone + head).

    Args:
        method: Model type (SimCLR, AE, MAE, Triplet_age, Triplet_cit_36mo)
        model_path: Path to the .pth file
        input_size: Temporal length of the signals
        n_channels: Number of EEG channels
        hidden_size: Latent space dimensionality
        sfreq: Sampling frequency
        device: Device (cuda/cpu)
        pretrained: If True, loads only backbone; if False, loads backbone + head

    Returns:
        Model loaded in evaluation mode
    """
    model_params = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "n_channels": n_channels,
        "sfreq": sfreq,
        "lstm_hidden_size": hidden_size // 2,
    }

    # Select backbone architecture based on method
    if method == "SimCLR" or method.startswith("Triplet_") or method == "supervised":
        backbone = EnhancedAttentionLSTM(**model_params)
    elif method == "AE":
        backbone = AttentionLSTMAutoencoder(**model_params)
    elif method == "MAE":
        backbone = MaskedAttentionLSTMAutoencoder(**model_params)
    else:
        raise ValueError(f"Unknown method: {method}")

    if pretrained:
        # Load only the backbone (pre-trained model)
        backbone.load_state_dict(torch.load(model_path, map_location=device))
        backbone.to(device)
        backbone.eval()
        return backbone
    else:
        # Load complete model with head (fine-tuned)
        head = Head(hidden_size, 1)
        model = FullModel(backbone, head)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model


def extract_embeddings(model, X, batch_size=128, device='cuda'):
    """
    Extracts embeddings from the backbone of the fine-tuned model.

    Args:
        model: Loaded FullModel
        X: EEG data (N, C, T)
        batch_size: Batch size for inference
        device: Device

    Returns:
        Embeddings (N, hidden_size)
    """
    emb = infer_embeddings_in_batches(model, X, batch_size=batch_size, device=device)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norm + 1e-12)


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE visualisation by age for age fine-tuned models"
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

    # Zone and frequency configuration
    parser.add_argument(
        "--zone",
        type=str,
        default="all",
        help="Brain zone (all, frontal, central, etc.)",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="all",
        help="Frequency band (all, alpha, beta, etc.)",
    )

    # Methods to analyse
    parser.add_argument(
        "--methods",
        nargs="+",
        default=AVAILABLE_METHODS,
        choices=AVAILABLE_METHODS,
        help="Methods to analyse.",
    )

    # Output paths
    parser.add_argument(
        "--save-model-dir",
        type=str,
        default="save/models",
        help="Directory where pre-trained models are stored.",
    )
    parser.add_argument(
        "--save-fig-dir",
        type=str,
        default="save/figures/latent_spaces",
        help="Directory to save figures.",
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

    # Triplet Loss specific parameters
    parser.add_argument(
        "--triplet-margin",
        type=float,
        default=0.4,
        help="Margin used in Triplet Loss training.",
    )

    # Colormap
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for visualisation (viridis, plasma, coolwarm, etc.)",
    )

    args = parser.parse_args()

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    # Crear directorios de salida
    os.makedirs(args.save_fig_dir, exist_ok=True)

    # Cargar datos
    print("[INFO] Loading data...", flush=True)
    X = np.load(args.data_path)
    meta = pd.read_csv(args.meta_path)
    print(f"  EEG dimensions: {X.shape}", flush=True)
    print(f"  Number of subjects: {meta['subject'].nunique()}", flush=True)
    print(f"  Unique ages: {sorted(meta['age'].unique())}", flush=True)

    age_labels = meta['age'].values

    # Preparar cuartiles de IQ para Triplet_cit_36mo (igual que en train_triplet_loss.py)
    iq_quartile_labels = None
    iq_valid_mask = None
    iq_intervals = None
    iq_quartile_names = None  # Nombres descriptivos con rangos
    if 'Triplet_cit_36mo' in args.methods:
        if 'cit_36mo' in meta.columns:
            iq_valid_mask = ~meta['cit_36mo'].isna()
            try:
                iq_bins, iq_intervals = pd.qcut(
                    meta.loc[iq_valid_mask, 'cit_36mo'],
                    q=4, retbins=True, labels=False, duplicates='drop'
                )
                # Create label array with NaN for invalid values
                iq_quartile_labels = np.full(len(meta), np.nan)
                iq_quartile_labels[iq_valid_mask.values] = iq_bins.values

                # Create descriptive names with the range of each quartile
                iq_quartile_names = {}
                for i in range(len(iq_intervals) - 1):
                    low = iq_intervals[i]
                    high = iq_intervals[i + 1]
                    iq_quartile_names[i] = f"Q{i+1} [{low:.1f}, {high:.1f})"
                # The last quartile includes the maximum
                last_idx = len(iq_intervals) - 2
                iq_quartile_names[last_idx] = f"Q{last_idx+1} [{iq_intervals[last_idx]:.1f}, {iq_intervals[last_idx+1]:.1f}]"

                print(f"  IQ quartiles computed. Intervals: {iq_intervals}", flush=True)
                print(f"  Quartile names: {iq_quartile_names}", flush=True)
                print(f"  Valid samples for IQ: {iq_valid_mask.sum()}", flush=True)
            except ValueError as e:
                print(f"  [WARNING] Could not compute IQ quartiles: {e}", flush=True)
        else:
            print("  [WARNING] Column 'cit_36mo' not found in metadata.", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("t-SNE VISUALISATION — PRE-TRAINED AND FINE-TUNED MODELS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Methods: {args.methods}", flush=True)
    print(f"Zone: {args.zone}, Frequency: {args.frequency}", flush=True)

    # Process each method in both versions (pretrained and finetuned) if applicable
    for method in args.methods:
        # Determine which versions to process
        versions_to_process = []

        # Fine-tuned version (all methods can have fine-tuned)
        if method in FINETUNED_METHODS:
            versions_to_process.append(('finetuned', False))

        # Pre-trained version (Triplet only)
        if method in PRETRAINED_METHODS:
            versions_to_process.append(('pretrained', True))

        for version_name, is_pretrained in versions_to_process:
            model_path = get_model_path(
                method=method,
                zone=args.zone,
                frequency=args.frequency,
                save_model_dir=args.save_model_dir,
                pretrained=is_pretrained,
                embedding_size=args.hidden_size,
                margin=args.triplet_margin
            )

            # Check that the model exists
            if not os.path.exists(model_path):
                print(f"\n[{method}] [{version_name}] [WARNING] Model not found: {model_path}. Skipping...", flush=True)
                continue

            print(f"\n[{method}] [{version_name}] Processing model...", flush=True)
            print(f"  Model: {model_path}", flush=True)

            # Determine which data and labels to use
            if method == "Triplet_cit_36mo":
                if iq_quartile_labels is None or iq_valid_mask is None or iq_quartile_names is None:
                    print(f"  [WARNING] No IQ quartile labels available. Skipping...", flush=True)
                    continue
                # Use only samples with valid IQ
                X_method = X[iq_valid_mask.values]
                # Convert quartile indices to descriptive names with ranges
                quartile_indices = iq_quartile_labels[iq_valid_mask.values].astype(int)
                labels_method = np.array([iq_quartile_names[idx] for idx in quartile_indices])
                label_name = 'IQ Quartile'
                method_name = f"{method}_{version_name}"
            else:
                X_method = X
                labels_method = age_labels
                label_name = 'Age'
                method_name = f"{method}_{version_name}"

            # Load model
            model = load_model(
                method=method,
                model_path=model_path,
                input_size=X.shape[2],
                n_channels=X.shape[1],
                hidden_size=args.hidden_size,
                sfreq=args.sampling_freq,
                device=device,
                pretrained=is_pretrained
            )

            # Extract representations
            print(f"  Extracting representations...", flush=True)
            representations = extract_embeddings(
                model, X_method, batch_size=args.batch_size, device=device
            )
            print(f"  Representation dimensions: {representations.shape}", flush=True)

            # Generate t-SNE visualisation
            print(f"  Generating t-SNE visualisation (coloured by {label_name})...", flush=True)
            visualization_emb_by_label(
                embeddings=representations,
                labels=labels_method,
                label_name=label_name,
                name_method=method_name,
                save_fig_dir=args.save_fig_dir,
                cmap=args.cmap
            )
            print(f"  [OK] Visualisation completed", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("VISUALISATION COMPLETED", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Figures saved to: {args.save_fig_dir}", flush=True)


if __name__ == "__main__":
    main()
