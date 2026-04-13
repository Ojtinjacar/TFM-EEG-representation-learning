import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from models import AttentionLSTMAutoencoder
from utils import split_dataset, create_dataloader

# -------------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------------
def fit_model(
    epochs,
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    device,
    model_name="model",
    save_model_dir=None,
    save_fig_dir=None,
    max_lr=1e-3,
):
    """
    Trains an autoencoder model with OneCycleLR and saves:
      - loss curve (figure)
      - model weights
    """
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=100,
        three_phase=False
    )

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            output, _ = model(x_batch)
            loss = criterion(output, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                output, _ = model(x_batch)
                loss = criterion(output, x_batch)
                val_loss += loss.item() * x_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"[{model_name}] Epoch {epoch+1}/{epochs} "
              f"- Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    # Save loss figure
    if save_fig_dir:
        os.makedirs(save_fig_dir, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training')
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title(f'Loss per epoch — {model_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(save_fig_dir, f"train_loss_{model_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Loss figure saved to: {fig_path}")

    # Save model
    if save_model_dir:
        os.makedirs(save_model_dir, exist_ok=True)
        model_path = os.path.join(save_model_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Model saved to: {model_path}")

    return model

# -------------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Autoencoder training.")

    # Data paths
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/5_s/processed_windows.npy",
        help="Path to the .npy file with windows (X).",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="data/processed/5_s/processed_metadata.csv",
        help="Path to the metadata CSV.",
    )

    # Output paths
    parser.add_argument(
        "--save-model-dir",
        type=str,
        default="save/models",
        help="Directory to save models (.pth).",
    )
    parser.add_argument(
        "--save-fig-dir",
        type=str,
        default="save/figures/pretrain",
        help="Directory to save figures.",
    )

    # Model and training parameters
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Latent space size (hidden_size)."
    )
    parser.add_argument(
        "--sampling-freq",
        type=int,
        default=250,
        help="Sampling frequency."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Maximum learning rate for OneCycleLR."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay."
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="frontal",
        help="Head zone data"
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="alpha",
        help="Recognized band frequency"
    )
    parser.add_argument(
        "--exclude_subjects",
        nargs="+",
        default=None,
        help="List of subject IDs to exclude from training (for test set)"
    )
    parser.add_argument(
        "--fold_id",
        type=str,
        default=None,
        help="Fold identifier to include in model filename (e.g., 'fold0')."
    )

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    # Load data
    print("[INFO] Loading data...")
    X_np = np.load(args.data_path)
    meta_df = pd.read_csv(args.meta_path) # Load metadata to filter subjects

    # --- Exclude subjects for the test set ---
    if args.exclude_subjects:
        print(f"Excluding {len(args.exclude_subjects)} subjects for pre-training: {args.exclude_subjects}")
        keep_mask = ~meta_df['subject'].isin(args.exclude_subjects)

        X_np = X_np[keep_mask.values]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Number of windows for pre-training after exclusion: {len(X_np)}")

    X = torch.tensor(X_np, dtype=torch.float32)
    print("EEG dimensions: ", X.shape)

    # DataLoaders
    train_data, val_data = split_dataset(X)
    train_loader, val_loader = create_dataloader(
        train_data, val_data, batch_size=args.batch_size
    )

    # Model
    print("[INFO] Creating the AE model...")
    model = AttentionLSTMAutoencoder(
        input_size=X.shape[2],
        hidden_size=args.hidden_size,
        n_channels=X.shape[1],
        sfreq=args.sampling_freq,
        lstm_hidden_size=args.hidden_size // 2
    ).to(device)

    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fold_suffix = f"_{args.fold_id}" if args.fold_id else ""
    model_name = f"AE_{args.zone}_{args.frequency}{fold_suffix}_hidden{args.hidden_size}_e{args.epochs}"

    print(f"[INFO] Starting model training: {model_name}")
    fit_model(
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        save_model_dir=args.save_model_dir,
        save_fig_dir=args.save_fig_dir,
        max_lr=args.lr,
    )

    print("[INFO] Pipeline completed.")

if __name__ == "__main__":
    main()