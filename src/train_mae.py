import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from models import MaskedAttentionLSTMAutoencoder
from utils import split_dataset, create_dataloader

# -------------------------------------------------------------------------
# MAE LOSS FUNCTION
# -------------------------------------------------------------------------
def masked_mse_loss(reconstruction, original, mask):
    """
    Computes MSE only on masked regions.

    Args:
        reconstruction: (B, C, T) reconstructed signal
        original: (B, C, T) original signal
        mask: (B, T) binary mask (1=masked, 0=visible)

    Returns:
        loss: average MSE only on masked regions
    """
    # Expand mask for all channels
    mask_expanded = mask.unsqueeze(1).expand_as(reconstruction)  # (B, C, T)

    # Compute error only on masked regions
    masked_reconstruction = reconstruction * mask_expanded
    masked_original = original * mask_expanded

    # MSE normalised by number of masked elements
    mse = (masked_reconstruction - masked_original) ** 2
    loss = mse.sum() / mask_expanded.sum()

    return loss

# -------------------------------------------------------------------------
# RECONSTRUCTION VISUALISATION
# -------------------------------------------------------------------------
def visualize_reconstructions(model, val_loader, device, save_dir, model_name, mask_ratio=0.5, num_samples=3):
    """
    Generates reconstruction visualisations from the MAE to validate learning.

    Shows:
    - Original signal (black line)
    - Visible points (blue) that the model saw
    - Full reconstruction (dashed red line)
    - Masked regions (grey shading)
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # Get a validation batch
        x_batch, _ = next(iter(val_loader))
        x_batch = x_batch[:num_samples].to(device)

        # Forward pass
        reconstruction, _, x_masked, mask = model(x_batch, mask_ratio=mask_ratio)

        # Compute metrics
        mask_expanded = mask.unsqueeze(1).expand_as(x_batch)

        # MSE only on masked regions
        masked_orig = x_batch[mask_expanded == 1]
        masked_recon = reconstruction[mask_expanded == 1]
        mse_masked = ((masked_orig - masked_recon) ** 2).mean().item()

        # Correlation on masked regions
        corr = np.corrcoef(
            masked_orig.cpu().numpy().flatten(),
            masked_recon.cpu().numpy().flatten()
        )[0, 1]

        print("\n" + "="*70)
        print("RECONSTRUCTION METRICS (masked regions only)")
        print("="*70)
        print(f"MSE:         {mse_masked:.6f}")
        print(f"Correlation: {corr:.4f}")
        print(f"Mask Ratio:  {mask_ratio*100:.0f}%")
        print("="*70 + "\n")

        # Visualise each sample
        for i in range(num_samples):
            n_channels = x_batch.shape[1]
            fig, axes = plt.subplots(n_channels, 1, figsize=(15, 4*n_channels))

            if n_channels == 1:
                axes = [axes]

            for ch in range(n_channels):
                ax = axes[ch]

                # Original signal
                original = x_batch[i, ch].cpu().numpy()
                time_axis = np.arange(len(original))
                ax.plot(time_axis, original, label='Original', color='black',
                       linewidth=2, alpha=0.7, zorder=1)

                # Visible points (that the model saw)
                mask_1d = mask[i].cpu().numpy()
                visible_indices = np.where(mask_1d == 0)[0]
                visible_values = x_batch[i, ch].cpu().numpy()[visible_indices]
                ax.scatter(visible_indices, visible_values,
                          color='blue', s=15, label='Visible (input)',
                          alpha=0.6, zorder=5, marker='o')

                # MAE reconstruction
                recon = reconstruction[i, ch].cpu().numpy()
                ax.plot(time_axis, recon, label='MAE Reconstruction',
                       color='red', linewidth=2, linestyle='--', alpha=0.8, zorder=3)

                # Shade masked regions
                masked_indices = np.where(mask_1d == 1)[0]
                if len(masked_indices) > 0:
                    # Group contiguous regions
                    splits = np.split(masked_indices,
                                     np.where(np.diff(masked_indices) != 1)[0] + 1)
                    for j, region in enumerate(splits):
                        if len(region) > 0:
                            label = 'Masked (predicted)' if j == 0 else ''
                            ax.axvspan(region[0], region[-1], alpha=0.15,
                                     color='gray', label=label, zorder=0)

                ax.set_xlabel('Time (samples)', fontsize=11)
                ax.set_ylabel('Amplitude', fontsize=11)
                ax.set_title(f'Canal {ch+1}', fontsize=12, fontweight='bold')
                ax.legend(loc='upper right', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

            plt.suptitle(f'MAE Reconstruction — {model_name}\n'
                        f'MSE: {mse_masked:.6f} | Correlation: {corr:.4f} | '
                        f'Mask: {mask_ratio*100:.0f}%',
                        fontsize=13, fontweight='bold')
            plt.tight_layout()

            fig_path = os.path.join(save_dir, f'reconstruction_{model_name}_sample{i+1}.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Saved visualisation: {fig_path}")

# -------------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------------
def fit_model(
    epochs,
    train_loader,
    val_loader,
    model,
    optimizer,
    device,
    model_name="model",
    save_model_dir=None,
    save_fig_dir=None,
    max_lr=1e-3,
    mask_ratio=0.5,
):
    """
    Trains a Masked Autoencoder with OneCycleLR and saves:
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

            # Forward pass with masking
            reconstruction, embedding, x_masked, mask = model(x_batch, mask_ratio=mask_ratio)

            # Compute loss only on masked regions
            loss = masked_mse_loss(reconstruction, x_batch, mask)

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
                reconstruction, embedding, x_masked, mask = model(x_batch, mask_ratio=mask_ratio)
                loss = masked_mse_loss(reconstruction, x_batch, mask)
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
        plt.ylabel('Masked MSE Loss')
        plt.title(f'Loss per epoch — {model_name} (Mask Ratio={mask_ratio})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(save_fig_dir, f"loss_curve_{model_name}.png")
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
    parser = argparse.ArgumentParser(description="Masked Autoencoder (MAE) training.")

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
        "--mask-ratio",
        type=float,
        default=0.5,
        help="Masking ratio (0.0-1.0). E.g.: 0.5 = 50%% of the signal masked."
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Size of contiguous blocks to mask (in samples). "
             "None = random point-wise masking. "
             "E.g.: 50 = blocks of 50 samples (200ms @ 250Hz)."
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="all",
        help="Head zone data"
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="all",
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
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    # Load data
    print("[INFO] Loading data...")
    X_np = np.load(args.data_path)
    meta_df = pd.read_csv(args.meta_path)

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
    mask_type = f"blocks of {args.block_size}" if args.block_size else "random point-to-point"
    print(f"[INFO] Creating MAE model:")
    print(f"       - Mask ratio: {args.mask_ratio}")
    print(f"       - Mask type: {mask_type}")
    model = MaskedAttentionLSTMAutoencoder(
        input_size=X.shape[2],
        hidden_size=args.hidden_size,
        n_channels=X.shape[1],
        sfreq=args.sampling_freq,
        lstm_hidden_size=args.hidden_size // 2,
        mask_ratio=args.mask_ratio,
        block_size=args.block_size
    ).to(device)

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fold_suffix = f"_{args.fold_id}" if args.fold_id else ""
    block_suffix = f"_block{args.block_size}" if args.block_size else ""
    model_name = f"MAE_{args.zone}_{args.frequency}{fold_suffix}_hidden{args.hidden_size}_mask{int(args.mask_ratio*100)}{block_suffix}_e{args.epochs}"

    print(f"[INFO] Starting model training: {model_name}")
    print(f"[INFO] Strategy: Mask {args.mask_ratio*100}% of the signal and predict only masked parts")
    fit_model(
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        save_model_dir=args.save_model_dir,
        save_fig_dir=args.save_fig_dir,
        max_lr=args.lr,
        mask_ratio=args.mask_ratio,
    )

    # Visualise reconstructions at end of training
    print("\n[INFO] Generating reconstruction visualisations...")
    visualize_reconstructions(
        model=model,
        val_loader=val_loader,
        device=device,
        save_dir=args.save_fig_dir,
        model_name=model_name,
        mask_ratio=args.mask_ratio,
        num_samples=3
    )

    print("[INFO] Pipeline completed.")

if __name__ == "__main__":
    main()
