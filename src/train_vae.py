import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import OneCycleLR

from models import (
    VariationalAttentionLSTMAutoencoder,
    ConditionalVariationalAttentionLSTMAutoencoder,
)
from checkpoint_naming import vae_checkpoint_name, write_sidecar
from utils import split_dataset, create_dataloader, ages_to_indices, CANONICAL_AGES, set_seed
from loss import build_prior, vae_elbo, kl_beta


def _unpack_batch(batch, device):
    x_batch = batch[0].to(device)
    cond = batch[2].to(device) if len(batch) == 3 else None
    return x_batch, cond


def fit_model(
    epochs,
    train_loader,
    val_loader,
    model,
    optimizer,
    device,
    target_beta,
    anneal_epochs,
    prior,
    conditional=False,
    free_bits=0.0,
    model_name="model",
    save_model_dir=None,
    save_fig_dir=None,
    max_lr=1e-3,
):
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
        beta = kl_beta(epoch, target_beta, anneal_epochs)

        model.train()
        train_loss, train_recon, train_kl = 0.0, 0.0, 0.0
        for batch in train_loader:
            x_batch, c_batch = _unpack_batch(batch, device)
            if conditional:
                reconstruction, mu, logvar, _ = model(x_batch, c_batch)
            else:
                reconstruction, mu, logvar, _ = model(x_batch)
            loss, recon, kl = vae_elbo(
                reconstruction, x_batch, mu, logvar, prior, beta,
                cond=c_batch, free_bits=free_bits,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            bs = x_batch.size(0)
            train_loss += loss.item() * bs
            train_recon += recon.item() * bs
            train_kl += kl.item() * bs

        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_recon /= n_train
        train_kl /= n_train
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_batch, c_batch = _unpack_batch(batch, device)
                if conditional:
                    reconstruction, mu, logvar, _ = model(x_batch, c_batch)
                else:
                    reconstruction, mu, logvar, _ = model(x_batch)
                loss, _, _ = vae_elbo(
                    reconstruction, x_batch, mu, logvar, prior, beta,
                    cond=c_batch, free_bits=free_bits,
                )
                val_loss += loss.item() * x_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"[{model_name}] Epoch {epoch+1}/{epochs} "
              f"- Train Loss = {train_loss:.6f} (recon = {train_recon:.6f}, "
              f"KL = {train_kl:.6f}, beta = {beta:.4f}), Val Loss = {val_loss:.6f}")

    if save_fig_dir:
        os.makedirs(save_fig_dir, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training')
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Negative ELBO')
        plt.title(f'Loss per epoch — {model_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(save_fig_dir, f"train_loss_{model_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Loss figure saved to: {fig_path}")

    if save_model_dir:
        os.makedirs(save_model_dir, exist_ok=True)
        model_path = os.path.join(save_model_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Model saved to: {model_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Variational Autoencoder (VAE) training.")

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

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Encoder/decoder feature width and latent size (hidden_size)."
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
        "--beta",
        type=float,
        default=0.003,
        help="Target weight of the KL term in the ELBO."
    )
    parser.add_argument(
        "--kl-anneal-epochs",
        type=int,
        default=20,
        help="Number of epochs to linearly ramp beta from 0 to its target (0 disables)."
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
    parser.add_argument(
        "--free-bits",
        type=float,
        default=0.0,
        help="Per-dimension KL floor (nats). 0 disables it."
    )
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Train a conditional VAE (CVAE) conditioned on the session age."
    )
    parser.add_argument(
        "--prior",
        type=str,
        default="standard",
        choices=["standard", "conditional"],
        help="Latent prior: 'standard' N(0,I) or 'conditional' per-age Gaussian (rich prior)."
    )
    parser.add_argument(
        "--cond-col",
        type=str,
        default="age",
        help="Metadata column holding the per-window session age used as CVAE condition."
    )
    parser.add_argument(
        "--cond-dim",
        type=int,
        default=16,
        help="Width of the learned condition (age) embedding in the CVAE."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional label prefix for the checkpoint filename (e.g. 'CVAE-SP'). "
             "Defaults to 'CVAE' when --conditional else 'VAE'. Lets ablation variants "
             "(same architecture, different prior) get distinct, comparable checkpoints."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weights, shuffling and augmentations."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device. 'auto' picks mps/cuda/cpu."
    )
    args = parser.parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Usando dispositivo: {device}")

    print("[INFO] Loading data...")
    X_np = np.load(args.data_path)
    meta_df = pd.read_csv(args.meta_path)

    if args.exclude_subjects:
        print(f"Excluding {len(args.exclude_subjects)} subjects for pre-training: {args.exclude_subjects}")
        keep_mask = ~meta_df['subject'].isin(args.exclude_subjects)

        X_np = X_np[keep_mask.values]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Number of windows for pre-training after exclusion: {len(X_np)}")

    X = torch.tensor(X_np, dtype=torch.float32)
    print("EEG dimensions: ", X.shape)

    needs_cond = args.conditional or args.prior == "conditional"
    cond = None
    n_conditions = len(CANONICAL_AGES)
    if needs_cond:
        if args.cond_col not in meta_df.columns:
            raise ValueError(
                f"Condition column '{args.cond_col}' not found in metadata "
                f"({args.meta_path}). Available: {list(meta_df.columns)}"
            )
        cond = ages_to_indices(meta_df[args.cond_col].values)
        print(f"[INFO] Conditioning on '{args.cond_col}' with {n_conditions} ages "
              f"{CANONICAL_AGES}.")

    train_data, val_data = split_dataset(X, cond=cond, seed=args.seed)
    train_loader, val_loader = create_dataloader(
        train_data, val_data, batch_size=args.batch_size, seed=args.seed
    )

    latent_dim = args.hidden_size
    prior = build_prior(args.prior, n_conditions=n_conditions, latent_dim=latent_dim).to(device)

    print(f"[INFO] Creating the {'CVAE' if args.conditional else 'VAE'} model "
          f"(prior={args.prior}, free_bits={args.free_bits})...")
    if args.conditional:
        model = ConditionalVariationalAttentionLSTMAutoencoder(
            input_size=X.shape[2],
            hidden_size=args.hidden_size,
            n_conditions=n_conditions,
            n_channels=X.shape[1],
            sfreq=args.sampling_freq,
            lstm_hidden_size=args.hidden_size // 2,
            cond_dim=args.cond_dim,
            prior=prior,
        ).to(device)
    else:
        model = VariationalAttentionLSTMAutoencoder(
            input_size=X.shape[2],
            hidden_size=args.hidden_size,
            n_channels=X.shape[1],
            sfreq=args.sampling_freq,
            lstm_hidden_size=args.hidden_size // 2,
            prior=prior,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    method_tag = args.tag if args.tag else ("CVAE" if args.conditional else "VAE")
    model_name = vae_checkpoint_name(
        method_tag, args.zone, args.frequency, args.fold_id,
        beta=args.beta, prior=args.prior, free_bits=args.free_bits,
        hidden_size=args.hidden_size, epochs=args.epochs,
    )[:-len(".pth")]

    print(f"[INFO] Starting model training: {model_name}")
    fit_model(
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        target_beta=args.beta,
        anneal_epochs=args.kl_anneal_epochs,
        prior=prior,
        conditional=args.conditional,
        free_bits=args.free_bits,
        model_name=model_name,
        save_model_dir=args.save_model_dir,
        save_fig_dir=args.save_fig_dir,
        max_lr=args.lr,
    )

    if args.save_model_dir:
        write_sidecar(os.path.join(args.save_model_dir, f"{model_name}.pth"), {
            "method": method_tag,
            "zone": args.zone,
            "frequency": args.frequency,
            "fold_id": args.fold_id,
            "exclude_subjects": sorted(str(s) for s in (args.exclude_subjects or [])),
            "seed": getattr(args, "seed", None),
            "hidden_size": args.hidden_size,
            "beta": args.beta,
            "prior": args.prior,
            "free_bits": args.free_bits,
            "conditional": bool(args.conditional),
            "cond_dim": args.cond_dim if args.conditional else None,
            "kl_anneal_epochs": args.kl_anneal_epochs,
            "epochs": args.epochs,
            "lr": args.lr,
            "n_windows": int(len(X)),
        })

    print("[INFO] Pipeline completed.")


if __name__ == "__main__":
    main()
