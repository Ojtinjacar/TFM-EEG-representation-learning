"""VAE pretraining over the shared EEG encoder.

Minimizes 0.5 * ||x - x_hat||^2 + beta * KL(q(z|x) || N(0, I)) per window, with
the reconstruction summed over data dims and the KL over latent dims, so beta=1
is the textbook VAE. The code reduces the reconstruction by mean instead, hence
beta_code = 2 * beta / (C * T). See --beta for the convention caveat.
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import OneCycleLR

from models import VariationalAttentionLSTMAutoencoder
from checkpoint_naming import vae_checkpoint_name, write_sidecar
from window_loading import NORM_PROVENANCE, load_windows
from utils import split_dataset, create_dataloader, set_seed
from loss import build_prior, vae_elbo, kl_beta

# Convention --beta is expressed in, recorded in every sidecar.
BETA_CONVENTION = "strict-gaussian-unit-variance"


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
    best_val, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(epochs):
        beta = kl_beta(epoch, target_beta, anneal_epochs)

        model.train()
        train_loss, train_recon, train_kl, train_kl_obj = 0.0, 0.0, 0.0, 0.0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            reconstruction, mu, logvar, _ = model(x_batch)
            loss, recon, kl_obj, kl_true = vae_elbo(
                reconstruction, x_batch, mu, logvar, prior, beta,
                free_bits=free_bits,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            bs = x_batch.size(0)
            train_loss += loss.item() * bs
            train_recon += recon.item() * bs
            train_kl += kl_true.item() * bs
            train_kl_obj += kl_obj.item() * bs

        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_recon /= n_train
        train_kl /= n_train
        train_kl_obj /= n_train
        train_losses.append(train_loss)

        model.eval()
        val_loss, val_recon, val_kl, val_kl_obj = 0.0, 0.0, 0.0, 0.0
        val_mus = []
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                reconstruction, mu, logvar, _ = model(x_batch)
                loss, recon, kl_obj, kl_true = vae_elbo(
                    reconstruction, x_batch, mu, logvar, prior, beta,
                    free_bits=free_bits,
                )
                bs = x_batch.size(0)
                val_loss += loss.item() * bs
                val_recon += recon.item() * bs
                val_kl += kl_true.item() * bs
                val_kl_obj += kl_obj.item() * bs
                val_mus.append(mu.detach().cpu())

        n_val = len(val_loader.dataset)
        val_loss /= n_val
        val_recon /= n_val
        val_kl /= n_val
        val_kl_obj /= n_val
        val_losses.append(val_loss)

        # Posterior-collapse diagnostics: dimensions whose mean responds to the
        # input (variance of mu across validation windows above 0.01).
        mu_all = torch.cat(val_mus)
        active_dims = int((mu_all.var(dim=0) > 0.01).sum())

        # Target beta and floored KL, so the criterion matches what is minimized.
        val_target = val_recon + target_beta * val_kl_obj
        if val_target < best_val:
            best_val = val_target
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        # KL is the true divergence; KL-obj is the floored one the loss uses.
        floored = (f", KL-obj = {train_kl_obj:.6f}" if free_bits > 0.0 else "")
        floored_val = (f", KL-obj = {val_kl_obj:.6f}" if free_bits > 0.0 else "")
        print(f"[{model_name}] Epoch {epoch+1}/{epochs} "
              f"- Train Loss = {train_loss:.6f} (recon = {train_recon:.6f}, "
              f"KL = {train_kl:.6f}{floored}, beta = {beta:.6g}), "
              f"Val Loss = {val_loss:.6f} "
              f"(recon = {val_recon:.6f}, KL = {val_kl:.6f}{floored_val}, "
              f"target-beta ELBO = {val_target:.6f}, active_dims = {active_dims})")

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
        last_dir = os.path.join(save_model_dir, "last")
        os.makedirs(last_dir, exist_ok=True)
        last_path = os.path.join(last_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), last_path)

        # The checkpoint the orchestrators pick up is the one the selection
        # criterion chose. Keeping the best in a subdirectory left the last
        # epoch to be evaluated instead, which made the criterion decorative.
        # The last epoch stays under last/ for inspection, out of the globs.
        if best_state is not None:
            torch.save(best_state, model_path)
            print(f"[INFO] Best-val model (epoch {best_epoch}, target-beta ELBO "
                  f"{best_val:.6f}) saved to: {model_path}")
            print(f"[INFO] Last-epoch model kept for inspection at: {last_path}")
        else:
            torch.save(model.state_dict(), model_path)
            print(f"[INFO] No validation split produced a best state; last epoch "
                  f"saved to: {model_path}")

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
        default=1.0,
        help="KL weight under the strict gaussian ELBO, where the reconstruction "
             "is 0.5*||x - x_hat||^2 summed over data dims and the KL is summed "
             "over latent dims: beta=1 is the textbook VAE, equivalent to a "
             "decoder noise of sigma^2=1. Rescaled internally to 2*beta/(C*T) to "
             "match the mean-reduced MSE. Implementations that drop the factor of "
             "one half would call this same setting beta=0.5."
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
        "--latent_dim",
        type=int,
        default=None,
        help="Latent code dimensionality. Defaults to --hidden_size, which keeps the "
             "code as wide as the encoder.",
    )
    parser.add_argument(
        "--free-bits",
        type=float,
        default=0.0,
        help="Per-dimension KL floor (nats). 0 disables it."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional label prefix for the checkpoint filename. Defaults to 'VAE'."
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
    # Normalisation statistics are refitted without the held-out subjects, so the
    # transform applied to the pre-training windows never saw them.
    X_np, meta_df = load_windows(
        args.data_path, args.meta_path, fit_stats_excluding=args.exclude_subjects
    )

    if args.exclude_subjects:
        print(f"Excluding {len(args.exclude_subjects)} subjects for pre-training: {args.exclude_subjects}")
        keep_mask = ~meta_df['subject'].isin(args.exclude_subjects)

        X_np = X_np[keep_mask.values]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Number of windows for pre-training after exclusion: {len(X_np)}")

    X = torch.tensor(X_np, dtype=torch.float32)
    beta_effective = 2.0 * args.beta / (X.shape[1] * X.shape[2])
    print(f"[INFO] Canonical beta {args.beta} ({BETA_CONVENTION}) -> code-scale "
          f"beta {beta_effective:.4e} = 2*beta/(C*T), C*T = "
          f"{X.shape[1] * X.shape[2]}")
    print("EEG dimensions: ", X.shape)

    # Validation picks the checkpoint, so it has to be independent of training:
    # splitting by subject keeps windows of the same recording on one side only.
    train_data, val_data = split_dataset(X, seed=args.seed,
                                         groups=meta_df['subject'].values)
    train_loader, val_loader = create_dataloader(
        train_data, val_data, batch_size=args.batch_size, seed=args.seed
    )

    prior = build_prior("standard").to(device)

    print(f"[INFO] Creating the VAE model (free_bits={args.free_bits})...")
    model = VariationalAttentionLSTMAutoencoder(
        input_size=X.shape[2],
        hidden_size=args.hidden_size,
        n_channels=X.shape[1],
        sfreq=args.sampling_freq,
        lstm_hidden_size=args.hidden_size // 2,
        latent_dim=args.latent_dim,
        prior=prior,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    method_tag = args.tag if args.tag else "VAE"
    model_name = vae_checkpoint_name(
        method_tag, args.zone, args.frequency, args.fold_id,
        beta=args.beta, prior="standard", free_bits=args.free_bits,
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
        target_beta=beta_effective,
        anneal_epochs=args.kl_anneal_epochs,
        prior=prior,
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
            "norm_stats": NORM_PROVENANCE,
            "seed": getattr(args, "seed", None),
            "hidden_size": args.hidden_size,
            "latent_dim": args.latent_dim if args.latent_dim else args.hidden_size,
            "beta": args.beta,
            "beta_convention": BETA_CONVENTION,
            "beta_effective": beta_effective,
            "prior": "standard",
            "free_bits": args.free_bits,
            "kl_anneal_epochs": args.kl_anneal_epochs,
            "checkpoint_selection": "best-val-target-beta",
            "epochs": args.epochs,
            "lr": args.lr,
            "n_windows": int(len(X)),
        })

    print("[INFO] Pipeline completed.")


if __name__ == "__main__":
    main()
