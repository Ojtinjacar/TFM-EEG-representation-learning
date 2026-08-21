"""Two-stage training for the InterFusion HVAE (Li et al., KDD 2021).

Stage 1 pretrains a plain z2-only VAE (paper Appendix C.1); because the conv
encoder, z2 heads and deconv decoder are shared module instances with the
main model, stage 1 IS the initialization of those parts. Stage 2 trains the
full hierarchical model. Optimization follows the official code: Adam,
gradient-norm clipping at 10, StepLR x0.5 every 10 epochs, batch 100, model
selection by validation loss (best-val checkpoint kept separately).
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import StepLR

from checkpoint_naming import interfusion_checkpoint_name, write_sidecar
from interfusion import InterFusionEEG, PretrainVAE
from utils import split_dataset, create_dataloader, set_seed


def run_stage(model, train_loader, val_loader, device, epochs, lr,
              lr_step, stage_name, track_latents=False):
    """Trains one stage (pretrain or full) with the paper's optimization.

    Args:
        model (nn.Module): Module exposing ``elbo(x) -> dict`` with 'loss'.
        train_loader (DataLoader): Training loader yielding (x, x) pairs.
        val_loader (DataLoader): Validation loader.
        device (torch.device): Compute device.
        epochs (int): Number of epochs.
        lr (float): Initial Adam learning rate.
        lr_step (int): StepLR period (factor is fixed at 0.5, as the paper).
        stage_name (str): Label for logging.
        track_latents (bool): Log per-latent KLs and active dimensions
            (only meaningful for the full model).

    Returns:
        tuple: (best_state dict on CPU, best validation loss, history list).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=0.5)

    best_val, best_state = float("inf"), None
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss, n_train = 0.0, 0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            out = model(x_batch)
            optimizer.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            bs = x_batch.size(0)
            train_loss += out["loss"].item() * bs
            n_train += bs
        scheduler.step()
        train_loss /= n_train

        model.eval()
        val_loss, val_recons, val_kl, n_val = 0.0, 0.0, 0.0, 0
        kl_z1, kl_z2 = 0.0, 0.0
        z1_means, z2_means = [], []
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                out = model(x_batch)
                bs = x_batch.size(0)
                val_loss += out["loss"].item() * bs
                val_recons += out["recons"].item() * bs
                val_kl += out["kl"].item() * bs
                if track_latents:
                    kl_z1 += out["kl_z1"].item() * bs
                    kl_z2 += out["kl_z2"].item() * bs
                    qz2_mean, _ = model.encode_z2(x_batch)
                    d = model.decoder(qz2_mean)
                    _, z1_mean, _ = model.qz1.sample(
                        model._arnn_features(d), deterministic=True
                    )
                    z1_means.append(z1_mean.mean(dim=1).cpu())
                    z2_means.append(qz2_mean.mean(dim=2).cpu())
                n_val += bs
        val_loss /= n_val
        val_recons /= n_val
        val_kl /= n_val

        extra = ""
        if track_latents:
            z1_all = torch.cat(z1_means)
            z2_all = torch.cat(z2_means)
            active_z1 = int((z1_all.var(dim=0) > 0.01).sum())
            active_z2 = int((z2_all.var(dim=0) > 0.01).sum())
            extra = (f", kl_z1 = {kl_z1 / n_val:.2f}, kl_z2 = {kl_z2 / n_val:.2f}"
                     f", active_z1 = {active_z1}/{z1_all.shape[1]}"
                     f", active_z2 = {active_z2}/{z2_all.shape[1]}")

        print(f"[{stage_name}] Epoch {epoch + 1}/{epochs} "
              f"- Train Loss = {train_loss:.2f}, Val Loss = {val_loss:.2f} "
              f"(recons = {val_recons:.2f}, kl = {val_kl:.2f}{extra})")
        history.append({"stage": stage_name, "epoch": epoch + 1,
                        "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    return best_state, best_val, history


def main():
    parser = argparse.ArgumentParser(
        description="InterFusion (KDD 2021) two-stage training on EEG windows."
    )
    parser.add_argument("--data-path", type=str,
                        default="data/processed/all_all/processed_windows.npy",
                        help="Path to the .npy file with windows (X).")
    parser.add_argument("--meta-path", type=str,
                        default="data/processed/all_all/processed_metadata.csv",
                        help="Path to the metadata CSV.")
    parser.add_argument("--save-model-dir", type=str, default="save/models",
                        help="Directory to save models (.pth).")
    parser.add_argument("--save-fig-dir", type=str, default="save/figures/pretrain",
                        help="Directory to save the loss figure.")
    parser.add_argument("--z-dim", type=int, default=4,
                        help="Inter-metric latent width M' (paper: 2-4).")
    parser.add_argument("--rnn-hidden", type=int, default=500,
                        help="Backward-GRU hidden units (paper: 500).")
    parser.add_argument("--dense-hidden", type=int, default=500,
                        help="Feature/fusion dense units (paper: 500).")
    parser.add_argument("--strides", type=int, nargs="+",
                        default=[2, 1, 2, 1, 2, 2, 2],
                        help="Conv-stack strides; controls W' "
                             "(paper: 2 1 2 1 2 for W=100).")
    parser.add_argument("--flow-levels", type=int, default=20,
                        help="RealNVP levels on q(z1) (paper: 20).")
    parser.add_argument("--pretrain-epochs", type=int, default=20,
                        help="Stage-1 (z2-only VAE) epochs (paper: 20).")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Stage-2 (full model) epochs (paper: 20).")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size (paper: 100).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial Adam learning rate (paper: 1e-3).")
    parser.add_argument("--lr-step", type=int, default=10,
                        help="Epochs between x0.5 LR decays (paper: 10).")
    parser.add_argument("--zone", type=str, default="all", help="Head zone data")
    parser.add_argument("--frequency", type=str, default="all",
                        help="Recognized band frequency")
    parser.add_argument("--exclude_subjects", nargs="+", default=None,
                        help="Subject IDs to exclude from training (test set).")
    parser.add_argument("--fold_id", type=str, default=None,
                        help="Fold identifier for the checkpoint filename.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for weights, shuffling and sampling.")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Compute device. 'auto' picks mps/cuda/cpu.")
    args = parser.parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading data...")
    X_np = np.load(args.data_path)
    meta_df = pd.read_csv(args.meta_path)

    if args.exclude_subjects:
        print(f"Excluding {len(args.exclude_subjects)} subjects for "
              f"pre-training: {args.exclude_subjects}")
        keep_mask = ~meta_df["subject"].isin(args.exclude_subjects)
        X_np = X_np[keep_mask.values]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Number of windows for pre-training after exclusion: {len(X_np)}")

    X = torch.tensor(X_np, dtype=torch.float32)
    print("EEG dimensions: ", X.shape)

    train_data, val_data = split_dataset(X, seed=args.seed)
    train_loader, val_loader = create_dataloader(
        train_data, val_data, batch_size=args.batch_size, seed=args.seed
    )

    model = InterFusionEEG(
        x_dim=X.shape[1], window=X.shape[2], z_dim=args.z_dim,
        strides=tuple(args.strides), rnn_hidden=args.rnn_hidden,
        dense_hidden=args.dense_hidden, flow_levels=args.flow_levels,
    ).to(device)
    print(f"[INFO] InterFusion: M={X.shape[1]}, W={X.shape[2]}, "
          f"M'={args.z_dim}, W'={model.w_prime}, "
          f"embedding_dim={model.embedding_dim}")

    model_name = interfusion_checkpoint_name(
        args.zone, args.frequency, args.fold_id,
        z_dim=args.z_dim, w_prime=model.w_prime, epochs=args.epochs,
    )[:-len(".pth")]

    # Stage 1: z2-only VAE; shares conv/deconv/z2 heads with the main model.
    pretrain = PretrainVAE(model).to(device)
    print(f"[INFO] Stage 1/2: pretraining the z2-only VAE "
          f"({args.pretrain_epochs} epochs)...")
    _, _, hist1 = run_stage(
        pretrain, train_loader, val_loader, device,
        epochs=args.pretrain_epochs, lr=args.lr, lr_step=args.lr_step,
        stage_name=f"{model_name}:pretrain",
    )

    print(f"[INFO] Stage 2/2: training the full hierarchical model "
          f"({args.epochs} epochs)...")
    best_state, best_val, hist2 = run_stage(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, lr_step=args.lr_step,
        stage_name=model_name, track_latents=True,
    )

    if args.save_model_dir:
        os.makedirs(args.save_model_dir, exist_ok=True)
        model_path = os.path.join(args.save_model_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Model saved to: {model_path}")
        if best_state is not None:
            best_dir = os.path.join(args.save_model_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            torch.save(best_state, os.path.join(best_dir, f"{model_name}.pth"))
            print(f"[INFO] Best-val model (val loss {best_val:.2f}) saved to: "
                  f"{os.path.join(best_dir, f'{model_name}.pth')}")

        write_sidecar(model_path, {
            "method": "InterFusion",
            "zone": args.zone,
            "frequency": args.frequency,
            "fold_id": args.fold_id,
            "exclude_subjects": sorted(str(s) for s in (args.exclude_subjects or [])),
            "seed": args.seed,
            "x_dim": int(X.shape[1]),
            "window": int(X.shape[2]),
            "z_dim": args.z_dim,
            "w_prime": int(model.w_prime),
            "rnn_hidden": args.rnn_hidden,
            "dense_hidden": args.dense_hidden,
            "strides": list(args.strides),
            "flow_levels": args.flow_levels,
            "pretrain_epochs": args.pretrain_epochs,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "embedding_dim": int(model.embedding_dim),
            "n_windows": int(len(X)),
        })

    if args.save_fig_dir:
        os.makedirs(args.save_fig_dir, exist_ok=True)
        df = pd.DataFrame(hist1 + hist2)
        plt.figure(figsize=(10, 5))
        for stage, group in df.groupby("stage", sort=False):
            plt.plot(group["epoch"] + (0 if "pretrain" in stage else args.pretrain_epochs),
                     group["val_loss"], label=f"val ({stage.split(':')[-1]})")
        plt.xlabel("Epoch (stages concatenated)")
        plt.ylabel("Negative ELBO")
        plt.title(f"InterFusion training - {model_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(args.save_fig_dir, f"train_loss_{model_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Loss figure saved to: {fig_path}")

    print("[INFO] Pipeline completed.")


if __name__ == "__main__":
    main()
