"""Pre-trains an encoder against a continuous expert descriptor.

The loss compares every pair of the batch, so the descriptor has to be complete and
comparable across its columns before training starts: rows without any measure are
dropped, partial gaps are filled with the column median, and every column is standardized
using the statistics of the training split alone, since its measures span orders of
magnitude and the largest would otherwise decide every distance on its own.

The run reports the effective dimensionality of the representation and the loss of a
perfectly equidistant geometry. That geometry is the least informative arrangement
available, so a loss sitting on it means the encoder settled for spreading the windows
evenly instead of ordering them.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset

from checkpoint_naming import expclr_checkpoint_name, write_sidecar
from build_expert_features import DESCRIPTORS, quality_path
from loss import ExpCLRLoss
from models import EnhancedAttentionLSTM
from utils import set_seed


class ExpertFeatureDataset(Dataset):
    def __init__(self, X, F):
        if len(X) != len(F):
            raise ValueError(f"Windows ({len(X)}) and expert features ({len(F)}) are not aligned")
        self.X = X
        self.F = F

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.F[idx]


def prepare_descriptor(features, *, quality=None, min_r2=None):
    keep = ~np.isnan(features).all(axis=1)
    if quality is not None and min_r2 is not None:
        keep &= np.nan_to_num(quality, nan=-np.inf) >= min_r2
    if not keep.any():
        raise ValueError("No window has a usable expert descriptor")

    F = features[keep].astype(np.float64)

    all_nan = np.isnan(F).all(axis=0)
    if all_nan.any():
        raise ValueError(f"{int(all_nan.sum())} descriptor columns are entirely NaN on this split")
    medians = np.nanmedian(F, axis=0)
    F = np.where(np.isnan(F), medians, F)

    mean = F.mean(axis=0)
    std = F.std(axis=0)
    constant = std < 1e-12
    if constant.any():
        print(f"  > {int(constant.sum())} constant descriptor columns zeroed out")
    std = np.where(constant, 1.0, std)
    F = (F - mean) / std
    F[:, constant] = 0.0

    return F, keep


def effective_dimensionality(z):
    if not np.isfinite(z).all():
        return float("nan")
    centred = z - z.mean(axis=0, keepdims=True)
    eigenvalues = np.linalg.svd(centred, compute_uv=False) ** 2
    total = eigenvalues.sum()
    if total <= 0:
        return 1.0
    p = eigenvalues / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def equidistant_reference(n, device):
    ref = torch.full((n, n), n / (n - 1.0), device=device)
    ref.fill_diagonal_(0.0)
    return ref


@torch.no_grad()
def diagnose_epoch(model, batches, criterion, device, loss_on):
    was_training = model.training
    model.eval()
    reps, references = [], []
    for x_batch, f_batch in batches:
        x_batch, f_batch = x_batch.to(device), f_batch.to(device)
        z = model(x_batch) if loss_on == "projection" else model.get_embedding(x_batch)
        reps.append(z.cpu().numpy())
        target = (1.0 - criterion.expert_similarity(f_batch)) * criterion.delta
        references.append(float(criterion.reduce(
            (target - equidistant_reference(len(target), target.device)).pow(2))))
    if was_training:
        model.train()
    return {
        "effective_dim": effective_dimensionality(np.concatenate(reps, axis=0)),
        "equidistant_loss": float(np.mean(references)),
    }


def max_pairwise_distance(F, *, max_samples=4096, seed=42):
    if len(F) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(F), size=max_samples, replace=False)
        F = F[idx]
    with torch.no_grad():
        t = torch.tensor(F, dtype=torch.float32)
        return float(torch.cdist(t, t, p=2).max().item())


MIN_BATCH_SIZE_ON_MPS = 26


def main(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "mps" and args.batch_size < MIN_BATCH_SIZE_ON_MPS:
        raise ValueError(
            f"batch_size={args.batch_size} is below {MIN_BATCH_SIZE_ON_MPS} and torch.cdist has no "
            f"backward on MPS under that size. Raise the batch size or export "
            f"PYTORCH_ENABLE_MPS_FALLBACK=1 to fall back to CPU for that operator."
        )
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    X_np = np.load(os.path.join(args.data_path, "processed_windows.npy"))
    meta_df = pd.read_csv(os.path.join(args.data_path, "processed_metadata.csv"))
    features = np.load(args.expert_features)
    quality = None
    r2_path = quality_path(args.descriptor, os.path.dirname(args.expert_features))
    if args.min_apsd_r2 is not None:
        if not os.path.exists(r2_path):
            raise FileNotFoundError(
                f"--min_apsd_r2 was given but {r2_path} does not exist. The quality of the "
                "fit is written next to the descriptor, so rebuild it with "
                "src/build_expert_features.py rather than filtering on nothing."
            )
        quality = np.load(r2_path)

    if len(features) != len(X_np):
        raise ValueError(
            f"Expert features ({len(features)}) do not align with the window set ({len(X_np)}). "
            "Rebuild them with code/src/build_expert_features.py against this metadata."
        )
    expected_columns = len(DESCRIPTORS[args.descriptor])
    if features.shape[1] != expected_columns:
        raise ValueError(
            f"{args.expert_features} has {features.shape[1]} columns but {args.descriptor} "
            f"is defined as {expected_columns}. The descriptor name reaches the checkpoint "
            "filename, so a mismatch would label the encoder as trained on something it "
            "never saw."
        )
    print(f"Windows: {X_np.shape}, expert descriptor: {features.shape}")

    if args.exclude_subjects:
        keep_mask = ~meta_df["subject"].isin(args.exclude_subjects)
        print(f"Excluding {len(args.exclude_subjects)} subjects for pre-training: {args.exclude_subjects}")
        X_np = X_np[keep_mask.values]
        features = features[keep_mask.values]
        if quality is not None:
            quality = quality[keep_mask.values]
        print(f"Number of windows for pre-training after exclusion: {len(X_np)}")

    F_np, keep = prepare_descriptor(features, quality=quality, min_r2=args.min_apsd_r2)
    dropped = int((~keep).sum())
    if dropped:
        print(f"Dropped {dropped} windows without a usable descriptor ({dropped / len(keep):.2%})")
    X = torch.tensor(X_np[keep], dtype=torch.float32)
    F = torch.tensor(F_np, dtype=torch.float32)
    print(f"Training on {len(X)} windows with a {F.shape[1]}-dimensional descriptor")

    feat_max_dist = None
    if args.sim_max == "train":
        feat_max_dist = max_pairwise_distance(F_np)
        print(f"max_kl ||f_k - f_l|| over train: {feat_max_dist:.4f}")

    set_seed(args.seed)
    dataset = ExpertFeatureDataset(X, F)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(args.seed))

    model = EnhancedAttentionLSTM(
        input_size=X.shape[2],
        hidden_size=args.embedding_size,
        n_channels=X.shape[1],
        sfreq=args.sampling_frequency,
        lstm_hidden_size=args.embedding_size // 2,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = ExponentialLR(optimizer, gamma=args.lr_gamma)
    criterion = ExpCLRLoss(delta=args.delta,
        temperature=None if args.no_hard_negative_mining else args.temperature,
        feat_max_dist=feat_max_dist,
        squared_similarity=not args.linear_similarity,
    )

    diag_rng = torch.Generator().manual_seed(args.seed + 1)
    diag_idx = torch.randperm(len(dataset), generator=diag_rng)[:8 * args.batch_size]
    diag_batches = [(X[diag_idx[i:i + args.batch_size]], F[diag_idx[i:i + args.batch_size]])
                    for i in range(0, len(diag_idx), args.batch_size)
                    if len(diag_idx[i:i + args.batch_size]) >= 2]

    baseline = diagnose_epoch(model, diag_batches, criterion, device, args.loss_on)
    print(f"Before training: effective dim {baseline['effective_dim']:.2f} | "
          f"equidistant-geometry loss {baseline['equidistant_loss']:.4f} "
          f"(the loss cannot go meaningfully below this unless Delta matches the descriptor)")

    losses, diagnostics = [], [baseline]
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        for x_batch, f_batch in train_loader:
            x_batch, f_batch = x_batch.to(device), f_batch.to(device)
            optimizer.zero_grad()
            z = model(x_batch) if args.loss_on == "projection" else model.get_embedding(x_batch)
            loss = criterion(z, f_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        epoch_loss = total_loss / len(train_loader)
        losses.append(epoch_loss)
        if epoch % args.diagnose_every == 0 or epoch == args.num_epochs - 1:
            diag = diagnose_epoch(model, diag_batches, criterion, device, args.loss_on)
            diagnostics.append(diag)
            print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {epoch_loss:.4f} | "
                  f"effective dim {diag['effective_dim']:.2f} | "
                  f"equidistant ref {diag['equidistant_loss']:.4f}")
        else:
            print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {epoch_loss:.4f}")

    final = diagnostics[-1]
    if final["effective_dim"] < baseline["effective_dim"]:
        print(f"WARNING: effective dimensionality fell from {baseline['effective_dim']:.2f} to "
              f"{final['effective_dim']:.2f}. The encoder contracted the representation; the probe "
              f"has fewer directions to work with than before training.")
    if losses[-1] <= final["equidistant_loss"] * 1.05:
        print(f"WARNING: final loss {losses[-1]:.4f} is at the equidistant-geometry floor "
              f"{final['equidistant_loss']:.4f}. Delta is mismatched to the descriptor scale, so "
              f"most of the objective is an irreducible bias rather than learnable signal.")

    model_path = os.path.join(
        args.save_dir,
        expclr_checkpoint_name(
            args.zone, args.frequency, args.fold_id, args.descriptor,
            batch_size=args.batch_size, lr=args.lr,
            temperature=args.temperature, delta=args.delta,
        ),
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    write_sidecar(model_path, {
        "method": "ExpCLR",
        "zone": args.zone,
        "frequency": args.frequency,
        "fold_id": args.fold_id,
        "descriptor": args.descriptor,
        "descriptor_dim": int(F.shape[1]),
        "n_windows": int(len(X)),
        "feat_max_dist": feat_max_dist,
        "delta": args.delta,
        "temperature": None if args.no_hard_negative_mining else args.temperature,
        "loss_on": args.loss_on,
        "sim_max": args.sim_max,
        "lr": args.lr,
        "lr_gamma": args.lr_gamma,
        "dropout": args.dropout,
        "embedding_size": args.embedding_size,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "seed": args.seed,
        "exclude_subjects": sorted(args.exclude_subjects),
        "final_loss": losses[-1],
        "effective_dim": final["effective_dim"],
    })

    plt.plot(range(1, args.num_epochs + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (ExpCLR)")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(
        args.plot_dir,
        f"ExpCLR_{args.zone}_{args.frequency}"
        f"{('_' + args.fold_id) if args.fold_id else ''}_training_loss_curve.png"
    )
    plt.savefig(plot_path)
    print(f"Loss curve saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-train the EEG encoder with the ExpCLR objective on expert features."
    )
    parser.add_argument("--data_path", default="data/processed/5_s",
                        help="Directory holding processed_windows.npy and processed_metadata.csv.")
    parser.add_argument("--expert_features", required=True,
                        help="Aligned expert-feature .npy from build_expert_features.py.")
    parser.add_argument("--descriptor", default="P_madurativo", choices=sorted(DESCRIPTORS),
                        help="Descriptor label, recorded in the checkpoint name.")
    parser.add_argument("--zone", default="all", help="Brain zone (naming only).")
    parser.add_argument("--frequency", default="all", help="Frequency band (naming only).")
    parser.add_argument("--fold_id", default="", help="Fold identifier for the checkpoint name.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the encoder initialisation and the batch order. The same "
                             "value is used for every fold, so what varies across folds is the "
                             "held-out subject and not the starting point of the optimisation.")
    parser.add_argument("--exclude_subjects", nargs="*", default=[],
                        help="Subjects held out for testing in this fold.")
    parser.add_argument("--sampling_frequency", type=int, default=250, help="Sampling rate (Hz).")
    parser.add_argument("--embedding_size", type=int, default=128,
                        help="Encoder embedding dimension (inherited d=128).")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size. The paper uses 64 (App. A.2).")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate. The paper's grid picks 5e-3 for EEG (Table 9).")
    parser.add_argument("--lr_gamma", type=float, default=0.99,
                        help="Exponential LR decay.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--delta", type=float, default=1.0, help="Margin Delta of Eq. 3/4.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Hard-negative mining temperature tau of Eq. 4.")
    parser.add_argument("--no_hard_negative_mining", action="store_true",
                        help="Use the plain quadratic loss of Eq. 3 (the NHNM ablation).")
    parser.add_argument("--linear_similarity", action="store_true",
                        help="Use Eq. 2 instead of the squared similarity of Eq. 5.")
    parser.add_argument("--sim_max", choices=["train", "batch"], default="train",
                        help="Whether max_kl ||f_k - f_l|| is taken over the training split or "
                             "per batch. 'train' avoids leaking the held-out fold.")
    parser.add_argument("--loss_on", choices=["projection", "embedding"], default="projection",
                        help="Where the loss is applied. Defaults to 'projection', which here is the "
                             "output of the full encoder: the paper describes E as the base network "
                             "plus a two-layer fully connected network on top (Sec. 4.2), so those "
                             "layers are part of E and not a discardable head as in SimCLR. "
                             "Evaluation must use the same representation, see extract_embeddings. "
                             "'embedding' yields the encoder two layers shorter and is kept as an "
                             "ablation.")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Encoder dropout. Zero by default: in the paper dropout is an "
                             "augmentation of the SimCLR baseline (Sec. 4.3 and App. B.4), never a "
                             "regulariser of ExpCLR, whose claim is that it uses no transformations. "
                             "It would also make E(x_i) stochastic exactly where the loss measures "
                             "distances, injecting mask noise into D_ij. The inherited 0.25 remains "
                             "available for the ablation.")
    parser.add_argument("--diagnose_every", type=int, default=5,
                        help="Epoch interval for the geometry diagnostics (effective "
                             "dimensionality and equidistant-geometry reference).")
    parser.add_argument("--min_apsd_r2", type=float, default=None,
                        help="Optional quality filter on the mean specparam R^2 per window.")
    parser.add_argument("--save_dir", default="save/models", help="Checkpoint directory.")
    parser.add_argument("--plot_dir", default="save/figures/pretrain", help="Loss-curve directory.")
    main(parser.parse_args())
