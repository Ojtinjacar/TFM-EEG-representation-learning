"""Pre-train the EEG encoder with ExpCLR (Nonnenmacher et al., ICML 2022).

ExpCLR replaces augmentation-based views with a continuous expert descriptor: the loss asks the
embedding geometry to reproduce the geometry of the expert features (properties P1/P2 of the
paper, Sec. 3.2). Consequently this script differs structurally from ``train_simclr.py``: a batch
carries a single, un-augmented view per window plus its expert feature vector, and the objective
is computed over the full NxN pairwise matrix.

The descriptor is the 106 curated expert features of E3, precomputed and aligned to the window
order by ``code/src/build_expert_features.py``.

Usage:
    python src/train_expclr.py \
        --data_path data/processed/all_all \
        --expert_features data/processed/expert_features/expert_features_P_full.npy \
        --zone all --frequency all --fold_id fold0 \
        --exclude_subjects B010 B011
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset

from loss import ExpCLRLoss
from models import EnhancedAttentionLSTM


class ExpertFeatureDataset(Dataset):
    """Pairs each EEG window with its expert feature vector.

    ExpCLR needs ``f_i`` for every sample in the batch, so windows without a descriptor are
    dropped at construction time rather than imputed wholesale.
    """

    def __init__(self, X, F):
        """Initialises the dataset.

        Args:
            X: Window tensor of shape (N, C, T).
            F: Expert feature tensor of shape (N, d), already imputed and standardised.

        Raises:
            ValueError: If the two tensors disagree on the number of windows.
        """
        if len(X) != len(F):
            raise ValueError(f"Windows ({len(X)}) and expert features ({len(F)}) are not aligned")
        self.X = X
        self.F = F

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.F[idx]


def prepare_descriptor(features, *, quality=None, min_r2=None):
    """Imputes, standardises and validates the expert descriptor on the training split.

    The paper does not specify any normalisation of the expert features, but it is mandatory here:
    our 106 features span scales from 1e-11 (band powers) to 10 (aperiodic offset), so without a
    per-feature z-score two or three columns would dominate the Euclidean distance that defines
    ``s_ij``. Statistics are fitted on the given rows only, which under LOSO are train-only.

    Args:
        features: Raw descriptor matrix of shape (N, d), possibly containing NaN.
        quality: Optional per-window specparam R^2 used to drop poorly fitted windows.
        min_r2: Optional threshold applied to ``quality``.

    Returns:
        Tuple ``(F, keep)`` where ``F`` is the standardised (M, d) matrix and ``keep`` is the
        boolean vector of rows retained from the input.

    Raises:
        ValueError: If no window survives, or if every value of some feature is NaN.
    """
    keep = ~np.isnan(features).all(axis=1)
    if quality is not None and min_r2 is not None:
        keep &= np.nan_to_num(quality, nan=-np.inf) >= min_r2
    if not keep.any():
        raise ValueError("No window has a usable expert descriptor")

    F = features[keep].astype(np.float64)

    # Checked before nanmedian: an all-NaN column would only yield a NaN median and a warning.
    all_nan = np.isnan(F).all(axis=0)
    if all_nan.any():
        raise ValueError(f"{int(all_nan.sum())} descriptor columns are entirely NaN on this split")
    medians = np.nanmedian(F, axis=0)
    F = np.where(np.isnan(F), medians, F)

    mean = F.mean(axis=0)
    std = F.std(axis=0)
    # Constant columns carry no similarity information; neutralise them instead of dividing by 0.
    constant = std < 1e-12
    if constant.any():
        print(f"  > {int(constant.sum())} constant descriptor columns zeroed out")
    std = np.where(constant, 1.0, std)
    F = (F - mean) / std
    F[:, constant] = 0.0

    return F, keep


def max_pairwise_distance(F, *, max_samples=4096, seed=42):
    """Computes ``max_kl ||f_k - f_l||`` over the training descriptor.

    Args:
        F: Standardised descriptor of shape (M, d).
        max_samples: Cap on the number of rows used, to bound the MxM distance matrix.
        seed: Seed for the subsample when the cap applies.

    Returns:
        The maximum pairwise Euclidean distance as a float.
    """
    if len(F) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(F), size=max_samples, replace=False)
        F = F[idx]
    with torch.no_grad():
        t = torch.tensor(F, dtype=torch.float32)
        return float(torch.cdist(t, t, p=2).max().item())


def main(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    X_np = np.load(os.path.join(args.data_path, "processed_windows.npy"))
    meta_df = pd.read_csv(os.path.join(args.data_path, "processed_metadata.csv"))
    features = np.load(args.expert_features)
    quality = None
    r2_path = args.expert_features.replace(".npy", "_r2.npy")
    if args.min_apsd_r2 is not None and os.path.exists(r2_path):
        quality = np.load(r2_path)

    if len(features) != len(X_np):
        raise ValueError(
            f"Expert features ({len(features)}) do not align with the window set ({len(X_np)}). "
            "Rebuild them with code/src/build_expert_features.py against this metadata."
        )
    print(f"Windows: {X_np.shape}, expert descriptor: {features.shape}")

    # --- Exclude the test subjects of this fold (same convention as train_simclr.py) ---
    if args.exclude_subjects:
        keep_mask = ~meta_df["subject"].isin(args.exclude_subjects)
        print(f"Excluding {len(args.exclude_subjects)} subjects for pre-training: {args.exclude_subjects}")
        X_np = X_np[keep_mask.values]
        features = features[keep_mask.values]
        if quality is not None:
            quality = quality[keep_mask.values]
        print(f"Number of windows for pre-training after exclusion: {len(X_np)}")

    # --- Descriptor preparation, fitted on this fold's training windows only ---
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

    dataset = ExpertFeatureDataset(X, F)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = EnhancedAttentionLSTM(
        input_size=X.shape[2],
        hidden_size=args.embedding_size,
        n_channels=X.shape[1],
        sfreq=args.sampling_frequency,
        lstm_hidden_size=args.embedding_size // 2,
    ).to(device)

    # Adam with exponential decay, as specified in the paper (Sec. 4.2), not the repo's AdamW.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = ExponentialLR(optimizer, gamma=args.lr_gamma)
    criterion = ExpCLRLoss(
        device,
        delta=args.delta,
        temperature=None if args.no_hard_negative_mining else args.temperature,
        feat_max_dist=feat_max_dist,
        squared_similarity=not args.linear_similarity,
    )

    losses = []
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
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {epoch_loss:.4f}")

    fold_suffix = f"_{args.fold_id}" if args.fold_id else ""
    model_path = os.path.join(
        args.save_dir,
        f"ExpCLR_{args.zone}_{args.frequency}{fold_suffix}_{args.descriptor}"
        f"_batch_{args.batch_size}_lr_{args.lr}_tau_{args.temperature}_delta_{args.delta}.pth",
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    sidecar = model_path.replace(".pth", "_config.json")
    with open(sidecar, "w") as fh:
        json.dump(
            {
                "descriptor": args.descriptor,
                "descriptor_dim": int(F.shape[1]),
                "n_windows": int(len(X)),
                "feat_max_dist": feat_max_dist,
                "delta": args.delta,
                "temperature": None if args.no_hard_negative_mining else args.temperature,
                "loss_on": args.loss_on,
                "final_loss": losses[-1] if losses else None,
            },
            fh,
            indent=2,
        )

    plt.plot(range(1, args.num_epochs + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (ExpCLR)")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(
        args.plot_dir, f"ExpCLR_{args.zone}_{args.frequency}{fold_suffix}_training_loss_curve.png"
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
    parser.add_argument("--descriptor", default="P_full",
                        help="Descriptor label, recorded in the checkpoint name.")
    parser.add_argument("--zone", default="all", help="Brain zone (naming only).")
    parser.add_argument("--frequency", default="all", help="Frequency band (naming only).")
    parser.add_argument("--fold_id", default="", help="Fold identifier for the checkpoint name.")
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
                        help="Exponential LR decay, as in the paper (Sec. 4.2).")
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
                        help="Apply the loss to the projection head output (repo convention for "
                             "SimCLR) or directly to the evaluated embedding.")
    parser.add_argument("--min_apsd_r2", type=float, default=None,
                        help="Optional quality filter on the mean specparam R^2 per window.")
    parser.add_argument("--save_dir", default="save/models", help="Checkpoint directory.")
    parser.add_argument("--plot_dir", default="save/figures/pretrain", help="Loss-curve directory.")
    main(parser.parse_args())
