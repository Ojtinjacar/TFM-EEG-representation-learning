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
import random

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


def checkpoint_is_reusable(checkpoint_path, expected):
    """Decides whether an existing checkpoint was trained under the requested settings.

    Reuse used to be decided by file name alone, which silently returned encoders trained under a
    different objective whenever a setting was absent from the name: the switch from
    ``--loss_on projection`` to ``embedding`` became a no-op because 45 checkpoints already existed
    at the expected paths. Comparing against the sidecar closes that class of bug for every
    setting, including ones added later.

    A checkpoint with no sidecar is treated as not reusable: it predates this check and its
    provenance is unknown.

    Args:
        checkpoint_path: Path of the ``.pth`` under consideration.
        expected: Mapping of sidecar keys to the values the caller requires.

    Returns:
        True if the checkpoint exists and every requested key matches.
    """
    checkpoint_path = str(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        return False
    sidecar = checkpoint_path.replace(".pth", "_config.json")
    if not os.path.exists(sidecar):
        return False
    with open(sidecar) as fh:
        recorded = json.load(fh)
    for key, value in expected.items():
        if key not in recorded:
            return False
        if isinstance(value, float) and isinstance(recorded[key], (int, float)):
            if not np.isclose(recorded[key], value):
                return False
        elif recorded[key] != value:
            return False
    return True


def effective_dimensionality(z):
    """Measures how many directions a representation actually uses.

    Computed as ``exp(H)`` over the normalised PCA eigenvalues, so a cloud spread evenly over k
    directions scores k and one collapsed onto a single axis scores 1. This is the diagnostic that
    exposes a contracting encoder: the ExpCLR loss can fall steadily while the representation
    collapses onto two or three directions, in which case a falling loss looks like convergence and
    is in fact damage. Tracking it alongside the loss is what makes that visible during training.

    Args:
        z: Representation matrix of shape (n_samples, n_features).

    Returns:
        Effective dimensionality as a float, between 1 and n_features.
    """
    # A diverged encoder produces NaN or inf, on which np.linalg.svd raises and would otherwise
    # abort a whole hyperparameter sweep instead of just discarding that configuration.
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
    """Builds the distance matrix of a perfectly equidistant cloud, as ``mu_i`` normalises it.

    Not a matrix of ones. ``mu_i`` averages over all N entries of row i including ``dist_ii = 0``,
    so if every off-diagonal distance equals some c, the row mean is ``c*(N-1)/N``; forcing that to
    the 1 the normalisation imposes gives ``c = N/(N-1)``. A matrix of ones has row mean
    ``(N-1)/N`` and is therefore produced by no embedding at all, which understated the reference
    floor by about 6 % at N=64 -- precisely in the regime where the warning has to fire.

    Args:
        n: Batch size.
        device: Torch device.

    Returns:
        Matrix of shape (n, n), zero on the diagonal and ``n/(n-1)`` elsewhere.
    """
    ref = torch.full((n, n), n / (n - 1.0), device=device)
    ref.fill_diagonal_(0.0)
    return ref


@torch.no_grad()
def diagnose_epoch(model, batches, criterion, device, loss_on):
    """Measures the geometry of the representation after a training epoch.

    Reports the effective dimensionality and the loss of the *equidistant* geometry, the degenerate
    optimum the objective falls into when ``Delta`` does not match the scale of the descriptor
    similarities. A training loss approaching that reference means the encoder is learning to place
    every pair at the same distance, i.e. the least informative geometry available, rather than
    reproducing the expert geometry.

    Takes pre-drawn batches rather than a DataLoader on purpose. Iterating a shuffling loader draws
    a fresh permutation from its generator even if the loop breaks early, which made
    ``--diagnose_every`` silently change the batch order of training and therefore the final
    weights: an observation-only flag acting as a hyperparameter. Fixed batches also make the
    estimate comparable across epochs instead of resampling noise.

    Args:
        model: Encoder under training.
        batches: Fixed list of (windows, features) tensors to measure on.
        criterion: The ExpCLR loss, used to recompute the reference on the same batches.
        device: Torch device.
        loss_on: Either "projection" or "embedding", matching the training objective.

    Returns:
        Dict with ``effective_dim`` and ``equidistant_loss``.
    """
    was_training = model.training
    model.eval()
    reps, references = [], []
    for x_batch, f_batch in batches:
        x_batch, f_batch = x_batch.to(device), f_batch.to(device)
        z = model(x_batch) if loss_on == "projection" else model.get_embedding(x_batch)
        reps.append(z.cpu().numpy())
        target = (1.0 - criterion.expert_similarity(f_batch)) * criterion.delta
        # Scored with the loss actually being optimised, so the two are comparable. Using the mean
        # (Eq. 3) against a log-sum-exp training loss (Eq. 4) made the warning unfirable by Jensen.
        references.append(float(criterion.reduce(
            (target - equidistant_reference(len(target), target.device)).pow(2))))
    if was_training:
        model.train()
    return {
        "effective_dim": effective_dimensionality(np.concatenate(reps, axis=0)),
        "equidistant_loss": float(np.mean(references)),
    }


def set_seed(seed):
    """Seeds every source of randomness that affects the trained encoder.

    Without this the encoder's initialisation and the batch order depend on torch's global state,
    so two runs of the same fold produce different weights and a small difference between methods
    could be attributed to the initialisation rather than to the method.

    ``torch.manual_seed`` covers the CPU and the accelerator (CUDA and MPS) generators. The
    DataLoader needs its own explicit generator, which :func:`main` passes in.

    Args:
        seed: Seed applied to ``random``, ``numpy`` and ``torch``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


#: Smallest batch whose backward pass works on MPS. ``torch.cdist``, which the ExpCLR loss uses for
#: both the descriptor and the embedding distances, switches to a matmul implementation once the
#: input has more than 25 rows. Only that path has a backward on MPS; the direct kernel raises
#: NotImplementedError for ``aten::_cdist_backward``. The paper's batch size of 64 is safely above.
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

    # Seeded before the loader and the model, the two consumers of torch's global RNG.
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

    # Fixed batches, drawn once from a generator of their own so the training loader's RNG is
    # untouched and the measurements stay comparable across epochs.
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
        # Diagnosed every few epochs: a falling loss with a falling effective dimensionality is
        # contraction, not convergence.
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
                # Everything a caller must match before reusing this checkpoint. Reuse is decided by
                # comparing against these, not by the file name: a setting missing from the name
                # would otherwise be silently inherited from a previous run.
                "delta": args.delta,
                "temperature": None if args.no_hard_negative_mining else args.temperature,
                "loss_on": args.loss_on,
                "sim_max": args.sim_max,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "seed": args.seed,
                "exclude_subjects": sorted(args.exclude_subjects),
                "final_loss": losses[-1] if losses else None,
                "effective_dim": final["effective_dim"],
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
    parser.add_argument("--loss_on", choices=["projection", "embedding"], default="embedding",
                        help="Where the loss is applied. Defaults to 'embedding' because ExpCLR "
                             "has no projection head: the paper optimises E(x), the very same "
                             "representation it evaluates (Secs. 3.6 and 4.2). Applying it to the "
                             "projection would shape a geometry behind a non-invertible ReLU MLP "
                             "that the probe never sees. 'projection' is kept for the SimCLR-style "
                             "ablation only.")
    parser.add_argument("--diagnose_every", type=int, default=5,
                        help="Epoch interval for the geometry diagnostics (effective "
                             "dimensionality and equidistant-geometry reference).")
    parser.add_argument("--min_apsd_r2", type=float, default=None,
                        help="Optional quality filter on the mean specparam R^2 per window.")
    parser.add_argument("--save_dir", default="save/models", help="Checkpoint directory.")
    parser.add_argument("--plot_dir", default="save/figures/pretrain", help="Loss-curve directory.")
    main(parser.parse_args())
