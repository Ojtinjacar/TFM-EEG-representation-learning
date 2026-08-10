import os
import sys
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from checkpoint_naming import simclr_checkpoint_name, write_sidecar
from utils import set_seed
from loss import NTXentLoss
from models import EnhancedAttentionLSTM

class CIMCYCDataset(Dataset):
    def __init__(self, X, aug_mode="legacy"):
        self.X = torch.FloatTensor(X)
        self.aug_mode = aug_mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        anchor = self.X[idx]

        # Generate two different augmentations
        aug1 = self.augment_sample(anchor)
        aug2 = self.augment_sample(anchor)
        return anchor, aug1, aug2

    def augment_sample(self, sample):
        if self.aug_mode == "legacy":
            return self._augment_legacy(sample)
        return self._augment_composed(sample)

    def _augment_legacy(self, sample):
        augmented = sample.clone()

        spatial_augmentations = [
            self.apply_channel_dropout,
            self.apply_channel_swap
        ]

        temporal_augmentations = [
            self.apply_time_shift,
            self.add_gaussian_noise,
            self.apply_zero_masking
        ]

        # Apply one augmentation from each category
        # augmented = np.random.choice(frequency_augmentations)(augmented)
        augmented = np.random.choice(spatial_augmentations)(augmented)
        augmented = np.random.choice(temporal_augmentations)(augmented)
        return augmented

    def validate_augmentation(self, original, augmented, min_correlation=0.3):
        """Ensure augmentation hasn't destroyed the signal"""
        corr = torch.corrcoef(torch.stack([original.flatten(), augmented.flatten()]))[0,1]
        return corr >= min_correlation

    def add_gaussian_noise(self, sample, std_range=(0.00001, 0.00005)):
        """Add random Gaussian noise with varying intensity"""
        std = torch.FloatTensor(1).uniform_(*std_range).item()
        noise = torch.randn_like(sample) * std
        return sample + noise

    def apply_time_shift(self, sample, max_shift_ratio=0.15):
        """Apply random time shift"""
        max_shift = int(sample.shape[1] * max_shift_ratio)
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        return torch.roll(sample, shifts=shift, dims=1)

    def apply_channel_dropout(self, sample, drop_prob=0.1):
        """Randomly drop entire channels"""
        mask = torch.rand(sample.shape[0]) >= drop_prob
        masked_sample = sample.clone()
        masked_sample[~mask] = 0
        return masked_sample

    def apply_channel_swap(self, sample, swap_prob=0.1):
        """Randomly swap adjacent channels"""
        augmented = sample.clone()
        for i in range(sample.shape[0] - 1):
            if torch.rand(1) < swap_prob:
                augmented[i], augmented[i+1] = augmented[i+1].clone(), augmented[i].clone()
        return augmented

    def apply_zero_masking(self, sample, mask_prob_range=(0.2, 0.3)):
        """Randomly mask values to zero"""
        mask_prob = torch.FloatTensor(1).uniform_(*mask_prob_range).item()
        mask = torch.rand_like(sample) < mask_prob
        masked_sample = sample.clone()
        masked_sample[mask] = 0
        return masked_sample

    def apply_amplitude_scaling(self, sample, scale_range=(0.8, 1.2)):
        """Scale the amplitude by a random factor"""
        scale = torch.FloatTensor(1).uniform_(*scale_range)
        return sample * scale

    def _pools_for_mode(self):
        dropout = [self.apply_channel_dropout]
        dropout_swap = [self.apply_channel_dropout, self.apply_channel_swap]
        legacy_temporal = [self.apply_time_shift, self.add_gaussian_noise, self.apply_zero_masking]
        psd_temporal = [
            self.apply_time_shift,
            self.add_gaussian_noise,
            self.apply_smooth_time_mask,
            self.apply_ft_surrogate,
            self.apply_sign_flip,
            self.apply_time_reverse,
        ]
        top2_temporal = [self.apply_ft_surrogate, self.apply_time_reverse]
        pools = {
            "no_swap": (dropout, legacy_temporal),
            "legacy_plus_psd": (dropout_swap, psd_temporal),
            "zone_preserving": (dropout, psd_temporal),
            "psd_ftsurrogate": (dropout, [self.apply_ft_surrogate]),
            "psd_smoothmask": (dropout, [self.apply_smooth_time_mask]),
            "psd_signflip": (dropout, [self.apply_sign_flip]),
            "psd_timereverse": (dropout, [self.apply_time_reverse]),
            "psd_top2": (dropout, top2_temporal),
        }
        return pools[self.aug_mode]

    def _augment_composed(self, sample, max_retries=3, min_correlation=0.3):
        spatial_pool, tempfreq_pool = self._pools_for_mode()
        validation_exempt = (self.apply_ft_surrogate, self.apply_sign_flip, self.apply_time_reverse)

        last = sample.clone()
        for _ in range(max_retries):
            candidate = sample.clone()
            candidate = spatial_pool[np.random.randint(len(spatial_pool))](candidate)
            tf = tempfreq_pool[np.random.randint(len(tempfreq_pool))]
            candidate = tf(candidate)
            last = candidate
            if tf in validation_exempt or self.validate_augmentation(sample, candidate, min_correlation):
                return candidate
        return last

    def apply_ft_surrogate(self, sample, phase_noise_max=None):
        if phase_noise_max is None:
            phase_noise_max = 0.9 * float(np.pi)
        n_times = sample.shape[1]
        spectrum = torch.fft.rfft(sample, dim=1)
        n_freqs = spectrum.shape[1]
        dphi = torch.empty(n_freqs).uniform_(0.0, phase_noise_max)
        dphi[0] = 0.0
        phase = torch.exp(1j * dphi).unsqueeze(0)
        surrogate = torch.fft.irfft(spectrum * phase, n=n_times, dim=1)
        return surrogate.to(sample.dtype)

    def apply_smooth_time_mask(self, sample, mask_len_ratio=0.15, sharpness=10.0):
        n_times = sample.shape[1]
        mask_len = max(1, int(n_times * mask_len_ratio))
        t_cut = torch.randint(0, max(1, n_times - mask_len), (1,)).item()
        t = torch.arange(n_times, dtype=sample.dtype)
        rise = torch.sigmoid(sharpness * (t - t_cut))
        fall = torch.sigmoid(sharpness * (t_cut + mask_len - t))
        keep = 1.0 - rise * fall
        return sample * keep.unsqueeze(0)

    def apply_time_reverse(self, sample):
        return torch.flip(sample, dims=[1])

    def apply_sign_flip(self, sample):
        return -sample

def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    data_path = os.path.join(args.data_path, "processed_windows.npy")
    meta_path = os.path.join(args.data_path, "processed_metadata.csv")

    X_np = np.load(data_path)
    meta_df = pd.read_csv(meta_path)

    # --- Exclude subjects for the test set ---
    if args.exclude_subjects:
        keep_mask = ~meta_df['subject'].isin(args.exclude_subjects)
        print(f"Excluding {len(args.exclude_subjects)} subjects for pre-training: {args.exclude_subjects}")
        X_np = X_np[keep_mask.values]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Number of windows for pre-training after exclusion: {len(X_np)}")
    else:
        keep_mask = pd.Series(True, index=range(len(X_np)))

    X = torch.tensor(X_np, dtype=torch.float32)
    meta = meta_df.to_numpy()

    if args.positives == "neighbor":
        from neighbor_positives import NeighborPositiveDataset

        nidx_path = os.path.join(args.neighbor_index_dir, f"neighbor_index_{args.neighbor_metric}.npy")
        nidx_full = np.load(nidx_path)
        if nidx_full.shape[0] != len(keep_mask):
            raise ValueError(
                f"neighbor_index ({nidx_full.shape[0]}) no alinea con el dataset ({len(keep_mask)}). "
                "El indice debe computarse sobre el mismo conjunto de ventanas (mismo N/orden)."
            )
        kept = keep_mask.values
        g2l = np.full(len(kept), -1, dtype=np.int64)
        g2l[kept] = np.arange(int(kept.sum()))
        nidx_local = nidx_full[kept]
        neighbor_index = np.full_like(nidx_local, -1)
        valid = nidx_local >= 0
        neighbor_index[valid] = g2l[nidx_local[valid]]

        view1_augmenter = CIMCYCDataset(X, aug_mode=args.aug_mode)
        full_dataset = NeighborPositiveDataset(
            X, neighbor_index, augment=view1_augmenter.augment_sample, fallback="duplicate",
        )
        print(f"Positives=neighbor metric={args.neighbor_metric} "
              f"coverage={full_dataset.coverage():.3f} view1_aug_mode={args.aug_mode}")
    else:
        full_dataset = CIMCYCDataset(X, aug_mode=args.aug_mode)
        print(f"Positives=augment aug_mode={args.aug_mode}")
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(args.seed))
    eval_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)

    model = EnhancedAttentionLSTM(
        input_size=X.shape[2],
        hidden_size=args.embedding_size,
        n_channels=X.shape[1],
        sfreq=args.sampling_frequency,
        lstm_hidden_size=args.embedding_size // 2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = NTXentLoss(device, batch_size=args.batch_size, temperature=args.temperature)

    losses = []
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for anchor, aug1, aug2 in train_loader:
            anchor, aug1, aug2 = anchor.to(device), aug1.to(device), aug2.to(device)
            optimizer.zero_grad()
            z1 = model(aug1)
            z2 = model(aug2)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        losses.append(epoch_loss)  
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {epoch_loss:.4f}')

    # Save model
    model_filename = simclr_checkpoint_name(
        args.zone, args.frequency, args.fold_id,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, temperature=args.temperature,
    )
    model_path = os.path.join(args.save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    write_sidecar(model_path, {
        "method": "SimCLR",
        "zone": args.zone,
        "frequency": args.frequency,
        "fold_id": args.fold_id,
        "exclude_subjects": sorted(str(s) for s in (args.exclude_subjects or [])),
        "seed": getattr(args, "seed", None),
        "aug_mode": args.aug_mode,
        "positives": args.positives,
        "neighbor_metric": args.neighbor_metric if args.positives == "neighbor" else None,
        "neighbor_index_dir": args.neighbor_index_dir if args.positives == "neighbor" else None,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "num_epochs": args.num_epochs,
        "n_windows": int(len(X)),
    })

    # Extract embeddings
    model.eval()
    embeddings, meta_list = [], []
    with torch.no_grad():
        for anchor, _, _ in eval_loader:
            anchor = anchor.to(device)
            embed = model.get_embedding(anchor).cpu().numpy()
            embeddings.append(embed)
            meta_list.append(meta[:len(embed)])
            meta = meta[len(embed):]

    print("Embeddings shape:", np.vstack(embeddings).shape)
    print("Metadata shape:", np.vstack(meta_list).shape)
    print("Original dataset size:", len(full_dataset))


    # np.save(os.path.join(args.save_dir, f"SimCLR_embeddings_{args.zone}_{args.frequency}{fold_suffix}.npy"), np.vstack(embeddings))
    # np.save(os.path.join(args.save_dir, f"SimCLR_metadata_{args.zone}_{args.frequency}{fold_suffix}.npy"), np.vstack(meta_list))
    # print(f"Embeddings and metadata saved to {args.save_dir}/")

    # Graph loss curve
    plt.plot(range(1, args.num_epochs + 1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (NT-Xent)')
    plt.grid(True)
    plt.tight_layout()

    fold_suffix = f"_{args.fold_id}" if args.fold_id else ""
    plot_path = os.path.join(args.plot_dir, f"SimCLR_{args.zone}_{args.frequency}{fold_suffix}_training_loss_curve.png")
    plt.savefig(plot_path)
    print(f"Loss curve saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train contrastive EEG model (SimCLR-style)."
    )
    parser.add_argument(
        "--sampling_frequency",
        type=int,
        default=250,
        help="Sampling frequency (Hz)."
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=128,
        help="Embedding size of the model."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Temperature parameter for NT-Xent loss."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save/models",
        help="Directory (relative to save_path) to save models."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="save/figures/pretrain",
        help="Directory (relative to save_path) to save plots."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/processed/5_s",
        help="Path to loading."
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
        help="List of subject IDs to exclude from training (for test set)."
    )
    parser.add_argument(
        "--fold_id",
        type=str,
        default=None,
        help="Fold identifier to include in model filename (e.g., 'fold0')."
    )
    parser.add_argument(
        "--aug_mode",
        type=str,
        default="legacy",
        choices=["legacy", "no_swap", "legacy_plus_psd", "zone_preserving",
                 "psd_ftsurrogate", "psd_smoothmask", "psd_signflip", "psd_timereverse", "psd_top2"],
        help=(
            "Augmentation strategy: 'legacy' (original: channel dropout/swap + "
            "time shift/gaussian/zero-mask); 'zone_preserving' (drops channel swap, adds "
            "PSD-preserving transforms FTSurrogate/SmoothTimeMask/SignFlip/TimeReverse and "
            "enables the correlation validation). The 'psd_*' modes are the fine ablation: "
            "dropout + a single transform ('psd_top2' = the two winners). "
            "Used for view1 when --positives neighbor."
        )
    )
    parser.add_argument(
        "--positives",
        type=str,
        default="augment",
        choices=["augment", "neighbor"],
        help="Positive pair source: 'augment' (two augmentations) or 'neighbor' (view2 = real "
             "nearest window, view1 = augmented anchor). See code/src/neighbor_positives.py."
    )
    parser.add_argument(
        "--neighbor_metric",
        type=str,
        default="cosine",
        choices=["cosine", "wasserstein", "riemann"],
        help="Distance used to find the neighbor positive (only if --positives neighbor)."
    )
    parser.add_argument(
        "--neighbor_index_dir",
        type=str,
        default="data/processed/neighbor_index",
        help="Directory with neighbor_index_<metric>.npy (from build_neighbor_index.py)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weights, shuffling and augmentations."
    )
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
