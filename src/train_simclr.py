import os
import argparse 
import numpy as np 
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from loss import NTXentLoss
from models import EnhancedAttentionLSTM
    
class CIMCYCDataset(Dataset):

    def __init__(self, X):
        self.X = torch.FloatTensor(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        anchor = self.X[idx]
        
        # Generate two different augmentations
        aug1 = self.augment_sample(anchor)
        aug2 = self.augment_sample(anchor)
        return anchor, aug1, aug2

    def augment_sample(self, sample):
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
        print(f"Excluding {len(args.exclude_subjects)} subjects for pre-training: {args.exclude_subjects}")
        keep_mask = ~meta_df['subject'].isin(args.exclude_subjects)

        X_np = X_np[keep_mask.values]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Number of windows for pre-training after exclusion: {len(X_np)}")

    X = torch.tensor(X_np, dtype=torch.float32)
    meta = meta_df.to_numpy()

    # Create datasets using subsets
    full_dataset = CIMCYCDataset(X)
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
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
    fold_suffix = f"_{args.fold_id}" if args.fold_id else ""
    model_path = os.path.join(args.save_dir, f"SimCLR_{args.zone}_{args.frequency}{fold_suffix}_batch_{args.batch_size}_lr_{args.lr}_wd_{args.weight_decay}_temperature_{args.temperature}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

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

    args = parser.parse_args()
    main(args)