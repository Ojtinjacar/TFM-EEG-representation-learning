import os
import argparse
import numpy as np
import pandas as pd
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from models import EnhancedAttentionLSTM

class TripletDataset(Dataset):
    """
    Dataset for Triplet Loss.
    Returns (anchor, positive, negative).
    - Positive: a random sample with the same label as the anchor.
    - Negative: a random sample with a different label.
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = y # numpy array of labels

        self.labels_to_indices = defaultdict(list)
        for i, label in enumerate(y):
            self.labels_to_indices[label].append(i)

        self.labels = list(self.labels_to_indices.keys())
        self.n_labels = len(self.labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        anchor_sample = self.X[index]
        anchor_label = self.y[index]

        # --- Positive sampling ---
        # Choose an index different from the anchor but with the same label
        positive_list = self.labels_to_indices[anchor_label]
        positive_idx = index
        # If there is only one sample for that label, use itself as positive
        if len(positive_list) > 1:
            while positive_idx == index:
                positive_idx = random.choice(positive_list)
        else:
            positive_idx = index # fallback
        positive_sample = self.X[positive_idx]

        # --- Negative sampling ---
        # Choose a different label
        negative_label = random.choice(self.labels)
        while negative_label == anchor_label:
            negative_label = random.choice(self.labels)

        # Choose a random sample from that negative label
        negative_idx = random.choice(self.labels_to_indices[negative_label])
        negative_sample = self.X[negative_idx]

        return anchor_sample, positive_sample, negative_sample

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # --- Data loading ---
    data_path = os.path.join(args.data_path, "processed_windows.npy")
    meta_path = os.path.join(args.data_path, "processed_metadata.csv")

    X = np.load(data_path)
    meta_df = pd.read_csv(meta_path)

    # --- Exclude subjects for the test set ---
    if args.exclude_subjects:
        print(f"Excluding {len(args.exclude_subjects)} subjects for test: {args.exclude_subjects}")
        keep_mask = ~meta_df['subject'].isin(args.exclude_subjects)

        X = X[keep_mask.values]
        meta_df = meta_df[keep_mask].reset_index(drop=True)
        print(f"Number of windows for training after exclusion: {len(X)}")

    # --- Label preparation (y) according to target ---
    if args.target == 'age':
        # Use age directly as label
        y = meta_df['age'].values
        print(f"Triplet Loss task based on 'age'. Number of unique ages: {len(np.unique(y))}")

    elif args.target == 'cit_36mo':
        # Create 4 intervals (quartiles) for IQ

        # Remove NaNs for quartile calculation and training
        valid_mask = ~meta_df[args.target].isna()
        X = X[valid_mask]
        meta_df = meta_df[valid_mask]

        # pd.qcut to create the 4 intervals
        try:
            iq_bins, intervals = pd.qcut(meta_df[args.target], q=4, retbins=True, labels=False, duplicates='drop')
            y = iq_bins.values
            print(f"Triplet Loss task based on 'IQ' ({args.target}).")
            print(f"IQ intervals (quartiles): {intervals}")
            print(f"Number of windows after filtering NaNs: {len(y)}")
        except ValueError as e:
            print(f"Error creating IQ quartiles: {e}")
            print("There may not be enough data or unique values. Aborting.")
            return
    else:
        raise ValueError(f"Target '{args.target}' not recognised.")

    # --- Dataset and DataLoader ---
    dataset = TripletDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # --- Model, Optimizer and Loss ---
    model = EnhancedAttentionLSTM(
        input_size=X.shape[2],
        hidden_size=args.embedding_size,
        n_channels=X.shape[1],
        sfreq=args.sampling_frequency,
        lstm_hidden_size=args.embedding_size // 2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)

    # --- Training loop ---
    losses = []
    active_triplets_percentages = [] # List to store % of active triplets
    epoch_gaps = [] # List to store average gap
    print("\nStarting Triplet Loss training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        total_gap = 0.0
        num_active_triplets = 0.0
        total_triplets_in_epoch = 0.0

        for anchor, positive, negative in train_loader:

            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()

            # Get embeddings
            emb_anchor = model.get_embedding(anchor)
            emb_positive = model.get_embedding(positive)
            emb_negative = model.get_embedding(negative)

            # L2 normalise embeddings
            emb_anchor = nn.functional.normalize(emb_anchor, p=2, dim=1)
            emb_positive = nn.functional.normalize(emb_positive, p=2, dim=1)
            emb_negative = nn.functional.normalize(emb_negative, p=2, dim=1)

            # Compute loss
            loss = criterion(emb_anchor, emb_positive, emb_negative)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Compute the gap (d_an - d_ap) for analysis
            with torch.no_grad():
                d_ap = torch.pairwise_distance(emb_anchor, emb_positive, p=2)
                d_an = torch.pairwise_distance(emb_anchor, emb_negative, p=2)
                gap_batch = d_an - d_ap
                total_gap += gap_batch.sum().item()

            # Count active triplets (where loss is > 0)
            num_active_triplets += (loss > 1e-5).sum().item()
            total_triplets_in_epoch += anchor.size(0)

        epoch_loss = total_loss / len(train_loader)
        losses.append(epoch_loss)

        avg_gap = total_gap / total_triplets_in_epoch if total_triplets_in_epoch > 0 else 0
        epoch_gaps.append(avg_gap)

        active_percentage = (num_active_triplets / total_triplets_in_epoch) * 100 if total_triplets_in_epoch > 0 else 0
        active_triplets_percentages.append(active_percentage)

        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {epoch_loss:.4f}, Active Triplets: {active_percentage:.2f}%, GAP: {avg_gap:.4f}')


    # --- Save model ---
    fold_suffix = f"_{args.fold_id}" if args.fold_id else ""
    model_filename = (
        f"Triplet_{args.target}_{args.zone}_{args.frequency}{fold_suffix}_"
        f"emb{args.embedding_size}_m{args.margin}.pth"
    )
    model_path = os.path.join(args.save_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # --- Loss curve plot ---
    plt.figure()
    plt.plot(range(1, args.num_epochs + 1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Loss')
    plt.title('Training Loss Curve (Triplet Loss)')
    plt.grid(True)
    plt.tight_layout()

    plot_filename = f"Triplet_{args.target}_{args.zone}_{args.frequency}{fold_suffix}_loss_curve.png"
    plot_path = os.path.join(args.plot_dir, plot_filename)
    plt.savefig(plot_path)
    print(f"Loss curve saved to: {plot_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pre-training with Triplet Loss for EEG."
    )

    ## Data and save arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/processed/5_s",
        help="Path to the folder with processed data."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save/models",
        help="Directory to save models."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="save/figures/pretrain",
        help="Directory to save plots."
    )
    parser.add_argument(
        "--exclude_subjects",
        nargs="+",
        default=None,
        help="List of subject IDs to exclude from training (for test)."
    )

    # Task arguments
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=['age', 'cit_36mo'],
        help="Objective for defining triplets: 'age' or 'IQ'."
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="all",
        help="Brain zone of the data."
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="all",
        help="Frequency band of the data."
    )

    # Model arguments
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=128,
        help="Embedding size of the model."
    )
    parser.add_argument(
        "--sampling_frequency",
        type=int,
        default=250,
        help="Sampling frequency (Hz)."
    )

    # Training arguments
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size."
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
        "--margin",
        type=float,
        default=0.4,
        help="Margin for TripletMarginLoss."
    )
    parser.add_argument(
        "--fold_id",
        type=str,
        default=None,
        help="Fold identifier to include in model filename (e.g., 'fold0')."
    )

    args = parser.parse_args()
    main(args)