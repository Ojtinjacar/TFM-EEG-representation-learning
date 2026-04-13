"""
ST-EEGFormer: loads and fine-tunes the ST-EEGFormer-largeV2 foundation model.

This script implements:
1. Exact ST-EEGFormer encoder architecture (Vision Transformer for EEG) matching the checkpoint
2. Loading of MAE pre-trained weights
3. Fine-tuning and linear probing for downstream regression tasks

Based on: https://github.com/LiuyinYang1101/STEEGFormer
Model: ST-EEGFormer-largeV2 (pre-trained on HBN dataset with MAE)

Checkpoint contents:
- Encoder: patch_embed, enc_channel_emd, enc_temporal_emd, cls_token, blocks.0-23, norm
- MAE decoder: decoder_* (not used for downstream tasks)

Strategy:
- Load ONLY the encoder from the checkpoint
- Add a SIMPLE linear layer (1024 → 1) for regression
- Freeze everything except the final linear layer (linear probe)
"""

import os
import math
import argparse
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from scipy.signal import resample
from timm.models.vision_transformer import Block
from timm.layers import trunc_normal_

# ============================================================================
# EXACT ST-EEGFormer ENCODER ARCHITECTURE (compatible with checkpoint)
# ============================================================================

class ChannelEmbedding(nn.Module):
    """
    Channel embedding via linear transformation.
    Matches enc_channel_emd.channel_transformation in the checkpoint.
    """
    def __init__(self, max_channels=256, embed_dim=1024):
        super().__init__()
        # The checkpoint uses a linear transformation, not nn.Embedding
        self.channel_transformation = nn.Linear(max_channels, embed_dim, bias=False)
        self.max_channels = max_channels

    def forward(self, channel_indices, batch_size, device):
        """
        Creates channel embeddings from one-hot encoding.

        Args:
            channel_indices: unique channel indices used
            batch_size: batch size
            device: target device
        Returns:
            (num_channels, embed_dim) channel embeddings
        """
        # Create one-hot encoding for each channel
        one_hot = torch.zeros(len(channel_indices), self.max_channels, device=device)
        for i, ch_idx in enumerate(channel_indices):
            one_hot[i, ch_idx] = 1.0

        # Transform: (num_channels, max_channels) -> (num_channels, embed_dim)
        channel_emb = self.channel_transformation(one_hot)
        return channel_emb


class TemporalEmbedding(nn.Module):
    """
    Learned temporal embedding.
    Matches enc_temporal_emd.pe in the checkpoint.
    """
    def __init__(self, max_len=512, embed_dim=1024):
        super().__init__()
        # Registered buffer (not a parameter, but loaded from checkpoint)
        self.register_buffer('pe', torch.zeros(1, max_len, embed_dim))

    def forward(self, seq_len):
        """
        Args:
            seq_len: temporal sequence length
        Returns:
            (1, seq_len, embed_dim) temporal embeddings
        """
        return self.pe[:, :seq_len, :]


class STEEGFormerEncoder(nn.Module):
    """
    ST-EEGFormer Encoder — EXACT architecture matching the checkpoint.

    Includes only the encoder layers present in the checkpoint:
    - patch_embed.proj: Linear(16 -> 1024)
    - enc_channel_emd.channel_transformation: Linear(256 -> 1024)
    - enc_temporal_emd.pe: Buffer (1, 512, 1024)
    - cls_token: Parameter (1, 1, 1024)
    - blocks.0-23: Transformer blocks
    - norm: LayerNorm(1024)
    """
    def __init__(
        self,
        num_channels=70,
        seq_len=640,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches_per_channel = seq_len // patch_size
        self.num_patches = num_channels * self.num_patches_per_channel

        # Patch embedding (matches checkpoint: patch_embed.proj)
        self.patch_embed = nn.ModuleDict({
            'proj': nn.Linear(patch_size, embed_dim)
        })

        # Channel embedding (matches checkpoint: enc_channel_emd)
        self.enc_channel_emd = ChannelEmbedding(max_channels=256, embed_dim=embed_dim)

        # Temporal embedding (matches checkpoint: enc_temporal_emd)
        self.enc_temporal_emd = TemporalEmbedding(max_len=512, embed_dim=embed_dim)

        # CLS token (matches checkpoint: cls_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks (matches checkpoint: blocks.0-23)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)
        ])

        # Final norm (matches checkpoint: norm)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Weight initialisation
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Encoder forward pass.

        Args:
            x: (B, C, T) EEG signals
        Returns:
            (B, embed_dim) CLS token embedding (for classification/regression)
        """
        B, C, T = x.shape
        device = x.device

        # 1. Patch embedding
        # (B, C, T) -> unfold -> (B, C, num_patches_per_channel, patch_size)
        x = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        # -> (B, C * num_patches_per_channel, patch_size)
        x = x.reshape(B, -1, self.patch_size)
        # -> (B, num_patches, embed_dim)
        x = self.patch_embed['proj'](x)

        # 2. Add channel embedding
        channel_indices = list(range(self.num_channels))
        channel_emb = self.enc_channel_emd(channel_indices, B, device)  # (C, embed_dim)
        # Repeat for each temporal patch
        channel_emb = channel_emb.unsqueeze(1).repeat(1, self.num_patches_per_channel, 1)  # (C, T_patches, embed_dim)
        channel_emb = channel_emb.reshape(-1, self.embed_dim)  # (num_patches, embed_dim)
        channel_emb = channel_emb.unsqueeze(0).expand(B, -1, -1)  # (B, num_patches, embed_dim)
        x = x + channel_emb

        # 3. Add temporal embedding
        temporal_emb = self.enc_temporal_emd(self.num_patches_per_channel)  # (1, T_patches, embed_dim)
        # Repeat for each channel
        temporal_emb = temporal_emb.repeat(1, self.num_channels, 1)  # (1, num_patches, embed_dim)
        x = x + temporal_emb

        # 4. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+num_patches, embed_dim)

        x = self.pos_drop(x)

        # 5. Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # 6. Extract CLS token for classification/regression
        cls_output = x[:, 0, :]  # (B, embed_dim)

        return cls_output


class STEEGFormerRegressor(nn.Module):
    """
    Full regression model: Encoder + MLP head.

    Strategy:
    - Encoder: loads checkpoint weights and is frozen
    - Head: deep MLP no wider than the embedding (~1.3M parameters)
    """
    def __init__(
        self,
        num_channels=70,
        seq_len=640,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        head_dropout=0.1,
    ):
        super().__init__()

        # Encoder (will load checkpoint weights)
        self.encoder = STEEGFormerEncoder(
            num_channels=num_channels,
            seq_len=seq_len,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Deep MLP head (progressively narrower, never wider than embed_dim)
        # 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 1
        # Parameters: ~660K params
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),       # 1024 -> 512: ~524K
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(512, 256),             # 512 -> 256: ~131K
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 128),             # 256 -> 128: ~33K
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 64),              # 128 -> 64: ~8K
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, 32),               # 64 -> 32: ~2K
            nn.GELU(),
            nn.Linear(32, num_classes),      # 32 -> 1: ~33
        )

        # Initialisation
        self._init_head()

    def _init_head(self):
        """Initialises MLP head weights."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, T) EEG signals
        Returns:
            (B, num_classes) predictions
        """
        features = self.encoder(x)  # (B, embed_dim)
        return self.head(features)  # (B, num_classes)

    def get_embedding(self, x):
        """
        Extracts encoder embeddings.
        """
        return self.encoder(x)


def create_steeeg_former_large(num_channels=70, seq_len=640, num_classes=1, **kwargs):
    """
    Creates the ST-EEGFormer-large model for regression.

    Configuration (matches checkpoint):
    - embed_dim: 1024
    - depth: 24 blocks
    - num_heads: 16
    """
    return STEEGFormerRegressor(
        num_channels=num_channels,
        seq_len=seq_len,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs
    )


# ============================================================================
# LOADING AND PREPROCESSING FUNCTIONS
# ============================================================================

def resample_eeg(X, orig_sfreq=250, target_sfreq=128):
    """
    Resamples EEG signals from one sampling frequency to another.

    Args:
        X: (N, C, T) EEG signal array
        orig_sfreq: original sampling frequency
        target_sfreq: target sampling frequency

    Returns:
        (N, C, T_new) resampled array
    """
    if orig_sfreq == target_sfreq:
        return X

    N, C, T = X.shape
    T_new = int(T * target_sfreq / orig_sfreq)

    X_resampled = np.zeros((N, C, T_new), dtype=X.dtype)

    for i in range(N):
        for c in range(C):
            X_resampled[i, c] = resample(X[i, c], T_new)

    return X_resampled


def load_pretrained_weights(model, checkpoint_path, strict=False):
    """
    Loads pre-trained weights into the model, handling key mismatches.

    The checkpoint has keys such as:
        - patch_embed.proj.weight
        - enc_channel_emd.channel_transformation.weight
        - blocks.0.attn.qkv.weight
        - etc.

    Our model has keys such as:
        - encoder.patch_embed.proj.weight
        - encoder.enc_channel_emd.channel_transformation.weight
        - encoder.blocks.0.attn.qkv.weight
        - head.weight (new layer, not in checkpoint)

    Args:
        model: STEEGFormerRegressor model
        checkpoint_path: path to the .pth file
        strict: if True, require exact key match

    Returns:
        Message with missing/unexpected keys
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model_state = model.state_dict()

    # Map checkpoint keys to our model (add 'encoder.' prefix)
    # Only map encoder keys (skip decoder_*)
    new_state_dict = {}
    skipped_keys = []
    loaded_keys = []

    for k, v in state_dict.items():
        # Skip MAE decoder keys (not needed for downstream)
        if k.startswith('decoder_') or k.startswith('dec_') or k == 'mask_token':
            skipped_keys.append(k)
            continue

        # Add 'encoder.' prefix to match our model's state dict
        new_key = f"encoder.{k}"

        if new_key in model_state:
            if model_state[new_key].shape == v.shape:
                new_state_dict[new_key] = v
                loaded_keys.append(new_key)
            else:
                print(f"  [SKIP] {new_key}: shape mismatch {v.shape} vs {model_state[new_key].shape}", flush=True)
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)

    print(f"  [INFO] Encoder keys loaded: {len(loaded_keys)}", flush=True)
    print(f"  [INFO] Skipped keys (decoder/incompatible): {len(skipped_keys)}", flush=True)

    # Load weights
    msg = model.load_state_dict(new_state_dict, strict=False)

    # Head is already initialised with trunc_normal_ in _init_head()
    print("  [INFO] Regression head ready (initialised with trunc_normal)", flush=True)

    return msg


def prepare_dataloaders(X, y, batch_size=32, train_ratio=0.8, seed=42):
    """
    Prepares dataloaders for training.

    Args:
        X: (N, C, T) EEG data
        y: (N,) labels
        batch_size: batch size
        train_ratio: fraction of data used for training
        seed: random seed for reproducibility

    Returns:
        train_loader, val_loader
    """
    torch.manual_seed(seed)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on a dataloader.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()

        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    mse = F.mse_loss(all_preds, all_targets).item()
    rmse = np.sqrt(mse)

    # nRMSE normalised by std
    std = all_targets.std().item()
    nrmse = rmse / std if std > 1e-8 else rmse

    return {
        "loss": total_loss / len(dataloader),
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse
    }


def fine_tune_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    lr=1e-4,
    weight_decay=0.01,
    warmup_epochs=5,
    save_path=None,
    freeze_backbone=False
):
    """
    Fine-tunes the ST-EEGFormer model.

    Args:
        model: STEEGFormer model
        train_loader, val_loader: DataLoaders
        device: device (cuda/cpu)
        epochs: number of epochs
        lr: learning rate
        weight_decay: weight decay
        warmup_epochs: number of warmup epochs
        save_path: path to save the best model
        freeze_backbone: if True, freezes the backbone (linear probe)

    Returns:
        Trained model and best metrics
    """
    model = model.to(device)

    # Freeze backbone if requested (linear probe)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        print("[INFO] Backbone frozen (linear probe mode)", flush=True)

    # Parameters to optimise
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_metrics = None

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1
        )

        # Eval
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val nRMSE: {val_metrics['nrmse']:.4f}", flush=True)

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_metrics = val_metrics
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  [SAVE] Best model saved to {save_path}", flush=True)

    return model, best_metrics


@torch.no_grad()
def extract_embeddings(model, X, batch_size=32, device='cuda'):
    """
    Extracts model embeddings for downstream analysis.

    Args:
        model: STEEGFormer model
        X: (N, C, T) EEG data
        batch_size: batch size
        device: target device

    Returns:
        (N, embed_dim) embeddings
    """
    model = model.to(device)
    model.eval()

    all_embeddings = []

    for i in range(0, len(X), batch_size):
        batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
        embeddings = model.get_embedding(batch)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ST-EEGFormer for downstream tasks"
    )

    # Paths
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/5_s/processed_windows.npy",
        help="Path to EEG data (.npy)"
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="data/processed/5_s/processed_metadata.csv",
        help="Path to metadata (.csv)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="save/models/steeg_former.pth",
        help="Path to the pre-trained checkpoint"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="save/models/steeg_former_linear_probe.pth",
        help="Path to save the fine-tuned model"
    )

    # Target
    parser.add_argument(
        "--target",
        type=str,
        default="age",
        choices=["age", "cit_36mo"],
        help="Target variable"
    )

    # Model configuration
    parser.add_argument(
        "--orig-sfreq",
        type=int,
        default=250,
        help="Original sampling frequency"
    )
    parser.add_argument(
        "--target-sfreq",
        type=int,
        default=128,
        help="Target sampling frequency (128 Hz for ST-EEGFormer)"
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone (linear probe)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Mode
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract embeddings without training"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    # Load data
    print("[INFO] Loading data...", flush=True)
    X = np.load(args.data_path)
    meta = pd.read_csv(args.meta_path)
    print(f"  Original shape: {X.shape}", flush=True)

    # Resample if needed
    if args.orig_sfreq != args.target_sfreq:
        print(f"[INFO] Resampling from {args.orig_sfreq} Hz to {args.target_sfreq} Hz...", flush=True)
        X = resample_eeg(X, args.orig_sfreq, args.target_sfreq)
        print(f"  Resampled shape: {X.shape}", flush=True)

    # Filter samples with valid target
    valid_mask = ~meta[args.target].isna()
    X = X[valid_mask]
    y = meta.loc[valid_mask, args.target].values
    print(f"  Samples with valid {args.target}: {len(y)}", flush=True)

    # Create model
    print("[INFO] Creating ST-EEGFormer-large model...", flush=True)
    model = create_steeeg_former_large(
        num_channels=X.shape[1],
        seq_len=X.shape[2],
        num_classes=1
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Load pre-trained weights
    if os.path.exists(args.checkpoint_path):
        print(f"[INFO] Loading pre-trained weights from {args.checkpoint_path}...", flush=True)
        msg = load_pretrained_weights(model, args.checkpoint_path)
        print(f"  Missing keys: {len(msg.missing_keys)}", flush=True)
        print(f"  Unexpected keys: {len(msg.unexpected_keys)}", flush=True)
    else:
        print(f"[WARNING] Checkpoint not found: {args.checkpoint_path}", flush=True)
        print("  Using model without pre-training", flush=True)

    if args.extract_only:
        # Extract embeddings only
        print("[INFO] Extracting embeddings...", flush=True)
        embeddings = extract_embeddings(model, X, args.batch_size, device)

        # Save embeddings
        emb_path = args.save_path.replace('.pth', '_embeddings.npy')
        np.save(emb_path, embeddings)
        print(f"[INFO] Embeddings saved to {emb_path}", flush=True)
        print(f"  Shape: {embeddings.shape}", flush=True)

    else:
        # Fine-tuning
        print("[INFO] Preparing dataloaders...", flush=True)
        train_loader, val_loader = prepare_dataloaders(
            X, y, args.batch_size, train_ratio=0.8, seed=args.seed
        )
        print(f"  Train: {len(train_loader.dataset)} samples", flush=True)
        print(f"  Val: {len(val_loader.dataset)} samples", flush=True)

        print("[INFO] Starting fine-tuning...", flush=True)
        mode = "linear probe" if args.freeze_backbone else "fine-tuning"
        print(f"  Mode: {mode}", flush=True)

        model, best_metrics = fine_tune_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=0.01,
            warmup_epochs=5,
            save_path=args.save_path,
            freeze_backbone=args.freeze_backbone
        )

        print("\n[RESULTS]", flush=True)
        print(f"  Best Val Loss: {best_metrics['loss']:.4f}", flush=True)
        print(f"  Best Val nRMSE: {best_metrics['nrmse']:.4f}", flush=True)
        print(f"  Best Val RMSE: {best_metrics['rmse']:.4f}", flush=True)

if __name__ == "__main__":
    main()
