import os
import argparse
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA

from models import (EnhancedAttentionLSTM, AttentionLSTMAutoencoder,
                    MaskedAttentionLSTMAutoencoder, VariationalAttentionLSTMAutoencoder)
from interfusion import EMBEDDING_STATS, InterFusionEEG
from checkpoint_naming import sidecar_path
from utils import set_seed

# ============================
#   DATASETS
# ============================

class EEGSupervisedDataset(Dataset):
    """
    Dataset for downstream tasks (regression).
    """
    def __init__(self, X, y, subjects):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        self.X = X
        self.y = torch.tensor(y, dtype=torch.float32)
        self.subjects = subjects

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subjects[idx]


# ============================
#   HEAD / FULL MODEL
# ============================

class Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Linear(in_dim // 2, out_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Which representation each pre-training objective actually shapes. A loss applied to a
# projection head that is thrown away afterwards, as in SimCLR, leaves the embedding
# underneath as the representation to evaluate; a loss applied to the encoder output
# itself leaves that output. Evaluating the other one measures something the objective
# never optimized.
LOSS_REPRESENTATION = {"ExpCLR": "projection"}
DEFAULT_REPRESENTATION = "embedding"


class FullModel(nn.Module):
    """Backbone and head for the downstream task."""

    def __init__(self, backbone, head, representation=DEFAULT_REPRESENTATION):
        """Initializes the model.

        Args:
            backbone (nn.Module): Pre-trained encoder.
            head (nn.Module): Task head.
            representation (str): ``embedding`` reads the encoder below its projection,
                ``projection`` reads the encoder output.

        Raises:
            ValueError: If the representation is not one of the two.
        """
        super().__init__()
        if representation not in ("embedding", "projection"):
            raise ValueError(
                f"representation must be 'embedding' or 'projection', got {representation!r}"
            )
        self.backbone = backbone
        self.head = head
        self.representation = representation

    def forward(self, x):
        """Maps a batch of windows to a prediction.

        Args:
            x (torch.Tensor): Windows, shape (batch, n_channels, n_samples).

        Returns:
            torch.Tensor: Head output.
        """
        # The backbone is already frozen; no need for no_grad here
        # if the optimizer only acts on the head.
        emb = (self.backbone(x) if self.representation == "projection"
               else self.backbone.get_embedding(x))
        return self.head(emb)


# ============================
#   SUBJECT SPLIT
# ============================

def subject_split_indices(subject_ids, test_subjects, seed=42):
    """
    Splits data into train/test according to a list of test subject IDs.
    """

    subjects = np.array(subject_ids)
    unique_subjects = np.unique(subjects)

    if not test_subjects:
        raise ValueError("The list of test subjects cannot be empty.")

    # Check that test subjects are present in the dataset
    missing_subjects = set(test_subjects) - set(unique_subjects)
    if missing_subjects:
        raise ValueError(f"The following test subjects were not found in the data: {list(missing_subjects)}")

    train_subj = [s for s in unique_subjects if s not in test_subjects]
    test_subj = test_subjects

    train_mask = np.isin(subjects, train_subj)
    test_mask = np.isin(subjects, test_subj)

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    print(f"# Test subject IDs: {test_subj}")
    print(f"# subjects: total={len(unique_subjects)}, train={len(train_subj)}, test={len(test_subj)}")
    print(f"# windows: train={len(train_idx)}, test={len(test_idx)}")

    return train_idx, test_idx


def get_loaders_by_subject(X, y, subject_ids,
                           batch_size,
                           test_subjects,
                           seed=None):
    n_samples = len(X)
    assert len(y) == n_samples
    assert len(subject_ids) == n_samples

    train_idx, test_idx = subject_split_indices(subject_ids, test_subjects=test_subjects)

    X_train, y_train, subjects_train = X[train_idx], y[train_idx], subject_ids[train_idx]
    X_test, y_test, subjects_test = X[test_idx], y[test_idx], subject_ids[test_idx]

    train_ds = EEGSupervisedDataset(X_train, y_train, subjects_train)
    test_ds = EEGSupervisedDataset(X_test, y_test, subjects_test)

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=False,
                              generator=generator)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader


# ============================
#   TRAINING / METRICS
# ============================

def run_epoch_regression(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_subjects = []

    with torch.set_grad_enabled(train):
        for X, y, subjects in loader:
            X = X.to(device)
            y = y.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(X)
            loss = criterion(outputs.squeeze(), y.squeeze())

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * X.size(0)
            all_preds.append(outputs.detach().cpu().squeeze().reshape(-1))
            all_targets.append(y.detach().cpu().squeeze().reshape(-1))
            all_subjects.extend(subjects)

    running_loss /= len(loader.dataset)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_subjects = np.array(all_subjects)

    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    std_y = all_targets.std()
    nrmse = rmse if std_y < 1e-8 else rmse / std_y

    return running_loss, nrmse, rmse, all_targets, all_preds, all_subjects


def train_model_regression(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    model_tag,
    save_dir,
):

    for epoch in range(num_epochs):
        train_loss, train_nrmse, _, _, _, _= run_epoch_regression(
            model, train_loader, criterion, optimizer, device, train=True
        )

        print(
            f"[{model_tag}] Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_loss:.4f}, train_nRMSE={train_nrmse:.4f}"
        )

    # Save the model at the end of training
    final_state = model.state_dict()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model_tag}.pth"
    save_path = os.path.join(save_dir, filename)

    torch.save(final_state, save_path)
    print(f"Final model '{model_tag}' saved to: {save_path}")

    return model


def evaluate_and_plot(
    model,
    test_loader,
    criterion,
    device,
    model_tag,
    task_description,
    plot_dir,
    fold_id=None,
):
    test_loss, _, _, y_true, y_pred, test_subjects = run_epoch_regression(
        model, test_loader, criterion, optimizer=None, device=device, train=False
    )

    # 2. Compute per-subject metrics and then average them
    df_results = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'subject': test_subjects
    })

    subject_metrics = []
    unique_test_subjects = sorted(df_results['subject'].unique())
    for subject_id in unique_test_subjects:
        df_subj = df_results[df_results['subject'] == subject_id]
        subj_y_true = df_subj['y_true'].values
        subj_y_pred = df_subj['y_pred'].values

        if len(df_subj) > 0:
            subj_mse = mean_squared_error(subj_y_true, subj_y_pred)
            subj_rmse = np.sqrt(subj_mse)
            subj_std_y = subj_y_true.std()
            subj_nrmse = subj_rmse if subj_std_y < 1e-8 else subj_rmse / subj_std_y
            subj_r2 = r2_score(subj_y_true, subj_y_pred) if len(df_subj) > 1 else np.nan

            subject_metrics.append({
                'rmse': subj_rmse,
                'nrmse': subj_nrmse,
                'r2': subj_r2
            })

    # 3. Aggregate subject metrics to obtain the final result
    df_subject_metrics = pd.DataFrame(subject_metrics)

    # Average each subject's metrics to give them equal weight
    final_nrmse = df_subject_metrics['nrmse'].mean()
    final_rmse = df_subject_metrics['rmse'].mean()
    final_r2 = df_subject_metrics['r2'].mean()

    print(f"[{model_tag}] Test loss={test_loss:.4f}, Test nRMSE (Subject-Avg)={final_nrmse:.4f}, Test RMSE (Subject-Avg)={final_rmse:.2f}, Test R2 (Subject-Avg)={final_r2:.4f}")

    # 4. Print subject-level average predictions for global R² computation in LOO
    # Simple format: SUBJECT_AVG_PRED: y_true_mean y_pred_mean
    for subject_id in unique_test_subjects:
        df_subj = df_results[df_results['subject'] == subject_id]
        subj_y_true_mean = df_subj['y_true'].mean()
        subj_y_pred_mean = df_subj['y_pred'].mean()
        print(f"SUBJECT_AVG_PRED: {subj_y_true_mean:.6f} {subj_y_pred_mean:.6f}")

    # --- Regression plot ---
    os.makedirs(plot_dir, exist_ok=True)
    if fold_id:
        fig_name = f"{model_tag}_{fold_id}_regression_plot.png"
    else:
        fig_name = f"{model_tag}_regression_plot.png"
    fig_path = os.path.join(plot_dir, fig_name)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label="Windows (test)")

    if len(y_true) > 1:
        # The fit line is still computed over all windows
        a, b = np.polyfit(y_true, y_pred, 1)
        x_line = np.linspace(y_true.min(), y_true.max(), 100)
        y_line = a * x_line + b
        plt.plot(x_line, y_line, label="Lineal fit")

    mean_true = y_true.mean()
    plt.axhline(mean_true, color="red", linestyle="--", label="True Mean")

    plt.xlabel("True")
    plt.ylabel("Prediction")
    # Update title with subject-aggregated metrics
    plt.title(f"{task_description}\n Method: {args.method}, Zone: {args.zone}, Frequency: {args.frequency}\n nRMSE={final_nrmse:.3f}, R2={final_r2:.3f}, (RMSE={final_rmse:.2f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"Regression plot saved to: {fig_path}")
    

# ============================
#   MAIN
# ============================

def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Load data
    X = np.load(args.data_path)
    meta_df = pd.read_csv(args.meta_path)

    # Define target column and task description
    target_col = args.target
    if target_col == "age":
        task_description = "Chronological age (months)"
    elif "cit" in target_col and "36mo" in target_col:
        task_description = f"Cognitive age at 36 (IQ)"
    else:
        task_description = f"Prediction of {target_col}"

    # Prepare regression task data: drop windows with missing target label
    mask = ~meta_df[target_col].isna()
    X_task = X[mask]
    y_task = meta_df.loc[mask, target_col].values
    subject_ids_task = meta_df.loc[mask, args.subject_column].values

    print(f"\nTarea: {task_description}")
    print(f"Ventanas totales tras filtrado: {len(X_task)}")

    # DataLoaders with subject-based split
    train_loader, test_loader = get_loaders_by_subject(
        X_task, y_task, subject_ids_task,
        batch_size=args.batch_size,
        test_subjects=args.test_subjects,
        seed=args.seed
    )

    if args.method in ["SimCLR", "AE", "MAE", "TripletLoss", "VAE", "ExpCLR", "InterFusion"]:

        if not args.model_path:
            raise ValueError(f"The --model_path argument is required for method '{args.method}'")

        embedding_size = args.embedding_size

        if args.method == "InterFusion":
            # Architecture hyperparameters live in the checkpoint sidecar so
            # train and eval can never drift apart.
            with open(sidecar_path(args.model_path)) as fh:
                sc = json.load(fh)
            # The readout mode is not part of the checkpoint: it decides how the
            # trained model is read, not how it was trained, so it comes from the
            # caller and the same weights serve both modes.
            backbone = InterFusionEEG(
                x_dim=X.shape[1], window=X.shape[2], z_dim=sc["z_dim"],
                strides=tuple(sc.get("strides", (2, 1, 2, 1, 2, 2, 2))),
                rnn_hidden=sc["rnn_hidden"],
                dense_hidden=sc.get("dense_hidden", 500),
                flow_levels=sc["flow_levels"],
                embedding_stats=getattr(args, "embedding_stats", "mean"),
            ).to(device)
            backbone.load_state_dict(
                torch.load(args.model_path, map_location=device), strict=True
            )
            embedding_size = backbone.embedding_dim
            print(f"Pre-trained backbone ('InterFusion') loaded from: "
                  f"{args.model_path} (embedding_dim={embedding_size})")
        else:
            # 1) Load pre-trained backbone
            model_params = {
                "input_size": X.shape[2],
                "hidden_size": embedding_size,
                "n_channels": X.shape[1],
                "sfreq": args.sampling_frequency,
                "lstm_hidden_size": embedding_size // 2,
            }

            model_map = {
                "SimCLR": EnhancedAttentionLSTM,
                "ExpCLR": EnhancedAttentionLSTM,
                "AE": AttentionLSTMAutoencoder,
                "MAE": MaskedAttentionLSTMAutoencoder,
                "VAE": VariationalAttentionLSTMAutoencoder,
                "TripletLoss": EnhancedAttentionLSTM
            }
            model_class = model_map.get(args.method)

            backbone = model_class(**model_params).to(device)
            state_dict = torch.load(args.model_path, map_location=device)
            if args.method == "VAE":
                # The latent prior has no parameters to restore for inference; drop
                # its keys and load the rest strictly, so a genuine mismatch fails.
                state_dict = {k: v for k, v in state_dict.items()
                              if not k.startswith("prior.")}
            backbone.load_state_dict(state_dict, strict=True)
            print(f"Pre-trained backbone ('{args.method}') loaded from: {args.model_path}")

        # 2) Build full model and optionally freeze backbone
        head = Head(embedding_size, 1).to(device)
        representation = LOSS_REPRESENTATION.get(args.method, DEFAULT_REPRESENTATION)
        model = FullModel(backbone, head, representation=representation).to(device)
        print(f"Evaluating the representation the objective shaped: {representation}.")

        if args.eval_mode == 'linear_probe': 
            for p in model.backbone.parameters():
                p.requires_grad = False
            print("Backbone frozen for linear probing.")
            optimizer = torch.optim.AdamW(
                model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print("Backbone not frozen for fine-tuning.")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )

        fold_suffix = f"_{args.fold_id}" if args.fold_id else ""
        model_tag = f"{args.method}_{args.zone}_{args.frequency}_{args.target}_{args.eval_mode}{fold_suffix}"
    
    elif args.method == "PCA":
        embedding_size = args.embedding_size

        # Extract data from the subject-split DataLoaders
        X_train, y_train = train_loader.dataset.X, train_loader.dataset.y
        X_test, y_test = test_loader.dataset.X, test_loader.dataset.y

        # Flatten and apply PCA (fit on train only)
        X_train_flat = X_train.numpy().reshape(X_train.shape[0], -1)
        X_test_flat = X_test.numpy().reshape(X_test.shape[0], -1)

        pca = PCA(n_components=embedding_size, random_state=args.seed)
        X_train_pca = pca.fit_transform(X_train_flat)
        X_test_pca = pca.transform(X_test_flat)
        print(f"PCA: {X_train_flat.shape[1]} -> {embedding_size} components. Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

        # Get subjects from the original datasets
        subjects_train = train_loader.dataset.subjects
        subjects_test = test_loader.dataset.subjects

        # Overwrite DataLoaders with PCA embeddings
        train_loader = DataLoader(EEGSupervisedDataset(X_train_pca, y_train, subjects_train),
                                  batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed))
        test_loader = DataLoader(EEGSupervisedDataset(X_test_pca, y_test, subjects_test), batch_size=args.batch_size, shuffle=False)

        # The model is just the 'Head' that learns from the PCA components
        model = Head(embedding_size, 1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        fold_suffix = f"_{args.fold_id}" if args.fold_id else ""
        model_tag = f"PCA_{args.zone}_{args.frequency}_{args.target}{fold_suffix}"

    elif args.method == "supervised":
        print(f"\n=== Training supervised from scratch ===")
        embedding_size = args.embedding_size

        # 1) Initialise backbone from scratch (using EnhancedAttentionLSTM as base architecture)
        model_params = {
            "input_size": X.shape[2], 
            "hidden_size": embedding_size, 
            "n_channels": X.shape[1],
            "sfreq": args.sampling_frequency, 
            "lstm_hidden_size": embedding_size // 2,
        }
        backbone = EnhancedAttentionLSTM(**model_params).to(device)
        print("Backbone initialised from scratch.")

        # 2) Build full model
        head = Head(embedding_size, 1).to(device)
        model = FullModel(backbone, head).to(device)
        print("Backbone NOT frozen. The full model will be trained.")

        # 3) Optimiser for the full model
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        fold_suffix = f"_{args.fold_id}" if args.fold_id else ""
        model_tag = f"supervised_{args.zone}_{args.frequency}_{args.target}{fold_suffix}"

    else:
        raise ValueError(f"Unrecognised method '{args.method}'.")

    criterion_reg = nn.MSELoss()
    model= train_model_regression(
        model,
        train_loader,
        criterion_reg,
        optimizer,
        device,
        num_epochs=args.num_epochs,
        model_tag=model_tag,
        save_dir=args.save_dir,
    )

    # ---- Evaluation and plot ----
    evaluate_and_plot(
        model,
        test_loader,
        criterion_reg,
        device,
        model_tag=model_tag,
        task_description=task_description,
        plot_dir=args.plot_dir,
        fold_id=args.fold_id,
    )
    
    print("\nProcess completed.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Validation of a pre-trained encoder on a regression task (linear probing / fine-tuning)."
    )

    # Paths
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/5_s/processed_windows.npy",
        help="Path to processed_windows.npy."
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="data/processed/5_s/processed_metadata.csv",
        help="Path to processed_metadata.csv."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the pre-trained backbone model (.pth). Not used in 'supervised' mode."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save/downstream_models",
        help="Directory to save the linear probing model."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="save/figures/prediction",
        help="Directory to save the regression plot."
    )

    # Model
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=128,
        help="Backbone embedding size."
    )
    parser.add_argument(
        "--sampling_frequency",
        type=int,
        default=250,
        help="EEG sampling frequency."
    )

    # Split y DataLoaders
    parser.add_argument(
        "--subject_column",
        type=str,
        default="subject",
        help="Column with the subject ID in the metadata."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for the regression task."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the data split."
    )

    # Training hyperparameters (Linear Probe)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Epochs to train the linear head."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the linear head."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for the optimiser."
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
        "--method",
        type=str,
        choices=["SimCLR", "AE", "supervised", "MAE", "PCA", "TripletLoss", "VAE", "ExpCLR",
                 "InterFusion"],
        required=True,
        help="Method to use."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="age",
        help="Objective variable"
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="linear_probe",
        choices=["linear_probe", "fine_tuning"],
        help="Evaluation mode: linear probe or fine-tuning."
    )
    parser.add_argument(
        "--test_subjects",
        nargs="+",
        default=None,
        help="List test subjects."
    )
    parser.add_argument(
        "--fold_id",
        type=str,
        default=None,
        help="Fold identifier to include in output filenames."
    )
    parser.add_argument(
        "--embedding_stats",
        type=str,
        default="mean",
        choices=sorted(EMBEDDING_STATS),
        help="InterFusion readout: 'mean' concatenates the averaged z2 and z1 means "
             "(channels + z_dim); 'mean_std' appends their standard deviations and doubles "
             "the width. Ignored by every other method."
    )

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)