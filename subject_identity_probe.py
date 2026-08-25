"""Subject-identity probe on frozen embeddings.

Measures how much subject identity a frozen representation carries: a linear
classifier is trained to name the subject a window comes from, splitting
*windows* (never subjects) into train and test. High accuracy means the encoder
memorised who the child is rather than what their EEG matured into.

Brookshire et al. (2024) report clinical EEG classifiers reaching 99.8% accuracy
with segment-wise splits that collapse to chance under subject holdout, so this
probe is read alongside the age probe: an age gain that arrives together with an
identity gain is suspect, not a success.

Example:
    python subject_identity_probe.py --fold_id fold0 \
        --methods SimCLR SimCLR-xsubj-cosine VAE InterFusion
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import run_downstream as rd
from checkpoint_naming import sidecar_path
from downstream import BACKBONE_BY_METHOD
from interfusion import InterFusionEEG
from utils import infer_embeddings_in_batches


class CheckpointNotFound(FileNotFoundError):
    """Raised when no checkpoint matches a requested method and fold."""


def expected_sidecar(method, fold_id, zone, frequency):
    """Returns the sidecar fields that identify the checkpoint of a method.

    Args:
        method (str): Method name, including the SimCLR and ExpCLR variants.
        fold_id (str): Fold identifier, e.g. ``fold0``.
        zone (str): Head zone the encoder trained on.
        frequency (str): Frequency band.

    Returns:
        dict: Fields a matching sidecar must record.
    """
    expected = {"zone": zone, "frequency": frequency, "fold_id": fold_id}
    simclr = rd.build_simclr_variants(zone)
    if method in simclr:
        tag = simclr[method]["tag"]
        expected["method"] = "SimCLR"
        # A variant carries its tag inside the fold id: fold0_xscosine.
        expected["fold_id"] = f"{fold_id}_{tag}" if tag else fold_id
    elif method in rd.EXPCLR_VARIANTS:
        # The variants differ in the descriptor, which the filename shares otherwise.
        expected["method"] = "ExpCLR"
        expected["descriptor"] = rd.EXPCLR_VARIANTS[method]["descriptor"]
    else:
        expected["method"] = method
    return expected


def find_checkpoint(method, fold_id, model_dirs, zone, frequency):
    """Locates the checkpoint a method wrote for one fold, zone and band.

    The sidecar decides rather than the filename. A name cannot tell two zones of equal
    channel count apart, and several checkpoints of one fold can differ in something it does
    not carry at all, so an ambiguous match is refused instead of resolved alphabetically:
    loading the wrong one would measure a different encoder under the right label.

    Args:
        method (str): Method name, including variants.
        fold_id (str): Fold identifier.
        model_dirs (list): Directories to search, in priority order.
        zone (str): Head zone the encoder must have trained on.
        frequency (str): Frequency band it must have trained on.

    Returns:
        str: Path to the checkpoint.

    Raises:
        CheckpointNotFound: If nothing matches, or if more than one does.
    """
    expected = expected_sidecar(method, fold_id, zone, frequency)
    for directory in model_dirs:
        matches = []
        for sidecar in sorted(glob.glob(os.path.join(directory, "*_config.json"))):
            checkpoint = sidecar.replace("_config.json", ".pth")
            if not os.path.exists(checkpoint):
                continue
            with open(sidecar) as fh:
                recorded = json.load(fh)
            if all(recorded.get(key) == value for key, value in expected.items()):
                matches.append(checkpoint)
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise CheckpointNotFound(
                f"'{method}' {fold_id} on {zone}/{frequency} matches {len(matches)} "
                f"checkpoints in {directory}: {[os.path.basename(m) for m in matches]}. "
                "They differ in something their sidecars do not distinguish, so pick one "
                "with --model_dirs rather than have it chosen alphabetically."
            )
    raise CheckpointNotFound(
        f"No checkpoint for '{method}' {fold_id} on zone {zone!r}, band {frequency!r} in "
        f"{model_dirs}. Checkpoints without a sidecar are not eligible."
    )


def load_backbone(method, model_path, x_dim, window, embedding_size, sfreq, device):
    """Rebuilds a frozen backbone exposing ``get_embedding``.

    Args:
        method (str): Method name.
        model_path (str): Checkpoint path.
        x_dim (int): Number of channels.
        window (int): Samples per window.
        embedding_size (int): Embedding width for the legacy encoders.
        sfreq (int): Sampling frequency.
        device (torch.device): Target device.

    Returns:
        torch.nn.Module: Backbone in eval mode.

    Raises:
        ValueError: If the method has no frozen-backbone mapping.
    """
    if method == "InterFusion":
        with open(sidecar_path(model_path)) as fh:
            sidecar = json.load(fh)
        backbone = InterFusionEEG(
            x_dim=x_dim, window=window, z_dim=sidecar["z_dim"],
            strides=tuple(sidecar.get("strides", (2, 1, 2, 1, 2, 2, 2))),
            rnn_hidden=sidecar["rnn_hidden"],
            dense_hidden=sidecar.get("dense_hidden", 500),
            flow_levels=sidecar["flow_levels"],
            embedding_stats=sidecar.get("embedding_stats", "mean"),
        ).to(device)
    else:
        base = "SimCLR" if method in rd.build_simclr_variants("all") else method
        base = "ExpCLR" if method in rd.EXPCLR_VARIANTS else base
        model_class = BACKBONE_BY_METHOD.get(base)
        if model_class is None:
            raise ValueError(f"Method '{method}' has no frozen-backbone mapping")
        backbone = model_class(
            input_size=window, hidden_size=embedding_size, n_channels=x_dim,
            sfreq=sfreq, lstm_hidden_size=embedding_size // 2,
        ).to(device)

    backbone.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    backbone.eval()
    return backbone


def identity_accuracy(embeddings, subject_ids, seed, test_size=0.3):
    """Trains a linear classifier to name the subject of each window.

    Windows are split, not subjects: every subject appears in both halves, which
    is exactly what makes the score a memorisation measure.

    Args:
        embeddings (np.ndarray): Frozen embeddings (N, D).
        subject_ids (np.ndarray): Subject label per window (N,).
        seed (int): Split and solver seed.
        test_size (float): Held-out window fraction.

    Returns:
        dict: accuracy, chance level, ratio to chance and number of subjects.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        embeddings, subject_ids, test_size=test_size,
        random_state=seed, stratify=subject_ids,
    )
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(scaler.transform(x_train), y_train)
    accuracy = float(clf.score(scaler.transform(x_test), y_test))
    n_subjects = int(len(np.unique(subject_ids)))
    chance = 1.0 / n_subjects
    return {
        "identity_accuracy": accuracy,
        "chance": chance,
        "ratio_to_chance": accuracy / chance,
        "n_subjects": n_subjects,
    }


def main(args):
    """Runs the probe for every requested method and writes one CSV.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
    """
    win_path, meta_path = rd.config_data_paths(args.zone, args.frequency)
    X = np.load(win_path)
    meta_df = pd.read_csv(meta_path)

    # The probe runs on the windows the encoder was trained on: the fold's
    # training subjects. Held-out subjects were never seen, so memorising them
    # was impossible and including them would dilute the measure.
    folds = {f"fold{idx}": (tr, te) for idx, tr, te in _folds(args, meta_path)}
    if args.fold_id not in folds:
        raise ValueError(f"Unknown fold '{args.fold_id}'")
    train_subjects, _ = folds[args.fold_id]

    mask = meta_df[args.subject_column].isin(train_subjects).values
    X_fold = X[mask]
    subjects = meta_df.loc[mask, args.subject_column].values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] {args.fold_id}: {len(train_subjects)} subjects, {len(X_fold)} windows, "
          f"device={device}", flush=True)

    rows = []
    for method in args.methods:
        try:
            model_path = find_checkpoint(method, args.fold_id, args.model_dirs,
                                         args.zone, args.frequency)
            backbone = load_backbone(
                method, model_path, X.shape[1], X.shape[2],
                args.embedding_size, args.sampling_frequency, device,
            )
            embeddings = infer_embeddings_in_batches(
                backbone, X_fold, batch_size=args.batch_size, device=device
            )
            result = identity_accuracy(embeddings, subjects, args.seed, args.test_size)
            result.update({
                "method": method, "fold": args.fold_id,
                "embedding_dim": int(embeddings.shape[1]),
                "checkpoint": os.path.basename(model_path),
            })
            rows.append(result)
            print(f"  {method:24s} identity={result['identity_accuracy']:.3f} "
                  f"(chance {result['chance']:.3f}, x{result['ratio_to_chance']:.1f})",
                  flush=True)
            del backbone
            torch.cuda.empty_cache()
        except (CheckpointNotFound, ValueError, RuntimeError) as exc:
            print(f"  [ERROR] {method}: {exc}", flush=True)

    if not rows:
        raise SystemExit("[ERROR] No method produced a result.")

    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, f"identity_probe_{args.fold_id}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[INFO] Results written to: {out}", flush=True)


def _folds(args, meta_path):
    """Rebuilds the protocol's fold split.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        meta_path (str): Metadata CSV path.

    Yields:
        tuple[int, list[str], list[str]]: fold index, train and test subjects.
    """
    from sklearn.model_selection import KFold

    meta_df = pd.read_csv(meta_path)
    subjects = sorted(meta_df[~meta_df[args.target].isna()][args.subject_column]
                      .unique().tolist())
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.base_seed)
    for idx, (train_idx, test_idx) in enumerate(kf.split(subjects)):
        yield (idx, [subjects[i] for i in train_idx], [subjects[i] for i in test_idx])


def parse_args():
    """Parses the command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Subject-identity probe on frozen embeddings."
    )
    parser.add_argument("--methods", nargs="+", required=True,
                        help="Methods or SimCLR variants to probe.")
    parser.add_argument("--fold_id", type=str, default="fold0", help="Fold to probe.")
    parser.add_argument("--model_dirs", nargs="+", default=["save/models"],
                        help="Checkpoint directories, in priority order.")
    parser.add_argument("--save_dir", type=str, default="save/identity",
                        help="Output directory.")
    parser.add_argument("--zone", type=str, default="all", help="Head zone data.")
    parser.add_argument("--frequency", type=str, default="all", help="Frequency band.")
    parser.add_argument("--target", type=str, default="age",
                        help="Target used to build the fold split.")
    parser.add_argument("--subject_column", type=str, default="subject",
                        help="Metadata column holding the subject id.")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of folds.")
    parser.add_argument("--base_seed", type=int, default=1234,
                        help="Seed shared with the main protocol.")
    parser.add_argument("--test_size", type=float, default=0.3,
                        help="Fraction of windows held out to score the probe.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Seed for the probe split and solver.")
    parser.add_argument("--embedding_size", type=int, default=128,
                        help="Embedding width of the legacy encoders.")
    parser.add_argument("--sampling_frequency", type=int, default=250,
                        help="EEG sampling frequency.")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for embedding extraction.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
