"""K-fold driver for InterFusion architecture variants (C1a/C1b/C1c).

``run_downstream.py`` pre-trains InterFusion with the paper defaults only, and
``interfusion_checkpoint_name`` does not encode the recurrent/dense widths, so an
architecture ablation launched through it would silently collide with the
baseline checkpoints. This driver runs the very same protocol (identical fold
split, identical per-fold seeds, identical evaluation entry point) while keeping
each variant in its own model and results directory.

Example:
    python run_interfusion_variant.py --tag c1a --rnn-hidden 128 --dense-hidden 128
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import run_downstream as rd
from checkpoint_naming import sidecar_path


class PretrainingError(RuntimeError):
    """Raised when the pre-training subprocess fails for a fold."""


def build_folds(meta_path, target, n_folds, base_seed):
    """Rebuilds the fold split that ``run_downstream.py`` produces.

    Mirrors the k-fold configuration of ``run_downstream.main``: subjects with a
    non-null target, sorted, split by ``KFold(shuffle=True,
    random_state=base_seed)``. Single-target only, which is why the union step of
    the original (irrelevant with one target) is omitted.

    Args:
        meta_path (str): Path to the processed metadata CSV.
        target (str): Target column name (e.g. "age").
        n_folds (int): Number of folds.
        base_seed (int): Seed shared with the main protocol.

    Returns:
        list[tuple[int, list[str], list[str]]]: (fold index, train subjects,
            test subjects) per fold.

    Raises:
        ValueError: If the target column is absent from the metadata.
    """
    meta_df = pd.read_csv(meta_path)
    if target not in meta_df.columns:
        raise ValueError(f"Target '{target}' not found in {meta_path}")

    subjects = sorted(meta_df[~meta_df[target].isna()]["subject"].unique().tolist())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
    return [
        (idx, [subjects[i] for i in train_idx], [subjects[i] for i in test_idx])
        for idx, (train_idx, test_idx) in enumerate(kf.split(subjects))
    ]


def find_checkpoint(model_dir, zone, frequency, fold_id, z_dim, epochs):
    """Locates the variant checkpoint of a fold, if it exists.

    The compressed length W' in the filename depends on the stride stack, so it
    is globbed instead of recomputed: duplicating the model's arithmetic here is
    exactly how train and eval drift apart.

    Args:
        model_dir (str): Directory holding this variant's checkpoints.
        zone (str): Head zone of the processed data.
        frequency (str): Frequency band of the processed data.
        fold_id (str): Fold identifier (e.g. "fold0").
        z_dim (int): Inter-metric latent width M'.
        epochs (int): Stage-2 epochs.

    Returns:
        str | None: Path to the checkpoint, or None if absent.

    Raises:
        PretrainingError: If several checkpoints match (ambiguous stride config).
    """
    pattern = os.path.join(
        model_dir, f"InterFusion_{zone}_{frequency}_{fold_id}_m{z_dim}_w*_e{epochs}.pth"
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) > 1:
        raise PretrainingError(
            f"Ambiguous checkpoints for {fold_id} in {model_dir}: {matches}. "
            "Mixing stride configurations in one variant directory is not supported."
        )
    return matches[0] if matches else None


def sidecar_agrees(model_path, expected):
    """Checks that an existing checkpoint was trained with the requested widths.

    Args:
        model_path (str): Path to the .pth checkpoint.
        expected (dict): Sidecar fields that must match.

    Returns:
        bool: True if the sidecar exists and every expected field matches.
    """
    path = sidecar_path(model_path)
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        sidecar = json.load(fh)
    return all(sidecar.get(key) == value for key, value in expected.items())


def pretrain_fold(args, fold_id, test_subjects, seed, model_dir):
    """Pre-trains one fold of the variant, reusing a valid checkpoint if present.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        fold_id (str): Fold identifier (e.g. "fold0").
        test_subjects (list[str]): Held-out subjects, excluded from training.
        seed (int): Per-fold seed (base_seed + fold index).
        model_dir (str): Directory for this variant's checkpoints.

    Returns:
        str: Path to the checkpoint.

    Raises:
        PretrainingError: If training fails or writes no checkpoint.
    """
    expected = {
        "rnn_hidden": args.rnn_hidden,
        "dense_hidden": args.dense_hidden,
        "flow_levels": args.flow_levels,
        "free_bits": args.free_bits,
        "strides": list(args.strides),
        "z_dim": args.z_dim,
        "seed": seed,
    }
    existing = find_checkpoint(
        model_dir, args.zone, args.frequency, fold_id, args.z_dim, args.epochs
    )
    if existing and not args.no_skip and sidecar_agrees(existing, expected):
        print(f"  > [REUSE] {os.path.basename(existing)}: sidecar matches, skipping pre-training.",
              flush=True)
        return existing

    win_path, meta_path = rd.config_data_paths(args.zone, args.frequency)
    command = [
        "python", "src/train_interfusion.py",
        "--zone", args.zone,
        "--frequency", args.frequency,
        "--fold_id", fold_id,
        "--data-path", win_path,
        "--meta-path", meta_path,
        "--save-model-dir", model_dir,
        "--save-fig-dir", os.path.join("save", "figures", f"pretrain_{args.tag}"),
        "--z-dim", str(args.z_dim),
        "--rnn-hidden", str(args.rnn_hidden),
        "--dense-hidden", str(args.dense_hidden),
        "--flow-levels", str(args.flow_levels),
        "--free-bits", str(args.free_bits),
        "--strides", *[str(s) for s in args.strides],
        "--pretrain-epochs", str(args.pretrain_epochs),
        "--epochs", str(args.epochs),
        "--seed", str(seed),
        "--exclude_subjects", *[str(s) for s in test_subjects],
    ]
    print(f"  > Running pre-training: {' '.join(command)}", flush=True)
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise PretrainingError(f"train_interfusion.py failed for {fold_id} "
                               f"(exit {result.returncode})")

    produced = find_checkpoint(
        model_dir, args.zone, args.frequency, fold_id, args.z_dim, args.epochs
    )
    if produced is None:
        raise PretrainingError(f"No checkpoint written for {fold_id} in {model_dir}")
    return produced


def evaluate_fold(args, fold_idx, fold_id, test_subjects, model_path, seed, method_label):
    """Runs the downstream evaluations of one fold.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        fold_idx (int): Fold index.
        fold_id (str): Fold identifier.
        test_subjects (list[str]): Held-out subjects.
        model_path (str): Variant checkpoint to evaluate.
        seed (int): Per-fold seed.
        method_label (str): Method name recorded in the results (e.g.
            "InterFusion-c1a"), distinct from the "InterFusion" code path.

    Returns:
        list[dict]: One result row per evaluation mode that produced metrics.
    """
    rows = []
    for eval_mode in args.eval_modes:
        print(f"  > Evaluating: {method_label} | {args.target} | {eval_mode}", flush=True)
        nrmse, r2, rmse, subject_avgs, session_avgs = rd.run_downstream_experiment(
            method=method_label,
            eval_mode=eval_mode,
            target=args.target,
            zone=args.zone,
            frequency=args.frequency,
            model_path=model_path,
            seed=seed,
            test_subjects=test_subjects,
            fold_id=fold_id,
            downstream_method="InterFusion",
        )
        if not subject_avgs:
            print(f"    [ERROR] No metrics for {eval_mode}; fold row dropped.", flush=True)
            continue

        rows.append({
            "fold": fold_idx,
            "method": method_label,
            "eval_mode": eval_mode,
            "target": args.target,
            "nRMSE": np.nan if nrmse is None else nrmse,
            "R2": np.nan if r2 is None else r2,
            "RMSE": np.nan if rmse is None else rmse,
            # Same serialization as run_downstream.py: the aggregation scripts
            # parse these columns positionally, not as Python literals.
            "subject_avgs": ";".join(f"{yt},{yp}" for yt, yp in subject_avgs),
            "session_avgs": ";".join(f"{s},{yt},{yp}" for s, yt, yp in session_avgs),
        })
        print(f"    OK nRMSE={rows[-1]['nRMSE']:.4f}, R2={rows[-1]['R2']:.4f}, "
              f"RMSE={rows[-1]['RMSE']:.2f}", flush=True)
    return rows


def main(args):
    """Runs the variant over the requested folds, one CSV per fold.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
    """
    _, meta_path = rd.config_data_paths(args.zone, args.frequency)
    folds = build_folds(meta_path, args.target, args.n_folds, args.base_seed)

    if args.print_folds:
        for fold_idx, _, test_subjects in folds:
            print(f"fold{fold_idx}: {' '.join(test_subjects)}")
        return

    model_dir = args.model_dir or os.path.join("save", f"models_{args.tag}")
    save_dir = args.save_dir or os.path.join("save", args.tag)
    method_label = f"InterFusion-{args.tag}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    start, end = args.fold_range if args.fold_range else (0, args.n_folds)
    for fold_idx, _, test_subjects in folds[start:end]:
        fold_id = f"fold{fold_idx}"
        csv_path = os.path.join(
            save_dir, f"downstream_raw_results_kfold_folds{fold_idx}-{fold_idx}.csv"
        )
        if os.path.exists(csv_path) and not args.no_skip:
            print(f"[SKIP] {fold_id}: {csv_path} already exists.", flush=True)
            continue

        seed = args.base_seed + fold_idx
        print(f"\n{'='*80}\n{method_label} | {fold_id} | test={test_subjects}\n{'='*80}",
              flush=True)

        model_path = pretrain_fold(args, fold_id, test_subjects, seed, model_dir)
        rows = evaluate_fold(
            args, fold_idx, fold_id, test_subjects, model_path, seed, method_label
        )
        if not rows:
            print(f"[ERROR] {fold_id}: no results, CSV not written.", flush=True)
            continue

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[INFO] Results saved to: {csv_path}", flush=True)


def parse_args():
    """Parses the command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="K-fold driver for InterFusion architecture variants."
    )
    parser.add_argument("--tag", type=str, required=True,
                        help="Variant tag; names the method, model dir and save dir.")
    parser.add_argument("--rnn-hidden", type=int, default=500,
                        help="Backward-GRU hidden units (paper: 500).")
    parser.add_argument("--dense-hidden", type=int, default=500,
                        help="Feature/fusion dense units (paper: 500).")
    parser.add_argument("--flow-levels", type=int, default=20,
                        help="RealNVP levels on q(z1) (paper: 20).")
    parser.add_argument("--free-bits", type=float, default=0.0,
                        help="Per-element KL floor (nats) on z1 (A3 variant).")
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 1, 2, 1, 2, 2, 2],
                        help="Conv-stack strides; controls W'.")
    parser.add_argument("--z-dim", type=int, default=4,
                        help="Inter-metric latent width M'.")
    parser.add_argument("--pretrain-epochs", type=int, default=20,
                        help="Stage-1 epochs.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Stage-2 epochs.")
    parser.add_argument("--zone", type=str, default="all", help="Head zone data.")
    parser.add_argument("--frequency", type=str, default="all", help="Frequency band.")
    parser.add_argument("--target", type=str, default="age", help="Regression target.")
    parser.add_argument("--eval_modes", nargs="+", default=["linear_probe", "fine_tuning"],
                        help="Downstream evaluation modes.")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of folds.")
    parser.add_argument("--base_seed", type=int, default=1234,
                        help="Seed shared with the main protocol.")
    parser.add_argument("--fold_range", type=int, nargs=2, default=None,
                        metavar=("START", "END"), help="Half-open fold range.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Checkpoint directory (default: save/models_<tag>).")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Results directory (default: save/<tag>).")
    parser.add_argument("--no_skip", action="store_true",
                        help="Recompute even if checkpoints or CSVs exist.")
    parser.add_argument("--print_folds", action="store_true",
                        help="Print the fold split and exit (split verification).")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
