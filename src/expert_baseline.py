"""Expert-descriptor baseline: the 78 spectral measures regressed against a target.

The descriptor built by ``build_expert_features.py`` guides the contrastive loss of ExpCLR,
and until now the only supervised reading it received was baseline B1 of
``run_expclr_folds.py``, pinned to the 32 measures of ``P_madurativo``. This asks the plain
question the whole comparison rests on: how far does the descriptor get on its own, with a
ridge and no encoder at all. A representation that cannot beat it has not earned its
complexity.

Nothing is recomputed here. The descriptor is read from the matrix ``build_expert_features``
already materialised, which is the one the encoder was trained against, so the two are
answering with the same numbers. Everything from the feature matrix onwards is
``tabular_baseline``, shared with ``apsd_baseline.py``.

A note on the widest descriptor. ``P_full`` is the sparse one: ``sp_alpha_cf`` and
``sp_alpha_pw`` are missing wherever the fit finds no alpha peak, so it needs more imputation
than the narrow descriptors. The fraction of filled cells is reported per fold rather than
left implicit, because a descriptor that is mostly medians is not measuring what it claims.

Usage (from the repository root):
    python src/expert_baseline.py --descriptor P_full --targets age cit_36mo \\
        --cv_strategy kfold --n_folds 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_expert_features import DESCRIPTORS, DESCRIPTOR_DIR, descriptor_path, zone_dir
from folds import BASE_SEED, N_FOLDS
from tabular_baseline import BaselineError, evaluate_targets, parse_fold_ranges

METHOD_NAME = "Expert_Ridge"
PREFIX = "expert"


def columns_path(descriptor, output_dir=DESCRIPTOR_DIR):
    """Returns the path of the provenance sidecar of a materialised descriptor.

    Args:
        descriptor (str): Descriptor name.
        output_dir (str): Directory the descriptors were written to.

    Returns:
        str: Path of the ``_columns.json`` sidecar.
    """
    return os.path.join(output_dir, f"{descriptor}_columns.json")


def load_descriptor(descriptor, zone="all", base_dir=DESCRIPTOR_DIR):
    """Loads a materialised descriptor together with the names of its columns.

    Args:
        descriptor (str): One of the descriptors of ``build_expert_features``.
        zone (str): ``all`` or one region of the montage.
        base_dir (str): Directory the descriptors were written to.

    Returns:
        pd.DataFrame: One row per window, one named column per measure.

    Raises:
        BaselineError: If the matrix or its sidecar is missing, or if the two disagree on
            how many columns the descriptor has.
    """
    directory = zone_dir(zone, base_dir)
    matrix_path = descriptor_path(descriptor, directory)
    sidecar_path = columns_path(descriptor, directory)

    if not os.path.exists(matrix_path):
        raise BaselineError(
            f"{matrix_path} does not exist. Build it first:\n"
            f"    python src/build_expert_features.py --descriptor {descriptor} "
            f"--zone {zone} --raw_path <windows with physical amplitude> "
            f"--meta_path <their metadata>"
        )
    if not os.path.exists(sidecar_path):
        raise BaselineError(
            f"{matrix_path} has no {os.path.basename(sidecar_path)} beside it, so its "
            "columns cannot be named. Rebuild the descriptor."
        )

    matrix = np.load(matrix_path)
    with open(sidecar_path) as fh:
        sidecar = json.load(fh)
    columns = sidecar["columns"]

    if matrix.shape[1] != len(columns):
        raise BaselineError(
            f"{matrix_path} has {matrix.shape[1]} columns and its sidecar names "
            f"{len(columns)}: the two were written by different runs."
        )
    print(f"[INFO] {descriptor}: {matrix.shape[0]} windows x {matrix.shape[1]} measures "
          f"(zone {sidecar.get('zone', zone)!r}, "
          f"aperiodic mode {sidecar.get('apsd_aperiodic_mode', 'unknown')!r})", flush=True)
    return pd.DataFrame(matrix, columns=columns)


def main(args):
    """Evaluates the expert descriptor against every requested target.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Raises:
        BaselineError: If the descriptor and the metadata describe different datasets, or if
            no target could be evaluated.
    """
    print(f"\n[1] Loading the descriptor and the metadata...", flush=True)
    features = load_descriptor(args.descriptor, args.zone, args.descriptor_dir)
    meta = pd.read_csv(args.meta_path)

    if len(features) != len(meta):
        raise BaselineError(
            f"The descriptor has {len(features)} rows and {args.meta_path} has {len(meta)}. "
            "The descriptor is indexed by position, so the two have to come from the same "
            "windows in the same order."
        )
    print(f"  {len(meta)} windows, {meta['subject'].nunique()} subjects", flush=True)

    print(f"\n[2] Evaluating {args.descriptor} ({len(features.columns)} measures)...",
          flush=True)
    _, agg, _ = evaluate_targets(
        features=features,
        meta=meta,
        targets=args.targets,
        prefix=f"{PREFIX}_{args.descriptor}",
        save_dir=args.save_dir,
        method_name=f"{METHOD_NAME}_{args.descriptor}",
        cv_strategy=args.cv_strategy,
        n_folds=args.n_folds,
        base_seed=args.base_seed,
        fold_range=tuple(args.fold_range) if args.fold_range else None,
        fold_ranges_dict=parse_fold_ranges(args.fold_ranges) or None,
        aggregation=args.aggregation,
    )

    print("\n[3] Summary", flush=True)
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expert spectral descriptor + ridge regression, evaluated by subject folds."
    )
    parser.add_argument("--descriptor", type=str, default="P_full",
                        choices=sorted(DESCRIPTORS),
                        help="Descriptor to evaluate. P_full is the 78 measures.")
    parser.add_argument("--descriptor_dir", type=str, default=DESCRIPTOR_DIR,
                        help="Directory holding the materialised descriptors.")
    parser.add_argument("--zone", type=str, default="all",
                        help="Zone the descriptor was built for.")
    parser.add_argument("--meta_path", type=str,
                        default="data/processed/all_all/processed_metadata.csv",
                        help="Per-window metadata of the same windows the descriptor "
                             "was built from, in the same order.")
    parser.add_argument("--targets", nargs="+", default=["age", "cit_36mo"],
                        help="Targets to evaluate.")
    parser.add_argument("--cv_strategy", type=str, default="kfold",
                        choices=["kfold", "leave_one_out"],
                        help="Cross-validation strategy.")
    parser.add_argument("--n_folds", type=int, default=N_FOLDS,
                        help="Number of folds of the k-fold.")
    parser.add_argument("--base_seed", type=int, default=BASE_SEED,
                        help="Seed of the fold shuffle; the partition of the whole project.")
    parser.add_argument("--fold_range", nargs=2, type=int, default=None,
                        metavar=("START", "END"),
                        help="Half-open range of folds to run, for parallel shards.")
    parser.add_argument("--fold_ranges", nargs="+", type=str, default=None,
                        metavar="TARGET:START:END",
                        help="Half-open range of folds per target, leave-one-out only.")
    parser.add_argument("--save_dir", type=str, default="save/downstream_results",
                        help="Directory the result tables are written to.")
    parser.add_argument("--aggregation", type=str, default="mean",
                        choices=["mean", "median"],
                        help="How the windows of a row are pooled.")
    main(parser.parse_args())
