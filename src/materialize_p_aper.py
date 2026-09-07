"""Materialises the P_aper descriptor by selecting columns from P_full.

P_aper is the aperiodic slope and offset of every region of interest, and every one of
those columns is already present in P_full, computed with the same specparam settings.
Rebuilding it from the raw recording would re-run the fit over every window to arrive at
numbers that are already on disk, so this selects them instead and records the same
provenance the builder writes.

Run from the repository root::

    python src/materialize_p_aper.py
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from build_expert_features import DESCRIPTORS, DESCRIPTOR_DIR, descriptor_path


class DescriptorSelectionError(RuntimeError):
    """Raised when the source descriptor cannot supply the requested columns."""


def _columns_path(descriptor: str, output_dir: str) -> str:
    """Returns the path of the sidecar describing a materialised descriptor.

    Args:
        descriptor (str): Descriptor name.
        output_dir (str): Directory the descriptors live in.

    Returns:
        str: Path of the ``_columns.json`` sidecar.
    """
    return os.path.join(output_dir, f"{descriptor}_columns.json")


def select_descriptor(source: str, target: str, output_dir: str = DESCRIPTOR_DIR) -> np.ndarray:
    """Writes a descriptor whose columns are a subset of another descriptor's.

    Args:
        source (str): Name of the descriptor already on disk.
        target (str): Name of the descriptor to write.
        output_dir (str): Directory holding the descriptors.

    Returns:
        np.ndarray: The materialised matrix, of shape (n_windows, n_target_columns).

    Raises:
        DescriptorSelectionError: If the target is not defined, if the source sidecar does
            not list its columns, or if any target column is missing from the source.
    """
    if target not in DESCRIPTORS:
        raise DescriptorSelectionError(
            f"{target} is not defined in build_expert_features.DESCRIPTORS, so the name "
            "would reach a checkpoint without meaning anything."
        )

    matrix = np.load(descriptor_path(source, output_dir))
    with open(_columns_path(source, output_dir)) as fh:
        meta = json.load(fh)
    source_columns = meta.get("columns")
    if not source_columns:
        raise DescriptorSelectionError(f"{_columns_path(source, output_dir)} lists no columns")

    wanted = DESCRIPTORS[target]
    missing = [c for c in wanted if c not in source_columns]
    if missing:
        raise DescriptorSelectionError(
            f"{source} does not carry {missing}, so {target} cannot be selected from it."
        )

    selected = matrix[:, [source_columns.index(c) for c in wanted]]
    np.save(descriptor_path(target, output_dir), selected)

    sidecar = dict(meta)
    sidecar["descriptor"] = target
    sidecar["columns"] = wanted
    sidecar["n_complete"] = int((~np.isnan(selected).any(axis=1)).sum())
    sidecar["selected_from"] = source
    with open(_columns_path(target, output_dir), "w") as fh:
        json.dump(sidecar, fh, indent=2, sort_keys=True)

    return selected


def main(args):
    """Materialises the descriptor and reports what was written.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    selected = select_descriptor(args.source, args.target, args.output_dir)
    nan_rate = float(np.isnan(selected).mean())
    print(f"Wrote {descriptor_path(args.target, args.output_dir)}: shape {selected.shape}, "
          f"NaN {nan_rate:.3%}, selected from {args.source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=str, default="P_full",
                        help="Descriptor already on disk to select the columns from.")
    parser.add_argument("--target", type=str, default="P_aper", choices=sorted(DESCRIPTORS),
                        help="Descriptor to write.")
    parser.add_argument("--output_dir", type=str, default=DESCRIPTOR_DIR,
                        help="Directory holding the descriptors.")
    main(parser.parse_args())
