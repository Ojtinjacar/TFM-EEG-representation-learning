"""Build the ExpCLR expert-feature matrix aligned to the window dataset.

ExpCLR (Nonnenmacher et al., ICML 2022) needs an expert descriptor ``f_i`` for *every* sample in
the batch: the loss is computed over the full NxN pairwise similarity matrix, so a missing
descriptor is not a missing label but a missing row of the geometry. This script materialises that
descriptor as a dense array whose row ``i`` corresponds to window ``i`` of
``processed_windows.npy``, the same positional convention used by
:mod:`build_neighbor_index` (``index == position in X``).

Source of truth is ``features_106_por_ventana.parquet`` (notebook 05), which holds the 106 curated
expert features for all 2609 windows. The older ``spectral_features_full.csv`` covers only the 2566
windows that survive ``window_quality >= 2`` and is accepted as a fallback, at the cost of leaving
43 windows without a descriptor.

Alignment is verified, never assumed: the (subject, age, epoch_index) key must be unique on both
sides and the join must cover the metadata exactly once.

Usage:
    python code/src/build_expert_features.py \
        --features_path data/eda_outputs/features_por_edad/features_106_por_ventana.parquet \
        --meta_path code/TFM-EEG-representation-learning/data/processed/all_all/processed_metadata.csv \
        --output_dir code/TFM-EEG-representation-learning/data/processed/expert_features \
        --descriptor P_full
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

# --- Descriptor definitions -------------------------------------------------------------------
# The 26 base features that, x 4 ROIs, give 104, plus the 2 topographic contrasts = 106.
# Mirrors BASE_A/BASE_B/BASE_C in code/notebooks/_build_05_features_por_edad.py:238-243 and
# code/experimentos/expclr_expert_features_definicion.md section 3.

ROIS = ["frontal", "central", "parietal", "occipital"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

BASE_A = (
    ["paf_freq", "paf_power", "sp_alpha_cf", "sp_alpha_pw"]
    + [f"rel_{b}" for b in BANDS]
    + [f"osc_{b}" for b in BANDS]
    + ["apsd_slope", "apsd_offset", "total_power"]
)  # 17 -> 68 columns
BASE_B = [
    "theta_alpha",
    "delta_theta",
    "delta_alpha",
    "low_high",
    "spec_entropy",
    "sef95",
    "medfreq",
]  # 7 -> 28 columns
BASE_C = ["alpha_burst_count", "alpha_burst_occ"]  # 2 -> 8 columns
TOPO = ["paf_central_minus_occipital", "paf_central_minus_parietal"]  # 2

# Ablation axes (expclr_expert_features_definicion.md section 6).
# The prose of that section reads "PAF, relative, oscillatory and aperiodic", which would be 14
# base features, but every recorded count says 36. The ambiguity was settled empirically on
# 2026-07-24 by reproducing the Ridge of section 5.2 (subject-aggregated features, 152 sessions,
# LeaveOneGroupOut by subject): only PAF + relative + aperiodic reproduces the registered
# R2 = 0.894 / RMSE = 3.73 (obtained 0.895 / 3.71); adding the oscillatory block gives 0.898 / 3.65
# and the other 36-column candidates land between 0.882 and 0.908. The oscillatory block is
# therefore excluded, and "oscillatory" in the prose is the error.
BASE_MADURATIVO = (
    ["paf_freq", "paf_power"]
    + [f"rel_{b}" for b in BANDS]
    + ["apsd_slope", "apsd_offset"]
)  # 9 -> 36 columns
BASE_APER = ["apsd_slope", "apsd_offset"]  # 2 -> 8 columns


def _expand(bases: list[str], *, topo: bool) -> list[str]:
    """Expand base feature names over the four ROIs, preserving notebook 05 ordering.

    Args:
        bases: Base feature names (without ROI suffix).
        topo: If True, append the two topographic contrast columns.

    Returns:
        Ordered list of concrete column names.
    """
    cols = [f"{base}_{roi}" for base in bases for roi in ROIS]
    return cols + TOPO if topo else cols


def _load_v2_descriptor(name: str) -> list[str] | None:
    """Loads a v2 catalogue descriptor from its manifest, if present.

    The v2 descriptors are not hardcoded here: they are built by
    ``code/eda_utils/descriptors.py`` with a declared selection criterion and persisted with a
    manifest. Reading the manifest keeps a single source of truth for which columns each one has.

    Args:
        name: Descriptor name, matching ``<name>_columns.json`` under the v2 output directory.

    Returns:
        The column list, or None when the manifest is not on disk.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    manifest = os.path.join(root, "data", "eda_outputs", "features_v2", "descriptores",
                            f"{name}_columns.json")
    if not os.path.exists(manifest):
        return None
    with open(manifest) as fh:
        return json.load(fh)["columns"]


DESCRIPTORS: dict[str, list[str]] = {
    "P_full": _expand(BASE_A + BASE_B + BASE_C, topo=True),
    "P_madurativo": _expand(BASE_MADURATIVO, topo=False),
    "P_aper": _expand(BASE_APER, topo=False),
}

# Descriptores del catálogo v2, cargados de su manifiesto cuando existe. `P_diverso` reparte 5
# columnas por cada una de las 8 familias, elegidas SIN mirar la edad: seleccionar por correlación
# con el target colapsa la geometría (dimensionalidad efectiva 6 frente a 20), y ExpCLR necesita
# ejes complementarios para que ||f_i - f_j|| no mida una sola dirección.
for _v2_name in ("P_diverso", "P_fiable", "P_todo", "P_edad"):
    _v2_cols = _load_v2_descriptor(_v2_name)
    if _v2_cols:
        DESCRIPTORS[_v2_name] = _v2_cols

JOIN_KEY = ["subject", "age", "epoch_index"]
QUALITY_COLS = [f"apsd_r2_{roi}" for roi in ROIS]


class DescriptorAlignmentError(RuntimeError):
    """Raised when the expert features cannot be aligned 1:1 with the window metadata."""


def load_features(path: str) -> pd.DataFrame:
    """Load the expert-feature table from parquet or CSV.

    Args:
        path: Path to a .parquet or .csv feature table.

    Returns:
        The feature table as a DataFrame.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature table format: {path!r} (expected .parquet or .csv)")


def align_to_metadata(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    columns: list[str],
    *,
    quality: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Align the expert features to the window order of the metadata.

    Args:
        features: Expert-feature table, one row per computed window.
        meta: ``processed_metadata.csv`` contents, one row per window in ``processed_windows.npy``.
        columns: Descriptor columns to extract, in order.
        quality: If True, also return the mean specparam R^2 per window.

    Returns:
        Tuple ``(matrix, mask, r2)`` where ``matrix`` has shape (len(meta), len(columns)) with NaN
        for windows without a descriptor, ``mask`` is the boolean coverage vector, and ``r2`` is
        the per-window mean specparam R^2 (or None when unavailable).

    Raises:
        DescriptorAlignmentError: If the join key is missing, non-unique, or matches more than
            once per window.
        KeyError: If any requested descriptor column is absent from the feature table.
    """
    missing_key = [k for k in JOIN_KEY if k not in features.columns or k not in meta.columns]
    if missing_key:
        raise DescriptorAlignmentError(f"Join key columns missing on one side: {missing_key}")

    missing_cols = [c for c in columns if c not in features.columns]
    if missing_cols:
        raise KeyError(f"{len(missing_cols)} descriptor columns absent from features: {missing_cols[:5]}...")

    if features.duplicated(JOIN_KEY).any():
        raise DescriptorAlignmentError("(subject, age, epoch_index) is not unique in the features")
    if meta.duplicated(JOIN_KEY).any():
        raise DescriptorAlignmentError("(subject, age, epoch_index) is not unique in the metadata")

    take = JOIN_KEY + columns + ([c for c in QUALITY_COLS if c in features.columns] if quality else [])
    merged = meta[JOIN_KEY].merge(features[take], on=JOIN_KEY, how="left", validate="one_to_one")
    if len(merged) != len(meta):
        raise DescriptorAlignmentError(
            f"Join changed the row count: {len(meta)} windows -> {len(merged)} rows"
        )

    matrix = merged[columns].to_numpy(dtype=np.float64)
    mask = ~np.isnan(matrix).all(axis=1)

    r2 = None
    present_quality = [c for c in QUALITY_COLS if c in merged.columns]
    if quality and present_quality:
        r2 = merged[present_quality].to_numpy(dtype=np.float64).mean(axis=1)

    return matrix, mask, r2


def main(args: argparse.Namespace) -> None:
    """Build and persist the aligned expert-feature matrix."""
    columns = DESCRIPTORS[args.descriptor]
    features = load_features(args.features_path)
    meta = pd.read_csv(args.meta_path)

    print(f"Features table: {features.shape} from {args.features_path}")
    print(f"Window metadata: {meta.shape} from {args.meta_path}")
    print(f"Descriptor {args.descriptor}: {len(columns)} columns")

    matrix, mask, r2 = align_to_metadata(features, meta, columns)

    coverage = float(mask.mean())
    print(f"Coverage: {int(mask.sum())}/{len(mask)} windows ({coverage:.2%})")
    if coverage < 1.0:
        print(
            f"  > {int((~mask).sum())} windows have no descriptor and will be dropped from "
            "ExpCLR pre-training (the loss needs f_i for every sample in the batch)."
        )
    nan_rate = float(np.isnan(matrix[mask]).mean()) if mask.any() else float("nan")
    print(f"Per-cell NaN rate within covered windows: {nan_rate:.2%} (imputed at training time)")

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.join(args.output_dir, f"expert_features_{args.descriptor}")
    np.save(f"{stem}.npy", matrix.astype(np.float32))
    np.save(f"{stem}_mask.npy", mask)
    with open(f"{stem}_columns.json", "w") as fh:
        json.dump({"descriptor": args.descriptor, "columns": columns}, fh, indent=2)
    if r2 is not None:
        np.save(f"{stem}_r2.npy", r2.astype(np.float32))
        print(f"Saved specparam R^2 per window (mean over ROIs) to {stem}_r2.npy")

    print(f"Saved {matrix.shape} descriptor to {stem}.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the ExpCLR expert-feature matrix aligned to processed_windows.npy."
    )
    parser.add_argument(
        "--features_path",
        default="data/eda_outputs/features_por_edad/features_106_por_ventana.parquet",
        help="Expert-feature table (.parquet from notebook 05, or .csv fallback).",
    )
    parser.add_argument(
        "--meta_path",
        required=True,
        help="processed_metadata.csv defining the window order of the training set.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/expert_features",
        help="Directory where the aligned .npy artefacts are written.",
    )
    parser.add_argument(
        "--descriptor",
        default="P_full",
        choices=sorted(DESCRIPTORS.keys()),
        help="Which descriptor variant to build (P_full is the main one).",
    )
    main(parser.parse_args())
