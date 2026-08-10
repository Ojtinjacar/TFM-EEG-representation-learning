"""Precompute neighbor-positive indices for contrastive SSL on EEG.

For a window dataset (``processed_windows.npy`` + ``processed_metadata.csv``) this script builds
the ``reprs`` required by :mod:`neighbor_positives` (spectral features, per-ROI PSD, spatial
covariance), finds the nearest real-window positives per anchor for each requested metric, and
saves an aligned ``neighbor_index_<metric>.npy`` array plus a diagnostics CSV.

The neighbor index is meant to be computed once on a canonical broadband, all-channels dataset and
reused for any (zone, band) training set, since windowing does not depend on zone/band and the
window order is identical across ``data/processed/*`` directories (verify before reusing).

Usage:
    python code/src/build_neighbor_index.py \
        --data_path data/processed/broadband_noz/processed_windows.npy \
        --meta_path data/processed/broadband_noz/processed_metadata.csv \
        --channels_txt data/raw/channel_names.txt \
        --output_dir data/processed/neighbor_index \
        --metrics cosine wasserstein riemann
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# neighbor_positives.py lives next to this file (code/src); eda_utils lives in code/;
# apsd_baseline (imported by eda_utils.compute_epoch_features for specparam) lives in the
# heredado repo's src/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_HERE)
# Inside this repo the file lives at <repo>/src, so eda_utils and the heredado
# repo are two levels up (the shared code/ directory); the original external
# layout (code/src) is also covered.
_OUTER_CODE = os.path.dirname(_CODE_DIR)
for _p in (_HERE, _CODE_DIR, _OUTER_CODE,
           os.path.join(_CODE_DIR, "RL-DaSCI-CIMCYC-C", "src"),
           os.path.join(_OUTER_CODE, "RL-DaSCI-CIMCYC-C", "src")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import eda_utils  # noqa: F401
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This index builder needs the shared eda_utils package (lives in the "
        "TFM's code/ directory, outside this repo). Run it from a checkout "
        "that has code/eda_utils available."
    ) from exc

from eda_utils import PRESET_ZONES, compute_epoch_features, compute_epoch_roi_psd  # noqa: E402
from neighbor_positives import (  # noqa: E402
    VALID_METRICS,
    build_neighbor_positive_table,
    neighbor_index_array,
)

_DESCRIPTOR_PREFIXES = ("bp_", "apsd_")
_EPS = 1e-12


def load_channel_names(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def roi_indices_from_channels(all_channels: list[str]) -> dict[str, list[int]]:
    """Map each ROI to the indices of its channels present in ``all_channels``."""
    return {
        roi: [i for i, ch in enumerate(all_channels) if ch in chans]
        for roi, chans in PRESET_ZONES.items()
    }


def compute_feat_z(X, meta: pd.DataFrame, roi_indices, sfreq: float) -> np.ndarray:
    """(N, D) intra-subject z-scored spectral descriptors (cosine metric input)."""
    feats = compute_epoch_features(X, meta, roi_indices, sfreq=sfreq, progress=False)
    if len(feats) != len(meta):
        raise ValueError(f"compute_epoch_features returned {len(feats)} rows, expected {len(meta)}")
    cols = [c for c in feats.columns if c.startswith(_DESCRIPTOR_PREFIXES)]
    matrix = feats[cols].to_numpy(dtype=np.float64)
    # Impute NaNs (failed specparam fits) with the column mean.
    col_mean = np.nanmean(matrix, axis=0)
    nan_r, nan_c = np.where(np.isnan(matrix))
    matrix[nan_r, nan_c] = np.take(col_mean, nan_c)
    # Intra-subject z-score.
    subjects = feats["subject"].to_numpy()
    out = np.zeros_like(matrix)
    for s in np.unique(subjects):
        m = subjects == s
        mu = matrix[m].mean(axis=0)
        sd = matrix[m].std(axis=0)
        sd[sd < _EPS] = 1.0
        out[m] = (matrix[m] - mu) / sd
    return out


def compute_covariances(X, reg: float = 1e-6):
    """Per-window SPD covariance (N, C, C) and the identity gpos->row mapping."""
    n, n_ch, _ = X.shape
    cov = np.empty((n, n_ch, n_ch), dtype=np.float64)
    eye = np.eye(n_ch)
    for i in range(n):
        xi = np.asarray(X[i], dtype=np.float64)
        c = xi @ xi.T / xi.shape[1]
        # Regularize to guarantee SPD (needed for the Riemannian distance).
        c += reg * np.trace(c) / n_ch * eye
        cov[i] = c
    gpos_to_cov = {i: i for i in range(n)}
    return cov, gpos_to_cov


def _diagnostics(table: pd.DataFrame, meta: pd.DataFrame, n_total: int) -> dict:
    if table.empty:
        return {"n_pairs": 0, "coverage": 0.0, "mean_lag": float("nan"),
                "frac_same_subject": float("nan"), "frac_same_block": float("nan")}
    anc = table["anchor_gpos"].to_numpy()
    nei = table["neigh_gpos"].to_numpy()
    same_subj = (meta["subject"].to_numpy()[anc] == meta["subject"].to_numpy()[nei]).mean()
    same_block = (meta["block"].to_numpy()[anc] == meta["block"].to_numpy()[nei]).mean()
    coverage = np.unique(anc).size / n_total
    return {
        "n_pairs": int(len(table)),
        "coverage": float(coverage),
        "mean_lag": float(table["lag"].mean()),
        "frac_same_subject": float(same_subj),
        "frac_same_block": float(same_block),
    }


def main(args) -> None:
    X = np.load(args.data_path, mmap_mode="r")
    meta = pd.read_csv(args.meta_path).reset_index(drop=True)  # index == position in X
    all_channels = load_channel_names(args.channels_txt)
    n_ch_data = X.shape[1]
    if len(all_channels) != n_ch_data:
        all_channels = all_channels[:n_ch_data]
    roi_indices = roi_indices_from_channels(all_channels)
    sfreq = X.shape[-1] / args.seconds
    n_total = len(X)
    print(f"[neighbor_index] X={X.shape} sfreq={sfreq} subjects={meta['subject'].nunique()}")

    metrics = [m for m in args.metrics if m in VALID_METRICS]

    reprs: dict[str, object] = {}
    if "cosine" in metrics:
        print("[neighbor_index] computing feat_z (cosine)...")
        reprs["feat_z"] = compute_feat_z(X, meta, roi_indices, sfreq)
    if "wasserstein" in metrics:
        print("[neighbor_index] computing psd_norm (wasserstein)...")
        _, reprs["psd_norm"] = compute_epoch_roi_psd(X, roi_indices, sfreq=sfreq)
    if "riemann" in metrics:
        print("[neighbor_index] computing covariances (riemann)...")
        cov, g2c = compute_covariances(X)
        reprs["cov"], reprs["gpos_to_cov"] = cov, g2c

    os.makedirs(args.output_dir, exist_ok=True)
    diag_rows = []
    for metric in metrics:
        print(f"[neighbor_index] building table for metric={metric} (k={args.k})...")
        table = build_neighbor_positive_table(
            reprs, meta, metric,
            k=args.k, same_activity=args.same_activity, exclude_lag=args.exclude_lag,
        )
        idx = neighbor_index_array(table, n_total, args.k)
        out_path = os.path.join(args.output_dir, f"neighbor_index_{metric}.npy")
        np.save(out_path, idx)
        d = _diagnostics(table, meta, n_total)
        d["metric"] = metric
        diag_rows.append(d)
        print(f"  saved {out_path} shape={idx.shape} | coverage={d['coverage']:.3f} "
              f"mean_lag={d['mean_lag']:.1f} same_subj={d['frac_same_subject']:.3f} "
              f"same_block={d['frac_same_block']:.3f}")

    # Persist metadata subset used for alignment checks and the diagnostics.
    meta[["subject", "age", "epoch_index", "block"]].to_csv(
        os.path.join(args.output_dir, "neighbor_index_meta.csv"), index=False)
    pd.DataFrame(diag_rows).to_csv(
        os.path.join(args.output_dir, "neighbor_index_diagnostics.csv"), index=False)
    print("[neighbor_index] done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute neighbor-positive indices for SSL.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--channels_txt", type=str, default="data/raw/channel_names.txt")
    parser.add_argument("--output_dir", type=str, default="data/processed/neighbor_index")
    parser.add_argument("--metrics", nargs="+", default=list(VALID_METRICS),
                        choices=list(VALID_METRICS))
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--exclude_lag", type=int, default=1,
                        help="Exclude neighbors with |epoch_index lag| <= this (EDA: spectral "
                             "similarity holds only within a few windows).")
    parser.add_argument("--same_activity", action="store_true", default=True,
                        help="Restrict neighbor to the same block (activity).")
    parser.add_argument("--seconds", type=float, default=5.0)
    main(parser.parse_args())
