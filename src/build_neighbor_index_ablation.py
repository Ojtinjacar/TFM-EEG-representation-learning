"""Ablation neighbor-positive indices: disentangle developmental signal vs session identity.

Same machinery as build_neighbor_index.py (features + per-metric distances), but two alternative
grouping/candidate rules for the positive:

- ``crosssubj`` (cross-subject, same age): group by ``age`` (and ``block`` if --same_activity), and
  the positive must come from a DIFFERENT subject. Tests whether the gain is developmental (same
  maturational stage) rather than subject identity.
- ``diffage`` (same subject, different age): group by ``subject`` (and ``block``), positive must come
  from a DIFFERENT age. Breaks the age==session shortcut; measures the subject-identity contribution.

Outputs ``neighbor_index_<strategy>_<metric>.npy`` (aligned (N,k) global-position arrays, same format
as the main index) plus a diagnostics CSV with frac_same_subject / frac_same_age / coverage.

Build on the same canonical broadband dataset as the main index (data/processed/broadband_noz) so the
window order and features are identical and the results are directly comparable.

Usage (from the repo root, with the dasci-cimcyc env):
    python code/src/build_neighbor_index_ablation.py --strategy crosssubj \
        --data_path data/processed/broadband_noz/processed_windows.npy \
        --meta_path data/processed/broadband_noz/processed_metadata.csv \
        --channels_txt data/raw/channel_names.txt \
        --output_dir data/processed/neighbor_index_crosssubj \
        --metrics cosine wasserstein riemann
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_HERE)
# Inside this repo the file lives at <repo>/src, so eda_utils and the heredado
# repo are two levels up (the shared code/ directory); the original external
# layout (code/src) is also covered.
_OUTER_CODE = os.path.dirname(_CODE_DIR)
_CANDIDATE_SRCS = [
    _HERE, _CODE_DIR, _OUTER_CODE,
    os.path.join(_CODE_DIR, "TFM-EEG-representation-learning", "src"),
    os.path.join(_CODE_DIR, "RL-DaSCI-CIMCYC-C", "src"),
    os.path.join(_OUTER_CODE, "RL-DaSCI-CIMCYC-C", "src"),
]
for _p in _CANDIDATE_SRCS:
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

from eda_utils import compute_epoch_roi_psd  # noqa: E402
from neighbor_positives import VALID_METRICS, _session_distance_matrix, neighbor_index_array  # noqa: E402
# Reuse the exact feature/covariance helpers of the main builder for identical representations.
from build_neighbor_index import (  # noqa: E402
    compute_covariances,
    compute_feat_z,
    load_channel_names,
    roi_indices_from_channels,
)

_STRATEGIES = {
    # strategy: (group_cols_base, exclude_column)  -- candidate must DIFFER in exclude_column
    "crosssubj": ("age", "subject"),
    "diffage": ("subject", "age"),
}


def build_ablation_table(reprs, meta_q, metric, strategy, *, k=2, same_activity=True, exclude_lag=0):
    """Nearest-neighbor positives under an ablation grouping rule.

    Args:
        reprs: precomputed representations (see neighbor_positives._session_distance_matrix).
        meta_q: window metadata (index == global position in X); needs subject, age, epoch_index, block.
        metric: 'cosine' | 'wasserstein' | 'riemann'.
        strategy: 'crosssubj' (group by age, positive from another subject) or
            'diffage' (group by subject, positive from another age).
        k: max positives per anchor.
        same_activity: also require the positive to share the anchor's block.
        exclude_lag: exclude candidates with |epoch_index lag| <= this (0 = no temporal exclusion;
            irrelevant across subjects/sessions).

    Returns:
        DataFrame with one row per (anchor, neighbor).
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"metric no valida: {metric!r}")
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy no valida: {strategy!r} (usar {list(_STRATEGIES)})")
    group_base, exclude_col = _STRATEGIES[strategy]
    group_cols = [group_base] + (["block"] if same_activity else [])

    rows = []
    for _keys, grp in meta_q.groupby(group_cols):
        grp = grp.sort_values("epoch_index")
        gpos = grp.index.values.astype(int)
        ep = grp["epoch_index"].values
        excl = grp[exclude_col].values
        n = len(gpos)
        if n < 2:
            continue
        d = _session_distance_matrix(metric, gpos, reprs)
        for i in range(n):
            mask = np.ones(n, bool)
            mask[i] = False
            mask &= (excl != excl[i])  # different subject (crosssubj) / different age (diffage)
            if exclude_lag > 0:
                mask &= np.abs(ep - ep[i]) > exclude_lag
            cand = np.where(mask)[0]
            if len(cand) == 0:
                continue
            order = cand[np.argsort(d[i, cand])[:k]]
            for rank, j in enumerate(order, start=1):
                rows.append({
                    "subject": grp["subject"].iloc[i],
                    "age": int(grp["age"].iloc[i]),
                    "block": int(grp["block"].iloc[i]),
                    "metric": metric,
                    "strategy": strategy,
                    "anchor_gpos": int(gpos[i]),
                    "neigh_gpos": int(gpos[j]),
                    "dist": float(d[i, j]),
                    "rank": rank,
                })
    cols = ["subject", "age", "block", "metric", "strategy",
            "anchor_gpos", "neigh_gpos", "dist", "rank"]
    return pd.DataFrame(rows, columns=cols)


def _diagnostics(table, meta, n_total):
    if table.empty:
        return {"n_pairs": 0, "coverage": 0.0, "frac_same_subject": float("nan"),
                "frac_same_age": float("nan")}
    anc = table["anchor_gpos"].to_numpy()
    nei = table["neigh_gpos"].to_numpy()
    subj = meta["subject"].to_numpy()
    age = meta["age"].to_numpy()
    return {
        "n_pairs": int(len(table)),
        "coverage": float(np.unique(anc).size / n_total),
        "frac_same_subject": float((subj[anc] == subj[nei]).mean()),
        "frac_same_age": float((age[anc] == age[nei]).mean()),
    }


def main(args):
    X = np.load(args.data_path, mmap_mode="r")
    meta = pd.read_csv(args.meta_path).reset_index(drop=True)
    all_channels = load_channel_names(args.channels_txt)
    if len(all_channels) != X.shape[1]:
        all_channels = all_channels[:X.shape[1]]
    roi_indices = roi_indices_from_channels(all_channels)
    sfreq = X.shape[-1] / args.seconds
    n_total = len(X)
    metrics = [m for m in args.metrics if m in VALID_METRICS]
    print(f"[ablation:{args.strategy}] X={X.shape} sfreq={sfreq} subjects={meta['subject'].nunique()} "
          f"ages={sorted(meta['age'].unique())} metrics={metrics}")

    reprs = {}
    if "cosine" in metrics:
        print("[ablation] computing feat_z (cosine)...")
        reprs["feat_z"] = compute_feat_z(X, meta, roi_indices, sfreq)
    if "wasserstein" in metrics:
        print("[ablation] computing psd_norm (wasserstein)...")
        _, reprs["psd_norm"] = compute_epoch_roi_psd(X, roi_indices, sfreq=sfreq)
    if "riemann" in metrics:
        print("[ablation] computing covariances (riemann)...")
        cov, g2c = compute_covariances(X)
        reprs["cov"], reprs["gpos_to_cov"] = cov, g2c

    os.makedirs(args.output_dir, exist_ok=True)
    diag_rows = []
    for metric in metrics:
        print(f"[ablation:{args.strategy}] building table for metric={metric} (k={args.k})...")
        table = build_ablation_table(
            reprs, meta, metric, args.strategy,
            k=args.k, same_activity=args.same_activity, exclude_lag=args.exclude_lag,
        )
        idx = neighbor_index_array(table, n_total, args.k)
        # The strategy is encoded by the output DIRECTORY; the filename stays
        # neighbor_index_<metric>.npy so train_simclr.py finds it via --neighbor_index_dir.
        out_path = os.path.join(args.output_dir, f"neighbor_index_{metric}.npy")
        np.save(out_path, idx)
        d = _diagnostics(table, meta, n_total)
        d["metric"] = metric
        d["strategy"] = args.strategy
        diag_rows.append(d)
        print(f"  saved {out_path} shape={idx.shape} | coverage={d['coverage']:.3f} "
              f"same_subj={d['frac_same_subject']:.3f} same_age={d['frac_same_age']:.3f}")

    meta[["subject", "age", "epoch_index", "block"]].to_csv(
        os.path.join(args.output_dir, "neighbor_index_meta.csv"), index=False)
    pd.DataFrame(diag_rows).to_csv(
        os.path.join(args.output_dir, f"neighbor_index_{args.strategy}_diagnostics.csv"), index=False)
    print("[ablation] done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build ablation neighbor-positive indices.")
    p.add_argument("--strategy", required=True, choices=list(_STRATEGIES))
    p.add_argument("--data_path", required=True)
    p.add_argument("--meta_path", required=True)
    p.add_argument("--channels_txt", default="data/raw/channel_names.txt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--metrics", nargs="+", default=list(VALID_METRICS), choices=list(VALID_METRICS))
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--exclude_lag", type=int, default=0)
    p.add_argument("--same_activity", action="store_true", default=True)
    p.add_argument("--seconds", type=float, default=5.0)
    main(p.parse_args())
