"""Nearest-neighbour positive sampling for contrastive SSL on EEG.

Instead of building the positive with a synthetic perturbation (blind augmentation), this
strategy takes as positive the **closest real window** to an anchor within the same session
(and, by default, the same activity/block), under one of three distances:

- ``cosine``: on the 24-D feature vector z-scored within subject (spectral profile);
- ``wasserstein``: on the ROI-normalised PSD (spectral shape);
- ``riemann``: on the SPD spatial covariance matrix (cross-channel pattern).

This is neighbour-based positive sampling, in the line of NNCLR (Dwibedi et al., ICCV 2021,
arXiv:2104.14548). New for EEG and proposed by this work: here the neighbour is searched in the
input feature space (not in a learned embedding space), so its reliability rests on these fixed
distances. Restricting the neighbour to the same activity mitigates the risk of false positives
(positives crossing task boundaries).

This module does NOT modify the loss. ``NeighborPositiveDataset`` yields, per item, the anchor
and two views ``(anchor, view1, view2)`` compatible with the inherited pipeline
(``CIMCYCDataset`` in ``train_simclr.py``); with ``k > 1`` positives, wiring them into the loss
(expanding to pairs or using a multi-positive loss such as SupCon) is left to the caller.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

VALID_METRICS = ("cosine", "wasserstein", "riemann")
_EPS = 1e-30


def _session_distance_matrix(
    metric: str,
    gpos: np.ndarray,
    reprs: Mapping[str, object],
) -> np.ndarray:
    """Distance matrix (n x n) between the windows of a group, under ``metric``.

    Args:
        metric: 'cosine' | 'wasserstein' | 'riemann'.
        gpos: global positions (indices into meta_epochs / X) of the windows in the group.
        reprs: precomputed representations. Keys required per metric:
            'cosine' -> 'feat_z' (N, D); 'wasserstein' -> 'psd_norm' (N, n_roi, n_freq);
            'riemann' -> 'cov' (M, C, C) and 'gpos_to_cov' (dict global position -> row in cov).

    Returns:
        Symmetric (n, n) distance matrix with zero diagonal.

    Raises:
        ValueError: if ``metric`` is not valid or a representation is missing.
    """
    n = len(gpos)
    if metric == "cosine":
        feat_z = np.asarray(reprs["feat_z"])
        v = feat_z[gpos].astype(np.float64)
        vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + _EPS)
        d = 1.0 - vn @ vn.T
        np.fill_diagonal(d, 0.0)
        return d
    if metric == "wasserstein":
        from eda_utils import wasserstein_fourier_distance  # noqa: PLC0415

        psd = np.asarray(reprs["psd_norm"])[gpos]
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dij = float(wasserstein_fourier_distance(psd[i], psd[j]).mean())
                d[i, j] = d[j, i] = dij
        return d
    if metric == "riemann":
        from pyriemann.geometry.distance import pairwise_distance  # noqa: PLC0415

        cov = reprs["cov"]
        g2c = reprs["gpos_to_cov"]
        covs = np.stack([np.asarray(cov[g2c[int(g)]], dtype=np.float64) for g in gpos])
        d = pairwise_distance(covs, metric="riemann")
        np.fill_diagonal(d, 0.0)
        return d
    raise ValueError(f"metric no válida: {metric!r} (usar una de {VALID_METRICS})")


def build_neighbor_positive_table(
    reprs: Mapping[str, object],
    meta_q: pd.DataFrame,
    metric: str,
    *,
    k: int = 2,
    same_activity: bool = True,
    exclude_lag: int = 0,
) -> pd.DataFrame:
    """Build the per-window table of positive neighbours under a given distance.

    For each quality window (anchor) it finds its ``k`` closest neighbours within the same
    session ``(subject, age)`` and, if ``same_activity``, within the same ``block``, excluding
    the anchor itself and (optionally) neighbours with ``|lag| <= exclude_lag``.

    Args:
        reprs: precomputed representations (see :func:`_session_distance_matrix`).
        meta_q: metadata of quality windows; its index is the global position in X.
            Must contain the columns ``subject, age, epoch_index, block``.
        metric: 'cosine' | 'wasserstein' | 'riemann'.
        k: maximum number of positive neighbours per anchor.
        same_activity: if True, the neighbour must share ``block`` with the anchor.
        exclude_lag: drops neighbours with ``|anchor_epoch_index - neighbour_epoch_index| <=
            exclude_lag``.

    Returns:
        DataFrame with one row per (anchor, neighbour): columns ``subject, age, block, metric,
        anchor_gpos, anchor_epoch, neigh_gpos, neigh_epoch, lag, dist, rank`` (``rank`` 1 =
        closest). Anchors without a valid neighbour produce no rows.

    Raises:
        ValueError: if ``metric`` is not valid.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"invalid metric: {metric!r} (use one of {VALID_METRICS})")

    group_cols = ["subject", "age"] + (["block"] if same_activity else [])
    rows: list[dict] = []
    for keys, grp in meta_q.groupby(group_cols):
        grp = grp.sort_values("epoch_index")
        gpos = grp.index.values.astype(int)
        ep = grp["epoch_index"].values
        blk = grp["block"].values
        n = len(gpos)
        if n < 2:
            continue
        d = _session_distance_matrix(metric, gpos, reprs)
        for i in range(n):
            lags = np.abs(ep - ep[i])
            mask = np.ones(n, bool)
            mask[i] = False
            if exclude_lag > 0:
                mask &= lags > exclude_lag
            cand = np.where(mask)[0]
            if len(cand) == 0:
                continue
            order = cand[np.argsort(d[i, cand])[:k]]
            for rank, j in enumerate(order, start=1):
                rows.append(
                    {
                        "subject": grp["subject"].iloc[i],
                        "age": int(grp["age"].iloc[i]),
                        "block": int(blk[i]),
                        "metric": metric,
                        "anchor_gpos": int(gpos[i]),
                        "anchor_epoch": int(ep[i]),
                        "neigh_gpos": int(gpos[j]),
                        "neigh_epoch": int(ep[j]),
                        "lag": int(lags[j]),
                        "dist": float(d[i, j]),
                        "rank": rank,
                    }
                )
    cols = ["subject", "age", "block", "metric", "anchor_gpos", "anchor_epoch",
            "neigh_gpos", "neigh_epoch", "lag", "dist", "rank"]
    return pd.DataFrame(rows, columns=cols)


def neighbor_index_array(table: pd.DataFrame, n_total: int, k: int) -> np.ndarray:
    """Turn the neighbour table into an (n_total, k) array of global indices.

    Args:
        table: output of :func:`build_neighbor_positive_table`.
        n_total: total number of windows (length of X), used to size the array.
        k: number of neighbour columns per anchor.

    Returns:
        Int array (n_total, k) where each entry is the global position of the neighbour at that
        rank, or -1 if it does not exist. Rows of anchors without neighbours stay at -1.
    """
    idx = np.full((n_total, k), -1, dtype=np.int64)
    for _, r in table.iterrows():
        rk = int(r["rank"]) - 1
        if rk < k:
            idx[int(r["anchor_gpos"]), rk] = int(r["neigh_gpos"])
    return idx


class NeighborPositiveDataset:
    """Neighbour-positive dataset, compatible with the inherited SimCLR pipeline.

    Each item returns ``(anchor, view1, view2)`` where ``view2`` is a **real neighbouring
    window** (positive by distance) and ``view1`` is the anchor, either raw or augmented. It
    mirrors the conventions of ``CIMCYCDataset`` (``torch.FloatTensor`` tensors of shape
    ``(n_channels, n_samples)``), so the inherited ``train_loader``
    (``for anchor, aug1, aug2 in ...``) consumes it unchanged.

    Args:
        X: tensor/array ``(N, C, T)`` holding the EEG windows.
        neighbor_index: array ``(N, k)`` of global neighbour positions (-1 when absent), e.g.
            from :func:`neighbor_index_array`.
        augment: optional callable ``tensor(C,T) -> tensor(C,T)``. Used for ``view1`` when
            ``augment_anchor`` is True, and always for the fallback of anchors without a
            neighbour.
        augment_anchor: when False, ``view1`` is the untransformed anchor and the pair is made of
            two real windows. The augmenter is still used for the fallback, since otherwise the
            two views of an anchor without a neighbour would be identical and the loss would
            degenerate.
        fallback: what to do when an anchor has no valid neighbour: ``'duplicate'`` (view2 =
            anchor, or its augmentation if ``augment`` is not None) or ``'skip_none'``
            (view2 = None).
        seed: seed for the random choice of neighbour among those available.
    """

    def __init__(
        self,
        X,
        neighbor_index: np.ndarray,
        *,
        augment: Callable | None = None,
        augment_anchor: bool = True,
        fallback: str = "duplicate",
        seed: int = 42,
    ) -> None:
        import torch  # noqa: PLC0415

        self._torch = torch
        self.X = X if isinstance(X, torch.Tensor) else torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.neighbor_index = np.asarray(neighbor_index, dtype=np.int64)
        if self.neighbor_index.shape[0] != len(self.X):
            raise ValueError("neighbor_index desalineado con X")
        if fallback not in ("duplicate", "skip_none"):
            raise ValueError("fallback debe ser 'duplicate' o 'skip_none'")
        self.augment = augment
        self.augment_anchor = augment_anchor
        self.fallback = fallback
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def coverage(self) -> float:
        """Fraction of anchors with at least one valid neighbour."""
        return float((self.neighbor_index >= 0).any(axis=1).mean())

    def __getitem__(self, i: int):
        anchor = self.X[i]
        if self.augment is not None and self.augment_anchor:
            view1 = self.augment(anchor)
        else:
            view1 = anchor.clone()
        neighs = self.neighbor_index[i]
        neighs = neighs[neighs >= 0]
        if len(neighs) == 0:
            if self.fallback == "skip_none":
                return anchor, view1, None
            view2 = self.augment(anchor) if self.augment is not None else anchor.clone()
            return anchor, view1, view2
        j = int(neighs[self._rng.integers(len(neighs))])
        return anchor, view1, self.X[j]
