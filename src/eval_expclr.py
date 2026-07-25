"""Evaluation path for E3 (ExpCLR) and its baselines.

Deliberately separate from ``downstream.py``: the shared path has four defects that would make the
numbers uninterpretable, and fixing it in place would invalidate the results already produced for
E0-E2. The defects, and what this module does instead:

1. ``downstream.py`` calls ``model.train()`` during the linear probe, so the encoder's BatchNorm
   uses batch statistics and updates its running stats, and dropout stays active. Here the encoder
   is put in ``eval()`` mode and embeddings are extracted once under ``no_grad``, which makes the
   probe deterministic.
2. Its ``Head`` is two stacked linear layers without a non-linearity, trained 100 epochs with AdamW
   and no validation, so the result depends on lr, weight decay and epochs. Here the probe is a
   closed-form Ridge whose only hyperparameter is chosen by an inner subject-grouped CV, following
   the standard linear-evaluation protocol (Chen et al. 2020, SimCLR §4.2; Kornblith et al. 2019).
3. It aggregates predictions by subject and averages the target across visits, so a child seen at
   6 and 36 months contributes a target of 21 months that never happened. Here the unit is the
   **session** ``(subject, age, block)``, which is what the target is actually defined on: 274 of
   them across 152 visits and 45 subjects, since a visit can hold up to three blocks.
4. Metrics are computed per fold and averaged. Here predictions are pooled out-of-fold and the
   metric is computed once over all sessions, with confidence intervals from a cluster bootstrap
   over subjects, since sessions of the same child are not independent.

The representation used is ``get_embedding()``. Note this is *not* the SimCLR argument about ``h``
beating ``z`` for linear evaluation: ExpCLR is precisely the method that drops the projection head,
and the paper optimises ``E(x)`` -- the same representation it evaluates (Secs. 3.6, 4.2). So the
training loss must be applied here too, which is why ``train_expclr.py`` defaults to
``--loss_on embedding``. An earlier version of this pipeline trained on the projection and
evaluated the embedding, shaping a geometry behind a non-invertible ReLU MLP that the probe never
saw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

#: Ridge penalties explored by the inner CV, matching the log-spaced grid used in the
#: linear-evaluation literature.
ALPHA_GRID = np.logspace(-3, 4, 30)

#: Columns that jointly identify one recording session. The target is defined at this level.
SESSION_KEYS = ("subject", "age", "block")


@torch.no_grad()
def extract_embeddings(model, X: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    """Extracts frozen embeddings from a pre-trained encoder.

    Runs the encoder in ``eval()`` mode inside ``no_grad``, so BatchNorm uses its running
    statistics and dropout is disabled. Both are required for the probe to be deterministic and
    for the encoder to be genuinely frozen, which ``downstream.py`` does not guarantee.

    Args:
        model: Pre-trained encoder exposing ``get_embedding``.
        X: Window tensor of shape (n_windows, n_channels, n_samples).
        device: Torch device.
        batch_size: Windows per forward pass.

    Returns:
        Embedding matrix of shape (n_windows, embedding_dim).
    """
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        # np.array forces a writable copy: slices of a read-only memmap otherwise make torch warn.
        chunk = np.array(X[start:start + batch_size], dtype=np.float32)
        batch = torch.as_tensor(chunk, device=device)
        out.append(model.get_embedding(batch).cpu().numpy())
    return np.concatenate(out, axis=0)


def fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    n_splits: int = 5,
) -> tuple[StandardScaler, Ridge]:
    """Fits the linear probe, selecting the penalty with an inner subject-grouped CV.

    Selecting the hyperparameter on the same split used to estimate performance biases the estimate
    (Cawley and Talbot, 2010), so the penalty is chosen inside the training subjects only, grouped
    so that no subject spans the inner split either.

    Args:
        X_train: Representation of the training windows.
        y_train: Target per training window.
        groups_train: Subject identifier per training window.
        n_splits: Folds of the inner CV.

    Returns:
        The fitted scaler and Ridge model.
    """
    scaler = StandardScaler().fit(X_train)
    Z = scaler.transform(X_train)

    n_groups = len(np.unique(groups_train))
    splits = min(n_splits, n_groups)
    if splits < 2:
        best_alpha = 1.0
    else:
        cv = GroupKFold(n_splits=splits)
        scores = np.zeros(len(ALPHA_GRID))
        for inner_train, inner_val in cv.split(Z, y_train, groups_train):
            for i, alpha in enumerate(ALPHA_GRID):
                model = Ridge(alpha=alpha).fit(Z[inner_train], y_train[inner_train])
                pred = model.predict(Z[inner_val])
                scores[i] += np.mean(np.abs(pred - y_train[inner_val]))
        best_alpha = float(ALPHA_GRID[int(np.argmin(scores))])

    return scaler, Ridge(alpha=best_alpha).fit(Z, y_train)


def aggregate_to_sessions(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Aggregates window-level predictions to one row per session.

    The median is used rather than the mean because artefacted windows produce extreme predictions
    and the median is robust to them. Aggregating to the subject instead, as the shared pipeline
    does, would average visits months apart and destroy the longitudinal signal.

    Args:
        meta: Window metadata, aligned row by row with the predictions.
        y_true: True target per window.
        y_pred: Predicted target per window.

    Returns:
        Frame with the session keys, ``y_true`` and ``y_pred``, one row per session.
    """
    keys = [k for k in SESSION_KEYS if k in meta.columns]
    df = meta[keys].copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    return df.groupby(keys, as_index=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "median"))


def session_metrics(sessions: pd.DataFrame) -> dict[str, float]:
    """Computes the pooled metrics over sessions.

    MAE in months is the primary metric: with a target supported on only four visits, R2 is
    dominated by the 36-month level and can look high for a model that merely tells infants from
    preschoolers. Spearman's rho is reported as an ordinal companion.

    Args:
        sessions: Output of :func:`aggregate_to_sessions`.

    Returns:
        Dict with ``mae``, ``r2``, ``rho`` and ``n_sessions``.
    """
    y, p = sessions.y_true.values, sessions.y_pred.values
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "rho": float(spearmanr(y, p).statistic) if len(np.unique(p)) > 1 else np.nan,
        "n_sessions": int(len(sessions)),
    }


def bootstrap_ci(
    sessions: pd.DataFrame,
    metric: str = "mae",
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimates a confidence interval by resampling subjects, not sessions.

    Sessions of the same child are not independent, so resampling them directly would understate
    the uncertainty. Resampling whole subjects and carrying all their sessions respects the
    clustering. The spread across LOSO folds is not a valid interval either, which is why this is
    the reported uncertainty.

    Args:
        sessions: Output of :func:`aggregate_to_sessions`.
        metric: Metric key from :func:`session_metrics`.
        n_boot: Bootstrap replicates.
        seed: Seed for the resampling.

    Returns:
        The 2.5 and 97.5 percentiles of the metric.
    """
    rng = np.random.default_rng(seed)
    subjects = sessions.subject.unique()
    by_subject = {s: g for s, g in sessions.groupby("subject")}
    values = []
    for _ in range(n_boot):
        picked = rng.choice(subjects, size=len(subjects), replace=True)
        rep = pd.concat([by_subject[s] for s in picked], ignore_index=True)
        val = session_metrics(rep)[metric]
        if np.isfinite(val):
            values.append(val)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (np.nan, np.nan)


def paired_bootstrap_difference(
    sessions_a: pd.DataFrame,
    sessions_b: pd.DataFrame,
    metric: str = "mae",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Compares two methods with a paired cluster bootstrap over subjects.

    The same resampled subjects are used for both methods on every replicate, which preserves the
    pairing while respecting the clustering. Reporting the interval of the difference says more
    than whether one method 'wins'.

    Args:
        sessions_a: Sessions frame of the first method.
        sessions_b: Sessions frame of the second method.
        metric: Metric key to compare.
        n_boot: Bootstrap replicates.
        seed: Seed for the resampling.

    Returns:
        Dict with the observed difference and its 95 % interval.
    """
    rng = np.random.default_rng(seed)
    keys = [k for k in SESSION_KEYS if k in sessions_a.columns]
    merged = sessions_a.merge(sessions_b, on=keys, suffixes=("_a", "_b"))
    subjects = merged.subject.unique()
    by_subject = {s: g for s, g in merged.groupby("subject")}

    def _diff(frame):
        a = session_metrics(frame.rename(columns={"y_true_a": "y_true", "y_pred_a": "y_pred"}))[metric]
        b = session_metrics(frame.rename(columns={"y_true_b": "y_true", "y_pred_b": "y_pred"}))[metric]
        return a - b

    observed = _diff(merged)
    diffs = []
    for _ in range(n_boot):
        picked = rng.choice(subjects, size=len(subjects), replace=True)
        rep = pd.concat([by_subject[s] for s in picked], ignore_index=True)
        val = _diff(rep)
        if np.isfinite(val):
            diffs.append(val)
    return {
        "diff": float(observed),
        "ci_low": float(np.percentile(diffs, 2.5)) if diffs else np.nan,
        "ci_high": float(np.percentile(diffs, 97.5)) if diffs else np.nan,
        "n_paired_sessions": int(len(merged)),
    }


def metrics_by_visit(sessions: pd.DataFrame) -> pd.DataFrame:
    """Breaks the error down by visit, to expose the four-level structure of the target.

    With visits at 6, 9, 16 and 36 months and a 20-month gap before the last one, an aggregate R2
    can be high while the model only separates infants from preschoolers. The per-visit breakdown
    makes that visible.

    Args:
        sessions: Output of :func:`aggregate_to_sessions`.

    Returns:
        Frame with MAE, bias and count per visit.
    """
    rows = []
    for age, grp in sessions.groupby("age"):
        err = grp.y_pred - grp.y_true
        rows.append({"age": age, "n": len(grp), "mae": float(err.abs().mean()),
                     "bias": float(err.mean()), "pred_median": float(grp.y_pred.median())})
    return pd.DataFrame(rows)
