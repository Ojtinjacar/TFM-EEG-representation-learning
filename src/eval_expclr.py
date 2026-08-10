from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

ALPHA_GRID = np.logspace(-3, 4, 30)

SESSION_KEYS = ("subject", "age", "block")


@torch.no_grad()
def extract_embeddings(model, X: np.ndarray, device, batch_size: int = 256,
                       representation: str = "projection") -> np.ndarray:
    if representation not in ("projection", "embedding"):
        raise ValueError(f"representation debe ser 'projection' o 'embedding', no {representation!r}")
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        chunk = np.array(X[start:start + batch_size], dtype=np.float32)
        batch = torch.as_tensor(chunk, device=device)
        z = model(batch) if representation == "projection" else model.get_embedding(batch)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


def fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    n_splits: int = 5,
) -> tuple[StandardScaler, Ridge]:
    n_groups = len(np.unique(groups_train))
    splits = min(n_splits, n_groups)
    if splits < 2:
        best_alpha = 1.0
    else:
        cv = GroupKFold(n_splits=splits)
        scores = np.zeros(len(ALPHA_GRID))
        for inner_train, inner_val in cv.split(X_train, y_train, groups_train):
            # The scaler is fitted inside each inner split so that alpha
            # selection never sees statistics of its own validation fold.
            inner_scaler = StandardScaler().fit(X_train[inner_train])
            Z_tr = inner_scaler.transform(X_train[inner_train])
            Z_val = inner_scaler.transform(X_train[inner_val])
            for i, alpha in enumerate(ALPHA_GRID):
                model = Ridge(alpha=alpha).fit(Z_tr, y_train[inner_train])
                pred = model.predict(Z_val)
                scores[i] += np.mean(np.abs(pred - y_train[inner_val]))
        best_alpha = float(ALPHA_GRID[int(np.argmin(scores))])

    scaler = StandardScaler().fit(X_train)
    return scaler, Ridge(alpha=best_alpha).fit(scaler.transform(X_train), y_train)


def fit_knn_probe(X_train: np.ndarray, y_train: np.ndarray, k: int = 1):
    scaler = StandardScaler().fit(X_train)
    return scaler, KNeighborsRegressor(n_neighbors=k).fit(scaler.transform(X_train), y_train)


def subject_metrics(sessions: pd.DataFrame) -> dict[str, float]:
    by_subject = sessions.groupby("subject")[["y_true", "y_pred"]].mean()
    y, p = by_subject.y_true.values, by_subject.y_pred.values
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "nrmse_subject": rmse / float(np.std(y)) if np.std(y) > 0 else np.nan,
        "rmse_subject": rmse,
        "r2_subject": 1 - float(np.sum((y - p) ** 2)) / ss_tot if ss_tot > 0 else np.nan,
    }


def aggregate_to_sessions(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    keys = [k for k in SESSION_KEYS if k in meta.columns]
    df = meta[keys].copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    return df.groupby(keys, as_index=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "median"))


def session_metrics(sessions: pd.DataFrame) -> dict[str, float]:
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
    rows = []
    for age, grp in sessions.groupby("age"):
        err = grp.y_pred - grp.y_true
        rows.append({"age": age, "n": len(grp), "mae": float(err.abs().mean()),
                     "bias": float(err.mean()), "pred_median": float(grp.y_pred.median())})
    return pd.DataFrame(rows)
