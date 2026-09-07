"""Fold-safe loading of the processed EEG windows.

Every trainer and every evaluation entry point used to reach for
``processed_windows.npy`` with its own ``np.load``, which left the amplitude
normalisation wherever ``postprocessing.py`` had put it: fitted once over the
whole cohort. The held-out subjects of a fold therefore contributed to the mean
and the standard deviation that transformed the training windows, which is a
leak the per-fold protocol is meant to prevent.

This module is the single place where windows are read, so the statistics can be
refitted excluding the held-out subjects of the fold. The transform is then
applied to *every* window, held-out ones included, exactly like a scaler fitted
on train and applied to test.

Refitting on top of an already normalised tensor is exact, not an approximation.
The stored windows are ``X' = (X - m) / s`` with ``m`` and ``s`` per channel over
the whole cohort. Re-standardising per channel over a subset gives

    (X' - mean(X'_sub)) / std(X'_sub)
        = ((X - m)/s - (m_sub - m)/s) / (s_sub / s)
        = (X - m_sub) / s_sub,

which is what would have come out of normalising the raw recording with
train-only statistics. A per-channel z-score absorbs any previous per-channel
affine transform, so the processed directories do not need to be regenerated.
"""

import os

import numpy as np
import pandas as pd

WINDOWS_FILENAME = "processed_windows.npy"
METADATA_FILENAME = "processed_metadata.csv"

# Same guard as postprocessing.py, so a constant channel yields zeros instead of
# dividing by zero.
_STD_EPSILON = 1e-12

# Recorded in every checkpoint sidecar. Weights trained under cohort-wide
# statistics and weights trained under per-fold statistics are not the same
# model, and the reuse gate compares this string to tell them apart.
NORM_PROVENANCE = "per-fold-channel-zscore"


class WindowLoadingError(RuntimeError):
    """Raised when the windows and their metadata cannot be read consistently."""


def resolve_paths(source, meta_path=None):
    """Resolves the window and metadata paths from either calling convention.

    Half the callers pass the directory holding both files and the other half
    pass the ``.npy`` path with the ``.csv`` alongside it. Both are accepted so
    this module can replace either without touching the command-line contract.

    Args:
        source (str): Either the directory holding the processed files or the
            path to ``processed_windows.npy`` itself.
        meta_path (str | None): Explicit metadata path. When omitted it is
            derived from ``source``.

    Returns:
        tuple[str, str]: The windows path and the metadata path.

    Raises:
        WindowLoadingError: If either file is missing.
    """
    source = os.fspath(source)
    if os.path.isdir(source):
        windows_path = os.path.join(source, WINDOWS_FILENAME)
        derived_meta = os.path.join(source, METADATA_FILENAME)
    else:
        windows_path = source
        derived_meta = os.path.join(os.path.dirname(source), METADATA_FILENAME)

    resolved_meta = os.fspath(meta_path) if meta_path else derived_meta

    for path in (windows_path, resolved_meta):
        if not os.path.exists(path):
            raise WindowLoadingError(f"Processed data not found: {path}")

    return windows_path, resolved_meta


def fit_channel_stats(X, mask=None):
    """Fits per-channel amplitude statistics over the selected windows.

    Args:
        X (np.ndarray): Window tensor of shape ``(n_windows, n_channels,
            n_samples)``.
        mask (np.ndarray | None): Boolean mask over the first axis selecting the
            windows the statistics are fitted on. ``None`` uses every window.

    Returns:
        tuple[np.ndarray, np.ndarray]: Mean and standard deviation, both of
            shape ``(1, n_channels, 1)`` so they broadcast over the tensor.

    Raises:
        WindowLoadingError: If the mask leaves no window to fit on.
    """
    subset = X if mask is None else X[mask]
    if len(subset) == 0:
        raise WindowLoadingError(
            "The normalisation mask removed every window; no data left to fit "
            "the per-channel statistics on."
        )
    mean_ch = subset.mean(axis=(0, 2), keepdims=True)
    std_ch = subset.std(axis=(0, 2), keepdims=True)
    return mean_ch, std_ch


def apply_fold_normalisation(X, subjects, fit_stats_excluding, verbose=True):
    """Refits the per-channel normalisation of an in-memory tensor for one fold.

    ``load_windows`` covers the entry points that read from disk once per run.
    A driver that sweeps every fold over a single tensor cannot afford to reread
    hundreds of megabytes per subject, so it refits here instead. The tensor is
    not modified: a normalised copy is returned, since the caller still needs the
    original to build the next fold from.

    Args:
        X (np.ndarray): Window tensor of shape ``(n_windows, n_channels,
            n_samples)``, normalised with whatever statistics it was stored with.
        subjects (np.ndarray | pd.Series): Subject of each window, aligned with
            ``X``.
        fit_stats_excluding (list[str]): Subjects held out in this fold.
        verbose (bool): Whether to report what the statistics were fitted on.

    Returns:
        np.ndarray: A copy of ``X`` transformed with statistics fitted on the
            windows of the remaining subjects.

    Raises:
        WindowLoadingError: If ``subjects`` does not align with ``X`` or if the
            exclusion leaves no window to fit the statistics on.
    """
    subjects = np.asarray(subjects).astype(str)
    if len(subjects) != len(X):
        raise WindowLoadingError(
            f"{len(X)} windows but {len(subjects)} subject labels; the two do "
            "not describe the same window set."
        )

    excluded = sorted({str(s) for s in (fit_stats_excluding or [])})
    if not excluded:
        return np.asarray(X)

    stats_mask = ~np.isin(subjects, excluded)
    mean_ch, std_ch = fit_channel_stats(X, stats_mask)

    if verbose:
        print(f"[window_loading] per-channel statistics refitted excluding "
              f"{excluded} ({int((~stats_mask).sum())} windows held out of the fit).")

    return (X - mean_ch) / (std_ch + _STD_EPSILON)


def load_windows(source, meta_path=None, fit_stats_excluding=None, verbose=True):
    """Loads the processed windows, refitting the normalisation without the
    held-out subjects.

    The returned tensor always holds every window in the directory. Selecting
    the training rows stays the caller's job; what this function guarantees is
    that the amplitude transform applied to them never saw the held-out
    subjects.

    Args:
        source (str): Directory holding the processed files, or the path to
            ``processed_windows.npy``.
        meta_path (str | None): Explicit metadata path. Derived from ``source``
            when omitted.
        fit_stats_excluding (list[str] | None): Subjects held out in this fold.
            When empty or ``None`` the tensor is returned as stored, since the
            statistics already cover every subject available.
        verbose (bool): Whether to report what the statistics were fitted on.

    Returns:
        tuple[np.ndarray, pd.DataFrame]: The window tensor and its metadata,
            aligned row by row.

    Raises:
        WindowLoadingError: If the files are missing, if the tensor and the
            metadata disagree in length, or if the exclusion leaves no window to
            fit the statistics on.
    """
    windows_path, resolved_meta = resolve_paths(source, meta_path)

    X = np.load(windows_path)
    meta = pd.read_csv(resolved_meta)

    if len(X) != len(meta):
        raise WindowLoadingError(
            f"{len(X)} windows in {windows_path} but {len(meta)} rows in "
            f"{resolved_meta}; the two files do not describe the same run."
        )

    excluded = sorted({str(s) for s in (fit_stats_excluding or [])})
    if not excluded:
        if verbose:
            print(f"[window_loading] {len(X)} windows, normalisation as stored "
                  f"(statistics cover every subject).")
        return X, meta

    stats_mask = ~meta["subject"].astype(str).isin(excluded).values
    mean_ch, std_ch = fit_channel_stats(X, stats_mask)

    # In place: the tensor runs to hundreds of megabytes and a copy per fold is
    # avoidable.
    X -= mean_ch
    X /= (std_ch + _STD_EPSILON)

    if verbose:
        print(f"[window_loading] {len(X)} windows, per-channel statistics "
              f"refitted excluding {len(excluded)} subject(s) "
              f"({int((~stats_mask).sum())} windows held out of the fit).")

    return X, meta
