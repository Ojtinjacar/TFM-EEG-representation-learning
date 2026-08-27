"""Tests for the fold-safe window loader.

What matters here is not that the numbers change, but that the held-out subjects
stop contributing to the transform applied to the training windows, and that
refitting on top of an already normalised tensor lands on the same values as
normalising the raw one with train-only statistics.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from window_loading import (  # noqa: E402
    WindowLoadingError,
    fit_channel_stats,
    load_windows,
    resolve_paths,
)

N_CHANNELS, N_SAMPLES = 4, 32
SUBJECTS = ["S1", "S1", "S2", "S2", "S3", "S3"]


def _write_dataset(directory, X, subjects):
    """Writes a processed directory the loader can read.

    Args:
        directory (str): Destination directory.
        X (np.ndarray): Window tensor.
        subjects (list[str]): Subject of each window.

    Returns:
        str: The directory it wrote to.
    """
    os.makedirs(directory, exist_ok=True)
    np.save(os.path.join(directory, "processed_windows.npy"), X)
    pd.DataFrame({
        "subject": subjects,
        "age": [6, 9, 6, 16, 9, 36],
    }).to_csv(os.path.join(directory, "processed_metadata.csv"), index=False)
    return directory


@pytest.fixture
def raw_and_normalised(tmp_path):
    """Builds the same cohort twice: unnormalised, and z-scored per channel.

    Returns:
        tuple[str, str, np.ndarray]: Raw directory, globally normalised
            directory, and the raw tensor.
    """
    rng = np.random.default_rng(0)
    # A per-channel offset and scale, so a channel-wise transform is not a no-op.
    X_raw = (rng.normal(size=(len(SUBJECTS), N_CHANNELS, N_SAMPLES))
             * np.array([1.0, 5.0, 0.2, 30.0]).reshape(1, -1, 1)
             + np.array([0.0, -3.0, 10.0, 100.0]).reshape(1, -1, 1))
    X_raw = X_raw.astype(np.float32)

    mean_ch = X_raw.mean(axis=(0, 2), keepdims=True)
    std_ch = X_raw.std(axis=(0, 2), keepdims=True)
    X_norm = ((X_raw - mean_ch) / (std_ch + 1e-12)).astype(np.float32)

    raw_dir = _write_dataset(str(tmp_path / "raw"), X_raw, SUBJECTS)
    norm_dir = _write_dataset(str(tmp_path / "norm"), X_norm, SUBJECTS)
    return raw_dir, norm_dir, X_raw


def test_refit_on_normalised_matches_raw_with_train_only_stats(raw_and_normalised):
    """Refitting over a normalised tensor equals normalising the raw one.

    This is what lets the processed directories stay as they are: a per-channel
    z-score absorbs the per-channel affine transform already applied.
    """
    raw_dir, norm_dir, X_raw = raw_and_normalised
    held = ["S3"]

    from_normalised, _ = load_windows(norm_dir, fit_stats_excluding=held, verbose=False)

    keep = ~pd.Series(SUBJECTS).isin(held).values
    mean_ch, std_ch = fit_channel_stats(X_raw, keep)
    from_raw = (X_raw - mean_ch) / (std_ch + 1e-12)

    np.testing.assert_allclose(from_normalised, from_raw, rtol=1e-4, atol=1e-4)


def test_held_out_subject_does_not_shape_the_transform(raw_and_normalised):
    """The statistics are fitted on the training windows only."""
    _, norm_dir, _ = raw_and_normalised
    held = ["S3"]

    X, meta = load_windows(norm_dir, fit_stats_excluding=held, verbose=False)
    train = ~meta["subject"].isin(held).values

    np.testing.assert_allclose(X[train].mean(axis=(0, 2)), 0.0, atol=1e-5)
    np.testing.assert_allclose(X[train].std(axis=(0, 2)), 1.0, atol=1e-5)

    # The held-out windows are transformed too, and by a scaler that is not
    # theirs, so they have no reason to be centred.
    assert np.abs(X[~train].mean(axis=(0, 2))).max() > 1e-5


def test_every_window_is_returned(raw_and_normalised):
    """Excluding subjects changes the transform, never the number of rows."""
    _, norm_dir, _ = raw_and_normalised
    X, meta = load_windows(norm_dir, fit_stats_excluding=["S3"], verbose=False)
    assert len(X) == len(SUBJECTS)
    assert len(meta) == len(SUBJECTS)
    assert set(meta["subject"]) == set(SUBJECTS)


def test_without_exclusion_the_tensor_is_untouched(raw_and_normalised):
    """No exclusion means no refit, so runs without folds stay bit-identical."""
    _, norm_dir, _ = raw_and_normalised
    stored = np.load(os.path.join(norm_dir, "processed_windows.npy"))

    for empty in (None, []):
        X, _ = load_windows(norm_dir, fit_stats_excluding=empty, verbose=False)
        np.testing.assert_array_equal(X, stored)


def test_different_folds_give_different_transforms(raw_and_normalised):
    """Two folds hold out different subjects, so they cannot share a scaler."""
    _, norm_dir, _ = raw_and_normalised
    fold_a, _ = load_windows(norm_dir, fit_stats_excluding=["S1"], verbose=False)
    fold_b, _ = load_windows(norm_dir, fit_stats_excluding=["S2"], verbose=False)
    assert not np.allclose(fold_a, fold_b)


def test_both_calling_conventions_resolve(raw_and_normalised):
    """The directory and the .npy path address the same pair of files."""
    _, norm_dir, _ = raw_and_normalised
    from_dir = resolve_paths(norm_dir)
    from_npy = resolve_paths(os.path.join(norm_dir, "processed_windows.npy"))
    assert from_dir == from_npy


def test_missing_files_fail_loudly(tmp_path):
    """A missing directory is an error, not an empty tensor."""
    with pytest.raises(WindowLoadingError):
        resolve_paths(str(tmp_path / "does_not_exist"))


def test_excluding_everyone_fails_loudly(raw_and_normalised):
    """Holding out the whole cohort leaves nothing to fit on."""
    _, norm_dir, _ = raw_and_normalised
    with pytest.raises(WindowLoadingError):
        load_windows(norm_dir, fit_stats_excluding=list(set(SUBJECTS)), verbose=False)


def test_length_mismatch_fails_loudly(tmp_path):
    """A tensor and a metadata table of different length do not describe a run."""
    directory = str(tmp_path / "broken")
    os.makedirs(directory)
    np.save(os.path.join(directory, "processed_windows.npy"),
            np.zeros((3, N_CHANNELS, N_SAMPLES), dtype=np.float32))
    pd.DataFrame({"subject": ["S1", "S2"]}).to_csv(
        os.path.join(directory, "processed_metadata.csv"), index=False)

    with pytest.raises(WindowLoadingError):
        load_windows(directory, verbose=False)
