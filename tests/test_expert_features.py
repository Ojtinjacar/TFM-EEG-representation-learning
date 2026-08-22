"""Tests for the expert spectral descriptor."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from build_expert_features import (  # noqa: E402
    BANDS,
    DESCRIPTORS,
    DescriptorAlignmentError,
    check_amplitude_is_physical,
    compute_window_features,
    roi_indices,
    roi_measures,
)
from epoch_features import SFREQ  # noqa: E402


AMPLITUDE_VOLTS = 2e-5


def synthetic_recording(n_windows=3, n_channels=8, n_samples=1250, seed=0):
    """Builds a recording with an alpha rhythm on top of pink-ish noise.

    Args:
        n_windows (int): Number of windows.
        n_channels (int): Number of channels.
        n_samples (int): Samples per window.
        seed (int): Seed of the random generator.

    Returns:
        np.ndarray: Recording of shape (n_windows, n_channels, n_samples) in volts.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / SFREQ
    alpha = np.sin(2 * np.pi * 9.0 * t)
    noise = rng.standard_normal((n_windows, n_channels, n_samples))
    return AMPLITUDE_VOLTS * (alpha + 0.5 * noise)


def test_catalogue_sizes_and_uniqueness():
    assert len(DESCRIPTORS["P_full"]) == 78
    assert len(DESCRIPTORS["P_madurativo"]) == 32
    assert len(DESCRIPTORS["P_aper"]) == 8
    for name, cols in DESCRIPTORS.items():
        assert len(cols) == len(set(cols)), f"{name} repeats a column"
        assert set(cols) <= set(DESCRIPTORS["P_full"])
    # The oscillatory block is not part of the maturational ablation.
    assert not any(c.startswith("osc_") for c in DESCRIPTORS["P_madurativo"])


def test_bands_tile_the_analysis_range():
    edges = sorted(BANDS.values())
    for (_, hi), (lo, _) in zip(edges, edges[1:]):
        assert hi == lo, "the bands must tile the range without gaps or overlap"


def test_relative_powers_sum_to_one():
    signal = synthetic_recording(n_windows=1)[0].mean(axis=0)
    measures = roi_measures(signal)
    total = sum(measures[f"rel_{band}"] for band in BANDS)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_alpha_rhythm_lands_on_the_peak_frequency():
    signal = synthetic_recording(n_windows=1)[0].mean(axis=0)
    assert roi_measures(signal)["paf_freq"] == pytest.approx(9.0, abs=1.0)


def test_roi_indices_reject_a_montage_of_another_size():
    with pytest.raises(DescriptorAlignmentError):
        roi_indices(["Fp1", "Fp2"], 70)


def test_amplitude_guard_accepts_volts_and_rejects_a_per_channel_zscore():
    recording = synthetic_recording()
    assert check_amplitude_is_physical(recording) < 1e-3

    normalized = recording / recording.std(axis=2, keepdims=True)
    with pytest.raises(DescriptorAlignmentError):
        check_amplitude_is_physical(normalized)


def test_topographic_contrasts_are_differences_of_peak_frequencies():
    recording = synthetic_recording(n_windows=2)
    indices = {"frontal": [0, 1], "central": [2, 3], "parietal": [4, 5], "occipital": [6, 7]}
    features = compute_window_features(recording, indices, progress=0)

    assert len(features) == len(recording)
    for column in DESCRIPTORS["P_full"]:
        assert column in features.columns
    np.testing.assert_allclose(
        features["paf_central_minus_occipital"],
        features["paf_freq_central"] - features["paf_freq_occipital"],
    )


def test_a_region_without_channels_is_refused():
    recording = synthetic_recording(n_windows=1)
    indices = {"frontal": [0, 1], "central": [2, 3], "parietal": [4, 5], "occipital": []}
    with pytest.raises(KeyError):
        compute_window_features(recording, indices, progress=0)


def test_a_failed_fit_returns_the_same_keys_as_a_successful_one():
    from apsd_baseline import extract_specparam_features

    rng = np.random.default_rng(0)
    converged = extract_specparam_features(np.arange(1.0, 30.0), rng.random(29) + 1.0)
    failed = extract_specparam_features(np.array([1.0, 2.0]), np.array([np.nan, np.nan]))
    assert set(converged) == set(failed)


def test_fit_diagnostics_stay_out_of_the_baseline_features():
    from apsd_baseline import APSD_FIT_DIAGNOSTICS, extract_window_features

    recording = synthetic_recording(n_windows=1)[0]
    indices = {"frontal": [0, 1], "central": [2, 3], "parietal": [4, 5], "occipital": [6, 7]}
    features = extract_window_features(recording, indices, SFREQ)

    regressed = [c for c in features
                 if c.startswith("APSD_") and not c.endswith(APSD_FIT_DIAGNOSTICS)]
    assert len(regressed) == 36
    assert not any(c.endswith(APSD_FIT_DIAGNOSTICS) for c in regressed)
