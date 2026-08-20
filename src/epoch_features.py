"""Per-epoch spectral descriptors over regions of interest.

Estimates the power spectral density of each epoch by averaging the channels of a
region of interest and applying Welch's method, and derives from it the two
representations the neighbour index is built on: band powers together with the
aperiodic fit (:func:`compute_epoch_features`), and the full normalized spectrum
treated as a probability distribution over frequency (:func:`compute_epoch_roi_psd`),
which :func:`wasserstein_fourier_distance` compares in closed form.

Band definitions and the aperiodic fit come from :mod:`apsd_baseline`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

from apsd_baseline import FREQ_BANDS, extract_specparam_features

# Sampling frequency of the CIMCYC recordings (Hz).
SFREQ: float = 250.0
# Welch segment length: 1 s at SFREQ, giving a 1 Hz frequency resolution.
PSD_NPERSEG: int = int(1.0 * SFREQ)
# Guard against log10(0) and division by zero.
EPS: float = 1e-30
# Frequency range and aperiodic mode handed to specparam.
APSD_FREQ_RANGE: tuple[float, float] = (1.5, 20.0)
APSD_APERIODIC_MODE: str = "knee"

# NumPy <2.0 exposes trapz, >=2.0 exposes trapezoid.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _bandpower(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Integrates the PSD over a frequency band.

    Args:
        freqs (np.ndarray): Frequency bins of the PSD.
        psd (np.ndarray): Power spectral density values.
        lo (float): Lower bound of the band (Hz), inclusive.
        hi (float): Upper bound of the band (Hz), inclusive.

    Returns:
        float: Integrated power in the band, 0.0 when the band selects no bin.
    """
    mask = (freqs >= lo) & (freqs <= hi)
    return float(_trapz(psd[mask], freqs[mask])) if mask.any() else 0.0


def compute_epoch_features(
    X: np.ndarray,
    meta: pd.DataFrame,
    roi_indices: dict[str, list[int]],
    *,
    sfreq: float = SFREQ,
    do_apsd: bool = True,
    apsd_freq_range: tuple[float, float] = APSD_FREQ_RANGE,
    apsd_aperiodic_mode: str = APSD_APERIODIC_MODE,
    bands: dict[str, tuple[float, float]] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Computes spectral descriptors per epoch and ROI.

    Args:
        X (np.ndarray): Tensor of shape (n_epochs, n_channels, n_samples); may be a memmap.
        meta (pd.DataFrame): Metadata with subject, age, epoch_index, block, window_quality.
        roi_indices (dict): Mapping {roi_name: [channel_indices]}.
        sfreq (float): Sampling frequency in Hz.
        do_apsd (bool): Whether to fit the aperiodic component with specparam.
        apsd_freq_range (tuple): Frequency range handed to specparam (Hz).
        apsd_aperiodic_mode (str): 'knee' or 'fixed'.
        bands (dict | None): Frequency bands; defaults to apsd_baseline.FREQ_BANDS.
        progress (bool): Whether to print progress every 300 epochs.

    Returns:
        pd.DataFrame: One row per epoch with bp_<band>_<roi> and, when do_apsd is set,
            apsd_slope/offset/knee/r2/n_peaks per ROI, plus the carried metadata columns.
    """
    if bands is None:
        bands = FREQ_BANDS

    nperseg = PSD_NPERSEG

    rows: list[dict] = []
    n_epochs = X.shape[0]

    for i in range(n_epochs):
        if progress and i % 300 == 0:
            print(f"  {i}/{n_epochs}...")

        feats: dict = {}
        epoch_data = np.asarray(X[i], dtype=np.float64)

        for roi, idx in roi_indices.items():
            if not idx:
                continue
            sig = epoch_data[idx].mean(axis=0)
            freqs, psd = welch(sig, fs=sfreq, nperseg=nperseg)

            for band, (lo, hi) in bands.items():
                feats[f"bp_{band}_{roi}"] = _bandpower(freqs, psd, lo, hi)

            if do_apsd:
                try:
                    af = extract_specparam_features(
                        freqs, psd,
                        freq_range=list(apsd_freq_range),
                        aperiodic_mode=apsd_aperiodic_mode,
                    )
                    feats[f"apsd_slope_{roi}"]   = af.get("aperiodic_exponent", np.nan)
                    feats[f"apsd_offset_{roi}"]  = af.get("aperiodic_offset",   np.nan)
                    feats[f"apsd_knee_{roi}"]    = af.get("aperiodic_knee",     np.nan)
                    feats[f"apsd_r2_{roi}"]      = af.get("aperiodic_r2",       np.nan)
                    feats[f"apsd_n_peaks_{roi}"] = af.get("aperiodic_n_peaks",  0)
                except Exception:
                    feats[f"apsd_slope_{roi}"]   = np.nan
                    feats[f"apsd_offset_{roi}"]  = np.nan
                    feats[f"apsd_knee_{roi}"]    = np.nan
                    feats[f"apsd_r2_{roi}"]      = np.nan
                    feats[f"apsd_n_peaks_{roi}"] = 0

        rows.append(feats)

    df = pd.DataFrame(rows)
    for col in ["subject", "age", "epoch_index", "block", "window_quality"]:
        if col in meta.columns:
            df[col] = meta[col].values
    return df


def compute_epoch_roi_psd(
    X: np.ndarray,
    roi_indices: dict[str, list[int]],
    *,
    sfreq: float = SFREQ,
    fmin: float = 1.5,
    fmax: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the per-epoch, per-ROI normalized power spectral density.

    Reuses the same Welch estimation as :func:`compute_epoch_features` (ROI channel
    average, ``nperseg = PSD_NPERSEG``) but keeps the full spectrum instead of collapsing
    it into band powers, and normalizes each PSD to sum 1 so it can be treated as a
    probability distribution over frequency. This is the input representation for the
    Wasserstein-Fourier distance.

    Args:
        X (np.ndarray): Tensor of shape (n_epochs, n_channels, n_samples); may be a memmap.
        roi_indices (dict): Mapping {roi_name: [channel_indices]}. The ROI axis of the
            output follows this mapping's iteration order.
        sfreq (float): Sampling frequency in Hz.
        fmin (float): Lower bound of the frequency range to keep (Hz), inclusive.
        fmax (float): Upper bound of the frequency range to keep (Hz), inclusive.

    Returns:
        tuple: ``(freqs, psd_norm)`` where freqs is the 1-D array of retained frequency
            bins and psd_norm has shape (n_epochs, n_rois, n_freq), each row summing to 1.
            Empty ROIs yield a uniform distribution.

    Raises:
        ValueError: If the selected [fmin, fmax] range retains no frequency bin.
    """
    nperseg = PSD_NPERSEG
    n_epochs = X.shape[0]
    roi_names = list(roi_indices.keys())

    # Frequency grid from a probe Welch call, then the analysis-band mask.
    probe_freqs, _ = welch(np.zeros(X.shape[-1]), fs=sfreq, nperseg=nperseg)
    band_mask = (probe_freqs >= fmin) & (probe_freqs <= fmax)
    if not band_mask.any():
        raise ValueError(f"No frequency bin in range [{fmin}, {fmax}] Hz")
    freqs = probe_freqs[band_mask]

    n_freq = freqs.size
    n_rois = len(roi_names)
    psd_norm = np.empty((n_epochs, n_rois, n_freq), dtype=np.float64)
    uniform = np.full(n_freq, 1.0 / n_freq)

    for i in range(n_epochs):
        epoch_data = np.asarray(X[i], dtype=np.float64)
        for r, roi in enumerate(roi_names):
            idx = roi_indices[roi]
            if not idx:
                psd_norm[i, r] = uniform
                continue
            sig = epoch_data[idx].mean(axis=0)
            _, psd = welch(sig, fs=sfreq, nperseg=nperseg)
            psd_band = psd[band_mask]
            total = psd_band.sum()
            psd_norm[i, r] = psd_band / total if total > EPS else uniform

    return freqs, psd_norm


def wasserstein_fourier_distance(
    psd_a: np.ndarray,
    psd_b: np.ndarray,
    *,
    df: float = 1.0,
) -> np.ndarray:
    """Wasserstein-1 distance between two normalized power spectral densities.

    Treats each normalized PSD as a 1-D probability distribution over an evenly spaced
    frequency grid (spacing ``df``). For 1-D distributions the Wasserstein-1 distance has
    the closed form ``W1 = sum_f |CDF_a(f) - CDF_b(f)| * df``, so no optimal-transport
    solver is needed. This is the Wasserstein-Fourier distance of Cazelles et al. (2020),
    specialized to a shared, uniform frequency grid.

    Both inputs must already be normalized to sum 1 along the last axis and share the same
    grid. Any leading dimensions are broadcast and preserved, enabling vectorized batch
    evaluation over many epoch pairs and ROIs at once.

    Args:
        psd_a (np.ndarray): Normalized PSD(s); last axis is frequency, shape (..., n_freq).
        psd_b (np.ndarray): Normalized PSD(s) on the same grid, broadcastable to psd_a.
        df (float): Frequency bin spacing (Hz) used as the integration step.

    Returns:
        np.ndarray: Wasserstein-1 distances with the broadcast leading shape of the inputs.
    """
    cdf_a = np.cumsum(psd_a, axis=-1)
    cdf_b = np.cumsum(psd_b, axis=-1)
    return np.abs(cdf_a - cdf_b).sum(axis=-1) * df
