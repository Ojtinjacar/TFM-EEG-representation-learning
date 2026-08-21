"""Expert spectral descriptor for every EEG window.

Builds a dense matrix whose row ``i`` describes window ``i`` of the recording. The
descriptor has to be dense because the contrastive loss it feeds compares every pair of a
batch: a window without a descriptor is not a missing label but a missing row of the
geometry.

It is computed from the broadband recording with its amplitude untouched. Two families of
measures require it: the aperiodic fit needs a spectrum that a band-pass has not reshaped,
and absolute power is only meaningful while the signal keeps its physical scale. Windows
normalized per channel are therefore rejected rather than described.

Nineteen measures are computed per region of interest, over four regions, plus two
topographic contrasts, for a total of 78. Per region, from the Welch spectrum of the
channel average:

- ``paf_freq`` / ``paf_power``: alpha peak of the spectrum after dividing out its 1/f
  component.
- ``sp_alpha_cf`` / ``sp_alpha_pw``: centre frequency and power of the alpha peak of the
  specparam fit.
- ``rel_<band>``: power of each band divided by the power of the analysis range.
- ``osc_<band>``: power of each band in the flattened spectrum, i.e. after removing the
  aperiodic component.
- ``apsd_slope`` / ``apsd_offset``: exponent and offset of the aperiodic fit.
- ``total_power``: power of the whole analysis range.
- ``theta_alpha``, ``delta_theta``, ``delta_alpha``, ``low_high``: band ratios.

The four bands tile the analysis range without gaps or overlap, so the relative powers sum
to one. The two contrasts subtract the central peak frequency from the occipital and the
parietal ones: posterior alpha and central mu overlap in frequency and can only be told
apart spatially.

Three descriptors are exposed. ``P_full`` is the whole catalogue; ``P_madurativo`` keeps
peak alpha, relative powers and the aperiodic pair; ``P_aper`` keeps only the aperiodic
pair. They differ in how often they are complete: the centre frequency and power of the
specparam alpha peak are undefined when the fit finds no peak in the alpha range, which
is a statement about the spectrum rather than a failure, so the narrower descriptors are
the dense ones. The build reports how many rows each leaves complete.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import welch

from apsd_baseline import FREQ_BANDS, extract_specparam_features
from montage import PRESET_ZONES, MontageError, load_channel_names
from epoch_features import EPS, PSD_NPERSEG, SFREQ, _bandpower, find_paf

ROIS = ["frontal", "central", "parietal", "occipital"]
BANDS = {b: lim for b, lim in FREQ_BANDS.items() if b != "gamma"}
ANALYSIS_LO = min(lo for lo, _ in BANDS.values())
ANALYSIS_HI = max(hi for _, hi in BANDS.values())
APSD_FREQ_RANGE = (1.5, 20.0)
APSD_APERIODIC_MODE = "knee"

BASE_A = (
    ["paf_freq", "paf_power", "sp_alpha_cf", "sp_alpha_pw"]
    + [f"rel_{b}" for b in BANDS]
    + [f"osc_{b}" for b in BANDS]
    + ["apsd_slope", "apsd_offset", "total_power"]
)  # 15
BASE_B = ["theta_alpha", "delta_theta", "delta_alpha", "low_high"]  # 4
BASE_MADURATIVO = (
    ["paf_freq", "paf_power"] + [f"rel_{b}" for b in BANDS] + ["apsd_slope", "apsd_offset"]
)
BASE_APER = ["apsd_slope", "apsd_offset"]
TOPO = ["paf_central_minus_occipital", "paf_central_minus_parietal"]

QUALITY_COLS = [f"apsd_r2_{roi}" for roi in ROIS]

# A per-channel z-score leaves the channels at unit standard deviation; a recording in
# volts sits five orders of magnitude below it.
AMPLITUDE_UNIT_SCALE = (0.5, 2.0)


class DescriptorAlignmentError(MontageError):
    """Raised when the descriptor cannot be matched to the window metadata."""


def _expand(bases: list[str], *, topo: bool) -> list[str]:
    """Expands base measure names over the four regions of interest.

    Args:
        bases (list): Measure names without the region suffix.
        topo (bool): Whether to append the two topographic contrasts.

    Returns:
        list: Ordered concrete column names.
    """
    cols = [f"{base}_{roi}" for base in bases for roi in ROIS]
    return cols + TOPO if topo else cols


DESCRIPTORS: dict[str, list[str]] = {
    "P_full": _expand(BASE_A + BASE_B, topo=True),
    "P_madurativo": _expand(BASE_MADURATIVO, topo=False),
    "P_aper": _expand(BASE_APER, topo=False),
}


def roi_indices(channels: list[str], n_channels: int) -> dict[str, list[int]]:
    """Maps each region of interest to its channels of the recording.

    Args:
        channels (list): Channel names of the montage, in recording order.
        n_channels (int): Number of channels of the array to describe.

    Returns:
        dict: Region name to channel indices.

    Raises:
        DescriptorAlignmentError: If the names do not account for the channels of the
            array, or if a region ends up with no channel of its own.
    """
    if len(channels) != n_channels:
        raise DescriptorAlignmentError(
            f"{len(channels)} channel names for an array of {n_channels} channels: the "
            "regions are resolved by position, so both must describe the same montage."
        )
    indices = {roi: [i for i, ch in enumerate(channels) if ch in set(PRESET_ZONES[roi])]
               for roi in ROIS}
    empty = [roi for roi, idx in indices.items() if not idx]
    if empty:
        raise DescriptorAlignmentError(
            f"No channel of the montage belongs to {empty}. The descriptor spans four "
            "regions and cannot be built from a montage that misses one."
        )
    return indices


def check_amplitude_is_physical(recording: np.ndarray, n_probe: int = 60) -> float:
    """Checks the recording still carries its physical amplitude.

    Normalizing per channel leaves every channel with unit standard deviation, which
    rewrites total power and the aperiodic offset while leaving every other measure
    plausible, so the damage would be silent. A spread sample of windows is enough to tell.

    Args:
        recording (np.ndarray): Recording of shape (n_windows, n_channels, n_samples).
        n_probe (int): Number of windows to sample.

    Returns:
        float: Median standard deviation across the sampled channels.

    Raises:
        DescriptorAlignmentError: If that median sits at the unit scale of a z-score.
    """
    positions = np.linspace(0, len(recording) - 1, min(n_probe, len(recording)))
    probe = np.asarray(recording[positions.astype(int)], dtype=np.float64)
    spread = float(np.median(probe.std(axis=2)))
    low, high = AMPLITUDE_UNIT_SCALE
    if low <= spread <= high:
        raise DescriptorAlignmentError(
            f"The channels have a standard deviation of {spread:.3f}, the unit scale left "
            "by a per-channel normalization. The descriptor measures absolute power and "
            "the shape of the aperiodic component, both of which that normalization "
            "destroys."
        )
    return spread


def roi_measures(signal: np.ndarray, sfreq: float = SFREQ) -> dict[str, float]:
    """Computes the nineteen base measures of one region of interest.

    Args:
        signal (np.ndarray): Channel-averaged signal of the region, shape (n_samples,).
        sfreq (float): Sampling frequency in Hz.

    Returns:
        dict: Measure name (without region suffix) to value. Measures that cannot be
            estimated come back as NaN rather than as a silent zero.
    """
    freqs, psd = welch(signal, fs=sfreq, nperseg=PSD_NPERSEG)
    out: dict[str, float] = {}

    band_power = {b: _bandpower(freqs, psd, lo, hi) for b, (lo, hi) in BANDS.items()}
    total = _bandpower(freqs, psd, ANALYSIS_LO, ANALYSIS_HI)
    out["total_power"] = total
    for band in BANDS:
        out[f"rel_{band}"] = band_power[band] / (total + EPS)

    out["theta_alpha"] = band_power["theta"] / (band_power["alpha"] + EPS)
    out["delta_theta"] = band_power["delta"] / (band_power["theta"] + EPS)
    out["delta_alpha"] = band_power["delta"] / (band_power["alpha"] + EPS)
    out["low_high"] = ((band_power["delta"] + band_power["theta"])
                       / (band_power["alpha"] + band_power["beta"] + EPS))

    out["paf_freq"], out["paf_power"] = find_paf(freqs, psd)

    fit = extract_specparam_features(
        freqs, psd, freq_range=list(APSD_FREQ_RANGE),
        aperiodic_mode=APSD_APERIODIC_MODE,
    )
    out["apsd_slope"] = fit.get("aperiodic_exponent", np.nan)
    out["apsd_offset"] = fit.get("aperiodic_offset", np.nan)
    out["apsd_r2"] = fit.get("aperiodic_r2", np.nan)
    for band in BANDS:
        out[f"osc_{band}"] = fit.get(f"periodic_bp_{band}", np.nan)
    out["sp_alpha_cf"] = fit.get("alpha_peak_freq", np.nan)
    out["sp_alpha_pw"] = fit.get("alpha_peak_power", np.nan)
    return out


def compute_window_features(
    X: np.ndarray,
    roi_indices: dict[str, list[int]],
    *,
    sfreq: float = SFREQ,
    progress: int = 200,
) -> pd.DataFrame:
    """Computes the full catalogue for every window.

    Args:
        X (np.ndarray): Windows of shape (n_windows, n_channels, n_samples); may be a memmap.
        roi_indices (dict): Mapping from region name to channel indices.
        sfreq (float): Sampling frequency in Hz.
        progress (int): Print progress every N windows; 0 disables.

    Returns:
        pd.DataFrame: One row per window, columns named ``<measure>_<roi>`` plus the two
            topographic contrasts.

    Raises:
        KeyError: If a region of interest has no channels assigned.
    """
    empty = [roi for roi in ROIS if not roi_indices.get(roi)]
    if empty:
        raise KeyError(f"Regions of interest without channels: {empty}")

    rows: list[dict[str, float]] = []
    for i in range(X.shape[0]):
        if progress and i % progress == 0:
            print(f"  {i}/{X.shape[0]}...", flush=True)
        window = np.asarray(X[i], dtype=np.float64)
        feats: dict[str, float] = {}
        for roi in ROIS:
            for key, value in roi_measures(window[roi_indices[roi]].mean(axis=0), sfreq).items():
                feats[f"{key}_{roi}"] = value
        feats["paf_central_minus_occipital"] = feats["paf_freq_central"] - feats["paf_freq_occipital"]
        feats["paf_central_minus_parietal"] = feats["paf_freq_central"] - feats["paf_freq_parietal"]
        rows.append(feats)
    return pd.DataFrame(rows)


def build_descriptor(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Selects the descriptor columns and checks they describe the given windows.

    Args:
        features (pd.DataFrame): One row per window, in window order.
        meta (pd.DataFrame): Window metadata, in the same order.
        columns (list): Descriptor column names.

    Returns:
        tuple: (matrix of shape (len(meta), len(columns)), boolean mask of complete rows).

    Raises:
        DescriptorAlignmentError: If the feature table does not have exactly one row per
            window, or if the metadata key is not unique.
        KeyError: If a descriptor column is missing from the feature table.
    """
    if len(features) != len(meta):
        raise DescriptorAlignmentError(
            f"{len(features)} feature rows for {len(meta)} windows: the descriptor is "
            "indexed by position, so both must match exactly."
        )
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise KeyError(f"{len(missing)} descriptor columns absent: {missing[:5]}...")

    key = [c for c in ("subject", "age", "epoch_index") if c in meta.columns]
    if len(key) == 3 and meta.duplicated(key).any():
        raise DescriptorAlignmentError(
            "(subject, age, epoch_index) is not unique in the metadata, so a row of the "
            "descriptor cannot be traced back to a single window."
        )

    matrix = features[columns].to_numpy(dtype=np.float32)
    complete = ~np.isnan(matrix).any(axis=1)
    return matrix, complete


def main(args: argparse.Namespace) -> None:
    """Builds the descriptor and writes it next to its metadata.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    X = np.load(args.raw_path, mmap_mode="r")
    meta = pd.read_csv(args.meta_path)
    print(f"[INFO] {X.shape[0]} windows, {X.shape[1]} channels, {X.shape[2]} samples")

    spread = check_amplitude_is_physical(X)
    print(f"[INFO] median channel standard deviation: {spread:.3e}")

    channels = load_channel_names(args.channels_txt)
    indices = roi_indices(channels, X.shape[1])
    print(f"[INFO] channels per region: { {r: len(i) for r, i in indices.items()} }")

    features = compute_window_features(X, indices, sfreq=args.sfreq)
    columns = DESCRIPTORS[args.descriptor]
    matrix, complete = build_descriptor(features, meta, columns)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, f"{args.descriptor}.npy"), matrix)
    features.to_parquet(os.path.join(args.output_dir, "window_features.parquet"), index=False)
    with open(os.path.join(args.output_dir, f"{args.descriptor}_columns.json"), "w") as fh:
        json.dump({
            "descriptor": args.descriptor,
            "columns": columns,
            "n_windows": int(matrix.shape[0]),
            "n_complete": int(complete.sum()),
            "bands": {b: list(lim) for b, lim in BANDS.items()},
            "analysis_range": [ANALYSIS_LO, ANALYSIS_HI],
            "apsd_freq_range": list(APSD_FREQ_RANGE),
            "apsd_aperiodic_mode": APSD_APERIODIC_MODE,
            "nperseg": PSD_NPERSEG,
        }, fh, indent=2, sort_keys=True)

    print(f"[INFO] {args.descriptor}: {matrix.shape[0]} x {matrix.shape[1]}, "
          f"{int(complete.sum())} complete rows "
          f"({100 * complete.mean():.1f}%)")
    for name, cols in sorted(DESCRIPTORS.items()):
        dense = features[cols].notna().all(axis=1).mean()
        print(f"[INFO] {name}: {len(cols)} columns, {100 * dense:.1f}% complete rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the expert spectral descriptor from the broadband recording."
    )
    parser.add_argument("--raw_path", type=str,
                        default="data/raw/all_epochs_raw.npy",
                        help="Broadband recording (n_windows, n_channels, n_samples), with "
                             "its amplitude untouched.")
    parser.add_argument("--meta_path", type=str,
                        default="data/processed/all_all/processed_metadata.csv",
                        help="Window metadata, one row per window in the same order.")
    parser.add_argument("--channels_txt", type=str, default="data/raw/channel_names.txt",
                        help="Channel names, used to map channels to regions of interest.")
    parser.add_argument("--output_dir", type=str, default="data/processed/expert_features",
                        help="Directory for the descriptor matrix and its column manifest.")
    parser.add_argument("--descriptor", type=str, default="P_full",
                        choices=sorted(DESCRIPTORS),
                        help="Which descriptor to materialise.")
    parser.add_argument("--sfreq", type=float, default=SFREQ,
                        help="Sampling frequency in Hz.")
    main(parser.parse_args())
