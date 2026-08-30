"""APSD baseline: the aperiodic-periodic spectral descriptor regressed against a target.

This is the interpretable floor of the project: how much of a child's development a handful
of specparam measures already carry, with a ridge and no network involved. A representation
that cannot beat it has not earned its complexity.

Features per region: aperiodic exponent and offset (2), periodic bandpower over five bands
(5), and the frequency and power of the alpha peak (2). Nine measures over four regions, 36
in all. The knee and the goodness of fit are computed and cached beside them but never
regressed on: they qualify the fit rather than describe the spectrum.

Everything from the feature matrix onwards is ``tabular_baseline``, shared with
``expert_baseline.py``, so the two spectral baselines are evaluated by the same folds, at the
same row unit, with the same anti-leakage discipline.

Two settings changed on 2026-08-28 and the figures moved with them, so numbers produced
before that date do not reproduce:

- The Welch segment is one second, which is what the 1 Hz floor of the delta band assumes and
  what ``epoch_features.PSD_NPERSEG`` uses. It used to be the whole five-second window, which
  made Welch a single periodogram with no averaging at all.
- Band edges are half-open, ``[low, high)``, with only the last band closed. They used to be
  closed on both sides, so the bins at 3, 6, 9 and 20 Hz counted towards two bands each.

Note on comparability with the expert descriptor: this baseline fits 0.5-48 Hz with a fixed
aperiodic slope, while ``build_expert_features.py`` fits 1.5-20 Hz with a knee. That is a
deliberate difference of scope, not an oversight, and it is why the two are separate
baselines. ``--aperiodic_mode knee`` is available for the comparison.

Usage (from the repository root):
    python src/apsd_baseline.py --targets age cit_36mo --cv_strategy kfold --n_folds 10
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import welch
from specparam import SpectralModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from folds import BASE_SEED, N_FOLDS
from montage import (
    PRESET_ZONES,  # noqa: F401  re-exported: the test suite imports it from here.
    resolve_processed_montage,
    roi_indices as montage_roi_indices,
)
from tabular_baseline import BaselineError, evaluate_targets, parse_fold_ranges

# =============================================================================
# CONFIGURATION
# =============================================================================

# ROI definitions live in montage.py, imported above.

METHOD_NAME = "APSD_Ridge"
PREFIX = "apsd"

# Frequency bands. Edges are half-open, [low, high), so a bin belongs to one band only; the
# last band closes at its upper edge so that nothing is dropped at the top.
FREQ_BANDS = {
    # The Welch estimate uses a one-second segment, so the lowest bin above DC
    # sits at 1 Hz: a floor below that describes a resolution the spectrum does
    # not have.
    "delta": (1.0, 3),
    "theta": (3, 6),
    "alpha": (6, 9),
    "beta": (9, 20),
    "gamma": (20, 48)
}

# Specparam settings
SPECPARAM_FREQ_RANGE = [0.5, 48]  # Hz
SPECPARAM_PEAK_WIDTH_LIMITS = [2, 8]  # Min 2 Hz to avoid overfitting noise (>= 2x freq resolution)
SPECPARAM_MAX_N_PEAKS = 6
SPECPARAM_MIN_PEAK_HEIGHT = 0.1
SPECPARAM_PEAK_THRESHOLD = 2.0

# Alpha peak detection range
ALPHA_PEAK_RANGE = (5, 12)  # Hz, broader for infants

# Welch segment length, in seconds. One second gives a 1 Hz resolution and lets Welch average
# several segments of a five-second window instead of returning a single periodogram.
WELCH_SECONDS = 1.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_roi_channel_indices(all_channels):
    """Returns {roi_name: [channel indices]} for a montage of known size."""
    return montage_roi_indices(all_channels, len(all_channels))


def compute_psd(signal, sfreq, nperseg=None):
    """Computes the power spectral density of one channel by Welch's method.

    Args:
        signal (np.ndarray): One channel, shape (n_samples,).
        sfreq (float): Sampling frequency in Hz.
        nperseg (int | None): Segment length in samples. Defaults to one second, which
            gives a 1 Hz resolution and averages several segments per window.

    Returns:
        tuple: (frequency bins, power spectral density).
    """
    if nperseg is None:
        nperseg = int(sfreq * WELCH_SECONDS)
    nperseg = min(int(nperseg), len(signal))

    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, psd


def compute_roi_psd(window_data, roi_indices, sfreq, nperseg=None):
    """Computes the average PSD of each region.

    Args:
        window_data (np.ndarray): One window, shape (n_channels, n_samples).
        roi_indices (dict): Region name to its channel indices.
        sfreq (float): Sampling frequency in Hz.
        nperseg (int | None): Welch segment length in samples.

    Returns:
        dict: Region name to (frequency bins, PSD averaged over its channels).
    """
    roi_psds = {}
    for roi_name, indices in roi_indices.items():
        psds = []
        for idx in indices:
            freqs, psd = compute_psd(window_data[idx], sfreq, nperseg)
            psds.append(psd)

        # Average PSD across channels in ROI
        roi_psds[roi_name] = (freqs, np.mean(psds, axis=0))

    return roi_psds


# Reported per region but not regressed on: they qualify the fit itself.
APSD_FIT_DIAGNOSTICS = ('aperiodic_knee', 'aperiodic_r2')


def band_mask(freqs, low, high, closed_top):
    """Selects the bins of one frequency band.

    Args:
        freqs (np.ndarray): Frequency bins.
        low (float): Lower edge, inclusive.
        high (float): Upper edge, exclusive unless ``closed_top``.
        closed_top (bool): Whether the upper edge belongs to the band.

    Returns:
        np.ndarray: Boolean mask over ``freqs``.
    """
    upper = freqs <= high if closed_top else freqs < high
    return (freqs >= low) & upper


def extract_specparam_features(freqs, psd, freq_range=SPECPARAM_FREQ_RANGE,
                               aperiodic_mode='fixed'):
    """
    Extract aperiodic and periodic features using specparam.

    Returns:
        dict with keys:
            - aperiodic_exponent
            - aperiodic_offset
            - periodic_bp_<band> for each band
            - alpha_peak_freq
            - alpha_peak_power
            - aperiodic_knee (NaN unless aperiodic_mode is 'knee')
            - aperiodic_r2

    The 'knee' mode adds a bend to the aperiodic fit, which infant spectra need when the
    slope is not constant across the whole range; 'fixed' assumes a single slope.

    A failed fit returns the same keys with NaN throughout, so the caller can tell it apart
    by ``aperiodic_offset`` being missing, which never happens on the successful path.

    Note: Compatible with specparam 2.0 API.
    """
    features = {}

    # Initialize specparam model
    sm = SpectralModel(
        peak_width_limits=SPECPARAM_PEAK_WIDTH_LIMITS,
        max_n_peaks=SPECPARAM_MAX_N_PEAKS,
        min_peak_height=SPECPARAM_MIN_PEAK_HEIGHT,
        peak_threshold=SPECPARAM_PEAK_THRESHOLD,
        aperiodic_mode=aperiodic_mode
    )

    try:
        # Fit the model
        sm.fit(freqs, psd, freq_range)

        # Aperiodic parameters (specparam 2.0 API)
        aperiodic_params = sm.results.params.aperiodic.params
        features['aperiodic_offset'] = aperiodic_params[0]
        if aperiodic_mode == 'knee':
            features['aperiodic_knee'] = aperiodic_params[1]
            features['aperiodic_exponent'] = aperiodic_params[2]
        else:
            features['aperiodic_knee'] = np.nan
            features['aperiodic_exponent'] = aperiodic_params[1]
        features['aperiodic_r2'] = float(sm.results.metrics.results['gof_rsquared'])

        # Get model frequencies and data spectrum
        model_freqs = sm.data.freqs
        data_spectrum = sm.data.power_spectrum

        # Compute aperiodic fit to get flattened spectrum
        # Aperiodic model in log-power: 'fixed' is a single slope, 'knee' bends.
        if aperiodic_mode == 'knee':
            aperiodic_fit = features['aperiodic_offset'] - np.log10(
                features['aperiodic_knee'] + model_freqs ** features['aperiodic_exponent'])
        else:
            aperiodic_fit = features['aperiodic_offset'] - np.log10(
                model_freqs ** features['aperiodic_exponent'])
        flat_spec = data_spectrum - aperiodic_fit

        # Compute bandpower from the periodic (flattened) spectrum. The top band closes so
        # that the highest bin is not dropped; the rest are half-open so that no bin is
        # counted towards two bands.
        last_band = list(FREQ_BANDS)[-1]
        for band_name, (low, high) in FREQ_BANDS.items():
            mask = band_mask(model_freqs, low, high, closed_top=band_name == last_band)
            features[f'periodic_bp_{band_name}'] = (
                float(np.sum(flat_spec[mask])) if np.any(mask) else 0.0
            )

        # Alpha peak detection (specparam 2.0 API)
        # peaks format: [[CF, PW, BW], ...] where indices are cf=0, pw=1, bw=2
        peaks = sm.results.params.periodic.params
        alpha_low, alpha_high = ALPHA_PEAK_RANGE

        alpha_peaks = [p for p in peaks if alpha_low <= p[0] <= alpha_high] if len(peaks) else []
        if len(alpha_peaks) > 0:
            # Take the strongest peak in alpha range
            alpha_peaks = sorted(alpha_peaks, key=lambda x: x[1], reverse=True)
            features['alpha_peak_freq'] = alpha_peaks[0][0]
            features['alpha_peak_power'] = alpha_peaks[0][1]
        else:
            features['alpha_peak_freq'] = np.nan
            features['alpha_peak_power'] = np.nan

    except Exception:
        # If fitting fails, return NaN for all features
        features['aperiodic_offset'] = np.nan
        features['aperiodic_exponent'] = np.nan
        features['aperiodic_knee'] = np.nan
        features['aperiodic_r2'] = np.nan
        for band_name in FREQ_BANDS.keys():
            features[f'periodic_bp_{band_name}'] = np.nan
        features['alpha_peak_freq'] = np.nan
        features['alpha_peak_power'] = np.nan

    return features


def extract_window_features(window_data, roi_indices, sfreq, aperiodic_mode='fixed',
                            nperseg=None):
    """Extracts every APSD feature of one window.

    Args:
        window_data (np.ndarray): One window, shape (n_channels, n_samples).
        roi_indices (dict): Region name to its channel indices.
        sfreq (float): Sampling frequency in Hz.
        aperiodic_mode (str): ``fixed`` or ``knee``.
        nperseg (int | None): Welch segment length in samples.

    Returns:
        dict: Features named ``APSD_<roi>_<feature>``.
    """
    features = {}

    roi_psds = compute_roi_psd(window_data, roi_indices, sfreq, nperseg)

    for roi_name, (freqs, psd) in roi_psds.items():
        roi_features = extract_specparam_features(freqs, psd, aperiodic_mode=aperiodic_mode)

        # Add ROI prefix to feature names
        for feat_name, feat_value in roi_features.items():
            features[f'APSD_{roi_name}_{feat_name}'] = feat_value

    return features


def extract_all_features(X, roi_indices, sfreq, aperiodic_mode='fixed', nperseg=None,
                         verbose=True):
    """Extracts the APSD features of every window and reports the fit failures.

    A fit that fails leaves NaN throughout its region. That happens, but it has to be
    visible: a descriptor whose fits failed is a table of medians once imputation is done.

    Args:
        X (np.ndarray): Windows, shape (n_windows, n_channels, n_samples).
        roi_indices (dict): Region name to its channel indices.
        sfreq (float): Sampling frequency in Hz.
        aperiodic_mode (str): ``fixed`` or ``knee``.
        nperseg (int | None): Welch segment length in samples.
        verbose (bool): Whether to report progress.

    Returns:
        pd.DataFrame: One row per window.
    """
    all_features = []
    n_windows = X.shape[0]

    for i in range(n_windows):
        if verbose and i % 100 == 0:
            print(f"  Processing window {i+1}/{n_windows}...", flush=True)

        all_features.append(
            extract_window_features(X[i], roi_indices, sfreq, aperiodic_mode, nperseg)
        )

    frame = pd.DataFrame(all_features)

    offsets = [c for c in frame.columns if c.endswith('_aperiodic_offset')]
    failures = frame[offsets].isna().to_numpy()
    if failures.any():
        print(f"  [WARNING] specparam failed on {int(failures.sum())} of {failures.size} "
              f"region fits ({100 * failures.mean():.1f}%)", flush=True)
    return frame


def feature_columns(features_df):
    """Returns the columns that are regressed on, leaving out the fit diagnostics.

    Args:
        features_df (pd.DataFrame): Per-window features.

    Returns:
        list[str]: Column names to hand to the regressor.
    """
    return [c for c in features_df.columns
            if c.startswith('APSD_') and not c.endswith(APSD_FIT_DIAGNOSTICS)]


# =============================================================================
# FEATURE CACHE
# =============================================================================

def cache_signature(args, X, sfreq, nperseg):
    """Describes the run that produced a feature table.

    The cache is only reusable by a run that would have computed the same numbers, so
    everything that changes them goes into the signature. Without this, a table computed for
    another dataset is silently reused and the features stop describing the windows they are
    lined up against.

    Args:
        args (argparse.Namespace): Parsed arguments.
        X (np.ndarray): The loaded windows.
        sfreq (float): Sampling frequency in Hz.
        nperseg (int): Welch segment length in samples.

    Returns:
        dict: The signature, JSON-serialisable.
    """
    source = os.path.abspath(args.data_path)
    stat = os.stat(source)
    return {
        "data_path": source,
        "data_mtime_ns": stat.st_mtime_ns,
        "data_shape": list(X.shape),
        "sfreq": float(sfreq),
        "nperseg": int(nperseg),
        "aperiodic_mode": args.aperiodic_mode,
        "freq_range": list(SPECPARAM_FREQ_RANGE),
        "bands": {k: list(v) for k, v in FREQ_BANDS.items()},
    }


def cache_paths(save_dir, signature):
    """Returns the paths of a feature cache and its signature sidecar.

    Args:
        save_dir (str): Directory the cache lives in.
        signature (dict): Output of :func:`cache_signature`.

    Returns:
        tuple: (path of the CSV, path of the JSON sidecar).
    """
    digest = hashlib.md5(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    stem = os.path.join(save_dir, f"apsd_features_cache_{digest}")
    return f"{stem}.csv", f"{stem}.json"


def load_cached_features(csv_path, json_path, signature, n_windows):
    """Reads a cached feature table if it describes this run.

    Args:
        csv_path (str): Path of the cached table.
        json_path (str): Path of its signature sidecar.
        signature (dict): Signature this run would produce.
        n_windows (int): Number of windows the table must cover.

    Returns:
        pd.DataFrame | None: The table, or None when there is nothing usable.
    """
    if not (os.path.exists(csv_path) and os.path.exists(json_path)):
        return None
    with open(json_path) as fh:
        if json.load(fh) != signature:
            return None
    frame = pd.read_csv(csv_path)
    if len(frame) != n_windows:
        print(f"[WARNING] {csv_path} holds {len(frame)} rows for {n_windows} windows; "
              "recomputing.", flush=True)
        return None
    return frame


# =============================================================================
# MAIN
# =============================================================================

def main(args):
    """Computes the APSD features and evaluates them against every requested target.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Raises:
        BaselineError: If the features and the metadata describe different datasets, or if
            no target could be evaluated.
    """
    print("=" * 80)
    print("APSD BASELINE: Aperiodic-Periodic Spectral Descriptor")
    print("=" * 80)

    print("\n[1] Loading data...")
    X = np.load(args.data_path)
    meta = pd.read_csv(args.meta_path)

    print(f"  Data shape: {X.shape}")
    print(f"  Metadata shape: {meta.shape}")
    print(f"  Subjects: {meta['subject'].nunique()}")

    if len(X) != len(meta):
        raise BaselineError(
            f"{len(X)} windows against {len(meta)} metadata rows: the features are lined up "
            "by position, so the two have to come from the same run of postprocessing."
        )

    # Montage of the processed dataset. Read, never guessed: the channel
    # selection keeps channels by region membership, so truncating the full
    # montage to the channel count misassigns every region.
    all_channels = resolve_processed_montage(
        args.data_path, X.shape[1], channels_txt=args.channels_txt)
    print(f"  Channels in montage: {len(all_channels)}")

    roi_indices = get_roi_channel_indices(all_channels)
    for roi, indices in roi_indices.items():
        print(f"  ROI '{roi}': {len(indices)} channels")

    sfreq = X.shape[-1] / args.seconds
    nperseg = int(sfreq * args.welch_seconds)
    print(f"  Sampling frequency: {sfreq} Hz")
    print(f"  Welch segment: {args.welch_seconds} s ({nperseg} samples), "
          f"aperiodic mode {args.aperiodic_mode!r}")

    signature = cache_signature(args, X, sfreq, nperseg)
    csv_path, json_path = cache_paths(args.save_dir, signature)

    features_df = None
    if args.use_cache:
        features_df = load_cached_features(csv_path, json_path, signature, len(X))
        if features_df is not None:
            print(f"\n[2] Reusing the features cached in {csv_path}")

    if features_df is None:
        print("\n[2] Extracting APSD features...")
        features_df = extract_all_features(
            X, roi_indices, sfreq, args.aperiodic_mode, nperseg, verbose=True
        )
        os.makedirs(args.save_dir, exist_ok=True)
        features_df.to_csv(csv_path, index=False)
        with open(json_path, "w") as fh:
            json.dump(signature, fh, indent=2, sort_keys=True)
        print(f"  Features cached to: {csv_path}")

    regressed = feature_columns(features_df)
    print(f"  Features shape: {features_df.shape}, {len(regressed)} of them regressed on")

    print("\n[3] Evaluating...")
    _, agg, _ = evaluate_targets(
        features=features_df[regressed],
        meta=meta,
        targets=args.targets,
        prefix=PREFIX,
        save_dir=args.save_dir,
        method_name=METHOD_NAME,
        cv_strategy=args.cv_strategy,
        n_folds=args.n_folds,
        base_seed=args.base_seed,
        fold_range=tuple(args.fold_range) if args.fold_range else None,
        fold_ranges_dict=parse_fold_ranges(args.fold_ranges) or None,
        aggregation=args.aggregation,
    )

    print("\n[4] Summary")
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="APSD Baseline: Spectral features + Ridge regression for EEG prediction tasks."
    )
    parser.add_argument("--data_path", type=str,
                        default="data/processed/all_all/processed_windows.npy",
                        help="Windows with physical amplitude, shape (N, C, T).")
    parser.add_argument("--meta_path", type=str,
                        default="data/processed/all_all/processed_metadata.csv",
                        help="Per-window metadata of those same windows, in the same order.")
    parser.add_argument("--channels_txt", type=str, default=None,
                        help="Channel names of the processed dataset. Resolved from the "
                             "dataset sidecar when omitted.")
    parser.add_argument("--targets", nargs="+", default=["age", "cit_36mo"],
                        help="Targets to evaluate.")
    parser.add_argument("--cv_strategy", type=str, default="kfold",
                        choices=["kfold", "leave_one_out"],
                        help="Cross-validation strategy.")
    parser.add_argument("--n_folds", type=int, default=N_FOLDS,
                        help="Number of folds of the k-fold.")
    parser.add_argument("--base_seed", type=int, default=BASE_SEED,
                        help="Seed of the fold shuffle; the partition of the whole project.")
    parser.add_argument("--fold_range", nargs=2, type=int, default=None,
                        metavar=("START", "END"),
                        help="Half-open range of folds to run, for parallel shards.")
    parser.add_argument("--fold_ranges", nargs="+", type=str, default=None,
                        metavar="TARGET:START:END",
                        help="Half-open range of folds per target, leave-one-out only.")
    parser.add_argument("--save_dir", type=str, default="save/downstream_results",
                        help="Directory the result tables are written to.")
    parser.add_argument("--seconds", type=float, default=5.0,
                        help="Length of a window in seconds, used to derive the sampling "
                             "frequency.")
    parser.add_argument("--welch_seconds", type=float, default=WELCH_SECONDS,
                        help="Welch segment length in seconds. One second gives a 1 Hz "
                             "resolution, which is what the band edges assume.")
    parser.add_argument("--aperiodic_mode", type=str, default="fixed",
                        choices=["fixed", "knee"],
                        help="Aperiodic model. 'fixed' is a single slope over the whole "
                             "range; 'knee' bends, as the expert descriptor does.")
    parser.add_argument("--aggregation", type=str, default="mean",
                        choices=["mean", "median"],
                        help="How the windows of a row are pooled.")
    parser.add_argument("--use_cache", action="store_true",
                        help="Reuse a cached feature table when its signature matches.")
    main(parser.parse_args())
