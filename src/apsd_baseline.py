"""
APSD Baseline: Aperiodic-Periodic Spectral Descriptor for EEG

This script computes spectral features from resting-state EEG using specparam/FOOOF
and evaluates them with Ridge regression using the same cross-validation strategy
as run_downstream.py.

Features per ROI:
- Aperiodic: exponent, offset (2)
- Periodic bandpower: delta, theta, alpha, beta, gamma (5)
- Alpha peak: frequency, power (2)
Total: 9 features × 4 ROIs = 36 features

Usage:
    python src/apsd_baseline.py --targets age cit_36mo --cv_strategy kfold --n_folds 10
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch
from specparam import SpectralModel
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# CONFIGURATION
# =============================================================================

# ROI definitions (from postprocessing.py)
PRESET_ZONES = {
    "central": ['E35', 'E29', 'E13', 'E6', 'E112', 'E111', 'E110', 'E41', 'E36', 'E30',
                'E7', 'E106', 'E105', 'E104', 'E103', 'E47', 'E42', 'E37', 'E31', 'Cz',
                'E80', 'E87', 'E93', 'E98', 'E54', 'E55', 'E79'],
    "frontal": ['E33', 'E34', 'E27', 'E23', 'E18', 'E16', 'E10', 'E3', 'E123', 'E116',
                'E122', 'E28', 'E24', 'E19', 'E11', 'E4', 'E124', 'E117', 'E20', 'E12',
                'E5', 'E118'],
    "parietal": ['E52', 'E53', 'E61', 'E62', 'E78', 'E86', 'E60', 'E67', 'E72', 'E77',
                 'E85', 'E92', 'E59', 'E91'],
    "occipital": ['E66', 'E71', 'E76', 'E84', 'E70', 'E75', 'E83']
}

# Frequency bands (user-specified)
FREQ_BANDS = {
    "delta": (0.5, 3),
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_channel_names(channels_txt):
    """Load channel names from text file."""
    with open(channels_txt) as f:
        return [line.strip() for line in f.readlines()]


def get_roi_channel_indices(all_channels):
    """
    Get channel indices for each ROI.
    Returns dict: {roi_name: [indices]}
    """
    roi_indices = {}
    for roi_name, roi_channels in PRESET_ZONES.items():
        indices = [i for i, ch in enumerate(all_channels) if ch in roi_channels]
        roi_indices[roi_name] = indices
    return roi_indices


def compute_psd(signal, sfreq, nperseg=None):
    """
    Compute Power Spectral Density using Welch method.

    Args:
        signal: 1D array of shape (n_samples,)
        sfreq: Sampling frequency in Hz
        nperseg: Length of each segment for Welch. Default: sfreq (1 second)

    Returns:
        freqs: Frequency bins
        psd: Power spectral density
    """
    if nperseg is None:
        nperseg = int(sfreq * 5)  # 2 second segments

    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg, noverlap=nperseg//2)
    return freqs, psd


def compute_roi_psd(window_data, roi_indices, sfreq):
    """
    Compute average PSD for each ROI.

    Args:
        window_data: 2D array of shape (n_channels, n_samples)
        roi_indices: Dict of {roi_name: [channel_indices]}
        sfreq: Sampling frequency

    Returns:
        Dict of {roi_name: (freqs, avg_psd)}
    """
    roi_psds = {}
    for roi_name, indices in roi_indices.items():
        if len(indices) == 0:
            continue

        # Compute PSD for each channel in ROI
        psds = []
        for idx in indices:
            freqs, psd = compute_psd(window_data[idx], sfreq)
            psds.append(psd)

        # Average PSD across channels in ROI
        avg_psd = np.mean(psds, axis=0)
        roi_psds[roi_name] = (freqs, avg_psd)

    return roi_psds


def extract_specparam_features(freqs, psd, freq_range=SPECPARAM_FREQ_RANGE):
    """
    Extract aperiodic and periodic features using specparam.

    Returns:
        dict with keys:
            - aperiodic_exponent
            - aperiodic_offset
            - periodic_bp_<band> for each band
            - alpha_peak_freq
            - alpha_peak_power

    Note: Compatible with specparam 2.0 API.
    """
    features = {}

    # Initialize specparam model
    sm = SpectralModel(
        peak_width_limits=SPECPARAM_PEAK_WIDTH_LIMITS,
        max_n_peaks=SPECPARAM_MAX_N_PEAKS,
        min_peak_height=SPECPARAM_MIN_PEAK_HEIGHT,
        peak_threshold=SPECPARAM_PEAK_THRESHOLD,
        aperiodic_mode='fixed'
    )

    try:
        # Fit the model
        sm.fit(freqs, psd, freq_range)

        # Aperiodic parameters (specparam 2.0 API)
        aperiodic_params = sm.results.params.aperiodic.params
        features['aperiodic_offset'] = aperiodic_params[0]
        features['aperiodic_exponent'] = aperiodic_params[1]

        # Get model frequencies and data spectrum
        model_freqs = sm.data.freqs
        data_spectrum = sm.data.power_spectrum

        # Compute aperiodic fit to get flattened spectrum
        offset, exponent = aperiodic_params[0], aperiodic_params[1]
        aperiodic_fit = offset - np.log10(model_freqs ** exponent)
        flat_spec = data_spectrum - aperiodic_fit

        # Compute bandpower from the periodic (flattened) spectrum
        for band_name, (low, high) in FREQ_BANDS.items():
            band_mask = (model_freqs >= low) & (model_freqs <= high)
            if np.any(band_mask):
                band_power = np.sum(flat_spec[band_mask])
                features[f'periodic_bp_{band_name}'] = band_power
            else:
                features[f'periodic_bp_{band_name}'] = 0.0

        # Alpha peak detection (specparam 2.0 API)
        # peaks format: [[CF, PW, BW], ...] where indices are cf=0, pw=1, bw=2
        peaks = sm.results.params.periodic.params
        alpha_low, alpha_high = ALPHA_PEAK_RANGE

        if len(peaks) > 0:
            alpha_peaks = [p for p in peaks if alpha_low <= p[0] <= alpha_high]

            if len(alpha_peaks) > 0:
                # Take the strongest peak in alpha range
                alpha_peaks = sorted(alpha_peaks, key=lambda x: x[1], reverse=True)
                features['alpha_peak_freq'] = alpha_peaks[0][0]
                features['alpha_peak_power'] = alpha_peaks[0][1]
            else:
                features['alpha_peak_freq'] = np.nan
                features['alpha_peak_power'] = np.nan
        else:
            features['alpha_peak_freq'] = np.nan
            features['alpha_peak_power'] = np.nan

    except Exception:
        # If fitting fails, return NaN for all features
        features['aperiodic_offset'] = np.nan
        features['aperiodic_exponent'] = np.nan
        for band_name in FREQ_BANDS.keys():
            features[f'periodic_bp_{band_name}'] = np.nan
        features['alpha_peak_freq'] = np.nan
        features['alpha_peak_power'] = np.nan

    return features


def extract_window_features(window_data, roi_indices, sfreq):
    """
    Extract all APSD features for a single window.

    Args:
        window_data: 2D array of shape (n_channels, n_samples)
        roi_indices: Dict of {roi_name: [channel_indices]}
        sfreq: Sampling frequency

    Returns:
        dict of features with keys like 'APSD_<ROI>_<feature_name>'
    """
    features = {}

    # Compute PSD for each ROI
    roi_psds = compute_roi_psd(window_data, roi_indices, sfreq)

    # Extract specparam features for each ROI
    for roi_name, (freqs, psd) in roi_psds.items():
        roi_features = extract_specparam_features(freqs, psd)

        # Add ROI prefix to feature names
        for feat_name, feat_value in roi_features.items():
            features[f'APSD_{roi_name}_{feat_name}'] = feat_value

    return features


def extract_all_features(X, roi_indices, sfreq, verbose=True):
    """
    Extract APSD features for all windows.

    Args:
        X: 3D array of shape (n_windows, n_channels, n_samples)
        roi_indices: Dict of {roi_name: [channel_indices]}
        sfreq: Sampling frequency

    Returns:
        DataFrame with features for each window
    """
    all_features = []
    n_windows = X.shape[0]

    for i in range(n_windows):
        if verbose and i % 100 == 0:
            print(f"  Processing window {i+1}/{n_windows}...", flush=True)

        features = extract_window_features(X[i], roi_indices, sfreq)
        all_features.append(features)

    return pd.DataFrame(all_features)


def aggregate_subject_features(features_df, meta_df, method='mean'):
    """
    Aggregate window-level features to subject-level.

    Args:
        features_df: DataFrame with window-level features
        meta_df: DataFrame with metadata including 'subject' column
        method: Aggregation method ('mean', 'median')

    Returns:
        DataFrame with subject-level features
    """
    # Combine features with metadata
    combined = pd.concat([features_df, meta_df[['subject']].reset_index(drop=True)], axis=1)

    # Aggregate by subject
    if method == 'mean':
        subject_features = combined.groupby('subject').mean()
    elif method == 'median':
        subject_features = combined.groupby('subject').median()
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    return subject_features.reset_index()


# =============================================================================
# CROSS-VALIDATION AND EVALUATION
# =============================================================================

def run_ridge_cv(X_train, y_train, X_test, y_test, alphas=None):
    """
    Train Ridge regression with cross-validation for alpha selection.

    Returns:
        y_pred: Predictions on test set
        best_alpha: Selected alpha value
        model: Trained model
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 7)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Ridge with CV
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    return y_pred, model.alpha_, model, scaler


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    std_y = np.std(y_true)
    nrmse = rmse / std_y if std_y > 1e-8 else rmse

    return {
        'RMSE': rmse,
        'R2': r2,
        'nRMSE': nrmse
    }


def run_cv_evaluation(subject_features_df, meta_df, target_col, cv_strategy='kfold',
                      n_folds=10, base_seed=1234, fold_range=None, fold_ranges_dict=None):
    """
    Run cross-validation evaluation matching run_downstream.py logic.
    """
    # Get feature columns (exclude 'subject')
    feature_cols = [c for c in subject_features_df.columns if c.startswith('APSD_')]

    # Merge features with targets
    subject_targets = meta_df.groupby('subject')[target_col].first().reset_index()
    df = subject_features_df.merge(subject_targets, on='subject', how='inner')

    # Filter subjects with valid target
    df = df.dropna(subset=[target_col])

    subjects = df['subject'].values
    X = df[feature_cols].values
    y = df[target_col].values

    # Handle NaN features by imputing with column mean
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = col_means[j]

    print(f"\n  Target: {target_col}")
    print(f"  Subjects with valid target: {len(subjects)}")
    print(f"  Features: {len(feature_cols)}")

    results = []
    all_subject_preds = []

    if cv_strategy == 'kfold':
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
        cv_splits = list(kf.split(subjects))

        if fold_range:
            start, end = fold_range
            cv_splits = cv_splits[start:end]
            fold_offset = start
        else:
            fold_offset = 0

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            actual_fold = fold_idx + fold_offset

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            train_subjects = subjects[train_idx]
            test_subjects = subjects[test_idx]

            print(f"\n    FOLD {actual_fold+1}: Train subjects ({len(train_subjects)}): {list(train_subjects)}", flush=True)
            print(f"             Test subjects ({len(test_subjects)}): {list(test_subjects)}", flush=True)

            y_pred, best_alpha, model, scaler = run_ridge_cv(X_train, y_train, X_test, y_test)
            metrics = compute_metrics(y_test, y_pred)

            results.append({
                'fold': actual_fold,
                'target': target_col,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'best_alpha': best_alpha,
                **metrics
            })

            # Store subject-level predictions
            for subj, yt, yp in zip(test_subjects, y_test, y_pred):
                all_subject_preds.append((subj, yt, yp))

            print(f"    Results: RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}, nRMSE={metrics['nRMSE']:.3f}")

    elif cv_strategy == 'leave_one_out':
        loo = LeaveOneOut()
        cv_splits = list(loo.split(subjects))

        if fold_ranges_dict and target_col in fold_ranges_dict:
            start, end = fold_ranges_dict[target_col]
            cv_splits = cv_splits[start:end]
            fold_offset = start
        else:
            fold_offset = 0

        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_subject = subjects[test_idx[0]]

            y_pred, best_alpha, model, scaler = run_ridge_cv(X_train, y_train, X_test, y_test)

            all_y_true.append(y_test[0])
            all_y_pred.append(y_pred[0])
            all_subject_preds.append((test_subject, y_test[0], y_pred[0]))

            if (fold_idx + 1) % 10 == 0:
                print(f"    Completed {fold_idx+1}/{len(cv_splits)} LOO folds...", flush=True)

        # Compute overall metrics for LOO
        metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))
        results.append({
            'fold': 'all',
            'target': target_col,
            'n_subjects': len(all_y_true),
            **metrics
        })

        print(f"    LOO Overall: RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}, nRMSE={metrics['nRMSE']:.3f}")

    return pd.DataFrame(results), all_subject_preds


# =============================================================================
# MAIN
# =============================================================================

def parse_fold_ranges(fold_ranges_arg):
    """Parse fold ranges from command line argument."""
    if fold_ranges_arg is None:
        return None

    fold_ranges_dict = {}
    for range_str in fold_ranges_arg:
        parts = range_str.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid fold_range format: {range_str}. Expected 'target:start:end'")
        target, start, end = parts
        fold_ranges_dict[target] = (int(start), int(end))

    return fold_ranges_dict


def main(args):
    print("=" * 80)
    print("APSD BASELINE: Aperiodic-Periodic Spectral Descriptor")
    print("=" * 80)

    # Load data
    print(f"\n[1] Loading data...")
    X = np.load(args.data_path)
    meta = pd.read_csv(args.meta_path)

    print(f"  Data shape: {X.shape}")
    print(f"  Metadata shape: {meta.shape}")
    print(f"  Subjects: {meta['subject'].nunique()}")

    # Load channel names
    all_channels = load_channel_names(args.channels_txt)
    print(f"  Channels in file: {len(all_channels)}")

    # Determine actual channels in data
    n_channels_data = X.shape[1]
    if n_channels_data != len(all_channels):
        print(f"  [WARNING] Data has {n_channels_data} channels but file has {len(all_channels)}")
        print(f"  Using first {n_channels_data} channels from file")
        all_channels = all_channels[:n_channels_data]

    # Get ROI indices
    roi_indices = get_roi_channel_indices(all_channels)
    for roi, indices in roi_indices.items():
        print(f"  ROI '{roi}': {len(indices)} channels")

    # Compute sampling frequency
    sfreq = X.shape[-1] / args.seconds
    print(f"  Sampling frequency: {sfreq} Hz")

    # Check if precomputed features exist
    features_cache_path = os.path.join(args.save_dir, 'apsd_features_cache.csv')

    if args.use_cache and os.path.exists(features_cache_path):
        print(f"\n[2] Loading cached features from {features_cache_path}")
        features_df = pd.read_csv(features_cache_path)
    else:
        # Extract APSD features
        print(f"\n[2] Extracting APSD features...")
        features_df = extract_all_features(X, roi_indices, sfreq, verbose=True)

        # Save cache
        os.makedirs(args.save_dir, exist_ok=True)
        features_df.to_csv(features_cache_path, index=False)
        print(f"  Features cached to: {features_cache_path}")

    print(f"  Features shape: {features_df.shape}")

    # Aggregate to subject level
    print(f"\n[3] Aggregating features to subject level...")
    subject_features = aggregate_subject_features(features_df, meta, method=args.aggregation)
    print(f"  Subject features shape: {subject_features.shape}")

    # Run cross-validation for each target
    print(f"\n[4] Running cross-validation...")
    print(f"  CV strategy: {args.cv_strategy}")
    print(f"  Targets: {args.targets}")

    all_results = []
    all_predictions = {}

    for target in args.targets:
        print(f"\n{'='*60}")
        print(f"Target: {target}")
        print(f"{'='*60}")

        results_df, subject_preds = run_cv_evaluation(
            subject_features_df=subject_features,
            meta_df=meta,
            target_col=target,
            cv_strategy=args.cv_strategy,
            n_folds=args.n_folds,
            base_seed=args.base_seed,
            fold_range=args.fold_range,
            fold_ranges_dict=parse_fold_ranges(args.fold_ranges) if args.fold_ranges else None
        )

        results_df['method'] = 'APSD_Ridge'
        all_results.append(results_df)
        all_predictions[target] = subject_preds

    # Combine results
    final_results = pd.concat(all_results, ignore_index=True)

    # Compute aggregated statistics
    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS")
    print(f"{'='*80}")

    agg_results = []
    for target in args.targets:
        target_results = final_results[final_results['target'] == target]

        if args.cv_strategy == 'kfold':
            agg = {
                'target': target,
                'method': 'APSD_Ridge',
                'RMSE_mean': target_results['RMSE'].mean(),
                'RMSE_std': target_results['RMSE'].std(),
                'R2_mean': target_results['R2'].mean(),
                'R2_std': target_results['R2'].std(),
                'nRMSE_mean': target_results['nRMSE'].mean(),
                'nRMSE_std': target_results['nRMSE'].std(),
            }

            # Compute global R2 from all subject predictions
            preds = all_predictions[target]
            y_true = np.array([p[1] for p in preds])
            y_pred = np.array([p[2] for p in preds])
            agg['R2_global'] = r2_score(y_true, y_pred)
            rmse_global = np.sqrt(np.mean((y_true - y_pred) ** 2))
            agg['nRMSE_global'] = rmse_global / np.std(y_true) if np.std(y_true) > 1e-8 else rmse_global

        else:  # leave_one_out
            agg = {
                'target': target,
                'method': 'APSD_Ridge',
                'RMSE': target_results['RMSE'].iloc[0],
                'R2': target_results['R2'].iloc[0],
                'nRMSE': target_results['nRMSE'].iloc[0],
            }

        agg_results.append(agg)

        print(f"\n{target}:")
        for k, v in agg.items():
            if k not in ['target', 'method']:
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

    agg_df = pd.DataFrame(agg_results)

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate suffix for filenames
    if args.cv_strategy == 'kfold' and args.fold_range:
        start, end = args.fold_range
        fold_suffix = f"_folds{start}-{end-1}"
    elif args.fold_ranges:
        fold_ranges_dict = parse_fold_ranges(args.fold_ranges)
        parts = [f"{t}_{s}-{e-1}" for t, (s, e) in sorted(fold_ranges_dict.items())]
        fold_suffix = "_" + "_".join(parts)
    else:
        fold_suffix = ""

    raw_path = os.path.join(args.save_dir, f'apsd_raw_results_{args.cv_strategy}{fold_suffix}.csv')
    final_results.to_csv(raw_path, index=False)
    print(f"\n[INFO] Raw results saved to: {raw_path}")

    agg_path = os.path.join(args.save_dir, f'apsd_agg_results_{args.cv_strategy}{fold_suffix}.csv')
    agg_df.to_csv(agg_path, index=False)
    print(f"[INFO] Aggregated results saved to: {agg_path}")

    # Save subject-level predictions
    for target, preds in all_predictions.items():
        preds_df = pd.DataFrame(preds, columns=['subject', 'y_true', 'y_pred'])
        preds_path = os.path.join(args.save_dir, f'apsd_predictions_{target}_{args.cv_strategy}{fold_suffix}.csv')
        preds_df.to_csv(preds_path, index=False)
        print(f"[INFO] Predictions for {target} saved to: {preds_path}")

    print(f"\n{'='*80}")
    print("APSD BASELINE FINISHED SUCCESSFULLY")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="APSD Baseline: Spectral features + Ridge regression for EEG prediction tasks."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/5_s/processed_windows.npy",
        help="Path to processed EEG windows (.npy file)."
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="data/processed/5_s/processed_metadata.csv",
        help="Path to metadata CSV file."
    )
    parser.add_argument(
        "--channels_txt",
        type=str,
        default="data/raw/channel_names.txt",
        help="Path to channel names text file."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["age", "cit_36mo"],
        help="Target variables for regression."
    )
    parser.add_argument(
        "--cv_strategy",
        type=str,
        default="kfold",
        choices=["kfold", "leave_one_out"],
        help="Cross-validation strategy."
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of folds for k-fold CV."
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--fold_range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Process specific fold range for kfold CV."
    )
    parser.add_argument(
        "--fold_ranges",
        nargs="+",
        type=str,
        default=None,
        metavar="TARGET:START:END",
        help="Process specific fold ranges per target (for LOO)."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save/downstream_results",
        help="Directory to save results."
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Window duration in seconds (to compute sfreq)."
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Method for aggregating window features to subject level."
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cached features if available."
    )

    args = parser.parse_args()
    main(args)