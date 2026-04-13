import numpy as np
import pandas as pd
import argparse
import os
from mne.filter import filter_data

PRESET_ZONES = {
    "central" : ['E35', 'E29', 'E13', 'E6', 'E112', 'E111', 'E110', 'E41', 'E36', 'E30', 'E7', 'E106', 'E105', 'E104', 'E103', 'E47', 'E42', 'E37', 'E31', 'Cz', 'E80', 'E87', 'E93', 'E98', 'E54', 'E55', 'E79'],
    "frontal" : ['E33', 'E34', 'E27', 'E23', 'E18', 'E16', 'E10', 'E3', 'E123', 'E116', 'E122', 'E28', 'E24', 'E19', 'E11', 'E4', 'E124', 'E117', 'E20', 'E12','E5', 'E118' ],
    "parietal" : ['E52', 'E53', 'E61', 'E62', 'E78', 'E86', 'E60', 'E67', 'E72', 'E77', 'E85', 'E92', 'E59', 'E91'],
    "occipital" : ['E66', 'E71', 'E76', 'E84', 'E70', 'E75', 'E83']  
}

# Metadata map
COLUMN_MAP = {
    "Sexo": "sex",
    "Z_Income_6mo": "inc",
    "Z_Income_to_need_6mo": "inc_need",
    "Z_Niv_Edu_6mo": "edu",
    "Z_Cla_Lab_6mo": "lab",
    "Z_SES_Index_6mo": "ses",
    "IBQ_ES_9mo": "ibq_es",
    "IBQ_NA_9mo": "ibq_na",
    "IBQ_EC_9mo": "ibq_ec",
    "CBQ_EC_36mo": "cbq_ec",
    "CBQ_ES_36mo": "cbq_es",
    "CBQ_NA_36mo": "cbq_na",
    "IQ_ICV_48mo": "icv_48mo",
    "IQ_ICV_Perc_48mo": "icv_perc_48mo",
    "IQ_IVE_48mo": "ive_48mo",
    "IQ_IVE_Perc_48mo": "ive_perc_48mo",
    "IQ_IRF_48mo": "irf_48mo",
    "IQ_IRF_Perc_48mo": "irf_perc_48mo",
    "IQ_IMT_48mo": "imt_48mo",
    "IQ_IMT_Perc_48mo": "imt_perc_48mo",
    "IQ_CIT_48mo": "cit_48mo",
    "IQ_CIT_Perc_48mo": "cit_perc_48mo",
    "IQ_ICV_36mo": "icv_36mo",
    "IQ_ICV_Perc_36mo": "icv_perc_36mo",
    "IQ_IVE_36mo": "ive_36mo",
    "IQ_IVE_Perc_36mo": "ive_perc_36mo",
    "IQ_IRF_36mo": "irf_36mo",
    "IQ_IRF_Perc_36mo": "irf_perc_36mo",
    "IQ_IMT_36mo": "imt_36mo",
    "IQ_IMT_Perc_36mo": "imt_perc_36mo",
    "IQ_CIT_36mo": "cit_36mo",
    "IQ_CIT_Perc_36mo": "cit_perc_36mo",
}

def get_channel_indices(zones, all_channels):
    selected = set()
    for zone in zones:
        if zone not in PRESET_ZONES:
            raise ValueError(f"Zone {zone} not found. Available zones: {list(PRESET_ZONES.keys())}")
        selected.update(PRESET_ZONES[zone])
    return [i for i, ch in enumerate(all_channels) if ch in selected]

def apply_windows(X, meta, sfreq, window_sec):
    samples_per_window = int(window_sec * sfreq)
    all_windows = []
    new_meta = []

    for i, epoch_data in enumerate(X):  # epoch_data shape: (n_channels, n_times)
        start_times = list(range(0, epoch_data.shape[1], samples_per_window))
        for start in start_times:
            end = start + samples_per_window
            if end > epoch_data.shape[1]:
                continue  # skip incomplete window

            window = epoch_data[:, start:end].astype(np.float32)

            row = meta.iloc[i]
            markers = row["window_markers"].split("/") if "window_markers" in row else []
            quality = sum(1 for m in markers if m != "9999") / len(markers) if markers else 0
            new_row = row.to_dict()  # Copy all metadata row

            # Update specific 
            new_row.update({
                "window_quality": quality,
                "window_markers": row.get("window_markers", "unknown")
            })

            all_windows.append(window)
            new_meta.append(new_row)

    return np.stack(all_windows), pd.DataFrame(new_meta)


def main(args):
    X = np.load(args.data_path)
    meta = pd.read_csv(args.meta_path)
    socio_meta = pd.read_csv(args.socio_path).rename(columns={"ID": "subject"} | COLUMN_MAP)

    # Merge both metadata dataframes
    meta = meta.merge(socio_meta, on="subject", how="left")

    with open(args.channels_txt) as f:
        all_channels = [line.strip() for line in f.readlines()]

    # Channel selection
    if args.zones:
        indices = get_channel_indices(args.zones, all_channels)
        X = X[:, indices, :]

    # Frequency filtering
    SFREQ = X.shape[-1] / args.seconds 
    if args.l_freq is not None and args.h_freq is not None:
        X = filter_data(
            X,
            sfreq=SFREQ,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            verbose=True
        )

    # Standarization with mean, std from all signal per channel
    mean_ch = X.mean(axis=(0, 2), keepdims=True)  # shape: (1, n_channels, 1)
    std_ch = X.std(axis=(0, 2), keepdims=True)    # shape: (1, n_channels, 1)
    X = (X - mean_ch) / (std_ch + 1e-12)

    # Window extraction
    X_win, meta_win = apply_windows(X, meta, SFREQ, args.seconds)

    # Save processed data to arg output path
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    np.save(os.path.join(args.output_path, "processed_windows.npy"), X_win)
    meta_win.to_csv(os.path.join(args.output_path, "processed_metadata.csv"), index=False)  

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Process EEG data."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the EEG data file."
    )

    parser.add_argument(
        "--meta_path",
        type=str,
        required=True,
        help="Path to the metadata file."
    )

    parser.add_argument(
        "--socio_path",
        type=str,
        required=True,
        help="Path to the sociological metadata file."
    )

    parser.add_argument(
        "--channels_txt",
        type=str,
        required=True,
        help="Path to the channels text file."
    )

    parser.add_argument(
        "--l_freq",
        type=float,
        default=0.2,
        help="Low cutoff frequency for bandpass filter."
    )

    parser.add_argument(
        "--h_freq",
        type=float,
        default=48.0,
        help="High cutoff frequency for bandpass filter."
    )

    parser.add_argument(
        "--zones",
        nargs="+",
        help="Zones to include, e.g., central occipital"
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Seconds to build windows"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/5_s",
        help="Output path for processed data."
    )

    args = parser.parse_args()
    main(args)