import numpy as np 
import pandas as pd
from mne.io import read_epochs_eeglab
import re
import os

DATA_DIR = "../data/original/raw"
TARGET_SFREQ = 250
WINDOW_LENGTH = 5  # seconds (overlap?)

central = ['E35', 'E29', 'E13', 'E6', 'E112', 'E111', 'E110', 'E41', 'E36', 'E30', 'E7', 'E106', 'E105', 'E104', 'E103', 'E47', 'E42', 'E37', 'E31', 'Cz', 'E80', 'E87', 'E93', 'E98', 'E54', 'E55', 'E79']
frontal = ['E33', 'E34', 'E27', 'E23', 'E18', 'E16', 'E10', 'E3', 'E123', 'E116', 'E122', 'E28', 'E24', 'E19', 'E11', 'E4', 'E124', 'E117', 'E20', 'E12','E5', 'E118' ]
parietal = ['E52', 'E53', 'E61', 'E62', 'E78', 'E86', 'E60', 'E67', 'E72', 'E77', 'E85', 'E92', 'E59', 'E91']
occipital = ['E66', 'E71', 'E76', 'E84', 'E70', 'E75', 'E83']  


def extract_metadata_from_filename(filename):
    """Extract subject ID and age in months from filename like 'B010_6mo.set' or 'B010_9m.set'."""
    base = os.path.basename(filename).replace(".set", "")
    match = re.match(r"(B\d+)_([0-9]+)(mo|m)", base, re.IGNORECASE)
    
    if not match:
        raise ValueError(f"Filename format not recognized: {filename}")
    
    subject = match.group(1)
    age = int(match.group(2))
    return subject, age


def extract_block_code(event_label):
    """Extract block code from event label (e.g., '1111/2222/9999' -> 1 or 2)."""
    # Block 1 = Pompas, Block 2 = Video
    labels = event_label.split("/")
    if len(labels) < 3:
        return -1  # UWhen less than 3 labels - invalid block 
    first = event_label.split("/")[0]
    if first.endswith("1"):
        return 1  # Pompas
    elif first.endswith("2"):
        return 2  # Video 
    else:
        second = event_label.split("/")[1]
        if second.endswith("1"):
            return 1
        elif second.endswith("2"):
            return 2
        else:
            return -1  # Unknown block

def process_file(path):
    subject, age = extract_metadata_from_filename(path)
    print(f"Processing: {subject}, Age: {age} months")

    try:
        epochs = read_epochs_eeglab(path, verbose=True)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []
    
    if epochs.info["sfreq"] != TARGET_SFREQ:
        epochs.resample(TARGET_SFREQ)
    
    id_to_label = {v: k for k, v in epochs.event_id.items()}
    labels = [id_to_label[e[-1]] for e in epochs.events]

    valid_indices = [i for i, label in enumerate(labels)] #removed label check, ass now we use all labels

    epochs = epochs[valid_indices]
    labels = [labels[i] for i in valid_indices]

    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_timepoints)
    ch_names = epochs.ch_names

    meta = []
    for i, label in enumerate(labels):
        block = extract_block_code(label)
        window_markers = label.split("/")[:3]
        quality = sum(1 for m in window_markers if m != "9999")
        meta.append({
            "subject": subject,
            "age": age,
            "epoch_index": i,
            "block": block,
            "window_markers": "/".join(window_markers),
            "window_quality": quality
        })

    return data, meta, ch_names


def process_all(data_dir):
    all_data = []
    all_meta = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".set"):
            path = os.path.join(data_dir, fname)
            result = process_file(path)
            if result:
                data, meta, ch_names = result
                all_data.append(data)
                all_meta.extend(meta)

    if not all_data:
        print("No data processed.")
        return

    X = np.concatenate(all_data, axis=0)
    df = pd.DataFrame(all_meta)
    np.save("all_epochs_raw.npy", X)
    df.to_csv("all_epochs_metadata.csv", index=False)
    with open("channel_names.txt", "w") as f:
        f.write("\n".join(ch_names))

if __name__ == "__main__":
    process_all(DATA_DIR)