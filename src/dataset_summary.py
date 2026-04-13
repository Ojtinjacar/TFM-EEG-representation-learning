"""
Dataset summary: sex distribution and windows-per-subject statistics.
"""

import os
import numpy as np
import pandas as pd

# ---------- paths ----------
save_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "5_s"
)
meta_path = os.path.join(save_path, "processed_metadata.csv")

# ---------- load ----------
meta_df = pd.read_csv(meta_path)

# ---------- sex distribution (unique subjects) ----------
subject_sex = meta_df.drop_duplicates(subset="subject")[["subject", "sex"]]
n_boys = (subject_sex["sex"] == 1.0).sum()
n_girls = (subject_sex["sex"] == 2.0).sum()

print("=== Sex distribution (unique subjects) ===")
print(f"Boys   (sex=1): {n_boys}")
print(f"Girls  (sex=2): {n_girls}")
print(f"Total subjects: {n_boys + n_girls}")

# ---------- windows per subject ----------
windows_per_subject = meta_df.groupby("subject").size()

median_windows = np.median(windows_per_subject)
std_windows = np.std(windows_per_subject, ddof=1)

print("\n=== Windows per subject ===")
print(f"Median:           {median_windows:.1f}")
print(f"Std deviation:    {std_windows:.2f}")
print(f"Min:              {windows_per_subject.min()}")
print(f"Max:              {windows_per_subject.max()}")
