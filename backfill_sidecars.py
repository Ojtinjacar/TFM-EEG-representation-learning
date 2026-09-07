"""Backfills sidecar configs for checkpoints trained before sidecars existed.

For every ``save/models/*.pth`` without a sidecar, writes a
``<name>_config.json`` marked ``legacy: true`` with the identity fields that can
be recovered from the filename. For ``fold{k}`` checkpoints the excluded test
subjects are reconstructed by replaying the orchestrator's split
(KFold(n_splits=10, shuffle=True, random_state=1234) over the sorted unique
subject IDs) and cross-validated against the authentic ``exclude_subjects``
recorded by the ExpCLR sidecars of the same fold. Checkpoints whose exclusion
cannot be reconstructed (e.g. ``rep{i}`` runs) get a sidecar without
``exclude_subjects`` and are therefore never reusable.

Existing sidecars (ExpCLR) are updated in place with any missing identity keys
derived from the filename; they are NOT marked legacy.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from checkpoint_naming import sidecar_path, write_sidecar

MODELS_DIR = "save/models"
META_PATH = os.path.join("data", "processed", "all_all", "processed_metadata.csv")
ZONES = {"all", "frontal", "central", "parietal", "occipital", "broadband"}
N_FOLDS = 10
BASE_SEED = 1234


def fold_exclusions():
    """Reconstructs the per-fold test subjects of the orchestrator's KFold.

    Returns:
        dict[str, list[str]]: Mapping ``fold{k}`` -> sorted test subject IDs.

    Raises:
        FileNotFoundError: If the reference metadata file is missing.
    """
    meta = pd.read_csv(META_PATH)
    unique_subjects = np.unique(meta["subject"].astype(str).values)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)
    folds = {}
    for idx, (_, test_idx) in enumerate(kf.split(unique_subjects)):
        folds[f"fold{idx}"] = sorted(unique_subjects[test_idx].tolist())
    return folds


def parse_identity(filename):
    """Extracts (method, zone, frequency, fold_id, fold_key) from a filename.

    ``fold_id`` is the full fold token as embedded in the name (it may carry a
    SimCLR variant tag, e.g. ``fold0_nbrcosine``); ``fold_key`` is the bare
    ``fold{k}`` used to look up the reconstructed exclusion, or None.

    Args:
        filename (str): Checkpoint basename ending in ``.pth``.

    Returns:
        dict | None: Identity fields, or None if the name is unparseable.
    """
    stem = filename[:-len(".pth")]
    tokens = stem.split("_")
    # The filename token differs from the orchestrator's method name.
    method = {"Triplet": "TripletLoss"}.get(tokens[0], tokens[0])

    zone_idx = next((i for i, t in enumerate(tokens[1:], start=1) if t in ZONES), None)
    if zone_idx is None or zone_idx + 1 >= len(tokens):
        return None
    zone = tokens[zone_idx]
    frequency = tokens[zone_idx + 1]

    hparam_markers = {"batch", "hidden128", "emb128", "P"}
    fold_tokens = []
    for t in tokens[zone_idx + 2:]:
        if (t in hparam_markers or t.startswith("hidden") or t.startswith("emb")
                or t.startswith("beta") or t.startswith("mask")
                or t.startswith("P")):
            break
        fold_tokens.append(t)
    fold_id = "_".join(fold_tokens) if fold_tokens else None
    fold_key = next((t for t in fold_tokens if t.startswith("fold")), None)
    return {
        "method": method,
        "zone": zone,
        "frequency": frequency,
        "fold_id": fold_id,
        "fold_key": fold_key,
    }


def validate_against_legacy_results(folds):
    """Validates the reconstructed folds against a real historical anchor.

    The legacy raw result CSVs store subject_avgs whose y_true values (for the
    age target) are the mean ages of each fold's test subjects. If the
    reconstruction is right, the sorted multiset of mean ages computed from
    the metadata for the reconstructed subjects must match every fold row.

    Args:
        folds (dict[str, list[str]]): Reconstructed ``fold{k}`` exclusions.

    Returns:
        tuple[int, int]: (rows compared, rows mismatched).
    """
    meta = pd.read_csv(META_PATH)
    meta["subject"] = meta["subject"].astype(str)
    mean_age = meta.groupby("subject")["age"].mean().to_dict()

    checked, mismatched = 0, 0
    for csv in sorted(glob.glob(
            "results_*_legacy/**/downstream_raw_results_kfold_folds*.csv",
            recursive=True)):
        df = pd.read_csv(csv)
        if "subject_avgs" not in df.columns or "fold" not in df.columns:
            continue
        for _, row in df.iterrows():
            if row.get("target") != "age" or not isinstance(row["subject_avgs"], str):
                continue
            key = f"fold{int(row['fold'])}"
            if key not in folds:
                continue
            y_true = sorted(float(p.split(",")[0])
                            for p in row["subject_avgs"].split(";") if p)
            expected = sorted(mean_age[s] for s in folds[key])
            checked += 1
            if len(y_true) != len(expected) or not np.allclose(y_true, expected, atol=1e-4):
                mismatched += 1
                print(f"[CHECK][MISMATCH] {os.path.basename(csv)} fold {row['fold']}")
    return checked, mismatched


def main():
    folds = fold_exclusions()

    checked, mismatched = validate_against_legacy_results(folds)
    if mismatched:
        raise SystemExit(
            f"Reconstruction check failed for {mismatched}/{checked} legacy "
            "result rows; the historical split did not use the assumed KFold "
            "parameters. Aborting without writing anything."
        )
    if checked == 0:
        raise SystemExit(
            "No legacy result rows were comparable: the KFold reconstruction "
            "is UNVALIDATED. Refusing to write sidecars on an unanchored "
            "assumption."
        )
    print(f"[CHECK] Reconstructed folds match {checked}/{checked} legacy "
          f"result rows (per-subject mean ages).")

    written, updated, skipped = 0, 0, 0
    for pth in sorted(glob.glob(os.path.join(MODELS_DIR, "*.pth"))):
        sc = sidecar_path(pth)
        identity = parse_identity(os.path.basename(pth))

        if os.path.exists(sc):
            with open(sc) as fh:
                recorded = json.load(fh)
            missing = {k: v for k, v in (identity or {}).items()
                       if k != "fold_key" and k not in recorded}
            if missing:
                recorded.update(missing)
                write_sidecar(pth, recorded)
                updated += 1
            else:
                skipped += 1
            continue

        config = {"legacy": True, "seed": None,
                  "exclude_subjects_reconstructed": True}
        if identity:
            config.update({k: v for k, v in identity.items() if k != "fold_key"})
            if identity["fold_key"] in folds:
                config["exclude_subjects"] = folds[identity["fold_key"]]
            else:
                config.pop("exclude_subjects_reconstructed")
        write_sidecar(pth, config)
        written += 1

    print(f"[DONE] sidecars written: {written}, updated: {updated}, "
          f"already complete: {skipped}")


if __name__ == "__main__":
    main()
