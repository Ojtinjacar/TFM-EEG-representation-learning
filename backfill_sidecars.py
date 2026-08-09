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
    method = tokens[0]

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


def main():
    folds = fold_exclusions()

    # Cross-validate the reconstruction against authentic ExpCLR sidecars.
    mismatches = 0
    for sc in glob.glob(os.path.join(MODELS_DIR, "ExpCLR_*_config.json")):
        with open(sc) as fh:
            recorded = json.load(fh)
        identity = parse_identity(os.path.basename(sc).replace("_config.json", ".pth"))
        if not identity or identity["fold_key"] not in folds:
            continue
        expected = folds[identity["fold_key"]]
        actual = sorted(str(s) for s in recorded.get("exclude_subjects", []))
        if actual and actual != expected:
            mismatches += 1
            print(f"[CHECK][MISMATCH] {os.path.basename(sc)}: reconstructed "
                  f"{expected} != recorded {actual}")
    if mismatches:
        raise SystemExit(
            f"Reconstruction check failed for {mismatches} ExpCLR sidecars; "
            "the historical split did not use the assumed KFold parameters. "
            "Aborting without writing anything."
        )
    print("[CHECK] Reconstructed folds match all authentic ExpCLR sidecars.")

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
