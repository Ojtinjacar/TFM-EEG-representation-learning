"""Tests for the resolution of trained checkpoints."""

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from checkpoint_naming import sidecar_path  # noqa: E402
from subject_identity_probe import (  # noqa: E402
    CheckpointNotFound,
    expected_sidecar,
    find_checkpoint,
)


def _checkpoint(directory, name, **fields):
    """Writes a checkpoint and the sidecar that describes it."""
    path = os.path.join(directory, f"{name}.pth")
    open(path, "wb").write(b"weights")
    with open(sidecar_path(path), "w") as fh:
        json.dump(fields, fh)
    return path


def test_the_zone_decides_which_checkpoint_is_loaded(tmp_path):
    """A name cannot tell two zones apart; the sidecar can."""
    common = dict(method="ExpCLR", frequency="all", fold_id="fold0",
                  descriptor="P_madurativo")
    _checkpoint(tmp_path, "ExpCLR_all_all_fold0_P_madurativo", zone="all", **common)
    wanted = _checkpoint(tmp_path, "ExpCLR_occipital_all_fold0_P_madurativo",
                         zone="occipital", **common)

    assert find_checkpoint("ExpCLR", "fold0", [str(tmp_path)], "occipital", "all") == wanted


def test_an_ambiguous_match_is_refused_rather_than_resolved_alphabetically(tmp_path):
    """Two checkpoints of one fold differing in something the sidecar does not record."""
    common = dict(method="VAE", zone="all", frequency="all", fold_id="fold0")
    _checkpoint(tmp_path, "VAE_all_all_fold0_beta0.001", **common)
    _checkpoint(tmp_path, "VAE_all_all_fold0_beta0.01", **common)

    with pytest.raises(CheckpointNotFound, match="matches 2 checkpoints"):
        find_checkpoint("VAE", "fold0", [str(tmp_path)], "all", "all")


def test_a_checkpoint_without_a_sidecar_is_not_eligible(tmp_path):
    open(os.path.join(tmp_path, "AE_all_all_fold0.pth"), "wb").write(b"weights")

    with pytest.raises(CheckpointNotFound, match="No checkpoint"):
        find_checkpoint("AE", "fold0", [str(tmp_path)], "all", "all")


def test_triplet_loss_is_reachable(tmp_path):
    """Its file is named Triplet_, so matching on the name alone never found it."""
    wanted = _checkpoint(tmp_path, "Triplet_age_all_all_fold0", method="TripletLoss",
                         zone="all", frequency="all", fold_id="fold0", target="age")

    assert find_checkpoint("TripletLoss", "fold0", [str(tmp_path)], "all", "all") == wanted


def test_each_expclr_variant_finds_its_own_descriptor(tmp_path):
    common = dict(method="ExpCLR", zone="all", frequency="all", fold_id="fold0")
    narrow = _checkpoint(tmp_path, "ExpCLR_all_all_fold0_P_madurativo",
                         descriptor="P_madurativo", **common)
    wide = _checkpoint(tmp_path, "ExpCLR_all_all_fold0_P_full", descriptor="P_full", **common)

    assert find_checkpoint("ExpCLR", "fold0", [str(tmp_path)], "all", "all") == narrow
    assert find_checkpoint("ExpCLR-full", "fold0", [str(tmp_path)], "all", "all") == wide


def test_a_simclr_variant_carries_its_tag_in_the_fold_id():
    plain = expected_sidecar("SimCLR", "fold0", "all", "all")
    variant = expected_sidecar("SimCLR-nbr-cosine", "fold0", "all", "all")

    assert plain["method"] == variant["method"] == "SimCLR"
    assert plain["fold_id"] == "fold0"
    assert variant["fold_id"].startswith("fold0_") and variant["fold_id"] != "fold0"
