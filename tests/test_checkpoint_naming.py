import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from checkpoint_naming import (
    ae_checkpoint_name,
    checkpoint_is_reusable,
    mae_checkpoint_name,
    sidecar_path,
    simclr_checkpoint_name,
    triplet_checkpoint_name,
    write_sidecar,
)


# ---------------------------------------------------------------------------
# Naming contract: these strings are frozen. They must match both the names
# produced historically by the train scripts and the names expected by
# run_downstream.py. Changing any of them silently orphans every existing
# checkpoint.
# ---------------------------------------------------------------------------

def test_simclr_name_contract():
    assert simclr_checkpoint_name("all", "all", "fold0") == (
        "SimCLR_all_all_fold0_batch_512_lr_0.001_wd_0.0001_temperature_0.05.pth"
    )


def test_simclr_name_with_variant_tag():
    assert simclr_checkpoint_name("all", "all", "fold3_nbrcosine") == (
        "SimCLR_all_all_fold3_nbrcosine_batch_512_lr_0.001_wd_0.0001_temperature_0.05.pth"
    )


def test_ae_name_contract():
    assert ae_checkpoint_name("all", "all", "fold0") == "AE_all_all_fold0_hidden128_e100.pth"


def test_mae_name_contract():
    assert mae_checkpoint_name("all", "all", "fold0") == (
        "MAE_all_all_fold0_hidden128_mask30_block25_e100.pth"
    )


def test_triplet_name_contract():
    assert triplet_checkpoint_name("age", "all", "all", "fold0") == (
        "Triplet_age_all_all_fold0_emb128_m0.4.pth"
    )


def test_no_fold_omits_suffix():
    assert ae_checkpoint_name("all", "all", None) == "AE_all_all_hidden128_e100.pth"


# ---------------------------------------------------------------------------
# Reuse gate
# ---------------------------------------------------------------------------

def _make_checkpoint(tmp_path, config):
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"fake")
    write_sidecar(str(ckpt), config)
    return str(ckpt)


def test_reuse_requires_sidecar(tmp_path):
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"fake")
    assert not checkpoint_is_reusable(str(ckpt), {})


def test_reuse_matches_and_mismatches(tmp_path):
    ckpt = _make_checkpoint(tmp_path, {
        "zone": "all", "exclude_subjects": ["B1", "B2"], "beta": 0.003,
    })
    assert checkpoint_is_reusable(ckpt, {"zone": "all", "exclude_subjects": ["B1", "B2"]})
    assert checkpoint_is_reusable(ckpt, {"beta": 0.003})
    assert not checkpoint_is_reusable(ckpt, {"zone": "frontal"})
    assert not checkpoint_is_reusable(ckpt, {"exclude_subjects": ["B1", "B3"]})
    assert not checkpoint_is_reusable(ckpt, {"missing_key": 1})


def test_legacy_refused_without_flag(tmp_path):
    ckpt = _make_checkpoint(tmp_path, {
        "legacy": True, "exclude_subjects": ["B1"], "zone": "all",
    })
    assert not checkpoint_is_reusable(ckpt, {"exclude_subjects": ["B1"]})
    assert checkpoint_is_reusable(ckpt, {"exclude_subjects": ["B1"]}, allow_legacy=True)


def test_legacy_skips_absent_keys_but_checks_exclusion(tmp_path):
    ckpt = _make_checkpoint(tmp_path, {
        "legacy": True, "exclude_subjects": ["B1"], "zone": "all", "seed": None,
    })
    # beta is not recorded and seed is recorded as null: skipped, not fatal.
    assert checkpoint_is_reusable(
        ckpt, {"exclude_subjects": ["B1"], "beta": 1.0, "seed": 1234}, allow_legacy=True
    )
    # A wrong exclusion is always fatal.
    assert not checkpoint_is_reusable(
        ckpt, {"exclude_subjects": ["B2"]}, allow_legacy=True
    )


def test_legacy_without_exclusion_never_reusable(tmp_path):
    ckpt = _make_checkpoint(tmp_path, {"legacy": True, "zone": "all"})
    assert not checkpoint_is_reusable(ckpt, {"zone": "all"}, allow_legacy=True)


def test_sidecar_round_trip(tmp_path):
    config = {"method": "AE", "exclude_subjects": ["B2", "B9"], "seed": 42}
    ckpt = _make_checkpoint(tmp_path, config)
    with open(sidecar_path(ckpt)) as fh:
        assert json.load(fh) == config
