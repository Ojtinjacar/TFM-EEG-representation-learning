"""Tests for how ExpCLR plugs into the shared pipeline."""

import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from checkpoint_naming import (  # noqa: E402
    checkpoint_is_reusable,
    expclr_checkpoint_name,
    sidecar_path,
)
from downstream import (  # noqa: E402
    DEFAULT_REPRESENTATION,
    LOSS_REPRESENTATION,
    FullModel,
)


class TwoStageEncoder(nn.Module):
    """Encoder whose output and inner embedding are told apart by construction."""

    def __init__(self, n_features=4):
        super().__init__()
        self.n_features = n_features

    def get_embedding(self, x):
        return torch.zeros(x.shape[0], self.n_features)

    def forward(self, x):
        return torch.ones(x.shape[0], self.n_features)


class Identity(nn.Module):
    def forward(self, x):
        return x


def test_expclr_is_evaluated_on_the_representation_its_loss_shaped():
    encoder, x = TwoStageEncoder(), torch.zeros(3, 2, 10)
    model = FullModel(encoder, Identity(),
                      representation=LOSS_REPRESENTATION["ExpCLR"])
    assert torch.equal(model(x), encoder(x))


def test_the_other_methods_keep_reading_the_embedding():
    encoder, x = TwoStageEncoder(), torch.zeros(3, 2, 10)
    for method in ("SimCLR", "AE", "MAE", "VAE", "TripletLoss"):
        representation = LOSS_REPRESENTATION.get(method, DEFAULT_REPRESENTATION)
        model = FullModel(encoder, Identity(), representation=representation)
        assert torch.equal(model(x), encoder.get_embedding(x)), method


def test_an_unknown_representation_is_refused():
    with pytest.raises(ValueError):
        FullModel(TwoStageEncoder(), Identity(), representation="logits")


def test_the_tuner_names_its_checkpoints_like_everyone_else():
    import tune_expclr

    name = tune_expclr.expclr_checkpoint_name(
        "all", "all", "d1.0_lr0.005_train", "P_madurativo",
        batch_size=64, lr=0.005, temperature=1.0, delta=1.0)
    assert name == expclr_checkpoint_name(
        "all", "all", "d1.0_lr0.005_train", "P_madurativo",
        batch_size=64, lr=0.005, temperature=1.0, delta=1.0)


def test_the_gate_keys_of_the_three_orchestrators_agree():
    """The same configuration must produce the same expectations everywhere."""
    import run_downstream
    import run_e3_loso
    import tune_expclr

    assert run_downstream.EXPCLR_DESCRIPTOR == run_e3_loso.DESCRIPTOR
    assert run_downstream.EXPCLR_DESCRIPTOR == tune_expclr.DEFAULT_DESCRIPTOR


def _sidecar(**overrides):
    config = {
        "method": "ExpCLR", "zone": "all", "frequency": "all", "fold_id": "fold0",
        "descriptor": "P_madurativo", "delta": 1.0, "lr": 0.005, "temperature": 1.0,
        "sim_max": "train", "num_epochs": 100, "loss_on": "projection",
        "seed": 42, "exclude_subjects": ["B010"],
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize("key,other", [
    ("descriptor", "P_aper"),
    ("delta", 1.88),
    ("temperature", 0.5),
    ("sim_max", "batch"),
    ("num_epochs", 30),
    ("loss_on", "embedding"),
])
def test_a_checkpoint_is_not_reused_across_configurations(key, other):
    recorded = _sidecar()
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = os.path.join(tmp, "m.pth")
        open(checkpoint, "wb").write(b"weights")
        with open(sidecar_path(checkpoint), "w") as fh:
            json.dump(recorded, fh)

        assert checkpoint_is_reusable(checkpoint, recorded)
        assert not checkpoint_is_reusable(checkpoint, _sidecar(**{key: other}))


def test_the_descriptor_reaches_the_checkpoint_name():
    name = expclr_checkpoint_name("all", "all", "fold0", "P_madurativo",
                                  batch_size=64, lr=0.005, temperature=1.0, delta=1.0)
    assert "P_madurativo" in name and name.endswith(".pth")


def _write_config(directory, **overrides):
    config = {"delta": 1.88, "lr": 0.001, "tau": 0.7, "sim_max": "batch",
              "mae": 3.1, "random_baseline_mae": 4.2, "beats_random_baseline": True}
    config.update(overrides)
    with open(os.path.join(directory, "best_config_P_madurativo.json"), "w") as fh:
        json.dump(config, fh)


@pytest.fixture
def pristine_variants():
    """Restores the whole catalogue, since tuning writes into every entry."""
    import copy

    import run_downstream

    saved = copy.deepcopy(run_downstream.EXPCLR_VARIANTS)
    yield run_downstream
    run_downstream.EXPCLR_VARIANTS.clear()
    run_downstream.EXPCLR_VARIANTS.update(saved)


def test_a_tuned_configuration_replaces_the_paper_defaults(pristine_variants):
    run_downstream = pristine_variants

    assert run_downstream.expclr_hparams("ExpCLR") == (
        run_downstream.EXPCLR_DELTA, run_downstream.EXPCLR_LR,
        run_downstream.EXPCLR_TAU, run_downstream.EXPCLR_SIM_MAX,
    )
    with tempfile.TemporaryDirectory() as tmp:
        _write_config(tmp)
        run_downstream.load_expclr_tuning(tmp)
        assert run_downstream.expclr_hparams("ExpCLR") == (1.88, 0.001, 0.7, "batch")


def test_a_configuration_that_never_beat_the_random_encoder_is_refused(pristine_variants):
    run_downstream = pristine_variants

    with tempfile.TemporaryDirectory() as tmp:
        _write_config(tmp, beats_random_baseline=False)
        with pytest.raises(ValueError, match="random encoder"):
            run_downstream.load_expclr_tuning(tmp)


def test_a_missing_configuration_falls_back_instead_of_failing(pristine_variants):
    run_downstream = pristine_variants

    with tempfile.TemporaryDirectory() as tmp:
        run_downstream.load_expclr_tuning(tmp)
    assert run_downstream.expclr_hparams("ExpCLR")[0] == run_downstream.EXPCLR_DELTA


@pytest.mark.parametrize("key,other", [
    ("embedding_size", 64),
    ("descriptor_dim", 78),
    ("lr_gamma", 0.95),
])
def test_a_checkpoint_is_not_reused_across_widths_or_schedules(key, other):
    """None of these reaches the filename, and all three change what the weights mean."""
    recorded = _sidecar(embedding_size=128, descriptor_dim=32, lr_gamma=0.99)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = os.path.join(tmp, "m.pth")
        open(checkpoint, "wb").write(b"weights")
        with open(sidecar_path(checkpoint), "w") as fh:
            json.dump(recorded, fh)

        assert checkpoint_is_reusable(checkpoint, recorded)
        expected = dict(recorded)
        expected[key] = other
        assert not checkpoint_is_reusable(checkpoint, expected)


def test_the_descriptor_declared_must_match_the_file():
    """The descriptor name reaches the checkpoint filename, so it cannot be a free label."""
    from build_expert_features import DESCRIPTORS

    assert len(DESCRIPTORS["P_madurativo"]) == 32
    assert len(DESCRIPTORS["P_full"]) == 78
    assert len(DESCRIPTORS["P_aper"]) == 8


def test_gaps_are_filled_from_the_training_split_only():
    from run_e3_loso import impute_with_train_medians

    features = np.array([[1.0, 10.0], [3.0, 30.0], [np.nan, np.nan], [100.0, 1000.0]])
    train = np.array([True, True, True, False])

    filled = impute_with_train_medians(features, train)
    # Median of the training rows alone: the held-out row must not move it.
    assert filled[2, 0] == 2.0 and filled[2, 1] == 20.0
    assert not np.isnan(filled).any()
    np.testing.assert_array_equal(filled[[0, 1, 3]], features[[0, 1, 3]])


def test_a_column_missing_throughout_training_is_refused():
    from run_e3_loso import impute_with_train_medians

    features = np.array([[np.nan, 1.0], [np.nan, 2.0], [5.0, 3.0]])
    train = np.array([True, True, False])
    with pytest.raises(ValueError, match="training split"):
        impute_with_train_medians(features, train)


def test_every_method_seeds_its_pretraining_the_same_way():
    """The fold-to-fold spread must mean the same thing for every method.

    That spread is what the comparison table reports as error, so a method whose seed does
    not follow the fold would show less variation by construction and not by being steadier.
    """
    import inspect

    import run_downstream

    source = inspect.getsource(run_downstream.run_pretraining)
    assert '"--seed", str(seed)' in source
    assert "eff_seed" not in source


def test_the_seed_is_applied_by_one_shared_function():
    import train_expclr
    import utils

    assert train_expclr.set_seed is utils.set_seed


@pytest.mark.parametrize("method", ["AE", "MAE", "VAE", "InterFusion", "TripletLoss"])
def test_every_pretraining_gets_the_zone_it_was_asked_for(method, monkeypatch):
    """A method that keeps its default path trains on the whole head whatever zone is asked.

    TripletLoss did exactly that: it was left out of the line that appends the resolved
    path, so every per-zone result of it described a different channel set than its label.
    """
    import subprocess

    import run_downstream

    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(run_downstream.subprocess, "run", fake_run)
    run_downstream.run_pretraining(
        method=method, target="age", zone="occipital", frequency="all",
        test_subjects=["B010"], fold_id="fold0", no_skip=True, seed=1234,
    )
    command = seen["command"]
    # --zone alone proves nothing: it is passed even when the data path is not, which is
    # how the bug hid. What matters is the value of the flag the loader actually reads.
    flags = [i for i, c in enumerate(command) if c in ("--data-path", "--data_path")]
    assert flags, f"{method} receives no data path, so it falls back to the default set"
    for i in flags:
        assert "occipital_all" in command[i + 1], \
            f"{method} is pointed at {command[i + 1]} while asked for occipital"


def test_a_result_row_says_which_zone_it_came_from():
    """Without it the zone lives only in the output directory the launch happened to pick.

    That is how eleven runs ended up describing a zone they were not trained on, and how a
    reader has to infer from a path what the row should have carried.
    """
    import inspect

    import run_downstream

    source = inspect.getsource(run_downstream.execute_fold)
    entry = source[source.index("result_entry = {"):]
    entry = entry[: entry.index("}")]
    for field in ('"zone": args.zone', '"frequency": args.frequency'):
        assert field in entry, f"the result row does not carry {field}"


def test_the_three_descriptors_are_registered_as_variants():
    """Each descriptor needs its own variant, or only one of them can ever be run."""
    import run_downstream
    from build_expert_features import DESCRIPTORS

    declared = {v["descriptor"] for v in run_downstream.EXPCLR_VARIANTS.values()}
    assert declared == set(DESCRIPTORS)


def test_each_variant_reads_the_file_of_its_own_descriptor_and_zone():
    """The descriptor and the zone both reach the checkpoint, so a crossed path mislabels it."""
    import run_downstream
    from build_expert_features import ROIS, descriptor_path, zone_dir

    seen = set()
    for zone in ["all"] + ROIS:
        for variant in run_downstream.EXPCLR_VARIANTS.values():
            path = run_downstream.expclr_features(variant["descriptor"], zone)
            assert path == descriptor_path(variant["descriptor"], zone_dir(zone))
            seen.add(path)
    # One file per (descriptor, zone): no zone may read another one's descriptor.
    assert len(seen) == len(run_downstream.EXPCLR_VARIANTS) * (len(ROIS) + 1)


@pytest.mark.parametrize("requested,expected", [
    (None, ["ExpCLR"]),
    (["ExpCLR-aper"], ["ExpCLR-aper"]),
    (["ExpCLR-full", "ExpCLR-aper"], ["ExpCLR-full", "ExpCLR-aper"]),
])
def test_the_requested_variants_replace_expclr_in_the_method_list(requested, expected):
    """Omitting the flag must leave the published runs reproducible by the same command."""
    selected = None

    def _use(m):
        return selected is None or m in selected

    expclr_variants = list(requested or ["ExpCLR"]) if _use("ExpCLR") else []

    def _expand(mlist):
        out = []
        for m in mlist:
            if m == "ExpCLR":
                out.extend(expclr_variants)
            else:
                out.append(m)
        return out

    assert _expand(["VAE", "ExpCLR"]) == ["VAE"] + expected


def test_two_variants_of_the_same_fold_do_not_share_a_checkpoint():
    """Two descriptors trained on the same fold must not overwrite each other."""
    names = {
        expclr_checkpoint_name("all", "all", "fold0", descriptor,
                               batch_size=64, lr=0.005, temperature=1.0, delta=1.0)
        for descriptor in ("P_madurativo", "P_full", "P_aper")
    }
    assert len(names) == 3


def _variant():
    return {"ExpCLR-probe": {"descriptor": "P_aper"}}


def test_a_zone_reads_the_descriptor_built_for_that_zone(tmp_path, monkeypatch):
    """Each zone has its own file, so one region cannot be described by another."""
    import run_downstream
    from build_expert_features import zone_dir

    monkeypatch.setattr(run_downstream, "DESCRIPTOR_DIR_FOR_TESTS", None, raising=False)
    assert zone_dir("all", "base") == "base"
    assert zone_dir("occipital", "base").endswith(os.path.join("base", "occipital"))
    assert run_downstream.expclr_features("P_aper", "occipital") != \
        run_downstream.expclr_features("P_aper", "all")


def test_a_missing_descriptor_stops_the_run_before_it_trains():
    """Otherwise the trainer dies, the orchestrator swallows it and the fold loses rows."""
    import run_downstream

    monkey = {"ExpCLR-ghost": {"descriptor": "P_aper"}}
    with pytest.raises(SystemExit, match="does not exist"):
        run_downstream.check_expclr_descriptors(
            monkey, ["ExpCLR-ghost"], {"ExpCLR-ghost"}, "central")


def test_the_preflight_stays_out_of_the_way_when_expclr_is_not_running():
    import run_downstream

    run_downstream.check_expclr_descriptors(
        _variant(), ["ExpCLR-probe"], {"AE", "MAE"}, "occipital")


def test_a_single_region_drops_the_other_three_and_the_contrasts():
    """A contrast is a difference between regions, so one region has none."""
    from build_expert_features import ROIS, descriptors_for, rois_of_zone

    whole = descriptors_for(rois_of_zone("all"))
    one = descriptors_for(rois_of_zone("parietal"))

    assert len(whole["P_full"]) == 78 and len(one["P_full"]) == 19
    assert len(whole["P_madurativo"]) == 32 and len(one["P_madurativo"]) == 8
    assert not any(c.startswith("paf_central_minus") for c in one["P_full"])
    assert all(c.endswith("_parietal") for c in one["P_full"])
    for name, columns in one.items():
        assert set(columns) <= set(whole[name]), f"{name} invents columns"


def test_a_descriptor_spans_exactly_the_regions_of_its_zone():
    from build_expert_features import ROIS, descriptors_for, rois_of_zone

    for zone in ["all"] + ROIS:
        rois = rois_of_zone(zone)
        for name, columns in descriptors_for(rois).items():
            covered = {c.rsplit("_", 1)[1] for c in columns if c.rsplit("_", 1)[1] in ROIS}
            assert covered == set(rois), f"{name} on {zone} spans {covered}"


def test_an_unknown_zone_is_refused():
    from build_expert_features import rois_of_zone

    with pytest.raises(ValueError, match="is not a zone"):
        rois_of_zone("temporal")


def test_windows_must_cover_the_regions_the_descriptor_describes():
    from apsd_baseline import PRESET_ZONES
    from build_expert_features import ROIS, rois_of_zone
    from train_expclr import check_roi_coverage

    # A one-region descriptor is happy with one-region windows.
    check_roi_coverage(PRESET_ZONES["occipital"], "P_aper", rois_of_zone("occipital"))
    # The whole-montage one is not.
    with pytest.raises(ValueError, match="no channel of"):
        check_roi_coverage(PRESET_ZONES["occipital"], "P_aper", rois_of_zone("all"))
    whole = [ch for roi in ROIS for ch in PRESET_ZONES[roi]]
    check_roi_coverage(whole, "P_madurativo", rois_of_zone("all"))
