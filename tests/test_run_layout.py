"""Tests for where a run puts what it produces, and how it survives being interrupted.

These cover the failures that cost the most during the campaign: a pipeline that reported
success while collecting nothing, results whose zone lived only in the directory somebody
chose by hand, folds lost because everything was held in memory until the end, and thirteen
variants writing one filename between them.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import run_downstream as rd  # noqa: E402


def test_a_run_that_collects_nothing_fails_loudly():
    """Exiting 0 with no results let a chained launch carry on as if the fold had worked."""
    source = open(os.path.join(ROOT, "run_downstream.py")).read()
    block = source[source.index("    if not all_results:"):][:400]
    assert "raise SystemExit" in block, "an empty run must not return normally"


def test_the_result_name_carries_every_dimension_that_tells_runs_apart():
    name = rd.result_filename("SimCLR-xsubj-cosine", "occipital", "all", "age", 3)
    for piece in ("SimCLR-xsubj-cosine", "occipital", "all", "age", "fold3"):
        assert piece in name


def test_two_variants_of_one_zone_do_not_share_a_file():
    """Naming by the family is what made the thirteen SimCLR variants overwrite each other."""
    names = {rd.result_filename(v, "occipital", "all", "age", 0)
             for v in ("SimCLR", "SimCLR-xsubj-cosine", "SimCLR-nbr-riemann", "VAE")}
    assert len(names) == 4


def test_two_zones_of_one_variant_do_not_share_a_file():
    names = {rd.result_filename("VAE", z, "all", "age", 0)
             for z in ("occipital", "parietal", "frontal", "central", "all")}
    assert len(names) == 5


@pytest.mark.parametrize("variant,expected", [
    ("PCA", ["linear_probe"]),
    ("supervised", ["fine_tuning"]),
    ("VAE", ["linear_probe", "fine_tuning"]),
    ("SimCLR-xsubj-cosine", ["linear_probe", "fine_tuning"]),
    ("ExpCLR-full", ["linear_probe", "fine_tuning"]),
])
def test_the_modes_of_a_variant_are_not_assumed_constant(variant, expected):
    """A fixed row count per fold is wrong: PCA has no encoder to fine-tune."""
    assert rd.modes_of(variant) == expected


def test_a_run_directory_is_derived_and_cannot_escape(tmp_path):
    dirs = rd.run_dirs("simclr_zones", base=str(tmp_path))
    assert sorted(os.path.basename(p) for p in dirs.values()) == [
        "figures", "heads", "logs", "models", "results"]
    for path in dirs.values():
        assert os.path.isdir(path)
    for bad in ("../escape", "/absolute", ""):
        with pytest.raises(ValueError):
            rd.run_dirs(bad, base=str(tmp_path))


def _row(method, mode, zone="occipital", fold=0):
    return {"fold": fold, "method": method, "zone": zone, "frequency": "all",
            "eval_mode": mode, "target": "age", "nRMSE": 1.0, "R2": 0.5, "RMSE": 2.0,
            "subject_avgs": [(1.0, 2.0), (3.0, 4.0)]}


def test_a_fold_is_written_as_soon_as_it_closes(tmp_path):
    """Holding every fold in memory meant a run dying at the seventh lost the six before."""
    written = rd.write_fold_results(
        [_row("VAE", "linear_probe"), _row("VAE", "fine_tuning"),
         _row("SimCLR-xsubj-cosine", "linear_probe")], str(tmp_path))
    assert len(written) == 2, "one file per variant"
    frame = pd.read_csv(os.path.join(
        tmp_path, rd.result_filename("VAE", "occipital", "all", "age", 0)))
    assert set(frame["eval_mode"]) == {"linear_probe", "fine_tuning"}
    assert set(frame["zone"]) == {"occipital"}, "the zone must be in the row, not the path"
    assert frame["subject_avgs"].iloc[0] == "1.0,2.0;3.0,4.0"


def test_resuming_redoes_a_fold_that_is_missing_a_variant(tmp_path):
    """Counting rows passes a fold written by a launch that ran fewer variants."""
    rd.write_fold_results([_row("VAE", "linear_probe")], str(tmp_path))
    args = ["VAE", "SimCLR-xsubj-cosine"]
    assert not rd.fold_is_done(str(tmp_path), args, "occipital", "all", ["age"], 0,
                               ["linear_probe"])
    assert rd.fold_is_done(str(tmp_path), ["VAE"], "occipital", "all", ["age"], 0,
                           ["linear_probe"])


def test_resuming_redoes_a_fold_that_is_missing_a_mode(tmp_path):
    rd.write_fold_results([_row("VAE", "linear_probe")], str(tmp_path))
    assert not rd.fold_is_done(str(tmp_path), ["VAE"], "occipital", "all", ["age"], 0,
                               ["linear_probe", "fine_tuning"])


def test_downstream_takes_a_label_apart_from_the_method():
    """The family picks the architecture; the variant names what gets written."""
    source = open(os.path.join(ROOT, "src", "downstream.py")).read()
    assert '"--label"' in source
    assert "args.label or args.method" in source
    # The tag already ends in the fold; appending it again gave ..._fold0_fold0_....png.
    assert 'fig_name = f"{model_tag}.png"' in source
    assert 'f"{model_tag}_{fold_id}' not in source


def test_a_sweep_is_one_invocation_and_not_a_loop_outside():
    """The zone loop written in bash is what made each launch invent its own layout."""
    source = open(os.path.join(ROOT, "run_downstream.py")).read()
    # The first '"--zone"' is the flag being passed to a subprocess, not its definition.
    block = source[source.index('parser.add_argument(\n        "--zone"'):][:320]
    assert 'nargs="+"' in block, "--zone must accept several zones"
    assert "for i, zone in enumerate(zones)" in source
    assert "args.zone = zone" in source


@pytest.mark.parametrize("method,trainer", [
    ("SimCLR", "train_simclr.py"), ("ExpCLR", "train_expclr.py"),
    ("TripletLoss", "train_triplet_loss.py"), ("VAE", "train_vae.py"),
    ("AE", "train_auto.py"), ("MAE", "train_mae.py"),
    ("InterFusion", "train_interfusion.py"),
])
def test_the_orchestrator_knows_what_each_trainer_calls_its_output_dirs(method, trainer):
    """The trainers spell these two flags two different ways, and that is why neither was
    ever passed: every checkpoint and every loss curve went to the shared default instead of
    to the run that produced them. The mapping has to match what the script really accepts.
    """
    source = open(os.path.join(ROOT, "src", trainer)).read()
    for flag in rd.output_flags(method):
        assert f'"{flag}"' in source, f"{trainer} does not accept {flag}"


def test_a_variant_takes_the_output_flags_of_its_family():
    assert rd.output_flags("SimCLR-xsubj-cosine") == rd.output_flags("SimCLR")
    assert rd.output_flags("ExpCLR-full") == rd.output_flags("ExpCLR")


def test_the_checkpoint_path_is_not_hardcoded():
    """The run computed model_dir, created it, and then wrote somewhere else entirely."""
    source = open(os.path.join(ROOT, "run_downstream.py")).read()
    assert 'os.path.join("save/models"' not in source
    assert source.count("model_dir=model_dir, plot_dir=pretrain_plot_dir") == 7, \
        "every pre-training call site must hand down both directories"


def test_pretraining_passes_both_directories_to_the_trainer():
    source = open(os.path.join(ROOT, "run_downstream.py")).read()
    block = source[source.index("    model_flag, fig_flag = output_flags(method)"):][:400]
    assert "command += [model_flag, model_dir]" in block
    assert "command += [fig_flag, plot_dir]" in block


def test_an_interfusion_label_belongs_to_the_interfusion_family():
    """Otherwise it has no evaluation modes, and the resume check passes every fold as done
    without ever looking at what is in the file."""
    assert rd.family_of("InterFusion-interfusion_c1a") == "InterFusion"
    assert rd.modes_of("InterFusion-interfusion_c1a") == ["linear_probe", "fine_tuning"]


def test_interfusion_writes_through_the_shared_convention():
    """Its results used to be named after the fold range alone, so the zone lived in a
    directory somebody picked by hand rather than in the result."""
    source = open(os.path.join(ROOT, "run_interfusion_variant.py")).read()
    assert "downstream_raw_results_kfold_folds" not in source
    assert "rd.write_fold_results(rows, save_dir)" in source
    assert "rd.run_dirs(args.run_name)" in source
    assert '"--run_name"' in source
    # The row must carry what tells two runs apart, not the path it happens to sit in.
    row = source[source.index("        rows.append({"):][:600]
    for key in ('"zone": args.zone', '"frequency": args.frequency', '"target": args.target'):
        assert key in row


@pytest.mark.parametrize("trainer", [
    "train_simclr.py", "train_vae.py", "train_auto.py", "train_mae.py",
    "train_expclr.py", "train_interfusion.py", "train_triplet_loss.py",
])
def test_every_trainer_makes_its_own_plot_directory(trainer):
    """A missing plot directory threw away a fully trained encoder at the last line.

    train_simclr.py was the only one of the seven that did not create it, so wiping save/
    made every one of its pre-trainings crash after doing all the work.
    """
    source = open(os.path.join(ROOT, "src", trainer)).read()
    assert "makedirs" in source and ("plot_dir" in source or "fig_dir" in source), \
        f"{trainer} saves a figure into a directory it never creates"
