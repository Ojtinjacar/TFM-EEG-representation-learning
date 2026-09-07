"""Tests for the base every figure of the writeup is read off.

This is the last step between what the machines produced and what gets reported, so a
mistake here is invisible: the tables still come out, they are just wrong. The cases below
are the ones that would actually pass unnoticed -- a partial run pooled as if it were
complete, a cohort silently grown by a duplicate, a zone inferred instead of read.
"""

import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import consolidate_results as cr  # noqa: E402


def _row(variant="ExpCLR", zone="all", fold=0, mode="linear_probe",
         subjects=(("10.0", "11.0"), ("20.0", "19.0"))):
    return {
        "fold": fold, "method": variant, "zone": zone, "frequency": "all",
        "eval_mode": mode, "target": "age", "nRMSE": "1.0", "R2": "0.5", "RMSE": "2.0",
        "subject_avgs": ";".join(f"{a},{b}" for a, b in subjects),
    }


def _write(results_dir, name, rows):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, name)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run(tmp_path, n_folds=10, variant="ExpCLR", zone="all", folds=range(10)):
    results = os.path.join(tmp_path, "save", "expclr", "results")
    for k in folds:
        _write(results, f"{variant}_{zone}_all_age_fold{k}.csv", [_row(variant, zone, k)])
    return os.path.join(tmp_path, "save")


# --- what counts as a result -------------------------------------------------------

@pytest.mark.parametrize("name", [
    "ExpCLR_all_all_age_fold0.csv",
    "SimCLR-xsubj-cosine_occipital_all_age_fold9.csv",
    "InterFusion_parietal_all_cit_36mo_fold3.csv",
])
def test_a_name_of_the_convention_is_a_result(name):
    assert cr.is_result_file(name)


@pytest.mark.parametrize("name", [
    # The aggregates the pipeline drops next to the results: pooling them would count
    # every fold a second time.
    "downstream_raw_results_kfold_folds0-0.csv",
    "downstream_agg_results_kfold.csv",
    # The old convention, whose name says nothing about zone, target or variant.
    "downstream_raw_results_kfold.csv",
    "notes.txt",
])
def test_anything_else_is_not(name):
    assert not cr.is_result_file(name)


def test_the_aggregates_are_not_collected(tmp_path):
    source = _run(str(tmp_path))
    results = os.path.join(source, "expclr", "results")
    _write(results, "downstream_raw_results_kfold_folds0-0.csv", [_row()])
    _write(results, "downstream_agg_results_kfold.csv", [_row()])
    folds, _, manifest = cr.collect(source)
    assert len(folds) == 10, "only the per-fold results are results"
    assert all(cr.is_result_file(m["file"]) for m in manifest)


# --- the zone is read, never inferred ----------------------------------------------

def test_a_row_without_its_zone_fails_loudly(tmp_path):
    """Inferring it from the directory is what let a result mean whatever its path said."""
    results = os.path.join(str(tmp_path), "save", "expclr", "results")
    row = _row()
    del row["zone"]
    _write(results, "ExpCLR_all_all_age_fold0.csv", [row])
    with pytest.raises(cr.ConsolidationError, match="zone"):
        cr.collect(os.path.join(str(tmp_path), "save"))


def test_a_row_with_no_subject_averages_fails_loudly(tmp_path):
    results = os.path.join(str(tmp_path), "save", "expclr", "results")
    _write(results, "ExpCLR_all_all_age_fold0.csv", [_row(subjects=())])
    with pytest.raises(cr.ConsolidationError, match="subject averages"):
        cr.collect(os.path.join(str(tmp_path), "save"))


# --- the packed label ---------------------------------------------------------------

@pytest.mark.parametrize("variant,family,strategy,metric", [
    ("SimCLR", "SimCLR", "augment", ""),
    ("SimCLR-xsubj-cosine", "SimCLR", "xsubj", "cosine"),
    ("SimCLR-nbr-wasser", "SimCLR", "nbr", "wasserstein"),
    ("SimCLR-lag0-riemann", "SimCLR", "lag0", "riemann"),
    ("SimCLR-diffage-cosine", "SimCLR", "diffage", "cosine"),
])
def test_a_simclr_label_unpacks_into_strategy_and_metric(variant, family, strategy, metric):
    out = cr.normalise(variant)
    assert (out["family"], out["strategy"], out["metric"]) == (family, strategy, metric)


@pytest.mark.parametrize("variant,descriptor", [
    ("ExpCLR", "P_madurativo"), ("ExpCLR-full", "P_full"), ("ExpCLR-aper", "P_aper"),
])
def test_an_expclr_label_carries_the_descriptor_it_was_guided_by(variant, descriptor):
    """The label alone does not say it, and it is what tells these three runs apart."""
    out = cr.normalise(variant)
    assert out["family"] == "ExpCLR"
    assert out["descriptor"] == descriptor


def test_an_interfusion_label_is_its_own_family():
    """One configuration, so the label carries no tag to strip."""
    out = cr.normalise("InterFusion")
    assert out["family"] == "InterFusion"
    assert out["variant"] == "InterFusion"


# --- what must not reach the base ---------------------------------------------------

def test_a_run_short_of_a_fold_is_dropped_and_reported(tmp_path):
    """Pooled, it gives a figure over part of the cohort that reads like the real one."""
    source = _run(str(tmp_path), folds=range(9))
    folds, subjects, _ = cr.collect(source)
    folds, subjects, incomplete = cr.keep_complete(folds, subjects, 10)
    assert folds == [] and subjects == []
    assert list(incomplete.values()) == [9]


def test_a_complete_run_survives(tmp_path):
    source = _run(str(tmp_path))
    folds, subjects, _ = cr.collect(source)
    folds, subjects, incomplete = cr.keep_complete(folds, subjects, 10)
    assert len(folds) == 10 and not incomplete


def test_a_repeated_fold_does_not_grow_the_cohort():
    """Filtering the subject rows by the folds kept keeps every copy, because they share
    the key: the cohort then reads larger than it is."""
    folds, subjects = [], []
    for _ in range(2):
        common = {"variant": "ExpCLR", "zone": "all", "frequency": "all",
                  "target": "age", "eval_mode": "linear_probe", "fold": 0}
        folds.append(dict(common))
        subjects.extend(dict(common, subject_index=i) for i in range(3))
    folds, subjects, dropped = cr.deduplicate(folds, subjects)
    assert (len(folds), len(subjects), dropped) == (1, 3, 1)


# --- end to end ---------------------------------------------------------------------

def test_the_pooled_r2_matches_what_the_subject_table_says(tmp_path):
    """The reported figure is pooled over subjects, not the r2 column of the fold table."""
    results = os.path.join(str(tmp_path), "save", "expclr", "results")
    truth = [(10.0, 12.0), (20.0, 19.0), (30.0, 33.0)]
    for k in range(10):
        _write(results, f"ExpCLR_all_all_age_fold{k}.csv",
               [_row(fold=k, subjects=[(str(a), str(b)) for a, b in truth])])
    out = os.path.join(str(tmp_path), "results")
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "consolidate_results.py"),
                    "--source", os.path.join(str(tmp_path), "save"), "--out", out],
                   check=True, capture_output=True, cwd=ROOT)

    rows = list(csv.DictReader(open(os.path.join(out, "results_subjects.csv"))))
    ys = [(float(r["y_true"]), float(r["y_pred"])) for r in rows]
    mean = sum(y for y, _ in ys) / len(ys)
    r2 = 1 - (sum((y - p) ** 2 for y, p in ys)
              / sum((y - mean) ** 2 for y, _ in ys))
    expected = truth * 10
    mean_e = sum(y for y, _ in expected) / len(expected)
    assert r2 == pytest.approx(
        1 - sum((y - p) ** 2 for y, p in expected)
        / sum((y - mean_e) ** 2 for y, _ in expected))


def test_the_base_records_no_machine(tmp_path):
    """The split across machines is operational: the same campaign on one machine or on
    four has to give the same base."""
    source = _run(str(tmp_path))
    out = os.path.join(str(tmp_path), "results")
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "consolidate_results.py"),
                    "--source", source, "--out", out], check=True, capture_output=True,
                   cwd=ROOT)
    assert "machine" not in cr.FOLD_COLUMNS + cr.SUBJECT_COLUMNS
    manifest = json.load(open(os.path.join(out, "MANIFEST.json")))
    assert "machine" not in json.dumps(manifest)
    assert manifest["campaigns"] == ["expclr"]
    assert all(e["md5"] for e in manifest["files"])


def test_a_source_with_no_results_fails_instead_of_writing_an_empty_base(tmp_path):
    empty = os.path.join(str(tmp_path), "save")
    os.makedirs(os.path.join(empty, "expclr", "results"))
    done = subprocess.run(
        [sys.executable, os.path.join(ROOT, "src", "consolidate_results.py"),
         "--source", empty, "--out", os.path.join(str(tmp_path), "results")],
        capture_output=True, text=True, cwd=ROOT)
    assert done.returncode != 0
    assert "No result files" in done.stderr
