import itertools
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from montage import (  # noqa: E402
    MONTAGE_SIDECAR,
    PRESET_ZONES,
    ROIS,
    MontageError,
    load_channel_names,
    resolve_processed_montage,
    roi_indices,
    select_channel_indices,
)

RAW_MONTAGE = os.path.join(os.path.dirname(__file__), "..",
                           "data", "raw", "channel_names.txt")
needs_montage = pytest.mark.skipif(not os.path.exists(RAW_MONTAGE),
                                   reason="the recording montage is not available")

ROI_SIZES = {"central": 27, "frontal": 22, "parietal": 14, "occipital": 7}


def test_regions_are_disjoint_and_sum_to_seventy():
    for a, b in itertools.combinations(ROIS, 2):
        assert not set(PRESET_ZONES[a]) & set(PRESET_ZONES[b])
    assert {r: len(set(PRESET_ZONES[r])) for r in ROIS} == ROI_SIZES
    assert len(set().union(*(set(PRESET_ZONES[r]) for r in ROIS))) == 70


@needs_montage
def test_every_region_is_complete_in_the_recording():
    names = set(load_channel_names(RAW_MONTAGE))
    for roi in ROIS:
        assert set(PRESET_ZONES[roi]) <= names, f"{roi} has channels outside the montage"


@needs_montage
def test_zone_all_selects_seventy_channels_in_montage_order():
    names = load_channel_names(RAW_MONTAGE)
    assert len(names) == 109
    idx = select_channel_indices(list(ROIS), names)
    kept = [names[i] for i in idx]
    assert len(kept) == 70
    assert idx == sorted(idx), "the selection must preserve the montage order"
    assert kept[0] == "E3" and kept[-1] == "Cz"


@needs_montage
def test_thirty_nine_channels_belong_to_no_region():
    names = load_channel_names(RAW_MONTAGE)
    union = set().union(*(set(PRESET_ZONES[r]) for r in ROIS))
    assert len([c for c in names if c not in union]) == 39


@needs_montage
def test_truncating_the_montage_misassigns_every_channel():
    """Pins the defect the sidecar exists to prevent.

    Taking the first 70 names of the montage instead of the selected ones leaves
    45 of 70 channels assigned and not a single position in agreement, which is
    why a channel count can never stand in for the montage itself.
    """
    names = load_channel_names(RAW_MONTAGE)
    kept = [names[i] for i in select_channel_indices(list(ROIS), names)]
    truncated = names[:70]

    assert sum(a == b for a, b in zip(kept, truncated)) == 0
    union = set().union(*(set(PRESET_ZONES[r]) for r in ROIS))
    assert sum(c in union for c in truncated) == 45
    assert len(union - set(truncated)) == 25


def test_roi_indices_refuses_a_montage_of_another_size():
    with pytest.raises(MontageError):
        roi_indices(["E3", "E4"], 70)


def test_roi_indices_refuses_a_montage_missing_a_region():
    only_frontal = list(PRESET_ZONES["frontal"])
    with pytest.raises(MontageError):
        roi_indices(only_frontal, len(only_frontal))


def test_select_channel_indices_refuses_an_unknown_zone():
    with pytest.raises(MontageError):
        select_channel_indices(["temporal"], ["E3"])


def test_resolve_reads_the_sidecar(tmp_path):
    (tmp_path / MONTAGE_SIDECAR).write_text("E3\nE4\nE5\n")
    got = resolve_processed_montage(str(tmp_path / "processed_windows.npy"), 3)
    assert got == ["E3", "E4", "E5"]


def test_resolve_without_a_sidecar_names_the_way_out(tmp_path):
    with pytest.raises(MontageError) as err:
        resolve_processed_montage(str(tmp_path / "processed_windows.npy"), 70)
    assert "postprocessing.py" in str(err.value)


def test_resolve_refuses_a_sidecar_of_the_wrong_size(tmp_path):
    (tmp_path / MONTAGE_SIDECAR).write_text("E3\nE4\n")
    with pytest.raises(MontageError):
        resolve_processed_montage(str(tmp_path / "processed_windows.npy"), 70)


def test_an_explicit_montage_file_is_validated_too(tmp_path):
    other = tmp_path / "elsewhere.txt"
    other.write_text("\n".join(f"E{i}" for i in range(109)))
    with pytest.raises(MontageError):
        resolve_processed_montage(str(tmp_path / "processed_windows.npy"), 70,
                                  channels_txt=str(other))


@needs_montage
def test_roi_indices_of_the_selected_montage_recover_the_full_regions():
    names = load_channel_names(RAW_MONTAGE)
    kept = [names[i] for i in select_channel_indices(list(ROIS), names)]
    idx = roi_indices(kept, len(kept))
    assert {r: len(idx[r]) for r in ROIS} == ROI_SIZES
    assert sum(len(v) for v in idx.values()) == 70
    assert len(set(itertools.chain.from_iterable(idx.values()))) == 70
    for roi in ROIS:
        assert {kept[i] for i in idx[roi]} == set(PRESET_ZONES[roi])


def test_postprocessing_writes_a_sidecar_matching_its_output(tmp_path):
    """End to end: the montage beside the windows describes their channel axis."""
    import json
    import subprocess

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rng = np.random.default_rng(0)
    # Two frontal channels and one occipital, so the selection is a real subset.
    np.save(tmp_path / "raw.npy", rng.normal(size=(4, 3, 500)))
    pd.DataFrame({"subject": ["S1", "S1", "S2", "S2"]}).to_csv(
        tmp_path / "meta.csv", index=False)
    pd.DataFrame({"ID": ["S1", "S2"], "Sexo": [1, 2]}).to_csv(
        tmp_path / "socio.csv", index=False)
    (tmp_path / "channels.txt").write_text("E33\nE66\nE34\n")

    out = tmp_path / "out"
    res = subprocess.run(
        [sys.executable, "src/postprocessing.py",
         "--data_path", str(tmp_path / "raw.npy"),
         "--meta_path", str(tmp_path / "meta.csv"),
         "--socio_path", str(tmp_path / "socio.csv"),
         "--channels_txt", str(tmp_path / "channels.txt"),
         "--output_path", str(out),
         "--zones", "frontal"],
        capture_output=True, text=True, cwd=root)
    assert res.returncode == 0, res.stderr[-3000:]

    X = np.load(out / "processed_windows.npy")
    kept = load_channel_names(str(out / MONTAGE_SIDECAR))
    # The occipital channel sat between the two frontal ones, so a positional
    # prefix of the montage would have kept the wrong pair.
    assert kept == ["E33", "E34"]
    assert len(kept) == X.shape[1]
    assert json.load(open(out / "manifest.json"))["channels"] == kept
