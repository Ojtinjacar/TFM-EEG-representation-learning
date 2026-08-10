import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture()
def tiny_raw(tmp_path):
    rng = np.random.default_rng(0)
    # 4 epochs x 3 channels x 500 samples (5 s at 100 Hz -> 1 window/epoch).
    X = rng.normal(loc=[[0.0], [5.0], [-3.0]], scale=[[1.0], [2.0], [0.5]],
                   size=(4, 3, 500)).astype(np.float64)
    # Make subject S2's epochs clearly shifted so excluding it moves the stats.
    X[2:] += 10.0
    np.save(tmp_path / "raw.npy", X)
    pd.DataFrame({"subject": ["S1", "S1", "S2", "S2"]}).to_csv(
        tmp_path / "meta.csv", index=False)
    pd.DataFrame({"ID": ["S1", "S2"], "Sexo": [1, 2]}).to_csv(
        tmp_path / "socio.csv", index=False)
    (tmp_path / "channels.txt").write_text("E33\nE34\nE27\n")
    return tmp_path, X


def _run(tmp, out_name, extra):
    out = tmp / out_name
    cmd = [
        sys.executable, "src/postprocessing.py",
        "--data_path", str(tmp / "raw.npy"),
        "--meta_path", str(tmp / "meta.csv"),
        "--socio_path", str(tmp / "socio.csv"),
        "--channels_txt", str(tmp / "channels.txt"),
        "--output_path", str(out),
        "--norm_mode", "per_channel",
    ] + extra
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr[-3000:]
    return np.load(out / "processed_windows.npy"), out


def test_fit_stats_excluding_changes_transform_but_keeps_all_windows(tiny_raw):
    tmp, _ = tiny_raw
    X_all, _ = _run(tmp, "out_all", [])
    X_excl, out = _run(tmp, "out_excl", ["--fit_stats_excluding", "S2"])

    # All windows are kept (epochs 0-1 are S1, 2-3 are S2, one window each).
    assert X_all.shape == X_excl.shape == (4, 3, 500)

    # The band-pass filter runs before normalization, so instead of
    # replicating it we assert the defining property of each fit:
    # full fit -> per-channel z-score over ALL epochs.
    assert np.allclose(X_all.mean(axis=(0, 2)), 0.0, atol=1e-5)
    assert np.allclose(X_all.std(axis=(0, 2)), 1.0, atol=1e-4)
    # train-only fit -> z-score over S1's epochs only; S2 (shifted +10, though
    # the 0.2 Hz high-pass removes most of the DC offset) is transformed with
    # S1's stats, so the global mean moves away from zero.
    assert np.allclose(X_excl[:2].mean(axis=(0, 2)), 0.0, atol=1e-5)
    assert np.allclose(X_excl[:2].std(axis=(0, 2)), 1.0, atol=1e-4)
    assert np.abs(X_excl.mean(axis=(0, 2))).max() > 0.05
    assert not np.allclose(X_all, X_excl)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["args"]["fit_stats_excluding"] == ["S2"]
    assert manifest["output_shape"] == [4, 3, 500]


def test_manifest_written_without_exclusion(tiny_raw):
    tmp, _ = tiny_raw
    _, out = _run(tmp, "out_manifest", [])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["args"]["norm_mode"] == "per_channel"
    assert manifest["input_size_bytes"] > 0
