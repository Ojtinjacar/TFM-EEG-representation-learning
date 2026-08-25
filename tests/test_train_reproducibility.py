import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils import set_seed


def test_set_seed_makes_torch_numpy_and_random_deterministic():
    import random

    set_seed(7)
    drawn = (torch.randn(3).tolist(), np.random.rand(3).tolist(), random.random())
    set_seed(7)
    again = (torch.randn(3).tolist(), np.random.rand(3).tolist(), random.random())
    assert drawn == again


def test_set_seed_gives_different_draws_for_different_seeds():
    set_seed(1)
    a = torch.randn(5)
    set_seed(2)
    b = torch.randn(5)
    assert not torch.equal(a, b)


def test_dataloader_order_is_reproducible_under_a_seeded_generator():
    from torch.utils.data import DataLoader, TensorDataset

    data = TensorDataset(torch.arange(40).float().unsqueeze(1))

    def order(seed):
        loader = DataLoader(data, batch_size=8, shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(seed))
        return [int(v) for batch, in loader for v in batch.flatten()]

    assert order(42) == order(42)
    assert order(42) != order(43)


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("data")
    n = 96
    np.save(root / "processed_windows.npy", rng.normal(size=(n, 4, 250)).astype(np.float32))
    pd.DataFrame({
        "subject": np.repeat([f"S{i:02d}" for i in range(8)], n // 8),
        "age": np.tile([6, 9, 16, 36], n // 4),
        "block": 1,
        "epoch_index": np.arange(n),
    }).to_csv(root / "processed_metadata.csv", index=False)
    features = root / "features.npy"
    np.save(features, rng.normal(size=(n, 8)).astype(np.float32))  # P_aper
    return root, features


def _train(tiny_dataset, seed, save_dir, fold_id, diagnose_every=5):
    root, features = tiny_dataset
    cmd = [sys.executable, os.path.join(ROOT, "src", "train_expclr.py"),
           "--data_path", str(root), "--expert_features", str(features),
           "--descriptor", "P_aper", "--zone", "z", "--frequency", "f",
           "--fold_id", fold_id, "--num_epochs", "2", "--batch_size", "32",
           "--embedding_size", "16", "--seed", str(seed),
           "--diagnose_every", str(diagnose_every),
           "--save_dir", str(save_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr[-3000:]
    ckpts = list(save_dir.glob("*.pth"))
    assert len(ckpts) == 1, f"expected one checkpoint, found {len(ckpts)}"
    state = torch.load(ckpts[0], map_location="cpu")
    ckpts[0].unlink()
    return state


def test_same_seed_reproduces_the_encoder_bit_for_bit(tiny_dataset, tmp_path):
    a = _train(tiny_dataset, 42, tmp_path, "a")
    b = _train(tiny_dataset, 42, tmp_path, "b")
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), f"tensor {key} differs between two runs with seed 42"


def test_different_seeds_produce_different_encoders(tiny_dataset, tmp_path):
    a = _train(tiny_dataset, 42, tmp_path, "a")
    b = _train(tiny_dataset, 7, tmp_path, "b")
    assert any(not torch.equal(a[k], b[k]) for k in a), "changing the seed changed nothing"


def test_diagnostics_do_not_change_the_trained_weights(tiny_dataset, tmp_path):
    often = _train(tiny_dataset, 42, tmp_path, "often", diagnose_every=1)
    rarely = _train(tiny_dataset, 42, tmp_path, "rarely", diagnose_every=1000)
    for key in often:
        assert torch.equal(often[key], rarely[key]), (
            f"tensor {key} changes with --diagnose_every, so diagnostics alter training")
