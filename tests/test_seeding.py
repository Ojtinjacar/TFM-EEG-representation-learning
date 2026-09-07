import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import EnhancedAttentionLSTM
from utils import set_seed, split_dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _model_state(seed):
    set_seed(seed)
    model = EnhancedAttentionLSTM(
        input_size=1250, hidden_size=16, n_channels=2, sfreq=250,
        lstm_hidden_size=8,
    )
    return model.state_dict()


def test_set_seed_makes_init_deterministic():
    a = _model_state(7)
    b = _model_state(7)
    c = _model_state(8)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert any(not torch.equal(a[k], c[k]) for k in a)


def test_split_dataset_seed_controls_split():
    X = np.random.default_rng(0).normal(size=(40, 2, 8)).astype(np.float32)
    t1, _ = split_dataset(torch.tensor(X), seed=7)
    t2, _ = split_dataset(torch.tensor(X), seed=7)
    t3, _ = split_dataset(torch.tensor(X), seed=8)
    assert list(t1.indices) == list(t2.indices)
    assert list(t1.indices) != list(t3.indices)


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vae_data")
    rng = np.random.default_rng(0)
    n = 32
    X = rng.normal(size=(n, 2, 1250)).astype(np.float32)
    meta = pd.DataFrame({
        "subject": [f"B{i % 8}" for i in range(n)],
        "age": [6, 9, 16, 36][0] * np.ones(n, dtype=int),
    })
    data_path = tmp / "windows.npy"
    meta_path = tmp / "meta.csv"
    np.save(data_path, X)
    meta.to_csv(meta_path, index=False)
    return str(data_path), str(meta_path), str(tmp)


def _run_train_vae(data_path, meta_path, out_dir, seed):
    save_dir = os.path.join(out_dir, f"models_{seed}_{np.random.randint(1 << 30)}")
    cmd = [
        sys.executable, "src/train_vae.py",
        "--data-path", data_path,
        "--meta-path", meta_path,
        "--epochs", "2",
        "--batch-size", "16",
        "--hidden-size", "16",
        "--kl-anneal-epochs", "1",
        "--device", "cpu",
        "--seed", str(seed),
        "--save-model-dir", save_dir,
        "--save-fig-dir", "",
        "--zone", "test",
        "--frequency", "test",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr[-3000:]
    pths = [f for f in os.listdir(save_dir) if f.endswith(".pth")]
    assert len(pths) == 1
    return torch.load(os.path.join(save_dir, pths[0]), map_location="cpu")


def test_train_vae_is_reproducible_on_cpu(tiny_dataset):
    data_path, meta_path, tmp = tiny_dataset
    s1 = _run_train_vae(data_path, meta_path, tmp, seed=7)
    s2 = _run_train_vae(data_path, meta_path, tmp, seed=7)
    s3 = _run_train_vae(data_path, meta_path, tmp, seed=8)
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    assert any(not torch.equal(s1[k], s3[k]) for k in s1)
