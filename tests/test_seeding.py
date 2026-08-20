import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import EnhancedAttentionLSTM
from utils import set_seed, split_dataset

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
