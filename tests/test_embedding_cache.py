import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from run_e3_loso import EmbeddingCache, build_encoder

from eval_expclr import extract_embeddings


@pytest.fixture
def windows():
    rng = np.random.default_rng(0)
    return rng.normal(size=(6, 4, 250)).astype(np.float32)


@pytest.fixture
def checkpoints(tmp_path, windows):
    paths = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        model = build_encoder(windows, torch.device("cpu"), None, seed=seed)
        path = tmp_path / f"encoder_{seed}.pth"
        torch.save(model.state_dict(), path)
        paths.append(path)
    return paths


def test_cache_returns_exactly_what_a_direct_extraction_would(windows, checkpoints):
    device = torch.device("cpu")
    expected = extract_embeddings(build_encoder(windows, device, checkpoints[0]), windows, device)
    got = EmbeddingCache(windows, device).get(checkpoints[0])
    assert np.array_equal(got, expected)


def test_repeated_requests_extract_only_once(windows, checkpoints):
    cache = EmbeddingCache(windows, torch.device("cpu"))
    first = cache.get(checkpoints[0])
    second = cache.get(checkpoints[0])
    assert cache.stats == {"hits": 1, "misses": 1}
    assert np.array_equal(first, second)


def test_distinct_checkpoints_never_share_an_entry(windows, checkpoints):
    cache = EmbeddingCache(windows, torch.device("cpu"))
    a, b = cache.get(checkpoints[0]), cache.get(checkpoints[1])
    assert cache.stats == {"hits": 0, "misses": 2}
    assert not np.allclose(a, b), "two different encoders should not give the same embedding"


def test_random_encoder_is_computed_once_for_the_whole_sweep(windows):
    cache = EmbeddingCache(windows, torch.device("cpu"))
    per_fold = [cache.get(None) for _ in range(5)]
    assert cache.stats == {"hits": 4, "misses": 1}
    for emb in per_fold[1:]:
        assert np.array_equal(emb, per_fold[0])


def test_random_encoder_is_reproducible_across_caches(windows):
    device = torch.device("cpu")
    torch.manual_seed(999)
    a = EmbeddingCache(windows, device).get(None)
    torch.manual_seed(123)
    b = EmbeddingCache(windows, device).get(None)
    assert np.array_equal(a, b)


def test_random_entry_does_not_collide_with_a_checkpoint(windows, checkpoints):
    cache = EmbeddingCache(windows, torch.device("cpu"))
    assert not np.allclose(cache.get(None), cache.get(checkpoints[0]))
