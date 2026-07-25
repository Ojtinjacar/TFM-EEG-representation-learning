"""Tests for the embedding cache of the E3 orchestrator.

The cache exists only to avoid recomputing identical forward passes, so the property that must
hold is that it changes nothing: cached embeddings are bit-for-bit what a direct extraction would
have produced, and two different encoders never share an entry.

Run with:
    conda run -n dasci-cimcyc python -m pytest tests/test_embedding_cache.py
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from run_e3_loso import EmbeddingCache, build_encoder  # noqa: E402

from eval_expclr import extract_embeddings  # noqa: E402


@pytest.fixture
def windows():
    """A few short windows: enough to exercise the encoder without the cost of real data."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(6, 4, 250)).astype(np.float32)


@pytest.fixture
def checkpoints(tmp_path, windows):
    """Writes two distinct encoder checkpoints to disk."""
    paths = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        model = build_encoder(windows, torch.device("cpu"), None, seed=seed)
        path = tmp_path / f"encoder_{seed}.pth"
        torch.save(model.state_dict(), path)
        paths.append(path)
    return paths


def test_cache_returns_exactly_what_a_direct_extraction_would(windows, checkpoints):
    """The guarantee that matters: caching must not perturb a single value."""
    device = torch.device("cpu")
    expected = extract_embeddings(build_encoder(windows, device, checkpoints[0]), windows, device)
    got = EmbeddingCache(windows, device).get(checkpoints[0])
    assert np.array_equal(got, expected)


def test_repeated_requests_extract_only_once(windows, checkpoints):
    """ExpCLR and B7 share an encoder within a fold; the second request must be a hit."""
    cache = EmbeddingCache(windows, torch.device("cpu"))
    first = cache.get(checkpoints[0])
    second = cache.get(checkpoints[0])
    assert cache.stats == {"hits": 1, "misses": 1}
    assert np.array_equal(first, second)


def test_distinct_checkpoints_never_share_an_entry(windows, checkpoints):
    """A collision here would silently evaluate one fold's encoder on another fold's split."""
    cache = EmbeddingCache(windows, torch.device("cpu"))
    a, b = cache.get(checkpoints[0]), cache.get(checkpoints[1])
    assert cache.stats == {"hits": 0, "misses": 2}
    assert not np.allclose(a, b), "dos encoders distintos no deberian dar el mismo embedding"


def test_random_encoder_is_computed_once_for_the_whole_sweep(windows):
    """B2 is never trained, so its embeddings are identical in every fold."""
    cache = EmbeddingCache(windows, torch.device("cpu"))
    per_fold = [cache.get(None) for _ in range(5)]
    assert cache.stats == {"hits": 4, "misses": 1}
    for emb in per_fold[1:]:
        assert np.array_equal(emb, per_fold[0])


def test_random_encoder_is_reproducible_across_caches(windows):
    """The fixed seed must make B2 the same baseline on any run, not just within one process."""
    device = torch.device("cpu")
    torch.manual_seed(999)  # ensucia el estado global para comprobar que build_encoder lo fija
    a = EmbeddingCache(windows, device).get(None)
    torch.manual_seed(123)
    b = EmbeddingCache(windows, device).get(None)
    assert np.array_equal(a, b)


def test_random_entry_does_not_collide_with_a_checkpoint(windows, checkpoints):
    cache = EmbeddingCache(windows, torch.device("cpu"))
    assert not np.allclose(cache.get(None), cache.get(checkpoints[0]))
