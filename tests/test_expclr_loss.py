import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from loss import ExpCLRLoss
from train_expclr import (
    ExpertFeatureDataset,
    checkpoint_is_reusable,
    effective_dimensionality,
    equidistant_reference,
    max_pairwise_distance,
    prepare_descriptor,
)
from build_expert_features import (
    DESCRIPTORS,
    DescriptorAlignmentError,
    align_to_metadata,
)

B, E, D = 12, 16, 5
DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def batch():
    z = torch.randn(B, E, requires_grad=True)
    f = torch.randn(B, D)
    return z, f


def test_similarity_is_bounded_with_unit_diagonal(batch):
    _, f = batch
    sim = ExpCLRLoss(DEVICE).expert_similarity(f)
    assert sim.shape == (B, B)
    assert torch.all(sim >= 0.0) and torch.all(sim <= 1.0)
    assert torch.allclose(torch.diagonal(sim), torch.ones(B), atol=1e-5)


def test_similarity_is_symmetric(batch):
    _, f = batch
    sim = ExpCLRLoss(DEVICE).expert_similarity(f)
    assert torch.allclose(sim, sim.T, atol=1e-6)


def test_squared_similarity_is_the_square_of_the_linear_one(batch):
    _, f = batch
    linear = ExpCLRLoss(DEVICE, squared_similarity=False).expert_similarity(f)
    squared = ExpCLRLoss(DEVICE, squared_similarity=True).expert_similarity(f)
    assert torch.allclose(squared, linear.pow(2), atol=1e-6)


def test_train_max_distance_clamps_instead_of_flipping_sign(batch):
    _, f = batch
    tiny_max = 1e-3
    sim = ExpCLRLoss(DEVICE, feat_max_dist=tiny_max).expert_similarity(f)
    off_diagonal = sim[~torch.eye(B, dtype=torch.bool)]
    assert torch.all(off_diagonal == 0.0)


def test_normalized_distance_has_unit_row_mean(batch):
    z, _ = batch
    dist = ExpCLRLoss(DEVICE).normalized_distance(z)
    assert dist.shape == (B, B)
    assert torch.allclose(dist.mean(dim=1), torch.ones(B), atol=1e-4)


def test_loss_is_finite_scalar_and_differentiable(batch):
    z, f = batch
    loss = ExpCLRLoss(DEVICE)(z, f)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_quadratic_loss_is_non_negative(batch):
    z, f = batch
    loss = ExpCLRLoss(DEVICE, temperature=None)(z, f)
    assert loss.item() >= 0.0


def test_identical_features_reduce_to_squared_distances():
    z = torch.randn(B, E)
    f = torch.ones(B, D)
    criterion = ExpCLRLoss(DEVICE, temperature=None, feat_max_dist=1.0)
    expected = criterion.normalized_distance(z).pow(2).mean()
    assert torch.allclose(criterion(z, f), expected, atol=1e-6)


def test_proposition_3b_large_tau_converges_to_quadratic_loss(batch):
    z, f = (t.double() for t in batch)
    quadratic = ExpCLRLoss(DEVICE, temperature=None)(z, f).item()
    mined = ExpCLRLoss(DEVICE, temperature=1e6)(z, f).item()
    assert mined == pytest.approx(quadratic, rel=1e-6)


def test_proposition_3a_small_tau_converges_to_max_pair_loss(batch):
    z, f = (t.double() for t in batch)
    criterion = ExpCLRLoss(DEVICE, temperature=1e-5)
    sim = criterion.expert_similarity(f)
    dist = criterion.normalized_distance(z)
    worst = (((1.0 - sim) * criterion.delta - dist).pow(2)).max().item()
    assert criterion(z, f).item() == pytest.approx(worst, rel=1e-3)


def test_mining_weights_the_worst_pairs_more_than_the_mean(batch):
    z, f = batch
    quadratic = ExpCLRLoss(DEVICE, temperature=None)(z, f).item()
    mined = ExpCLRLoss(DEVICE, temperature=1.0)(z, f).item()
    assert mined > quadratic


def test_tolerates_incomplete_batches():
    criterion = ExpCLRLoss(DEVICE)
    for size in (2, 3, 7):
        loss = criterion(torch.randn(size, E), torch.randn(size, D))
        assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"feat_max_dist": 0.0}, "feat_max_dist"),
        ({"temperature": -1.0}, "temperature"),
    ],
)
def test_invalid_hyperparameters_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ExpCLRLoss(DEVICE, **kwargs)


def test_batch_mismatch_and_singleton_batch_raise():
    criterion = ExpCLRLoss(DEVICE)
    with pytest.raises(ValueError, match="Batch mismatch"):
        criterion(torch.randn(B, E), torch.randn(B + 1, D))
    with pytest.raises(ValueError, match="at least two samples"):
        criterion(torch.randn(1, E), torch.randn(1, D))


def test_prepare_descriptor_imputes_and_standardises():
    features = np.random.randn(50, D) * 100.0 + 5.0
    features[3, 0] = np.nan
    F, keep = prepare_descriptor(features)
    assert keep.all()
    assert F.shape == (50, D)
    assert not np.isnan(F).any()
    assert np.allclose(F.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(F.std(axis=0), 1.0, atol=1e-8)


def test_prepare_descriptor_drops_fully_missing_windows():
    features = np.random.randn(10, D)
    features[4, :] = np.nan
    F, keep = prepare_descriptor(features)
    assert keep.sum() == 9 and not keep[4]
    assert len(F) == 9


def test_prepare_descriptor_neutralises_constant_columns():
    features = np.random.randn(20, D)
    features[:, 2] = 7.0
    F, _ = prepare_descriptor(features)
    assert np.allclose(F[:, 2], 0.0)


def test_prepare_descriptor_applies_quality_filter():
    features = np.random.randn(10, D)
    quality = np.linspace(0.8, 1.0, 10)
    _, keep = prepare_descriptor(features, quality=quality, min_r2=0.95)
    assert keep.sum() == int((quality >= 0.95).sum())


def test_prepare_descriptor_rejects_an_all_nan_column():
    features = np.random.randn(10, D)
    features[:, 1] = np.nan
    with pytest.raises(ValueError, match="entirely NaN"):
        prepare_descriptor(features)


def test_max_pairwise_distance_matches_bruteforce():
    F = np.random.randn(30, D)
    expected = float(torch.cdist(torch.tensor(F, dtype=torch.float32),
                                 torch.tensor(F, dtype=torch.float32)).max())
    assert max_pairwise_distance(F) == pytest.approx(expected, rel=1e-5)


def test_dataset_pairs_windows_with_features_and_rejects_misalignment():
    X = torch.randn(6, 4, 32)
    F = torch.randn(6, D)
    dataset = ExpertFeatureDataset(X, F)
    x, f = dataset[2]
    assert torch.equal(x, X[2]) and torch.equal(f, F[2])
    with pytest.raises(ValueError, match="not aligned"):
        ExpertFeatureDataset(X, torch.randn(5, D))


def _synthetic(n_windows=6, n_missing=2):
    meta = pd.DataFrame({
        "subject": ["B010"] * 3 + ["B011"] * (n_windows - 3),
        "age": [6] * 3 + [9] * (n_windows - 3),
        "epoch_index": list(range(3)) + list(range(n_windows - 3)),
    })
    cols = DESCRIPTORS["P_aper"]
    features = meta.iloc[: n_windows - n_missing].copy()
    for j, c in enumerate(cols):
        features[c] = np.arange(len(features), dtype=float) + j
    return meta, features, cols


def test_align_preserves_window_order_and_flags_coverage():
    meta, features, cols = _synthetic()
    matrix, mask, _ = align_to_metadata(features, meta, cols)
    assert matrix.shape == (len(meta), len(cols))
    assert mask.sum() == 4 and not mask[-1] and not mask[-2]
    assert matrix[0, 0] == 0.0 and matrix[3, 0] == 3.0
    assert np.isnan(matrix[4]).all()


def test_align_rejects_duplicated_keys():
    meta, features, cols = _synthetic()
    dup = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    with pytest.raises(DescriptorAlignmentError, match="not unique in the features"):
        align_to_metadata(dup, meta, cols)


def test_align_rejects_missing_join_key():
    meta, features, cols = _synthetic()
    with pytest.raises(DescriptorAlignmentError, match="Join key"):
        align_to_metadata(features.drop(columns=["age"]), meta, cols)


def test_align_rejects_unknown_columns():
    meta, features, cols = _synthetic()
    with pytest.raises(KeyError):
        align_to_metadata(features, meta, cols + ["not_a_feature"])


def test_descriptor_catalogue_sizes():
    assert len(DESCRIPTORS["P_full"]) == 106
    assert len(DESCRIPTORS["P_madurativo"]) == 36
    assert not any(c.startswith("osc_") for c in DESCRIPTORS["P_madurativo"])
    assert len(DESCRIPTORS["P_aper"]) == 8
    assert len(set(DESCRIPTORS["P_full"])) == 106
    for name in ("P_madurativo", "P_aper"):
        assert set(DESCRIPTORS[name]).issubset(set(DESCRIPTORS["P_full"]))


def test_row_mean_of_normalised_distance_is_always_one():
    loss = ExpCLRLoss(torch.device("cpu"))
    for n, dim, scale in ((8, 3, 1.0), (64, 128, 1e-3), (16, 5, 1e4)):
        z = torch.randn(n, dim, generator=torch.Generator().manual_seed(n)) * scale
        row_means = loss.normalized_distance(z).mean(dim=1)
        assert torch.allclose(row_means, torch.ones(n), atol=1e-5), (
            f"row mean should be 1 for n={n}, dim={dim}, scale={scale}")


def test_dataset_wide_max_shrinks_the_target_far_below_the_imposed_row_mean():
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(1)
    population = torch.randn(2000, 40, generator=generator)
    batch = population[:64]

    dataset_max = float(torch.cdist(population, population, p=2).max())
    batch_max = float(torch.cdist(batch, batch, p=2).max())
    assert dataset_max > batch_max, "el maximo global debe superar al del batch"

    def mean_target(max_dist):
        criterion = ExpCLRLoss(device, delta=1.0, feat_max_dist=max_dist, temperature=None)
        return float((1.0 - criterion.expert_similarity(batch)).mean())

    with_dataset_max = mean_target(dataset_max)
    with_batch_max = mean_target(batch_max)

    assert abs(1.0 - with_dataset_max) > abs(1.0 - with_batch_max)
    assert with_dataset_max < with_batch_max


def test_equidistant_geometry_sits_at_n_over_n_minus_one_not_at_one():
    loss = ExpCLRLoss(torch.device("cpu"))
    for n in (8, 16, 64):
        z = torch.eye(n, dtype=torch.float64)
        D = loss.normalized_distance(z)
        off_diagonal = D[~torch.eye(n, dtype=bool)]
        assert torch.allclose(off_diagonal, torch.full_like(off_diagonal, n / (n - 1.0)),
                              atol=1e-9), f"para n={n} se esperaba {n / (n - 1.0)}"
        assert torch.allclose(equidistant_reference(n, D.device).double(), D, atol=1e-9)


def test_matching_delta_equalises_the_row_means():
    device = torch.device("cpu")
    f = torch.randn(64, 40, generator=torch.Generator().manual_seed(0))
    max_dist = float(torch.cdist(f, f, p=2).max())

    probe = ExpCLRLoss(device, feat_max_dist=max_dist, temperature=None)
    mean_target = float((1.0 - probe.expert_similarity(f)).mean())
    delta_star = 1.0 / mean_target

    def row_means_at(delta):
        criterion = ExpCLRLoss(device, delta=delta, feat_max_dist=max_dist, temperature=None)
        return ((1.0 - criterion.expert_similarity(f)) * delta).mean(dim=1)

    assert row_means_at(delta_star).mean() == pytest.approx(1.0, abs=1e-5)
    assert row_means_at(1.0).mean() < 0.95


def _write_checkpoint(tmp_path, **config):
    ckpt = tmp_path / "encoder.pth"
    ckpt.write_bytes(b"weights")
    (tmp_path / "encoder_config.json").write_text(json.dumps(config))
    return ckpt


def test_checkpoint_with_a_different_loss_target_is_not_reused(tmp_path):
    ckpt = _write_checkpoint(tmp_path, loss_on="projection", delta=1.0, seed=42)
    assert not checkpoint_is_reusable(ckpt, {"loss_on": "embedding"})
    assert checkpoint_is_reusable(ckpt, {"loss_on": "projection"})


def test_checkpoint_with_different_hyperparameters_is_not_reused(tmp_path):
    ckpt = _write_checkpoint(tmp_path, loss_on="embedding", delta=1.0, sim_max="train",
                             lr=0.005, seed=42, num_epochs=50)
    assert checkpoint_is_reusable(ckpt, {"delta": 1.0, "sim_max": "train", "lr": 0.005})
    for mismatch in ({"delta": 1.86}, {"sim_max": "batch"}, {"lr": 0.001}, {"num_epochs": 100},
                     {"seed": 0}):
        assert not checkpoint_is_reusable(ckpt, mismatch), f"{mismatch} deberia reentrenar"


def test_checkpoint_without_a_sidecar_is_never_reused(tmp_path):
    ckpt = tmp_path / "orphan.pth"
    ckpt.write_bytes(b"weights")
    assert not checkpoint_is_reusable(ckpt, {"loss_on": "embedding"})


def test_missing_checkpoint_is_not_reusable(tmp_path):
    assert not checkpoint_is_reusable(tmp_path / "absent.pth", {})


def test_reuse_tolerates_float_representation(tmp_path):
    ckpt = _write_checkpoint(tmp_path, delta=1.8599999999)
    assert checkpoint_is_reusable(ckpt, {"delta": 1.86})


def test_unknown_key_blocks_reuse(tmp_path):
    ckpt = _write_checkpoint(tmp_path, loss_on="embedding")
    assert not checkpoint_is_reusable(ckpt, {"sim_max": "train"})


def test_effective_dimensionality_counts_the_directions_actually_used():
    rng = np.random.default_rng(0)
    for k in (1, 3, 10):
        basis = np.linalg.qr(rng.normal(size=(64, k)))[0]
        z = rng.normal(size=(500, k)) @ basis.T
        assert effective_dimensionality(z) == pytest.approx(k, rel=0.05), f"esperaba ~{k}"


def test_effective_dimensionality_survives_a_diverged_encoder():
    z = np.full((10, 4), np.nan)
    assert np.isnan(effective_dimensionality(z))
    z = np.ones((10, 4)) * np.inf
    assert np.isnan(effective_dimensionality(z))


def test_delta_star_approaches_one_as_distances_concentrate():
    rng = np.random.default_rng(0)
    device = torch.device("cpu")

    def delta_star(matrix):
        max_dist = float(torch.cdist(matrix, matrix, p=2).max())
        criterion = ExpCLRLoss(device, feat_max_dist=max_dist, temperature=None)
        return 1.0 / float((1.0 - criterion.expert_similarity(matrix[:64])).mean())

    values = [delta_star(torch.tensor(rng.normal(size=(600, d)), dtype=torch.float32))
              for d in (10, 40, 200, 600)]
    assert all(a >= b for a, b in zip(values, values[1:])), values
    assert values[-1] == pytest.approx(1.0, abs=0.1), "en dimension alta Delta* deberia rondar 1"
    assert values[0] > values[-1], "en dimension baja Delta* deberia ser claramente mayor"


def test_loss_is_invariant_to_rescaling_the_embedding():
    generator = torch.Generator().manual_seed(3)
    z = torch.randn(32, 16, dtype=torch.float64, generator=generator)
    f = torch.randn(32, 8, dtype=torch.float64, generator=generator)
    criterion = ExpCLRLoss(torch.device("cpu"), feat_max_dist=10.0, temperature=None)

    reference = float(criterion(z, f))
    for scale in (1e-3, 0.5, 2.0, 1e3):
        assert float(criterion(z * scale, f)) == pytest.approx(reference, rel=1e-9), (
            f"la perdida deberia ser invariante al reescalado, falla con c={scale}")


def test_zero_loss_is_unreachable_at_delta_one():
    generator = torch.Generator().manual_seed(4)
    n = 32
    z = torch.randn(n, 16, dtype=torch.float64, generator=generator)
    f = torch.randn(n, 8, dtype=torch.float64, generator=generator)
    criterion = ExpCLRLoss(torch.device("cpu"), delta=1.0, feat_max_dist=10.0, temperature=None)

    row_sums_of_d = criterion.normalized_distance(z).sum(dim=1)
    assert torch.allclose(row_sums_of_d, torch.full((n,), float(n), dtype=torch.float64),
                          atol=1e-6)

    target = (1.0 - criterion.expert_similarity(f)) * criterion.delta
    assert target.sum(dim=1).max() <= n - 1 + 1e-6, "la diana no puede sumar mas de Delta*(N-1)"
    assert float(criterion(z, f)) > 0.0
