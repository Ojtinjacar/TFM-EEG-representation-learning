"""Fast, data-free unit tests for the ExpCLR objective and its descriptor pipeline.

Covers the loss of Nonnenmacher et al. (ICML 2022) -- similarity (Eq. 2/5), mean-normalised
distances, the quadratic pair term (Eq. 3) and the hard-negative-mined form (Eq. 4), including
the two limits of Proposition 3 -- plus the descriptor preparation (imputation, z-score) and the
window alignment performed by ``code/src/build_expert_features.py``. No EEG data or trained
weights required.

Run with:
    conda run -n dasci-cimcyc python -m pytest tests/test_expclr_loss.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# build_expert_features.py lives in the auxiliary package at code/src.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from loss import ExpCLRLoss  # noqa: E402
from train_expclr import (  # noqa: E402
    ExpertFeatureDataset,
    checkpoint_is_reusable,
    effective_dimensionality,
    equidistant_reference,
    max_pairwise_distance,
    prepare_descriptor,
)
from build_expert_features import (  # noqa: E402
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


# --- Similarity (Eq. 2 and Eq. 5) ---------------------------------------------------------

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
    """A batch pair beyond the train-fitted maximum must saturate at 0, never square back up."""
    _, f = batch
    tiny_max = 1e-3  # far smaller than any real pairwise distance in the batch
    sim = ExpCLRLoss(DEVICE, feat_max_dist=tiny_max).expert_similarity(f)
    off_diagonal = sim[~torch.eye(B, dtype=torch.bool)]
    assert torch.all(off_diagonal == 0.0)


# --- Normalised distances (Kim et al., 2021) ----------------------------------------------

def test_normalized_distance_has_unit_row_mean(batch):
    z, _ = batch
    dist = ExpCLRLoss(DEVICE).normalized_distance(z)
    assert dist.shape == (B, B)
    assert torch.allclose(dist.mean(dim=1), torch.ones(B), atol=1e-4)


# --- Loss behaviour -------------------------------------------------------------------------

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
    """With s_ij = 1 the target margin vanishes, so L_ij = D_ij^2 exactly."""
    z = torch.randn(B, E)
    f = torch.ones(B, D)
    criterion = ExpCLRLoss(DEVICE, temperature=None, feat_max_dist=1.0)
    expected = criterion.normalized_distance(z).pow(2).mean()
    assert torch.allclose(criterion(z, f), expected, atol=1e-6)


def test_proposition_3b_large_tau_converges_to_quadratic_loss(batch):
    """Prop. 3(b): as tau -> inf, the ExpCLR loss reduces to the plain quadratic loss (Eq. 3)."""
    # float64: the limit subtracts two nearly equal log-sum-exp terms and then rescales by tau,
    # which in float32 amplifies rounding error above any meaningful tolerance.
    z, f = (t.double() for t in batch)
    quadratic = ExpCLRLoss(DEVICE, temperature=None)(z, f).item()
    mined = ExpCLRLoss(DEVICE, temperature=1e6)(z, f).item()
    assert mined == pytest.approx(quadratic, rel=1e-6)


def test_proposition_3a_small_tau_converges_to_max_pair_loss(batch):
    """Prop. 3(a): as tau -> 0, the ExpCLR loss approaches max_ij L_ij.

    The residual is exactly the tau * log(N^2) normalisation term of Eq. 4, which vanishes with
    tau; at tau = 1e-5 over a 12x12 batch it is below 5e-5.
    """
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
    """Unlike NTXentLoss, ExpCLRLoss builds no fixed mask, so any batch size works."""
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


# --- Descriptor preparation -----------------------------------------------------------------

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


# --- Window alignment -----------------------------------------------------------------------

def _synthetic(n_windows=6, n_missing=2):
    """Builds a metadata frame and a feature table missing the last ``n_missing`` windows."""
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
    # Row i of the matrix must carry the features of window i, not of the i-th matched row.
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
    # 36, not 56: settled by reproducing the Ridge of the definition document (section 5.2).
    # The oscillatory block is not part of this ablation.
    assert len(DESCRIPTORS["P_madurativo"]) == 36
    assert not any(c.startswith("osc_") for c in DESCRIPTORS["P_madurativo"])
    assert len(DESCRIPTORS["P_aper"]) == 8
    assert len(set(DESCRIPTORS["P_full"])) == 106
    # Every ablation must be a strict subset of the main descriptor.
    for name in ("P_madurativo", "P_aper"):
        assert set(DESCRIPTORS[name]).issubset(set(DESCRIPTORS["P_full"]))


# --- The structural clash between Kim's normalisation and Delta ---------------------------
#
# These tests pin down the finding that explains why pre-training degraded the representation:
# normalising by mu_i forces every row mean of D_ij to equal 1, so a Delta that does not match the
# scale of the similarities leaves a bias no encoder can remove. Kept as a verified property of the
# objective rather than as a note in a document.

def test_row_mean_of_normalised_distance_is_always_one():
    """mu_i is the mean of row i, so D_ij = dist/mu_i has row mean 1 by construction."""
    loss = ExpCLRLoss(torch.device("cpu"))
    for n, dim, scale in ((8, 3, 1.0), (64, 128, 1e-3), (16, 5, 1e4)):
        z = torch.randn(n, dim, generator=torch.Generator().manual_seed(n)) * scale
        row_means = loss.normalized_distance(z).mean(dim=1)
        assert torch.allclose(row_means, torch.ones(n), atol=1e-5), (
            f"row mean should be 1 for n={n}, dim={dim}, scale={scale}")


def test_dataset_wide_max_shrinks_the_target_far_below_the_imposed_row_mean():
    """Where the bias comes from: a dataset-wide max_kl against batch-sized distances.

    ``--sim_max train`` takes ``max_kl ||f_k - f_l||`` over the whole training split (thousands of
    windows), while ``s_ij`` is evaluated within a batch of 64 whose pairs are merely typical. The
    ratio ``d_f / max_dist`` is therefore small, ``s_ij`` large, and the target ``(1 - s_ij)`` far
    below the 1 that the normalisation imposes. Taking the max per batch -- which the paper allows
    just as explicitly (Sec. 3.4) -- shrinks that gap considerably.
    """
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

    # El objetivo con max global queda más lejos de 1 que con max de batch: ese hueco es el sesgo.
    assert abs(1.0 - with_dataset_max) > abs(1.0 - with_batch_max)
    assert with_dataset_max < with_batch_max


def test_equidistant_geometry_sits_at_n_over_n_minus_one_not_at_one():
    """The reference geometry has off-diagonal N/(N-1), because mu_i averages in the zero diagonal.

    A matrix of ones has row mean (N-1)/N and is produced by no embedding whatsoever, so using it
    as the reference understates the floor. Checked against embeddings that really are equidistant.
    """
    loss = ExpCLRLoss(torch.device("cpu"))
    for n in (8, 16, 64):
        # Rows of a scaled identity are mutually equidistant.
        z = torch.eye(n, dtype=torch.float64)
        D = loss.normalized_distance(z)
        off_diagonal = D[~torch.eye(n, dtype=bool)]
        assert torch.allclose(off_diagonal, torch.full_like(off_diagonal, n / (n - 1.0)),
                              atol=1e-9), f"para n={n} se esperaba {n / (n - 1.0)}"
        assert torch.allclose(equidistant_reference(n, D.device).double(), D, atol=1e-9)


def test_matching_delta_equalises_the_row_means():
    """The fix: Delta* = 1 / E[1 - s_ij] puts the target's row mean at the imposed 1.

    Stated as the property that actually holds. The residual's minimiser is Delta*/(1 + CV^2), not
    Delta* itself, so asserting that Delta* minimises the residual would only pass for descriptors
    whose targets have a low coefficient of variation.
    """
    device = torch.device("cpu")
    f = torch.randn(64, 40, generator=torch.Generator().manual_seed(0))
    max_dist = float(torch.cdist(f, f, p=2).max())

    probe = ExpCLRLoss(device, feat_max_dist=max_dist, temperature=None)
    # Averaged over all N entries, including the diagonal where (1 - s_ii) = 0, because mu_i
    # averages over all N entries too, including dist_ii = 0.
    mean_target = float((1.0 - probe.expert_similarity(f)).mean())
    delta_star = 1.0 / mean_target

    def row_means_at(delta):
        criterion = ExpCLRLoss(device, delta=delta, feat_max_dist=max_dist, temperature=None)
        return ((1.0 - criterion.expert_similarity(f)) * delta).mean(dim=1)

    # Delta* equalises the mean across rows. Individual rows still vary -- that residual spread is
    # exactly the learnable part of the objective, the shape of the geometry.
    assert row_means_at(delta_star).mean() == pytest.approx(1.0, abs=1e-5)
    # Delta=1 leaves the target's row mean short of the 1 the normalisation imposes.
    assert row_means_at(1.0).mean() < 0.95


# --- Reuse of checkpoints -----------------------------------------------------------------
#
# Regression tests for the bug that silently invalidated a whole sweep: reuse was keyed on the file
# name, so switching --loss_on returned the 45 encoders trained under the previous setting.

def _write_checkpoint(tmp_path, **config):
    """Writes a dummy checkpoint plus the sidecar train_expclr.py would produce."""
    ckpt = tmp_path / "encoder.pth"
    ckpt.write_bytes(b"weights")
    (tmp_path / "encoder_config.json").write_text(json.dumps(config))
    return ckpt


def test_checkpoint_with_a_different_loss_target_is_not_reused(tmp_path):
    """The exact failure: a projection-trained encoder must not satisfy an embedding request."""
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
    """Checkpoints predating this check have unknown provenance, so they must be retrained."""
    ckpt = tmp_path / "orphan.pth"
    ckpt.write_bytes(b"weights")
    assert not checkpoint_is_reusable(ckpt, {"loss_on": "embedding"})


def test_missing_checkpoint_is_not_reusable(tmp_path):
    assert not checkpoint_is_reusable(tmp_path / "absent.pth", {})


def test_reuse_tolerates_float_representation(tmp_path):
    """A delta written as 1.8599999 must still match a request for 1.86."""
    ckpt = _write_checkpoint(tmp_path, delta=1.8599999999)
    assert checkpoint_is_reusable(ckpt, {"delta": 1.86})


def test_unknown_key_blocks_reuse(tmp_path):
    """A setting the sidecar predates cannot be assumed to have matched."""
    ckpt = _write_checkpoint(tmp_path, loss_on="embedding")
    assert not checkpoint_is_reusable(ckpt, {"sim_max": "train"})


# --- Effective dimensionality -------------------------------------------------------------

def test_effective_dimensionality_counts_the_directions_actually_used():
    """A cloud spread evenly over k orthogonal directions must score exactly k."""
    rng = np.random.default_rng(0)
    for k in (1, 3, 10):
        # Orthonormal basis and equal variance per direction, so the eigenvalues are identical and
        # exp(entropy) lands on k. A random mixing matrix would give unequal eigenvalues and a
        # value below k, which measures the mixing rather than the estimator.
        basis = np.linalg.qr(rng.normal(size=(64, k)))[0]
        z = rng.normal(size=(500, k)) @ basis.T
        assert effective_dimensionality(z) == pytest.approx(k, rel=0.05), f"esperaba ~{k}"


def test_effective_dimensionality_survives_a_diverged_encoder():
    """A diverged run must yield NaN, not abort the sweep with LinAlgError."""
    z = np.full((10, 4), np.nan)
    assert np.isnan(effective_dimensionality(z))
    z = np.ones((10, 4)) * np.inf
    assert np.isnan(effective_dimensionality(z))
