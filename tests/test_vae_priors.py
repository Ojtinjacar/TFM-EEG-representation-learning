import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loss import (
    StandardNormalPrior,
    ConditionalGaussianPrior,
    build_prior,
    vae_elbo,
    kl_beta,
)
from models import (
    VariationalAttentionLSTMAutoencoder,
    ConditionalVariationalAttentionLSTMAutoencoder,
)
from utils import split_dataset, ages_to_indices, CANONICAL_AGES

B, C, T = 8, 4, 1250
H = 32
N_COND = len(CANONICAL_AGES)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def posterior():
    mu = torch.randn(B, H)
    logvar = torch.randn(B, H)
    return mu, logvar


def test_standard_prior_kl_is_nonnegative_scalar(posterior):
    mu, logvar = posterior
    kl = StandardNormalPrior().kl_divergence(mu, logvar)
    assert kl.dim() == 0 and torch.isfinite(kl) and kl.item() >= 0.0


def test_free_bits_floors_the_kl(posterior):
    mu, logvar = posterior
    prior = StandardNormalPrior()
    assert prior.kl_divergence(mu, logvar, free_bits=0.5).item() >= \
        prior.kl_divergence(mu, logvar).item() - 1e-6


def test_conditional_prior_equals_standard_at_init(posterior):
    mu, logvar = posterior
    cond = torch.randint(0, N_COND, (B,))
    kl_c = ConditionalGaussianPrior(N_COND, H).kl_divergence(mu, logvar, cond=cond)
    kl_s = StandardNormalPrior().kl_divergence(mu, logvar)
    assert torch.allclose(kl_c, kl_s, atol=1e-5)


def test_conditional_prior_requires_cond(posterior):
    mu, logvar = posterior
    with pytest.raises(ValueError):
        ConditionalGaussianPrior(N_COND, H).kl_divergence(mu, logvar, cond=None)


def test_build_prior_factory():
    assert isinstance(build_prior("standard"), StandardNormalPrior)
    assert isinstance(build_prior("conditional", N_COND, H), ConditionalGaussianPrior)
    with pytest.raises(ValueError):
        build_prior("unknown")


def test_kl_beta_annealing():
    assert kl_beta(0, 0.1, 10) == pytest.approx(0.01)
    assert kl_beta(100, 0.1, 10) == 0.1
    assert kl_beta(5, 0.1, 0) == 0.1


def test_vae_forward_and_elbo_backward():
    x = torch.randn(B, C, T)
    vae = VariationalAttentionLSTMAutoencoder(input_size=T, hidden_size=H, n_channels=C,
                                              sfreq=250, lstm_hidden_size=H // 2)
    recon, mu, logvar, z = vae(x)
    assert recon.shape == (B, C, T) and mu.shape == (B, H) and z.shape == (B, H)
    loss, _, _ = vae_elbo(recon, x, mu, logvar, vae.prior, beta=0.1, free_bits=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert vae.get_embedding(x).shape == (B, H)


def test_vae_state_dict_has_no_prior_keys():
    vae = VariationalAttentionLSTMAutoencoder(input_size=T, hidden_size=H, n_channels=C,
                                              sfreq=250, lstm_hidden_size=H // 2)
    assert not any(k.startswith("prior.") for k in vae.state_dict())


def _make_cvae():
    return ConditionalVariationalAttentionLSTMAutoencoder(
        input_size=T, hidden_size=H, n_conditions=N_COND, n_channels=C, sfreq=250,
        lstm_hidden_size=H // 2, cond_dim=8, prior=build_prior("conditional", N_COND, H))


def test_cvae_forward_and_embedding():
    x = torch.randn(B, C, T)
    cond = torch.randint(0, N_COND, (B,))
    cvae = _make_cvae()
    recon, mu, logvar, z = cvae(x, cond)
    assert recon.shape == (B, C, T)
    loss, _, _ = vae_elbo(recon, x, mu, logvar, cvae.prior, beta=0.1, cond=cond)
    loss.backward()
    assert torch.isfinite(loss)
    assert cvae.get_embedding(x).shape == (B, H)
    assert cvae.get_embedding(x, cond).shape == (B, H)


def test_cvae_state_dict_has_prior_and_cond_keys():
    sd = _make_cvae().state_dict()
    assert any(k.startswith("prior.") for k in sd)
    assert any(k.startswith("cond_embed") for k in sd)


def test_cvae_checkpoint_round_trip(tmp_path):
    x = torch.randn(B, C, T)
    cvae = _make_cvae()
    ckpt = tmp_path / "cvae.pth"
    torch.save(cvae.state_dict(), ckpt)
    rebuilt = ConditionalVariationalAttentionLSTMAutoencoder(
        input_size=T, hidden_size=H, n_conditions=N_COND, n_channels=C, sfreq=250,
        lstm_hidden_size=H // 2, cond_dim=8)
    missing, unexpected = rebuilt.load_state_dict(torch.load(ckpt), strict=False)
    assert not missing and not unexpected
    assert rebuilt.get_embedding(x).shape == (B, H)


def test_ages_to_indices_maps_and_rejects():
    idx = ages_to_indices([6, 9, 16, 36, 6])
    assert idx.min() == 0 and idx.max() == N_COND - 1
    with pytest.raises(ValueError):
        ages_to_indices([99])


def test_split_dataset_carries_condition():
    X = np.random.randn(16, C, T).astype(np.float32)
    cond = ages_to_indices(list(CANONICAL_AGES) * 4)
    train, _ = split_dataset(X, cond=cond)
    assert len(train[0]) == 3
    with pytest.raises(ValueError):
        split_dataset(X, cond=cond[:4])
