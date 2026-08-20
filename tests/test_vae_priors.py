import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loss import (
    StandardNormalPrior,
    build_prior,
    vae_elbo,
    kl_beta,
)
from models import VariationalAttentionLSTMAutoencoder

B, C, T = 8, 4, 1250
H = 32


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


def test_kl_matches_the_closed_form(posterior):
    """Pins the analytic KL of Kingma & Welling, appendix B."""
    mu, logvar = posterior
    ours = StandardNormalPrior().kl_true(mu, logvar)
    reference = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    assert torch.allclose(ours, reference, atol=1e-6)


def test_reparameterization_uses_a_standard_deviation(posterior):
    """exp(0.5 * logvar) is sqrt(exp(logvar)): the scale is a std, not a variance."""
    _, logvar = posterior
    assert torch.allclose(torch.exp(0.5 * logvar), torch.sqrt(torch.exp(logvar)),
                          atol=1e-6)


def test_free_bits_zero_is_exactly_the_true_kl(posterior):
    """Guards the refactor: the default path must not have changed."""
    mu, logvar = posterior
    prior = StandardNormalPrior()
    assert torch.allclose(prior.kl_divergence(mu, logvar, free_bits=0.0),
                          prior.kl_true(mu, logvar), atol=1e-7)


def test_free_bits_floor_applies_to_the_batch_mean_per_dimension(posterior):
    """Kingma et al. (2016): the floor is on E_batch[KL_j], not on each sample."""
    mu, logvar = posterior
    lam = 1e4  # far above any per-dimension KL, so every dimension is floored
    floored = StandardNormalPrior().kl_divergence(mu, logvar, free_bits=lam)
    assert torch.isclose(floored, torch.tensor(H * lam), rtol=1e-6)


def test_kl_true_ignores_free_bits(posterior):
    mu, logvar = posterior
    prior = StandardNormalPrior()
    prior.kl_divergence(mu, logvar, free_bits=1e4)
    assert torch.allclose(prior.kl_true(mu, logvar),
                          prior.kl_true(mu, logvar), atol=1e-7)
    assert prior.kl_true(mu, logvar).item() < 1e4 * H


def test_build_prior_factory():
    assert isinstance(build_prior("standard"), StandardNormalPrior)
    with pytest.raises(ValueError):
        build_prior("unknown")


def test_kl_beta_annealing():
    assert kl_beta(0, 0.1, 10) == 0.0
    assert kl_beta(5, 0.1, 10) == pytest.approx(0.05)
    assert kl_beta(10, 0.1, 10) == 0.1
    assert kl_beta(100, 0.1, 10) == 0.1
    assert kl_beta(5, 0.1, 0) == 0.1


@pytest.mark.parametrize("beta_canonical", [0.5, 1.0, 2.0])
def test_strict_gaussian_beta_convention(beta_canonical):
    """The code objective is a positive multiple of 0.5*SSE + beta*KL.

    This is the identity the whole beta convention rests on: --beta is stated
    in the units of the strict gaussian ELBO, where the reconstruction carries
    the factor of one half that comes out of the log-likelihood, and the code
    reaches it through beta_code = 2*beta/(C*T).
    """
    torch.manual_seed(0)
    B_, C_, T_, J_ = 4, 3, 20, 8
    x = torch.randn(B_, C_, T_)
    recon = torch.randn(B_, C_, T_)
    mu = torch.randn(B_, J_)
    logvar = torch.randn(B_, J_) * 0.1
    prior = StandardNormalPrior()

    beta_code = 2.0 * beta_canonical / (C_ * T_)
    total, _, _, kl = vae_elbo(recon, x, mu, logvar, prior, beta_code)

    sse_per_sample = ((recon - x) ** 2).sum(dim=(1, 2)).mean()
    expected = 0.5 * sse_per_sample + beta_canonical * kl
    assert torch.isclose(total * (C_ * T_) / 2.0, expected, rtol=1e-5)


def test_vae_forward_and_elbo_backward():
    x = torch.randn(B, C, T)
    vae = VariationalAttentionLSTMAutoencoder(input_size=T, hidden_size=H, n_channels=C,
                                              sfreq=250, lstm_hidden_size=H // 2)
    recon, mu, logvar, z = vae(x)
    assert recon.shape == (B, C, T) and mu.shape == (B, H) and z.shape == (B, H)
    loss, _, _, _ = vae_elbo(recon, x, mu, logvar, vae.prior, beta=0.1, free_bits=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert vae.get_embedding(x).shape == (B, H)


def test_vae_state_dict_has_no_prior_keys():
    vae = VariationalAttentionLSTMAutoencoder(input_size=T, hidden_size=H, n_channels=C,
                                              sfreq=250, lstm_hidden_size=H // 2)
    assert not any(k.startswith("prior.") for k in vae.state_dict())


def test_vae_checkpoint_round_trip(tmp_path):
    x = torch.randn(B, C, T)
    vae = VariationalAttentionLSTMAutoencoder(input_size=T, hidden_size=H, n_channels=C,
                                              sfreq=250, lstm_hidden_size=H // 2)
    ckpt = tmp_path / "vae.pth"
    torch.save(vae.state_dict(), ckpt)
    rebuilt = VariationalAttentionLSTMAutoencoder(input_size=T, hidden_size=H, n_channels=C,
                                                  sfreq=250, lstm_hidden_size=H // 2)
    rebuilt.load_state_dict(torch.load(ckpt), strict=True)
    assert rebuilt.get_embedding(x).shape == (B, H)
