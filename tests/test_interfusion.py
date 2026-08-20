import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from checkpoint_naming import interfusion_checkpoint_name
from interfusion import (
    InterFusionEEG,
    PretrainVAE,
    RealNVP,
    RecurrentGaussian,
    gaussian_log_prob,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

M, W, B = 5, 200, 4


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def model():
    return InterFusionEEG(x_dim=M, window=W, z_dim=3,
                          strides=(2, 1, 2, 1, 2),
                          rnn_hidden=16, dense_hidden=16, flow_levels=3)


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------

def test_encoder_decoder_shapes_round_trip(model):
    x = torch.randn(B, M, W)
    mean, logstd = model.encode_z2(x)
    # W=200 with strides (2,1,2,1,2): 100 -> 100 -> 50 -> 50 -> 25
    assert mean.shape == (B, M, 25) and model.w_prime == 25
    assert logstd.max() <= 2.0 and logstd.min() >= -5.0
    d = model.decoder(mean)
    assert d.shape == (B, M, W)


def test_elbo_finite_and_backward(model):
    x = torch.randn(B, M, W)
    out = model(x)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_free_bits_defaults_to_the_plain_sgvb_objective():
    """free_bits=0 must leave the paper's objective bit-identical."""
    kwargs = dict(x_dim=M, window=W, z_dim=3, strides=(2, 1, 2, 1, 2),
                  rnn_hidden=16, dense_hidden=16, flow_levels=3)
    x = torch.randn(B, M, W)

    torch.manual_seed(1)
    plain = InterFusionEEG(**kwargs)(x)["loss"]
    torch.manual_seed(1)
    floored = InterFusionEEG(free_bits=0.0, **kwargs)(x)["loss"]

    assert torch.equal(plain, floored)


def test_free_bits_raises_the_loss_and_stays_differentiable():
    """A positive floor can only add KL cost, never remove it."""
    kwargs = dict(x_dim=M, window=W, z_dim=3, strides=(2, 1, 2, 1, 2),
                  rnn_hidden=16, dense_hidden=16, flow_levels=3)
    x = torch.randn(B, M, W)

    torch.manual_seed(2)
    baseline = InterFusionEEG(**kwargs)(x)["loss"]
    torch.manual_seed(2)
    model = InterFusionEEG(free_bits=0.5, **kwargs)
    out = model(x)

    assert torch.isfinite(out["loss"])
    assert out["loss"] >= baseline - 1e-4
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_embedding_shape_and_determinism(model):
    x = torch.randn(B, M, W)
    e1 = model.get_embedding(x)
    e2 = model.get_embedding(x)
    assert e1.shape == (B, M + 3)
    assert torch.equal(e1, e2), "the embedding must be deterministic"


def test_embedding_mean_std_doubles_dims_and_rejects_unknown(model):
    x = torch.randn(B, M, W)
    e = model.get_embedding(x, stats="mean_std")
    assert e.shape == (B, 2 * (M + 3))
    assert torch.isfinite(e).all()
    # the first half must equal the plain mean embedding
    assert torch.equal(e[:, : M + 3], model.get_embedding(x))
    with pytest.raises(ValueError):
        model.get_embedding(x, stats="median")


def test_get_score_shape_and_separates_fitted_data_from_noise(model):
    torch.manual_seed(3)
    x = torch.randn(B, M, W) * 0.1
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(30):
        out = model(x)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        score_train = model.get_score(x, n_samples=5)
        score_noise = model.get_score(torch.randn(B, M, W) * 5.0, n_samples=5)
    assert score_train.shape == (B,)
    assert torch.isfinite(score_train).all()
    assert score_train.mean() > score_noise.mean(), \
        "fitted windows must be more likely than out-of-scale noise"


# ---------------------------------------------------------------------------
# RealNVP
# ---------------------------------------------------------------------------

def test_realnvp_logdet_matches_autograd_jacobian():
    torch.manual_seed(1)
    flow = RealNVP(dim=3, n_levels=2, hidden_units=8)
    z = torch.randn(1, 3, dtype=torch.double)
    flow = flow.double()
    y, log_det = flow(z)
    jac = torch.autograd.functional.jacobian(lambda t: flow(t)[0], z)
    jac = jac.reshape(3, 3)
    expected = torch.logdet(jac)
    assert torch.isclose(log_det.squeeze(), expected, atol=1e-6)


def test_realnvp_is_injective_on_samples():
    flow = RealNVP(dim=4, n_levels=3)
    z = torch.randn(64, 4)
    y, _ = flow(z)
    assert torch.isfinite(y).all()
    # distinct inputs must map to distinct outputs (invertibility smoke)
    dists = torch.cdist(y, y) + torch.eye(64) * 1e9
    assert dists.min() > 1e-8


# ---------------------------------------------------------------------------
# RecurrentGaussian
# ---------------------------------------------------------------------------

def test_recurrent_log_prob_matches_manual_recurrence():
    torch.manual_seed(2)
    rg = RecurrentGaussian(input_dim=4, z_dim=2)
    cond = torch.randn(1, 3, 4)
    z = torch.randn(1, 3, 2)
    lp = rg.log_prob(z, cond)

    z_prev = torch.zeros(1, 2)
    manual = []
    for t in range(3):
        mean, logstd = rg._params(cond[:, t], z_prev)
        manual.append(gaussian_log_prob(z[:, t], mean, logstd))
        z_prev = z[:, t]
    manual = torch.stack(manual, dim=1)
    assert torch.allclose(lp, manual, atol=1e-6)


def test_recurrent_sampling_is_seeded_and_deterministic_mode_matches_mean():
    rg = RecurrentGaussian(input_dim=4, z_dim=2)
    cond = torch.randn(2, 5, 4)
    torch.manual_seed(7)
    z_a, _, _ = rg.sample(cond)
    torch.manual_seed(7)
    z_b, _, _ = rg.sample(cond)
    assert torch.equal(z_a, z_b)
    z_det, mean, _ = rg.sample(cond, deterministic=True)
    assert torch.equal(z_det, mean)


# ---------------------------------------------------------------------------
# prefiltering and weight sharing
# ---------------------------------------------------------------------------

def test_z1_sees_x_only_through_d(model):
    """The gradient of z1's params w.r.t. x must flow ONLY through d=g(z2)."""
    x = torch.randn(B, M, W, requires_grad=True)
    qz2_mean, _ = model.encode_z2(x)
    d = model.decoder(qz2_mean)
    a_seq = model._arnn_features(d)
    _, z1_mean, _ = model.qz1.sample(a_seq, deterministic=True)
    grad = torch.autograd.grad(z1_mean.sum(), x, retain_graph=True)[0]
    assert torch.isfinite(grad).all()
    # Cutting d must cut the whole path: gradient through a detached d is None
    a_cut = model._arnn_features(d.detach())
    _, z1_cut, _ = model.qz1.sample(a_cut, deterministic=True)
    grad_cut = torch.autograd.grad(z1_cut.sum(), x, allow_unused=True)[0]
    assert grad_cut is None


def test_pretrain_shares_conv_deconv_with_main(model):
    """Stage-1 training must move the main model's conv/deconv weights."""
    pre = PretrainVAE(model)
    x = torch.randn(B, M, W)
    before = model.encoder.convs[0].weight.detach().clone()
    opt = torch.optim.SGD(pre.parameters(), lr=0.05)
    out = pre(x)
    opt.zero_grad()
    out["loss"].backward()
    opt.step()
    after = model.encoder.convs[0].weight.detach()
    assert not torch.equal(before, after), \
        "pretraining must update the shared encoder weights in place"


# ---------------------------------------------------------------------------
# naming contract
# ---------------------------------------------------------------------------

def test_interfusion_name_contract():
    assert interfusion_checkpoint_name("all", "all", "fold0") == (
        "InterFusion_all_all_fold0_m4_w40_e20.pth"
    )
    assert interfusion_checkpoint_name("all", "all", None, z_dim=3,
                                       w_prime=25, epochs=5) == (
        "InterFusion_all_all_m3_w25_e5.pth"
    )


# ---------------------------------------------------------------------------
# end-to-end reproducibility (CPU)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("interfusion_data")
    rng = np.random.default_rng(0)
    n = 24
    X = rng.normal(size=(n, 3, 200)).astype(np.float32)
    meta = pd.DataFrame({"subject": [f"B{i % 6}" for i in range(n)]})
    np.save(tmp / "windows.npy", X)
    meta.to_csv(tmp / "meta.csv", index=False)
    return str(tmp / "windows.npy"), str(tmp / "meta.csv"), str(tmp)


def _run_train(data_path, meta_path, out_dir, seed):
    save_dir = os.path.join(out_dir, f"models_{seed}_{np.random.randint(1 << 30)}")
    cmd = [
        sys.executable, "src/train_interfusion.py",
        "--data-path", data_path,
        "--meta-path", meta_path,
        "--pretrain-epochs", "1",
        "--epochs", "1",
        "--batch-size", "8",
        "--z-dim", "2",
        "--rnn-hidden", "8",
        "--flow-levels", "2",
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


def test_train_interfusion_is_reproducible_on_cpu(tiny_dataset):
    data_path, meta_path, tmp = tiny_dataset
    s1 = _run_train(data_path, meta_path, tmp, seed=7)
    s2 = _run_train(data_path, meta_path, tmp, seed=7)
    s3 = _run_train(data_path, meta_path, tmp, seed=8)
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    assert any(not torch.equal(s1[k], s3[k]) for k in s1)
