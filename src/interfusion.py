"""InterFusion (Li et al., KDD 2021) ported to PyTorch for EEG windows.

Hierarchical VAE with two-view latent embeddings for multivariate time series:
z2 compresses the time axis (temporal embedding, shape (B, M, W')) and z1
compresses the metric axis (inter-metric embedding, shape (B, W, M')). z1 is
inferred from the deterministic "reconstructed input" d = g(z2) (prefiltering
strategy), never from x directly. The posterior of z1 is enriched with a
RealNVP flow. Training follows the paper's two-stage procedure: a plain
z2-only VAE is pretrained first and its conv/deconv weights initialize the
full model (they are shared module instances here, so the transfer is
implicit).

Port of the official TensorFlow implementation
https://github.com/zhhlee/InterFusion (MIT License, Copyright (c) 2021
zhhlee); reference: Li et al., "Multivariate Time Series Anomaly Detection
and Interpretation using Hierarchical Inter-Metric and Temporal Embedding",
KDD 2021, doi:10.1145/3447548.3467075.

Fidelity notes (verified against the official code):
- Objective: plain hierarchical SGVB (no beta, no KL annealing), see
  ``stack_train.sgvb_loss``.
- Conv/deconv stacks: kernel 5, channels = number of metrics, TF 'same'
  padding semantics (ceil(L/stride)); the deconv stack is a single shared
  instance used both for d (variational net) and e (generative net), which is
  what makes the deterministic variables cancel in the ELBO (paper Eq. 3).
- q(z1_t | z1_{t-1}, a_t) and p(z1_t | z1_{t-1}, e_t) are sequential
  Gaussians with shared dense heads and zero initial state
  (``recurrent_distribution.py``); log-std clipped to [-5, 2] everywhere.
- RealNVP: 20 levels of (invertible linear + affine coupling), coupling MLP
  with one hidden layer of 100 ReLU units, zero-initialized scale/shift head,
  sigmoid scale with bias 2 (``real_nvp.py``).
- Activation is ReLU (config ``use_leaky_relu=False`` in the official runs).

Adaptations for the TFM (documented deviations from the paper):
- Each 5 s EEG window (M channels x T samples) is one sample; there is no
  stride-1 sliding window over a continuous stream.
- The conv stack uses a longer stride list to compress T=1250 (vs W=100 in
  the paper) to a comparable W'.
- ``get_embedding`` exposes a deterministic (B, M + M') embedding for the
  downstream age-regression protocol: concat of the time-averaged z1 mean and
  the W'-averaged z2 mean. Anomaly scoring, MCMC imputation and IPS are out
  of scope.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Readout modes of ``get_embedding`` and how many blocks of (x_dim + z_dim) each one
# concatenates. The mode is a decision about how the trained model is read, not about
# how it is trained, so it never affects the weights and existing checkpoints stay valid.
EMBEDDING_STATS = {"mean": 1, "mean_std": 2}

LOGSTD_MIN = -5.0
LOGSTD_MAX = 2.0
_HALF_LOG_2PI = 0.9189385332046727  # 0.5 * log(2 * pi)


def gaussian_log_prob(x, mean, logstd):
    """Elementwise diagonal-Gaussian log density.

    Args:
        x (torch.Tensor): Points to evaluate.
        mean (torch.Tensor): Gaussian means, broadcastable to ``x``.
        logstd (torch.Tensor): Log standard deviations, broadcastable to ``x``.

    Returns:
        torch.Tensor: Elementwise log probabilities, same shape as ``x``.
    """
    precision = torch.exp(-2.0 * logstd)
    return -_HALF_LOG_2PI - logstd - 0.5 * precision * (x - mean) ** 2


def _same_pad(x, kernel_size, stride):
    """Applies TF 'same' asymmetric padding along the last axis."""
    length = x.shape[-1]
    out_len = -(-length // stride)  # ceil
    pad_total = max((out_len - 1) * stride + kernel_size - length, 0)
    left = pad_total // 2
    return F.pad(x, (left, pad_total - left))


class ConvStack(nn.Module):
    """TF-'same' Conv1d stack with per-layer ReLU (encoder for z2).

    Input and output layout is channels-first: (B, M, L). The number of
    channels is kept at ``x_dim`` in every layer, as in the paper.
    """

    def __init__(self, x_dim, strides, kernel_size=5):
        super().__init__()
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList(
            nn.Conv1d(x_dim, x_dim, kernel_size) for _ in self.strides
        )

    def forward(self, x):
        for conv, stride in zip(self.convs, self.strides):
            x = _same_pad(x, self.kernel_size, stride)
            x = F.relu(F.conv1d(x, conv.weight, conv.bias, stride=stride))
        return x

    def output_lengths(self, length):
        """Returns the per-layer output lengths for an input of ``length``."""
        lengths = []
        for stride in self.strides:
            length = -(-length // stride)
            lengths.append(length)
        return lengths


class DeconvStack(nn.Module):
    """Mirror ConvTranspose1d stack producing exact target lengths.

    A single instance is shared between the variational net (producing d) and
    the generative net (producing e); do NOT duplicate it, the analytic
    cancellation of the deterministic variables in the ELBO depends on the
    sharing (paper Eq. 3).
    """

    def __init__(self, x_dim, strides, output_lengths, kernel_size=5):
        super().__init__()
        assert len(strides) == len(output_lengths)
        self.strides = list(strides)
        self.output_lengths = list(output_lengths)
        self.kernel_size = kernel_size
        pad = (kernel_size - 1) // 2
        self.deconvs = nn.ModuleList(
            nn.ConvTranspose1d(x_dim, x_dim, kernel_size, stride=s, padding=pad)
            for s in self.strides
        )

    def forward(self, z):
        h = z
        for deconv, target in zip(self.deconvs, self.output_lengths):
            h = deconv(h)
            if h.shape[-1] > target:
                h = h[..., :target]
            elif h.shape[-1] < target:
                h = F.pad(h, (0, target - h.shape[-1]))
            h = F.relu(h)
        return h


class AffineCoupling(nn.Module):
    """RealNVP affine coupling on the last axis (paper: sigmoid scale, bias 2)."""

    def __init__(self, dim, hidden_units=100, sigmoid_scale_bias=2.0):
        super().__init__()
        self.n1 = dim // 2
        self.n2 = dim - self.n1
        self.hidden = nn.Linear(self.n1, hidden_units)
        self.head = nn.Linear(hidden_units, self.n2 * 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.sigmoid_scale_bias = sigmoid_scale_bias

    def forward(self, x):
        x1, x2 = x[..., : self.n1], x[..., self.n1:]
        params = self.head(F.relu(self.hidden(x1)))
        shift = params[..., : self.n2]
        scale = torch.sigmoid(params[..., self.n2:] + self.sigmoid_scale_bias)
        y2 = x2 * scale + shift
        log_det = torch.log(scale).sum(dim=-1)
        return torch.cat([x1, y2], dim=-1), log_det


class InvertibleLinear(nn.Module):
    """PLU-parameterized invertible linear map on the last axis.

    Equivalent role to tfsnippet's ``InvertibleDense`` (non-strict): a
    learnable change of basis between coupling layers with a cheap exact
    log-determinant.
    """

    def __init__(self, dim):
        super().__init__()
        w = torch.linalg.qr(torch.randn(dim, dim)).Q
        p, l, u = torch.linalg.lu(w)
        self.register_buffer("perm", p)
        self.l_raw = nn.Parameter(l)
        self.u_raw = nn.Parameter(torch.triu(u, diagonal=1))
        self.register_buffer("diag_sign", torch.sign(torch.diagonal(u)))
        self.log_diag = nn.Parameter(torch.log(torch.abs(torch.diagonal(u))))
        self.register_buffer("l_mask", torch.tril(torch.ones(dim, dim), -1))
        self.register_buffer("eye", torch.eye(dim))

    def _weight(self):
        l = self.l_raw * self.l_mask + self.eye
        u = torch.triu(self.u_raw, diagonal=1) + torch.diag(
            self.diag_sign * torch.exp(self.log_diag)
        )
        return self.perm @ l @ u

    def forward(self, x):
        y = x @ self._weight().T
        log_det = self.log_diag.sum().expand(x.shape[:-1])
        return y, log_det


class RealNVP(nn.Module):
    """Stack of (invertible linear + affine coupling) levels."""

    def __init__(self, dim, n_levels=20, hidden_units=100):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_levels):
            self.layers.append(InvertibleLinear(dim))
            self.layers.append(AffineCoupling(dim, hidden_units))

    def forward(self, x):
        log_det = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, ld = layer(x)
            log_det = log_det + ld
        return x, log_det


class RecurrentGaussian(nn.Module):
    """Sequential Gaussian z_t | z_{t-1}, c_t with shared dense heads.

    Port of ``recurrent_distribution.py``: at each step the conditioning
    input c_t is concatenated with the previous latent (zeros at t=0) and fed
    through shared mean/log-std linear heads; log-std is clipped.
    """

    def __init__(self, input_dim, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.mean_head = nn.Linear(input_dim + z_dim, z_dim)
        self.logstd_head = nn.Linear(input_dim + z_dim, z_dim)

    def _params(self, cond_t, z_prev):
        h = torch.cat([cond_t, z_prev], dim=-1)
        mean = self.mean_head(h)
        logstd = torch.clamp(self.logstd_head(h), LOGSTD_MIN, LOGSTD_MAX)
        return mean, logstd

    def sample(self, cond, deterministic=False):
        """Samples a latent sequence with the reparameterization trick.

        Args:
            cond (torch.Tensor): Conditioning sequence (B, W, F).
            deterministic (bool): If True, returns the mean sequence (no
                noise); used for downstream embeddings.

        Returns:
            tuple: (z (B, W, z), mean (B, W, z), logstd (B, W, z)).
        """
        batch, window, _ = cond.shape
        z_prev = cond.new_zeros(batch, self.z_dim)
        zs, means, logstds = [], [], []
        for t in range(window):
            mean, logstd = self._params(cond[:, t], z_prev)
            if deterministic:
                z_t = mean
            else:
                z_t = mean + torch.exp(logstd) * torch.randn_like(mean)
            zs.append(z_t)
            means.append(mean)
            logstds.append(logstd)
            z_prev = z_t
        return (torch.stack(zs, dim=1), torch.stack(means, dim=1),
                torch.stack(logstds, dim=1))

    def log_prob(self, z, cond):
        """Teacher-forced log density of a given latent sequence.

        Vectorized: with the whole sequence known, every step's conditioning
        (c_t, z_{t-1}) is available in parallel.

        Args:
            z (torch.Tensor): Latent sequence (B, W, z).
            cond (torch.Tensor): Conditioning sequence (B, W, F).

        Returns:
            torch.Tensor: Elementwise log density (B, W, z).
        """
        z_prev = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)
        mean, logstd = self._params(cond, z_prev)
        return gaussian_log_prob(z, mean, logstd)


class InterFusionEEG(nn.Module):
    """The full InterFusion HVAE adapted to fixed-length EEG windows.

    Args:
        x_dim (int): Number of channels (metrics) M.
        window (int): Samples per window W (e.g. 1250 for 5 s at 250 Hz).
        z_dim (int): Inter-metric latent width M' (paper: 2-4).
        strides (tuple[int]): Conv stack strides; W' is the resulting
            compressed length (paper: (2, 1, 2, 1, 2) for W=100 -> W'=13).
        rnn_hidden (int): Hidden units of the backward GRU (paper: 500).
        dense_hidden (int): Units of the feature/fusion dense layers (500).
        flow_levels (int): RealNVP levels on q(z1) (paper: 20).
    """

    def __init__(self, x_dim, window, z_dim=4, strides=(2, 1, 2, 1, 2, 2, 2),
                 rnn_hidden=500, dense_hidden=500, flow_levels=20,
                 embedding_stats="mean"):
        super().__init__()
        if embedding_stats not in EMBEDDING_STATS:
            raise ValueError(
                f"embedding_stats must be one of {sorted(EMBEDDING_STATS)}, "
                f"got {embedding_stats!r}"
            )
        self.x_dim = x_dim
        self.window = window
        self.z_dim = z_dim
        self.embedding_stats = embedding_stats

        self.encoder = ConvStack(x_dim, strides)
        lengths = self.encoder.output_lengths(window)
        self.w_prime = lengths[-1]
        # Mirror: reversed strides, target lengths are the encoder's
        # intermediate lengths reversed (ending at the original window).
        deconv_targets = list(reversed([window] + lengths[:-1]))
        self.decoder = DeconvStack(x_dim, list(reversed(strides)), deconv_targets)

        self.qz2_mean = nn.Conv1d(x_dim, x_dim, 1)
        self.qz2_logstd = nn.Conv1d(x_dim, x_dim, 1)

        self.arnn = nn.GRU(input_size=x_dim, hidden_size=rnn_hidden)
        self.a_feat = nn.Sequential(
            nn.Linear(rnn_hidden, dense_hidden), nn.ReLU(),
            nn.Linear(dense_hidden, dense_hidden), nn.ReLU(),
        )

        self.qz1 = RecurrentGaussian(dense_hidden, z_dim)
        self.pz1 = RecurrentGaussian(x_dim, z_dim)
        self.posterior_flow = RealNVP(z_dim, flow_levels) if flow_levels else None

        self.hz1_proj = nn.Linear(z_dim, x_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * x_dim, dense_hidden), nn.ReLU(),
            nn.Linear(dense_hidden, dense_hidden), nn.ReLU(),
        )
        self.px_mean = nn.Linear(dense_hidden, x_dim)
        self.px_logstd = nn.Linear(dense_hidden, x_dim)

        # The downstream head is sized from this, so it has to follow the readout mode.
        self.embedding_dim = EMBEDDING_STATS[embedding_stats] * (x_dim + z_dim)

    # ------------------------------------------------------------------
    # building blocks
    # ------------------------------------------------------------------
    def encode_z2(self, x):
        """q(z2 | x): conv stack + 1x1 conv heads. x is (B, M, W)."""
        h = self.encoder(x)
        mean = self.qz2_mean(h)
        logstd = torch.clamp(self.qz2_logstd(h), LOGSTD_MIN, LOGSTD_MAX)
        return mean, logstd

    def _arnn_features(self, d):
        """Backward GRU over d plus the two feature dense layers.

        Args:
            d (torch.Tensor): Reconstructed input (B, M, W).

        Returns:
            torch.Tensor: a-sequence (B, W, dense_hidden).
        """
        seq = d.permute(2, 0, 1).flip(0)          # (W, B, M), reversed time
        out, _ = self.arnn(seq)
        out = out.flip(0).permute(1, 0, 2)        # (B, W, rnn_hidden)
        return self.a_feat(out)

    def decode_x(self, z1, e):
        """p(x | z1, z2): fuse z1 with e and emit the Gaussian over x.

        Args:
            z1 (torch.Tensor): Inter-metric latent (B, W, z_dim).
            e (torch.Tensor): Deterministic decode of z2 (B, M, W).

        Returns:
            tuple: (x_mean (B, M, W), x_logstd (B, M, W)).
        """
        e_seq = e.permute(0, 2, 1)                          # (B, W, M)
        h = torch.cat([self.hz1_proj(z1), e_seq], dim=-1)   # (B, W, 2M)
        h = self.fusion(h)
        x_mean = self.px_mean(h).permute(0, 2, 1)
        x_logstd = torch.clamp(self.px_logstd(h), LOGSTD_MIN, LOGSTD_MAX)
        return x_mean, x_logstd.permute(0, 2, 1)

    # ------------------------------------------------------------------
    # objective
    # ------------------------------------------------------------------
    def elbo(self, x):
        """Computes the hierarchical SGVB objective for a batch.

        Args:
            x (torch.Tensor): Input windows (B, M, W).

        Returns:
            dict: loss (scalar, negative ELBO), recons (mean log p(x|z)),
            kl (mean KL estimate), plus per-latent diagnostics.
        """
        # q(z2 | x) and its sample
        qz2_mean, qz2_logstd = self.encode_z2(x)
        z2 = qz2_mean + torch.exp(qz2_logstd) * torch.randn_like(qz2_mean)
        logqz2 = gaussian_log_prob(z2, qz2_mean, qz2_logstd).sum(dim=(1, 2))

        # deterministic d (variational side) == e (generative side): the
        # SAME module instance decodes z2, so both terms cancel in the ELBO.
        d = self.decoder(z2)

        # q(z1 | d): sequential base sample, then the RealNVP flow.
        a_seq = self._arnn_features(d)
        z1_base, _, _ = self.qz1.sample(a_seq)
        logqz1 = self.qz1.log_prob(z1_base, a_seq).sum(dim=(1, 2))
        log_det_sum = None
        if self.posterior_flow is not None:
            z1, log_det = self.posterior_flow(z1_base)
            log_det_sum = log_det.sum(dim=1)
            logqz1 = logqz1 - log_det_sum
        else:
            z1 = z1_base

        # priors
        logpz2 = gaussian_log_prob(
            z2, torch.zeros_like(z2), torch.zeros_like(z2)
        ).sum(dim=(1, 2))
        e_seq_cond = d.permute(0, 2, 1)  # e == d (shared decode of z2)
        logpz1 = self.pz1.log_prob(z1, e_seq_cond).sum(dim=(1, 2))

        # likelihood
        x_mean, x_logstd = self.decode_x(z1, d)
        logpx = gaussian_log_prob(x, x_mean, x_logstd).sum(dim=(1, 2))

        kl_z1 = logqz1 - logpz1
        kl_z2 = logqz2 - logpz2

        loss = -(logpx + logpz1 + logpz2 - logqz1 - logqz2).mean()

        return {
            "loss": loss,
            "recons": logpx.mean(),
            "kl": (kl_z1 + kl_z2).mean(),
            "kl_z1": kl_z1.mean(),
            "kl_z2": kl_z2.mean(),
            "x_mean": x_mean,
        }

    def forward(self, x):
        """Alias of :meth:`elbo` so the module is callable."""
        return self.elbo(x)

    # ------------------------------------------------------------------
    # downstream interface
    # ------------------------------------------------------------------
    def get_embedding(self, x, stats=None):
        """Deterministic embedding for downstream heads.

        With ``stats="mean"`` (the protocol's original representation):
        concatenates the W'-averaged z2 posterior mean (M dims) with the
        time-averaged deterministic z1 mean sequence (M' dims) -> (B, M + M').
        With ``stats="mean_std"`` the temporal/W' standard deviations are
        appended -> (B, 2*(M + M')); motivated by the burst-like nature of
        infant EEG, where the temporal mean alone is degenerate. No sampling
        and no flow: stable representation, gradients flow for fine-tuning.

        Note that M is the channel count, so this representation is far
        narrower than the 128 dimensions every other method of the protocol
        uses, and it narrows further the fewer channels the region has.

        Args:
            x (torch.Tensor): Input windows (B, M, W).
            stats (str): "mean" or "mean_std". Defaults to the instance's
                ``embedding_stats``, so callers that do not care keep the
                mode the model was configured with.

        Returns:
            torch.Tensor: Embedding of shape (B, M + M') or (B, 2*(M + M')).

        Raises:
            ValueError: If ``stats`` is not a supported mode.
        """
        stats = self.embedding_stats if stats is None else stats
        if stats not in EMBEDDING_STATS:
            raise ValueError(f"Unknown stats mode: {stats!r} (use 'mean' or 'mean_std').")
        qz2_mean, _ = self.encode_z2(x)
        d = self.decoder(qz2_mean)
        a_seq = self._arnn_features(d)
        _, z1_mean, _ = self.qz1.sample(a_seq, deterministic=True)
        parts = [qz2_mean.mean(dim=2), z1_mean.mean(dim=1)]
        if stats == "mean_std":
            parts += [qz2_mean.std(dim=2), z1_mean.std(dim=1)]
        return torch.cat(parts, dim=1)

    def get_score(self, x, n_samples=10):
        """Per-window reconstruction log-probability (anomaly score basis).

        Monte Carlo estimate of E_q(z1,z2|x)[log p(x | z1, z2)], the
        "reconstruction probability" the paper uses for anomaly detection
        (its negative is the anomaly score). In the TFM this is used ONLY as
        an exploratory atypicality marker, never as a clinical claim.

        Args:
            x (torch.Tensor): Input windows (B, M, W).
            n_samples (int): Monte Carlo samples L (paper uses L=100 at test).

        Returns:
            torch.Tensor: Log-probability per window, shape (B,).
        """
        scores = []
        for _ in range(n_samples):
            qz2_mean, qz2_logstd = self.encode_z2(x)
            z2 = qz2_mean + torch.exp(qz2_logstd) * torch.randn_like(qz2_mean)
            d = self.decoder(z2)
            a_seq = self._arnn_features(d)
            z1_base, _, _ = self.qz1.sample(a_seq)
            if self.posterior_flow is not None:
                z1, _ = self.posterior_flow(z1_base)
            else:
                z1 = z1_base
            x_mean, x_logstd = self.decode_x(z1, d)
            scores.append(gaussian_log_prob(x, x_mean, x_logstd).sum(dim=(1, 2)))
        return torch.stack(scores, dim=0).mean(dim=0)


class PretrainVAE(nn.Module):
    """Stage-1 VAE with only the temporal latent z2 (paper Appendix C.1).

    Shares the conv encoder, the z2 heads and the deconv decoder with the
    main model (the very same module instances), so training this wrapper IS
    the initialization of those parts of the main model. Only the output
    heads (1x1 convs for the x Gaussian) are private to the pretrain stage.
    """

    def __init__(self, main: InterFusionEEG):
        super().__init__()
        self.main = main
        self.pre_px_mean = nn.Conv1d(main.x_dim, main.x_dim, 1)
        self.pre_px_logstd = nn.Conv1d(main.x_dim, main.x_dim, 1)

    def elbo(self, x):
        """Plain SGVB objective of the z2-only VAE.

        Args:
            x (torch.Tensor): Input windows (B, M, W).

        Returns:
            dict: loss, recons and kl scalars.
        """
        qz_mean, qz_logstd = self.main.encode_z2(x)
        z = qz_mean + torch.exp(qz_logstd) * torch.randn_like(qz_mean)
        logqz = gaussian_log_prob(z, qz_mean, qz_logstd).sum(dim=(1, 2))
        logpz = gaussian_log_prob(
            z, torch.zeros_like(z), torch.zeros_like(z)
        ).sum(dim=(1, 2))

        h = self.main.decoder(z)
        x_mean = self.pre_px_mean(h)
        x_logstd = torch.clamp(self.pre_px_logstd(h), LOGSTD_MIN, LOGSTD_MAX)
        logpx = gaussian_log_prob(x, x_mean, x_logstd).sum(dim=(1, 2))

        elbo = logpx + logpz - logqz
        return {
            "loss": -elbo.mean(),
            "recons": logpx.mean(),
            "kl": (logqz - logpz).mean(),
        }

    def forward(self, x):
        """Alias of :meth:`elbo`."""
        return self.elbo(x)
