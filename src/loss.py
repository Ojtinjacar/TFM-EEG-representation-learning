import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NTXentLoss(torch.nn.Module):
    """
    This is the same implementation used in TF-C and TS-TCC. 
    The comparison is between two positive augmentations, the rest of the batch are negative samples.
    
    """

    def __init__(self, device, batch_size, temperature, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).reshape(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].reshape(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """ Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        return loss / (2 * self.batch_size)


# =============================================================================
#  ExpCLR: contrastive learning with continuous expert features
# =============================================================================
class ExpCLRLoss(torch.nn.Module):
    """ExpCLR objective of Nonnenmacher et al. (ICML 2022), arXiv:2202.10337.

    Unlike NT-Xent there are no positive/negative pairs and no augmented views: the loss operates
    on the full NxN pairwise matrix of a single batch, pulling representation distances towards a
    target set by how similar the expert descriptors are. It implements, in order:

    - Eq. 5 (similarity):   ``s_ij = (1 - ||f_i - f_j|| / max_kl ||f_k - f_l||) ** 2``
    - Kim et al. (2021) normalisation: ``D_ij = ||E_i - E_j|| / mu_i``,
      with ``mu_i = (1/N) sum_j ||E_i - E_j||``
    - Eq. 3 (pair term):    ``L_ij = ((1 - s_ij) * Delta - D_ij) ** 2``
    - Eq. 4 (full loss):    ``L = tau * log[ (1/N^2) * sum_ij exp(L_ij / tau) ]``

    The log-sum-exp is the implicit hard-negative mining: the gradient of the loss w.r.t. each
    ``L_ij`` is proportional to ``exp(L_ij / tau)``, so the worst-fitted pairs dominate (App. F of
    the paper). Proposition 3 bounds the two regimes: ``tau -> 0`` minimises ``max_ij L_ij`` and
    ``tau -> inf`` reduces to the plain quadratic loss of Eq. 3.
    """

    def __init__(self, device, delta=1.0, temperature=1.0, feat_max_dist=None,
                 squared_similarity=True):
        """Initialises the ExpCLR objective.

        Args:
            device: Torch device the loss runs on.
            delta: Margin hyperparameter ``Delta`` of Eq. 3/4. The paper uses 1.0 for all datasets.
            temperature: Hard-negative mining strength ``tau`` of Eq. 4. The paper uses 1.0.
                Pass None to disable mining and fall back to the quadratic loss of Eq. 3 (the
                NHNM ablation of Fig. 2).
            feat_max_dist: Precomputed ``max_kl ||f_k - f_l||`` over the training split. When None,
                the maximum is taken per batch, which the paper also allows (Sec. 3.4) but which
                leaks test-fold information if the batch is not train-only.
            squared_similarity: If True use Eq. 5 (the variant the paper actually uses); if False
                use the plain Eq. 2.

        Raises:
            ValueError: If ``feat_max_dist`` or ``temperature`` are non-positive.
        """
        super(ExpCLRLoss, self).__init__()
        if feat_max_dist is not None and feat_max_dist <= 0:
            raise ValueError(f"feat_max_dist must be positive, got {feat_max_dist}")
        if temperature is not None and temperature <= 0:
            raise ValueError(f"temperature must be positive or None, got {temperature}")
        self.device = device
        self.delta = float(delta)
        self.temperature = None if temperature is None else float(temperature)
        self.feat_max_dist = None if feat_max_dist is None else float(feat_max_dist)
        self.squared_similarity = bool(squared_similarity)
        self.eps = 1e-8

    def expert_similarity(self, f):
        """Computes the continuous similarity matrix from expert features (Eq. 2 / Eq. 5).

        Args:
            f: Expert features, shape (B, d).

        Returns:
            Similarity matrix of shape (B, B) with values in [0, 1] and ones on the diagonal.
        """
        feat_dist = torch.cdist(f, f, p=2)
        if self.feat_max_dist is not None:
            max_dist = self.feat_max_dist
        else:
            max_dist = feat_dist.max().clamp_min(self.eps)
        # Clamped at zero: with a train-fitted max_dist a batch pair may exceed it, and squaring a
        # negative value would silently turn "very dissimilar" back into "similar".
        sim = (1.0 - feat_dist / max_dist).clamp_min(0.0)
        return sim.pow(2) if self.squared_similarity else sim

    def normalized_distance(self, z):
        """Computes the mean-normalised embedding distances ``D_ij`` (Kim et al., 2021).

        Args:
            z: Embeddings, shape (B, e).

        Returns:
            Distance matrix of shape (B, B).
        """
        dist = torch.cdist(z, z, p=2)
        mu = dist.mean(dim=1, keepdim=True).clamp_min(self.eps)
        return dist / mu

    def forward(self, z, f):
        """Evaluates the ExpCLR loss on a batch.

        Args:
            z: Embeddings produced by the encoder, shape (B, e).
            f: Expert features aligned with ``z``, shape (B, d). Treated as constant.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If ``z`` and ``f`` disagree on the batch dimension, or the batch has
                fewer than two samples (the pairwise loss is undefined).
        """
        if z.shape[0] != f.shape[0]:
            raise ValueError(f"Batch mismatch between embeddings {z.shape} and features {f.shape}")
        if z.shape[0] < 2:
            raise ValueError("ExpCLR needs at least two samples per batch")

        sim = self.expert_similarity(f.detach())
        dist = self.normalized_distance(z)
        return self.reduce(((1.0 - sim) * self.delta - dist).pow(2))

    def reduce(self, pair_loss):
        """Reduces per-pair terms to the scalar objective, Eq. 3 or Eq. 4.

        Exposed so that reference geometries can be scored with the very functional being
        optimised. Comparing a mean (Eq. 3) against a log-sum-exp (Eq. 4) is not meaningful: by
        Jensen the latter is always the larger of the two.

        Args:
            pair_loss: Matrix of per-pair terms ``L_ij``.

        Returns:
            Scalar loss tensor.
        """
        if self.temperature is None:
            # Eq. 3: no hard-negative mining (the NHNM ablation).
            return pair_loss.mean()

        # Eq. 4 via log-sum-exp for numerical stability:
        #   tau * log[ (1/N^2) sum exp(L_ij/tau) ] = tau * (logsumexp(L/tau) - log(N^2))
        n_pairs = pair_loss.numel()
        scaled = pair_loss.reshape(-1) / self.temperature
        return self.temperature * (torch.logsumexp(scaled, dim=0) - np.log(n_pairs))


# =============================================================================
#  VAE latent priors (Strategy pattern) and ELBO objective
# =============================================================================
def _apply_free_bits(kl_per_dim, free_bits):
    """Clamps each latent dimension's KL at a minimum floor (free bits).

    Free bits reserve a guaranteed amount of information per latent dimension to
    mitigate posterior collapse: dimensions whose KL is already below ``free_bits``
    stop being penalized, so the model is not pushed to switch them off.

    Args:
        kl_per_dim (torch.Tensor): Per-dimension KL of shape (B, J), non-negative.
        free_bits (float): Minimum KL (in nats) kept per dimension. 0 disables it.

    Returns:
        torch.Tensor: The (possibly clamped) per-dimension KL, same shape (B, J).
    """
    if free_bits and free_bits > 0.0:
        return torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim


class LatentPrior:
    """Strategy interface for the VAE latent prior p(z), possibly conditional.

    Implementations provide the KL divergence KL(q(z|x) || p(z)) that forms the
    regularization term of the ELBO. Encapsulating the prior behind this interface
    lets the VAE swap a plain N(0, I) prior for richer, conditional priors without
    touching the model or the training loop (Strategy pattern).
    """

    def kl_divergence(self, mu, logvar, cond=None, free_bits=0.0):
        """Computes the batch-averaged KL between q(z|x)=N(mu, sigma^2) and p(z).

        Args:
            mu (torch.Tensor): Posterior mean (B, J).
            logvar (torch.Tensor): Posterior log-variance (B, J).
            cond (torch.Tensor, optional): Condition indices (B,) for conditional
                priors; ignored by unconditional priors.
            free_bits (float): Per-dimension KL floor (see :func:`_apply_free_bits`).

        Returns:
            torch.Tensor: Scalar KL averaged over the batch.

        Raises:
            NotImplementedError: If the subclass does not implement it.
        """
        raise NotImplementedError

    def to(self, *args, **kwargs):
        """No-op device/dtype move for parameter-free priors; returns ``self``.

        Learnable priors that subclass ``nn.Module`` (e.g.
        :class:`ConditionalGaussianPrior`) override this via ``nn.Module.to`` through
        the method resolution order, so their parameters are moved as usual.
        """
        return self


class StandardNormalPrior(LatentPrior):
    """Standard isotropic Gaussian prior p(z) = N(0, I).

    Reproduces the original closed-form KL used by the VAE, optionally with free
    bits. Stateless (no learnable parameters).
    """

    def kl_divergence(self, mu, logvar, cond=None, free_bits=0.0):
        """See :meth:`LatentPrior.kl_divergence`. ``cond`` is ignored."""
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # (B, J), >= 0
        kl_per_dim = _apply_free_bits(kl_per_dim, free_bits)
        return kl_per_dim.sum(dim=1).mean()


class ConditionalGaussianPrior(nn.Module, LatentPrior):
    """Learnable per-condition Gaussian prior p(z | c) = N(mu_c, sigma_c^2).

    One Gaussian per discrete condition (e.g. one per session age). The
    per-condition means and log-variances are learned jointly with the model, so
    the latent space is organized into a region per condition. The KL between the
    posterior N(mu, sigma^2) and the conditional prior N(mu_c, sigma_c^2) has a
    closed form and is applied dimension-wise.
    """

    def __init__(self, n_conditions, latent_dim):
        """Initializes the conditional prior tables.

        Args:
            n_conditions (int): Number of discrete conditions (e.g. distinct ages).
            latent_dim (int): Dimensionality J of the latent vector z.
        """
        super().__init__()
        self.mu_table = nn.Embedding(n_conditions, latent_dim)      # mu_c
        self.logvar_table = nn.Embedding(n_conditions, latent_dim)  # log(sigma_c^2)
        # Init to N(0, I) so training starts equivalent to the standard prior.
        nn.init.zeros_(self.mu_table.weight)
        nn.init.zeros_(self.logvar_table.weight)

    def kl_divergence(self, mu, logvar, cond=None, free_bits=0.0):
        """See :meth:`LatentPrior.kl_divergence`. ``cond`` is required.

        Raises:
            ValueError: If ``cond`` is None.
        """
        if cond is None:
            raise ValueError(
                "ConditionalGaussianPrior requires `cond` (condition indices)."
            )
        mu_c = self.mu_table(cond)          # (B, J)
        logvar_c = self.logvar_table(cond)  # (B, J)
        kl_per_dim = 0.5 * (
            logvar_c - logvar
            + (logvar.exp() + (mu - mu_c).pow(2)) / logvar_c.exp()
            - 1.0
        )  # (B, J), >= 0
        kl_per_dim = _apply_free_bits(kl_per_dim, free_bits)
        return kl_per_dim.sum(dim=1).mean()


def vae_elbo(reconstruction, x, mu, logvar, prior, beta, cond=None, free_bits=0.0):
    """Computes the negative ELBO of a Gaussian VAE with a pluggable prior.

    The objective is the reconstruction MSE plus a beta-weighted KL divergence
    between the approximate posterior N(mu, sigma^2) and the prior ``prior``. The
    prior is a :class:`LatentPrior` strategy, so the same objective serves the
    standard, conditional or any future prior.

    Args:
        reconstruction (torch.Tensor): Decoder output (B, C, T).
        x (torch.Tensor): Ground-truth input (B, C, T).
        mu (torch.Tensor): Posterior mean (B, J).
        logvar (torch.Tensor): Posterior log-variance (B, J).
        prior (LatentPrior): Prior strategy providing ``kl_divergence``.
        beta (float): Weight applied to the KL term (KL annealing value for this step).
        cond (torch.Tensor, optional): Condition indices (B,) for conditional priors.
        free_bits (float): Per-dimension KL floor to mitigate posterior collapse.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, reconstruction
        term and (unweighted) KL term, all as scalar tensors.
    """
    recon = F.mse_loss(reconstruction, x, reduction="mean")
    kl = prior.kl_divergence(mu, logvar, cond=cond, free_bits=free_bits)
    total = recon + beta * kl
    return total, recon, kl


def build_prior(prior_type, n_conditions=None, latent_dim=None):
    """Factory that builds a latent prior strategy from a string name.

    Centralizing prior construction keeps training and downstream in sync, so a
    conditional prior is rebuilt with the exact same parameter shapes and its state
    dict loads without mismatches.

    Args:
        prior_type (str): Either ``"standard"`` (N(0, I)) or ``"conditional"``
            (per-condition Gaussian).
        n_conditions (int, optional): Number of conditions; required for
            ``"conditional"``.
        latent_dim (int, optional): Latent dimensionality J; required for
            ``"conditional"``.

    Returns:
        LatentPrior: The constructed prior strategy.

    Raises:
        ValueError: If ``prior_type`` is unknown, or required sizes are missing for
            the conditional prior.
    """
    if prior_type == "standard":
        return StandardNormalPrior()
    if prior_type == "conditional":
        if n_conditions is None or latent_dim is None:
            raise ValueError(
                "ConditionalGaussianPrior requires n_conditions and latent_dim."
            )
        return ConditionalGaussianPrior(n_conditions, latent_dim)
    raise ValueError(f"Unknown prior_type: {prior_type!r} (use 'standard' or 'conditional').")


def kl_beta(epoch, target_beta, anneal_epochs):
    """Linear KL warm-up schedule.

    Ramps the KL weight from 0 to ``target_beta`` over the first ``anneal_epochs``
    epochs to mitigate posterior collapse, then holds it constant.

    Args:
        epoch (int): Zero-based current epoch.
        target_beta (float): Final KL weight.
        anneal_epochs (int): Number of epochs over which to ramp up. 0 disables it.

    Returns:
        float: KL weight to use for this epoch.
    """
    if anneal_epochs <= 0:
        return target_beta
    return target_beta * min(1.0, (epoch + 1) / anneal_epochs)
