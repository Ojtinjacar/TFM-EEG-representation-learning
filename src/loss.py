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
