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
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """ Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)
        return loss / (2 * self.batch_size)

def _floor_free_bits(kl_per_dim_batch, free_bits):
    """Floors the batch-averaged per-dimension KL, per Kingma et al. (2016)."""
    if free_bits and free_bits > 0.0:
        return torch.clamp(kl_per_dim_batch, min=free_bits)
    return kl_per_dim_batch


class LatentPrior:
    """Interface for the latent prior p(z).

    ``kl_divergence`` is the term entering the objective, ``kl_true`` the
    actual divergence to report; they differ only under free bits.
    """

    def kl_divergence(self, mu, logvar, free_bits=0.0):
        """Returns the KL term of the objective, possibly floored."""
        raise NotImplementedError

    def kl_true(self, mu, logvar):
        """Returns the actual KL divergence, never floored."""
        raise NotImplementedError

    def to(self, *args, **kwargs):
        return self


class StandardNormalPrior(LatentPrior):
    """Standard normal prior p(z) = N(0, I), Kingma & Welling (2014) app. B::

        KL = -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j))

    The encoder emits log-variance, not log standard deviation.
    """

    def _kl_per_dim(self, mu, logvar):
        """Returns the per-sample, per-dimension KL, shape (B, D)."""
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)

    def kl_true(self, mu, logvar):
        """Returns the KL summed over dimensions and averaged over the batch."""
        return self._kl_per_dim(mu, logvar).sum(dim=1).mean()

    def kl_divergence(self, mu, logvar, free_bits=0.0):
        """Returns the objective KL; equals kl_true when free_bits is zero."""
        kl_per_dim_batch = self._kl_per_dim(mu, logvar).mean(dim=0)
        return _floor_free_bits(kl_per_dim_batch, free_bits).sum()


def vae_elbo(reconstruction, x, mu, logvar, prior, beta, free_bits=0.0):
    """Returns the negative ELBO to minimize, in code scale.

    Args:
        beta (float): KL weight already in code scale, not canonical units.

    Returns:
        tuple: (total, recon, kl_objective, kl_true). Backpropagate ``total``
        and report ``kl_true``; the two KLs coincide without free bits.
    """
    recon = F.mse_loss(reconstruction, x, reduction="mean")
    kl_objective = prior.kl_divergence(mu, logvar, free_bits=free_bits)
    kl_true = prior.kl_true(mu, logvar)
    total = recon + beta * kl_objective
    return total, recon, kl_objective, kl_true


def build_prior(prior_type):
    if prior_type == "standard":
        return StandardNormalPrior()
    raise ValueError(f"Unknown prior_type: {prior_type!r} (use 'standard').")


def kl_beta(epoch, target_beta, anneal_epochs):
    if anneal_epochs <= 0:
        return target_beta
    return target_beta * min(1.0, epoch / anneal_epochs)
