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


class ExpCLRLoss(torch.nn.Module):
    """Contrastive loss driven by a continuous expert descriptor.

    Augmentation-based contrastive losses need a rule that says which pairs are positive,
    and they get it from transformations assumed to preserve the label. This one replaces
    that rule with the descriptor itself: how close two windows should end up in the
    representation is read off how close an expert would call them, on a continuous scale
    rather than a positive/negative split. Binning the descriptor to recover discrete
    labels would throw away exactly the relative distances that make the target useful.

    The target for a pair is ``(1 - s_ij) * delta`` and the achieved value is the
    normalized distance between representations, so the loss is the squared gap between
    them, reduced over the batch by a soft maximum whose sharpness is set by the
    temperature.
    """

    def __init__(self, delta=1.0, temperature=1.0, feat_max_dist=None,
                 squared_similarity=True):
        """Initializes the loss.

        Args:
            delta (float): Scale of the target distance for maximally dissimilar pairs.
            temperature (float): Sharpness of the soft maximum over pairs; ``None``
                averages the pairs instead, which removes the hard-negative mining.
            feat_max_dist (float): Largest descriptor distance, used to normalize the
                similarity. ``None`` takes the largest distance within each batch.
            squared_similarity (bool): Square the similarity measure.

        Raises:
            ValueError: If ``feat_max_dist`` or ``temperature`` is not positive.
        """
        super(ExpCLRLoss, self).__init__()
        if feat_max_dist is not None and feat_max_dist <= 0:
            raise ValueError(f"feat_max_dist must be positive, got {feat_max_dist}")
        if temperature is not None and temperature <= 0:
            raise ValueError(f"temperature must be positive or None, got {temperature}")
        self.delta = float(delta)
        self.temperature = None if temperature is None else float(temperature)
        self.feat_max_dist = None if feat_max_dist is None else float(feat_max_dist)
        self.squared_similarity = bool(squared_similarity)
        self.eps = 1e-8

    def expert_similarity(self, f):
        """Turns descriptor distances into similarities in [0, 1].

        One means an identical descriptor and zero means the two furthest apart of the
        reference set. Squaring the whole expression, rather than the ratio inside it,
        keeps similarity high across the range of distances where most pairs sit and lets
        it fall off only for genuinely distant ones.

        Args:
            f (torch.Tensor): Expert descriptors, shape (batch, n_features).

        Returns:
            torch.Tensor: Similarity matrix, shape (batch, batch).
        """
        feat_dist = torch.cdist(f, f, p=2)
        if self.feat_max_dist is not None:
            max_dist = self.feat_max_dist
        else:
            max_dist = feat_dist.max().clamp_min(self.eps)
        sim = (1.0 - feat_dist / max_dist).clamp_min(0.0)
        return sim.pow(2) if self.squared_similarity else sim

    def normalized_distance(self, z):
        """Divides each representation distance by the mean distance of its row.

        This makes the loss invariant to rescaling the representation, so the encoder
        cannot lower it by shrinking or inflating the space instead of arranging it. The
        price is that every row of the result averages exactly one, which bounds how close
        the target can be approached for a given delta.

        Args:
            z (torch.Tensor): Representations, shape (batch, n_dims).

        Returns:
            torch.Tensor: Normalized distance matrix, shape (batch, batch).
        """
        dist = torch.cdist(z, z, p=2)
        mu = dist.mean(dim=1, keepdim=True).clamp_min(self.eps)
        return dist / mu

    def forward(self, z, f):
        """Computes the loss of a batch.

        Args:
            z (torch.Tensor): Representations, shape (batch, n_dims).
            f (torch.Tensor): Expert descriptors, shape (batch, n_features).

        Returns:
            torch.Tensor: Scalar loss.

        Raises:
            ValueError: If the two arguments disagree on the batch size, or the batch
                holds fewer than two samples, which leaves no pair to compare.
        """
        if z.shape[0] != f.shape[0]:
            raise ValueError(f"Batch mismatch between embeddings {z.shape} and features {f.shape}")
        if z.shape[0] < 2:
            raise ValueError("ExpCLR needs at least two samples per batch")

        sim = self.expert_similarity(f.detach())
        dist = self.normalized_distance(z)
        return self.reduce(((1.0 - sim) * self.delta - dist).pow(2))

    def reduce(self, pair_loss):
        """Reduces the per-pair losses to a scalar.

        A plain mean weights every pair alike. The soft maximum used instead concentrates
        the gradient on the pairs that are furthest from their target, which is
        hard-negative mining without having to select the pairs explicitly: a low
        temperature approaches optimizing the single worst pair, a high one approaches the
        mean.

        Args:
            pair_loss (torch.Tensor): Squared gaps, shape (batch, batch).

        Returns:
            torch.Tensor: Scalar loss.
        """
        if self.temperature is None:
            return pair_loss.mean()

        n_pairs = pair_loss.numel()
        scaled = pair_loss.reshape(-1) / self.temperature
        return self.temperature * (torch.logsumexp(scaled, dim=0) - np.log(n_pairs))


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
