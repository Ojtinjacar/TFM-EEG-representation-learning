import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NTXentLoss(torch.nn.Module):
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


class ExpCLRLoss(torch.nn.Module):
    def __init__(self, device, delta=1.0, temperature=1.0, feat_max_dist=None,
                 squared_similarity=True):
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
        feat_dist = torch.cdist(f, f, p=2)
        if self.feat_max_dist is not None:
            max_dist = self.feat_max_dist
        else:
            max_dist = feat_dist.max().clamp_min(self.eps)
        sim = (1.0 - feat_dist / max_dist).clamp_min(0.0)
        return sim.pow(2) if self.squared_similarity else sim

    def normalized_distance(self, z):
        dist = torch.cdist(z, z, p=2)
        mu = dist.mean(dim=1, keepdim=True).clamp_min(self.eps)
        return dist / mu

    def forward(self, z, f):
        if z.shape[0] != f.shape[0]:
            raise ValueError(f"Batch mismatch between embeddings {z.shape} and features {f.shape}")
        if z.shape[0] < 2:
            raise ValueError("ExpCLR needs at least two samples per batch")

        sim = self.expert_similarity(f.detach())
        dist = self.normalized_distance(z)
        return self.reduce(((1.0 - sim) * self.delta - dist).pow(2))

    def reduce(self, pair_loss):
        if self.temperature is None:
            return pair_loss.mean()

        n_pairs = pair_loss.numel()
        scaled = pair_loss.reshape(-1) / self.temperature
        return self.temperature * (torch.logsumexp(scaled, dim=0) - np.log(n_pairs))


def _apply_free_bits(kl_per_dim, free_bits):
    if free_bits and free_bits > 0.0:
        return torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim


class LatentPrior:
    def kl_divergence(self, mu, logvar, free_bits=0.0):
        raise NotImplementedError

    def to(self, *args, **kwargs):
        return self


class StandardNormalPrior(LatentPrior):
    def kl_divergence(self, mu, logvar, free_bits=0.0):
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        kl_per_dim = _apply_free_bits(kl_per_dim, free_bits)
        return kl_per_dim.sum(dim=1).mean()


def vae_elbo(reconstruction, x, mu, logvar, prior, beta, free_bits=0.0):
    recon = F.mse_loss(reconstruction, x, reduction="mean")
    kl = prior.kl_divergence(mu, logvar, free_bits=free_bits)
    total = recon + beta * kl
    return total, recon, kl


def build_prior(prior_type):
    if prior_type == "standard":
        return StandardNormalPrior()
    raise ValueError(f"Unknown prior_type: {prior_type!r} (use 'standard').")


def kl_beta(epoch, target_beta, anneal_epochs):
    if anneal_epochs <= 0:
        return target_beta
    return target_beta * min(1.0, epoch / anneal_epochs)
