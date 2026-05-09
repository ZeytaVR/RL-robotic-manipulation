"""Running mean/std normalizer with clipping. Required for HER+Fetch stability."""
import torch
import numpy as np


class Normalizer:
    def __init__(self, dim, eps=1e-2, clip_range=5.0, device="cuda"):
        self.eps = eps
        self.clip_range = clip_range
        self.device = device
        self.dim = dim

        self.sum    = torch.zeros(dim, device=device)
        self.sum_sq = torch.zeros(dim, device=device)
        self.count  = torch.tensor(0.0, device=device)

        self.mean = torch.zeros(dim, device=device)
        self.std  = torch.ones(dim, device=device)

    def update(self, data):
        """data: (N, dim) numpy array or torch tensor."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float().to(self.device)
        else:
            data = data.float().to(self.device)
        self.sum    += data.sum(dim=0)
        self.sum_sq += (data ** 2).sum(dim=0)
        self.count  += data.shape[0]

        self.mean = self.sum / self.count
        var = (self.sum_sq / self.count) - self.mean ** 2
        self.std = var.clamp(min=self.eps ** 2).sqrt()

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        return ((data - self.mean) / self.std).clamp(-self.clip_range, self.clip_range)