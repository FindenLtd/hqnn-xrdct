from __future__ import annotations

import torch
import torch.nn as nn


class FixedPMN(nn.Module):
    """Feature normalization with training statistics reusable on hardware data."""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(features, dtype=torch.float64))
        self.register_buffer("std", torch.ones(features, dtype=torch.float64))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, values: torch.Tensor) -> None:
        self.mean.copy_(values.mean(dim=0))
        self.std.copy_(values.std(dim=0).clamp_min(self.eps))
        self.fitted.fill_(True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            mean = values.mean(dim=0, keepdim=True)
            std = values.std(dim=0, keepdim=True).clamp_min(self.eps)
            return (values - mean) / std
        return (values - self.mean) / self.std.clamp_min(self.eps)


def _branch(hidden: int, widths: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = hidden
    for width in widths:
        layers.extend([nn.Linear(previous, width, dtype=torch.float64), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, 1, dtype=torch.float64))
    return nn.Sequential(*layers)


class DualHeadQNN(nn.Module):
    def __init__(self, qlayer: nn.Module, features: int, phases: int = 4, effective_shots: int = 0):
        super().__init__()
        self.qlayer = qlayer
        self.effective_shots = effective_shots
        self.pmn = FixedPMN(features)
        self.wt_shared = nn.Sequential(
            nn.Linear(features, 64, dtype=torch.float64), nn.ReLU(),
            nn.Linear(64, 64, dtype=torch.float64), nn.ReLU(),
        )
        self.wt_branches = nn.ModuleList([_branch(64, [64, 64, 32]) for _ in range(phases)])
        self.sf_shared = nn.Sequential(
            nn.Linear(features + 1, 64, dtype=torch.float64), nn.ReLU(),
            nn.Linear(64, 64, dtype=torch.float64), nn.ReLU(),
        )
        self.sf_branches = nn.ModuleList([_branch(64, [64, 64, 32]) for _ in range(phases)])

    def heads_from_features(self, features: torch.Tensor, pattern_max: torch.Tensor):
        normalized = self.pmn(features)
        wt_hidden = self.wt_shared(normalized)
        wt_logits = torch.cat([branch(wt_hidden) for branch in self.wt_branches], dim=1)
        wt = torch.softmax(wt_logits, dim=1)
        sf_hidden = self.sf_shared(torch.cat([normalized, pattern_max], dim=1))
        sf = torch.cat([branch(sf_hidden) for branch in self.sf_branches], dim=1)
        return wt, sf

    def forward(self, patterns: torch.Tensor, pattern_max: torch.Tensor):
        patterns = patterns / torch.linalg.vector_norm(patterns, dim=1, keepdim=True).clamp_min(1e-12)
        features = self.qlayer(patterns).to(patterns.dtype)
        if self.training and self.effective_shots > 0:
            probability = ((features.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
            counts = torch.binomial(torch.full_like(probability, float(self.effective_shots)), probability)
            sampled = 2.0 * counts / float(self.effective_shots) - 1.0
            features = features + (sampled - features).detach()
        return self.heads_from_features(features, pattern_max)
