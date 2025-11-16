"""Loss functions used during GRAVITY training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "WeightedFeatureTripletLoss",
]


class WeightedFeatureTripletLoss(nn.Module):
    """Feature-wise weighted triplet loss for RNA velocity alignment."""

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        weighted_diff_positive = weights * (anchor - positive) ** 2
        weighted_diff_negative = weights * (anchor - negative) ** 2
        positive_distance = torch.sqrt(torch.sum(weighted_diff_positive, dim=1))
        negative_distance = torch.sqrt(torch.sum(weighted_diff_negative, dim=1))
        losses = F.relu(positive_distance - negative_distance + self.margin)
        return torch.sum(losses)
