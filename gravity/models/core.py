"""Core neural components used across GRAVITY models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..utils import log_verbose

__all__ = [
    "FeedForwardNetwork",
    "MultiHeadAttention",
]


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward block implemented with 1×1 convolutions.

    The residual+LayerNorm arrangement mirrors transformer-style FFN blocks.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.ff1 = nn.Conv1d(d_model, d_ff, 1)
        self.ff2 = nn.Conv1d(d_ff, d_model, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)
        x = x.transpose(1, 2)
        return self.layer_norm(residual + x)


class MultiHeadAttention(nn.Module):
    """Multi-headed attention with optional boolean mask support.

    The mask is assumed to be broadcastable to the attention score shape and
    indicates entries to be suppressed.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = q
        batch_size = q.size(0)
        q = self._split_heads(self.W_q(q))
        k = self._split_heads(self.W_k(k))
        v = self._split_heads(self.W_v(v))

        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.depth ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.d_model)

        output = self.W_o(context)
        return self.layer_norm(residual + output), attn
