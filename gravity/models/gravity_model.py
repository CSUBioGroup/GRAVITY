"""GRAVITY neural network definition.

The model embeds unspliced/spliced inputs and cell coordinates, applies a
cross-attention block followed by a position-wise FFN, and finally uses a small
MLP translator to predict kinetics parameters (α, β, γ). The predicted
parameters are constrained via ``Softplus`` and rescaled by initial factors to
produce updated unspliced/spliced expressions.
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn

from .core import FeedForwardNetwork, MultiHeadAttention
from ..utils import time_section

__all__ = [
    "MLPTranslator",
    "GravityModel",
]


class MLPTranslator(nn.Module):
    """Flexible multi-layer perceptron mapping kinetics inputs to α/β/γ."""

    def __init__(
        self,
        num_fc_input: int,
        num_output_nodes: int,
        num_fc_layers: int,
        initial_dropout: float,
        act: nn.Module | None = None,
    ) -> None:
        super().__init__()
        act = act or nn.LeakyReLU()
        self.num_fc_layers = num_fc_layers
        self.linear_layer_names: list[str] = []

        if num_fc_layers == 1:
            self.fc0 = nn.Linear(num_fc_input, num_output_nodes)
            self.linear_layer_names.append('fc0')
        else:
            self.fc0 = nn.Linear(num_fc_input, 32)
            self.dropout0 = nn.Dropout(initial_dropout)
            self.linear_layer_names.append('fc0')
            if num_fc_layers == 2:
                self.fc1 = nn.Linear(32, num_output_nodes)
                self.linear_layer_names.append('fc1')
            else:
                hidden = []
                dropouts = []
                for idx in range(1, num_fc_layers - 1):
                    hidden.append(nn.Linear(32, 32))
                    dropouts.append(nn.Dropout(initial_dropout))
                    self.linear_layer_names.append(f'hidden.{idx-1}')
                self.hidden = nn.ModuleList(hidden)
                self.hidden_drop = nn.ModuleList(dropouts)
                self.fc_last = nn.Linear(32, num_output_nodes)
                self.linear_layer_names.append('fc_last')

        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_fc_layers == 1:
            return self.fc0(x)

        out = self.dropout0(self.fc0(x))
        if self.num_fc_layers == 2:
            return self.fc1(out)

        for layer, drop in zip(self.hidden, self.hidden_drop):
            out = self.act(drop(layer(out)))
        return self.fc_last(out)


class GravityModel(nn.Module):
    """Cross-attention based RNA velocity predictor."""

    def __init__(
        self,
        embedding_size: int,
        model_dimension: int,
        ffn_dimension: int,
        unsplice_num: int,
        splice_num: int,
        mlp_n_layers: int,
        transformer_n_layers: int,
    ) -> None:
        super().__init__()
        self.ffn = FeedForwardNetwork(model_dimension, ffn_dimension)
        self.cross = MultiHeadAttention(model_dimension, num_heads=8)
        self.unsplice_embedding = nn.Linear(unsplice_num, embedding_size // 2)
        self.splice_embedding = nn.Linear(splice_num, embedding_size // 2)
        self.cell_embedding = nn.Linear(2, embedding_size // 2)
        self.solver = MLPTranslator(2, 3, mlp_n_layers, 0.0)
        self.constrain = nn.Softplus()
        self.attention = None
        self.unsplice_enc = None
        self.splice_enc = None

    def forward(
        self,
        unsplice_mat: torch.Tensor,
        splice_mat: torch.Tensor,
        cell_mat: torch.Tensor,
        mask: torch.Tensor,
        unsplice_num: int,
        splice_num: int,
        alpha0: torch.Tensor,
        beta0: torch.Tensor,
        gamma0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with time_section("prep_embeddings", level=3):
            unsplice_mat_broadcast = einops.repeat(unsplice_mat, 'b g -> b g k', k=unsplice_mat.shape[1])
            unsplice_channel = self.unsplice_embedding(unsplice_mat_broadcast)
            splice_mat_broadcast = einops.repeat(splice_mat, 'b g -> b g k', k=splice_mat.shape[1])
            splice_channel = self.splice_embedding(splice_mat_broadcast)
            cell_channel = self.cell_embedding(cell_mat)

        cell_repeated = cell_channel.unsqueeze(1).repeat(1, unsplice_mat.shape[1], 1)
        us_final = torch.cat((unsplice_channel, cell_repeated), dim=2)
        s_final = torch.cat((splice_channel, cell_repeated), dim=2)

        with time_section("cross_attention", level=3):
            encoder_output, attn_matrix = self.cross(us_final, s_final, us_final, mask.to(unsplice_mat.device))
        with time_section("ffn", level=3):
            encoder_output = self.ffn(encoder_output)
        self.attention = attn_matrix
        self.unsplice_enc = encoder_output
        self.splice_enc = s_final

        new_unsplice = encoder_output.reshape(-1, encoder_output.shape[-1])
        previous_splice = s_final.reshape(-1, s_final.shape[-1])
        with time_section("solver", level=3):
            solver_input = torch.cat(
                (new_unsplice.unsqueeze(1), previous_splice.unsqueeze(1)),
                dim=1,
            )
            solver_input_final = solver_input.transpose(1, 2)
            solver_output = self.solver(solver_input_final).transpose(1, 2).mean(dim=-1)

        batch_size, gene_count = unsplice_mat.shape
        alphas = self.constrain(solver_output[:, 2]).view(batch_size, gene_count)
        betas = self.constrain(solver_output[:, 0]).view(batch_size, gene_count)
        gammas = self.constrain(solver_output[:, 1]).view(batch_size, gene_count)

        alphas = alphas * alpha0
        betas = betas * beta0
        gammas = gammas * gamma0

        unsplice_predict = unsplice_mat + (alphas - betas * unsplice_mat) * 0.5
        splice_predict = splice_mat + (betas * unsplice_mat - gammas * splice_mat) * 0.5
        return unsplice_predict, splice_predict, alphas, betas, gammas
