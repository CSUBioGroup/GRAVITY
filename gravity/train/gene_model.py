"""Lightning module implementing the GRAVITY gene-wise refinement stage.

This stage fine-tunes a subset of solver parameters on top of the stage-1
checkpoint. The wrapper mirrors the cell-wise module but operates with a
fixed mask, concentrating adjustments on gene-level kinetics.
"""

from __future__ import annotations

import ast
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from ..models.gravity_model import GravityModel
from ..utils import log_verbose

__all__ = [
    "get_unsplice",
    "get_splice",
    "velocity_calculate_gene",
    "FullModelGeneWise",
]


def get_unsplice(dim: int):
    """Return indices of unspliced features (interleaved layout)."""
    return list(range(0, dim - 3, 2))


def get_splice(dim: int):
    """Return indices of spliced features (interleaved layout)."""
    return list(range(1, dim - 3, 2))


def velocity_calculate_gene(
    origin: pd.DataFrame,
    cellindex: torch.Tensor,
    unsplice: torch.Tensor,
    splice: torch.Tensor,
    unsplice_predict: torch.Tensor,
    splice_predict: torch.Tensor,
    nbrs: np.ndarray,
):
    """Compute a mean-squared alignment loss against neighbor statistics."""
    indices = nbrs[cellindex.type(torch.int64)]
    if len(indices.shape) == 1:
        indices = np.expand_dims(indices, axis=0)
    unsplice_all = np.zeros((indices.shape[0], indices.shape[1], unsplice.shape[-1]), dtype=np.float32)
    splice_all = np.zeros_like(unsplice_all)

    for sample_idx, samples in enumerate(indices):
        sub_data = origin.iloc[samples].values
        for row_idx, row in enumerate(sub_data):
            eval_items = [ast.literal_eval(item) for item in row[3:]]
            unsplice_all[sample_idx, row_idx, :len(eval_items)] = [item[1] for item in eval_items]
            splice_all[sample_idx, row_idx, :len(eval_items)] = [item[2] for item in eval_items]

    unsplice_all = torch.tensor(unsplice_all, device=unsplice.device).squeeze(0)
    splice_all = torch.tensor(splice_all, device=splice.device).squeeze(0)

    def mse(unsplice_all, splice_all, unsplice, splice, unsplice_predict, splice_predict):
        uv, sv = unsplice_predict - unsplice, splice_predict - splice
        unv = unsplice_all - unsplice
        snv = splice_all - splice
        losses_u = torch.mean((uv - unv) ** 2, dim=1)
        losses_s = torch.mean((sv - snv) ** 2, dim=1)
        losses = 0.5 * losses_u + losses_s * 0.5
        return losses

    cost1 = mse(unsplice_all, splice_all, unsplice, splice, unsplice_predict, splice_predict)
    cost_fin = cost1.mean()
    return cost_fin, unsplice_predict, splice_predict


class FullModelGeneWise(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        TF_list: Sequence[str],
        TG_list: Sequence[str],
        TFTG_map: Dict[str, Sequence[str]],
        TFTG_map_reverse: Dict[str, Sequence[str]],
        embedding_size: int = 16,
        model_dimension: int = 16,
        ffn_dimension: int = 16,
        output_dim_trans: int = 10,
        output_dim_dense: int = 20,
        gene_list: Optional[Sequence[str]] = None,
        embedding_map: Optional[Dict[int, str]] = None,
        id_map: Optional[Dict[int, str]] = None,
        gene_select: Optional[Sequence[str]] = None,
        nbrs: Optional[np.ndarray] = None,
        origin_data: Optional[pd.DataFrame] = None,
        negs: Optional[np.ndarray] = None,
        output_csv: Optional[str] = None,
        learning_rate: float = 1e-2,
    ) -> None:
        super().__init__()
        self.csv = pd.DataFrame(columns=['cellIndex', 'gene_name', 'unsplice', 'splice', 'unsplice_predict', 'splice_predict', 'alpha', 'beta', 'gamma', 'embedding1', 'embedding2', 'clusters', 'cellID'])
        self.input_dim = input_dim
        self.output_dim_trans = output_dim_trans
        self.output_dim_dense = output_dim_dense
        self.TF_list = TF_list
        self.TG_list = TG_list
        self.TFTG_map = TFTG_map
        self.TFTG_map_reverse = TFTG_map_reverse
        self.gene_list = list(gene_list or [])
        self.gene_select = [g.upper() for g in (gene_select or self.gene_list)]
        self.GravityModel = GravityModel(embedding_size, model_dimension, ffn_dimension, output_dim_trans, output_dim_trans, 6, 2)
        self.attentions = None
        self.encoding = None
        self.attn_mask = None
        self.embed_mapper = embedding_map or {}
        self.id_mapper = id_map or {}
        self.nbrs = nbrs
        self.negs = negs
        self.origin_data = origin_data
        self.encodings = []
        self.output_csv = output_csv
        self.learning_rate = learning_rate

    def forward(self, x, mask=None):
        cell_info, gene_info = x[:, :3], x[:, 3:]
        points = cell_info[:, 1:]
        unsplice_idx = get_unsplice(self.input_dim)
        splice_idx = get_splice(self.input_dim)
        unsplice = gene_info[:, unsplice_idx] + 1e-8
        splice = gene_info[:, splice_idx] + 1e-8
        attn_mask = torch.zeros((unsplice.shape[1], splice.shape[1]), dtype=torch.bool, device=unsplice.device)
        umax = unsplice.max(dim=0)[0]
        smax = splice.max(dim=0)[0]
        alpha0 = umax * 2.0
        beta0 = torch.ones_like(umax)
        gamma0 = umax / (smax + 1e-8)
        return self.GravityModel(unsplice, splice, points, attn_mask, self.output_dim_trans, self.output_dim_trans, alpha0, beta0, gamma0)

    def training_step(self, batch, batch_idx):
        _, x = batch
        output = self(x)
        cell_info, gene_info = x[:, :3], x[:, 3:]
        points = cell_info[:, 1:]
        unsplice_idx = get_unsplice(x.shape[1])
        splice_idx = get_splice(x.shape[1])
        unsplice = gene_info[:, unsplice_idx]
        splice = gene_info[:, splice_idx]
        cell_index = cell_info[:, 0].to('cpu')
        loss = velocity_calculate_gene(self.origin_data, cell_index, unsplice, splice, output[0], output[1], self.nbrs)[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, x = batch
        output = self(x)
        cell_info, gene_info = x[:, :3], x[:, 3:]
        points = cell_info[:, 1:]
        unsplice_idx = get_unsplice(x.shape[1])
        splice_idx = get_splice(x.shape[1])
        unsplice = gene_info[:, unsplice_idx]
        splice = gene_info[:, splice_idx]
        cell_index = cell_info[:, 0].to('cpu')

        unsplice_predict, splice_predict, alphas, betas, gammas = output
        gene_list = self.gene_select
        gene_indices = torch.tensor([self.gene_list.index(gene) for gene in gene_list if gene in self.gene_list])

        rows = []
        for b in range(cell_info.shape[0]):
            key = int(cell_info[b, 0].item())
            for gene_index in gene_indices:
                gene = self.gene_list[gene_index]
                to_insert = [key, gene] + [float(tensor[b, gene_index].item()) for tensor in (
                    unsplice, splice, unsplice_predict, splice_predict, alphas, betas, gammas
                )] + [float(points[b, i].item()) for i in range(2)]
                to_insert += [self.embed_mapper.get(key, "NA"), self.id_mapper.get(key, str(key))]
                rows.append(to_insert)
        if rows:
            self.csv = pd.concat([self.csv, pd.DataFrame(rows, columns=self.csv.columns)], ignore_index=True)
        loss = velocity_calculate_gene(self.origin_data, cell_index, unsplice, splice, output[0], output[1], self.nbrs)[0]
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        if hasattr(self, "trainer") and getattr(self.trainer, "global_rank", 0) != 0:
            return
        self.csv = self.csv.sort_values(by=['gene_name', 'cellIndex'])
        self.csv.reset_index(drop=True, inplace=True)
        if self.output_csv:
            self.csv.to_csv(self.output_csv)

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
