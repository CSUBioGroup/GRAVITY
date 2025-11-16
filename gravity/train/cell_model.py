"""Lightning module implementing the GRAVITY cell-wise stage.

This module defines utilities and a :class:`pl.LightningModule` that execute
the cell-wise stage (stage 1) of GRAVITY. The forward pass delegates to the
``GravityModel`` model while this wrapper computes losses against neighborhood
statistics, exports attention-derived artefacts, and writes per-cell per-gene
predictions to CSV during a test-only pass.

Notes
-----
- Indices for unspliced/spliced features follow the convention of interleaved
  columns (``u,s,u,s,...``) with the last three columns reserved for
  ``cellIndex, embedding1, embedding2``.
- Attention-to-weight mapping averages over the key dimension, matching legacy
  behavior.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Dict, Iterable, Optional, Sequence

import ast
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.gravity_model import GravityModel
from ..train.attention import CSRStreamBuilder, TFScorer
from ..train.losses import WeightedFeatureTripletLoss
from ..utils import log_verbose, time_section

__all__ = [
    "get_unsplice_indices",
    "get_splice_indices",
    "mask_generate",
    "velocity_calculate",
    "FullModelCellWise",
]


def get_unsplice_indices(dim: int) -> Sequence[int]:
    """Return indices of unspliced features given the total feature dimension."""
    return list(range(0, dim - 3, 2))


def get_splice_indices(dim: int) -> Sequence[int]:
    """Return indices of spliced features given the total feature dimension."""
    return list(range(1, dim - 3, 2))


def mask_generate(q: torch.Tensor, k: torch.Tensor, gene_list: Sequence[str], reverse_mapper: Dict[str, Sequence[str]], mode: Optional[str] = None) -> torch.Tensor:
    """Generate a boolean attention mask of shape ``[G, G]``.

    If ``mode`` is ``'all'``, returns an all-false mask (no masking). Otherwise,
    prevent attention from gene to its mapped targets unless explicitly allowed
    by the provided reverse map.
    """
    mask = np.ones((q.shape[1], k.shape[1]))
    for index, gene in enumerate(gene_list):
        mask[index, index] = 1
        if gene in reverse_mapper:
            targets = reverse_mapper[gene]
            indices = [gene_list.index(item) for item in targets if item in gene_list]
            mask[index, indices] = 0
        else:
            mask[index, :] = 1
    if mode == 'all':
        mask = np.zeros((q.shape[1], k.shape[1]))
    return torch.tensor(mask, dtype=torch.bool)


def _prepare_origin_matrices(origin: pd.DataFrame, cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse and cache original unspliced/spliced matrices from the dataframe."""
    key = id(origin)
    if key not in cache:
        gene_cols = list(origin.columns[3:])
        parsed = [[ast.literal_eval(x) for x in row] for row in origin[gene_cols].values.tolist()]
        u = torch.tensor([[t[1] for t in row] for row in parsed], dtype=torch.float32)
        s = torch.tensor([[t[2] for t in row] for row in parsed], dtype=torch.float32)
        try:
            u = u.pin_memory()
            s = s.pin_memory()
        except Exception:
            pass
        cache[key] = (u, s)
    return cache[key]


def _gather_neighbors(mat_cpu: torch.Tensor, idx_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Gather rows by neighbor indices and return a ``[K, B, G]`` tensor on device."""
    if idx_np.ndim == 1:
        idx_np = idx_np[None, :]
    bb, k = idx_np.shape
    flat = torch.from_numpy(idx_np.reshape(-1)).long()
    gathered = mat_cpu.index_select(0, flat).view(bb, k, -1).permute(1, 0, 2).contiguous()
    return gathered.to(device, non_blocking=True)


def velocity_calculate(
    origin: pd.DataFrame,
    cellindex: torch.Tensor,
    unsplice: torch.Tensor,
    splice: torch.Tensor,
    unsplice_predict: torch.Tensor,
    splice_predict: torch.Tensor,
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    n_neighbors: int,
    loss_func: str,
    attn: Optional[torch.Tensor],
    nbrs: np.ndarray,
    negs: np.ndarray,
    _cache: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute loss and return predictions without altering model logic.

    The loss can be chosen among ``cosine``, ``rmse`` and ``triple``. Gene-wise
    weights are derived from attention with column-wise averaging.
    """
    _cache = _cache if _cache is not None else {}
    device = unsplice.device
    with time_section("origin_parse_cache", level=3):
        tu, ts = _prepare_origin_matrices(origin, _cache)

    with time_section("gather_neighbors", level=3):
        idx_np = nbrs[cellindex.long().cpu().numpy(), :]
        neg_np = negs[cellindex.long().cpu().numpy(), :]

        unsplice_all = _gather_neighbors(tu, idx_np, device)
        splice_all = _gather_neighbors(ts, idx_np, device)
        unsplice_all_n = _gather_neighbors(tu, neg_np, device)
        splice_all_n = _gather_neighbors(ts, neg_np, device)

    def _align_vec_to_g(w: torch.Tensor, g_needed: int) -> torch.Tensor:
        b, l = w.size()
        if l == g_needed:
            return w
        if l > g_needed:
            return w[:, :g_needed]
        pad = g_needed - l
        row_mean = w.mean(dim=1, keepdim=True)
        return torch.cat([w, row_mean.expand(-1, pad)], dim=1)

    def _make_gene_weights(attn_tensor: torch.Tensor | None, batch: int, genes: int) -> torch.Tensor:
        """Map attention to per-gene weights [B, G] using column-wise mean.

        This matches the original behavior (take mean over columns/key axis).
        """
        if attn_tensor is None:
            return torch.full((batch, genes), 1.0 / genes, device=device)

        if attn_tensor.dim() == 4:
            # [B, H, G, G] -> average over heads
            A = attn_tensor.mean(dim=1)
        elif attn_tensor.dim() == 3:
            # [B, G, G]
            A = attn_tensor
        elif attn_tensor.dim() == 2:
            # Already per-gene weights: [B, L] -> align to [B, G]
            return _align_vec_to_g(attn_tensor.to(device), genes)
        elif attn_tensor.dim() == 1:
            return torch.full((batch, genes), 1.0 / genes, device=device)
        else:
            return torch.full((batch, genes), 1.0 / genes, device=device)

        # Ensure [B, G, G]
        q, k = A.size(-2), A.size(-1)
        if (q != genes) or (k != genes):
            L = min(q, k)
            A = A[:, :L, :L]
            if L < genes:
                pad_q = genes - L
                pad_k = genes - L
                pad_val = A.mean()
                A = F.pad(A, (0, pad_k, 0, pad_q), value=float(pad_val))

        # Column-wise mean (key-centric): [B, G]
        w_col = A.mean(dim=-2)
        return w_col.to(device)

    with time_section("attn_to_gene_weights", level=3):
        loss_weight = _make_gene_weights(attn, unsplice.shape[0], unsplice.shape[1]).to(unsplice.dtype)

    def _weight_triple(us_all, sp_all, us_neg, sp_neg, us, sp, up, sp_pred):
        tl = WeightedFeatureTripletLoss()
        uv, sv = up - us, sp_pred - sp
        unv = us_all[1:, :, :] - us
        snv = sp_all[1:, :, :] - sp
        unv_neg = us_neg[:, :, :] - us
        snv_neg = sp_neg[:, :, :] - sp
        pos_u, pos_s = torch.mean(unv, dim=0), torch.mean(snv, dim=0)
        neg_u, neg_s = torch.mean(unv_neg, dim=0), torch.mean(snv_neg, dim=0)
        return 0.5 * tl(uv, pos_u, neg_u, loss_weight) + 0.5 * tl(sv, pos_s, neg_s, loss_weight)

    def _cosine(us_all, sp_all, us, sp, up, sp_pred):
        uv, sv = up - us, sp_pred - sp
        unv = us_all[1:, :, :] - us
        snv = sp_all[1:, :, :] - sp
        den = torch.sqrt(unv**2 + snv**2) * torch.sqrt(uv**2 + sv**2) + 1e-12
        cos = (unv * uv + snv * sv) / den
        cos_max, _ = torch.max(cos, dim=0)
        return 1.0 - cos_max

    def _rmse(us_all, sp_all, us, sp, up, sp_pred):
        uv, sv = up - us, sp_pred - sp
        unv = us_all[1:, :, :] - us
        snv = sp_all[1:, :, :] - sp
        err = torch.sqrt(0.5 * ((uv - unv) ** 2 + (sv - snv) ** 2))
        return torch.mean(err * loss_weight.unsqueeze(0), dim=0)

    with time_section("compute_loss", level=3):
        if loss_func == 'cosine':
            cost = _cosine(unsplice_all, splice_all, unsplice, splice, unsplice_predict, splice_predict).mean()
        elif loss_func == 'rmse':
            cost = _rmse(unsplice_all, splice_all, unsplice, splice, unsplice_predict, splice_predict).mean()
        elif loss_func == 'triple':
            tmp = _weight_triple(unsplice_all, splice_all, unsplice_all_n, splice_all_n, unsplice, splice, unsplice_predict, splice_predict)
            cost = tmp.mean() if tmp.ndim > 0 else tmp
        else:
            raise ValueError(f"Unknown loss_func: {loss_func}")
    return cost, unsplice_predict, splice_predict


class FullModelCellWise(pl.LightningModule):
    """PyTorch Lightning module encapsulating the cell-wise GRAVITY stage."""

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
        attention_output: bool = True,
        output_network_path: str = 'attentions',
        cell_to_type: Optional[Dict[str, str]] = None,
        gene_list_path: str = 'genes.txt',
        gene_mapper_path: str = 'genemap.json',
        csr_topk: int = 64,
        learning_rate: float = 1e-5,
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
        self.attention_output = attention_output
        self.cell_to_type = cell_to_type or {}
        self.gene_list_path = gene_list_path
        self.gene_mapper_path = gene_mapper_path
        self._origin_cache: dict = {}
        self.learning_rate = learning_rate
        if self.attention_output:
            self.output_network_path = output_network_path
            self.scorer = TFScorer(
                gene_list_path=self.gene_list_path,
                gene_mapper_json=self.gene_mapper_path,
                topk_frac=0.10,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self.builder = CSRStreamBuilder(n_cols=len(self.scorer.keys), per_row_topk=csr_topk, value_threshold=0.0)
            self.obs_names = []
            self.obs_celltype = []
            self.type_sums: Dict[str, torch.Tensor] = {}
            self.type_counts = defaultdict(int)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        cell_info, gene_info = x[:, :3], x[:, 3:]
        points = cell_info[:, 1:]
        unsplice_idx = get_unsplice_indices(self.input_dim)
        splice_idx = get_splice_indices(self.input_dim)
        unsplice = gene_info[:, unsplice_idx] + 1e-8
        splice = gene_info[:, splice_idx] + 1e-8
        if isinstance(mask, torch.Tensor):
            attn_mask = mask
        else:
            mask_mode = 'all' if mask == 'all' else None
            with time_section("mask_generate", level=3):
                attn_mask = mask_generate(unsplice, splice, self.gene_list, self.TFTG_map_reverse, mode=mask_mode)
        umax = unsplice.max(dim=0)[0]
        smax = splice.max(dim=0)[0]
        alpha0 = umax * 2.0
        beta0 = torch.ones_like(umax)
        gamma0 = umax / (smax + 1e-8)
        with time_section("gravity_forward", level=2):
            return self.GravityModel(unsplice, splice, points, attn_mask, self.output_dim_trans, self.output_dim_trans, alpha0, beta0, gamma0)

    def training_step(self, batch, batch_idx):
        _, x = batch
        with time_section("forward(cell)"):
            output = self(x)
        cell_info, gene_info = x[:, :3], x[:, 3:]
        points = cell_info[:, 1:]
        unsplice_idx = get_unsplice_indices(x.shape[1])
        splice_idx = get_splice_indices(x.shape[1])
        unsplice = gene_info[:, unsplice_idx]
        splice = gene_info[:, splice_idx]
        cell_index = cell_info[:, 0].to('cpu')
        with time_section("velocity_calculate", level=2):
            loss = velocity_calculate(self.origin_data, cell_index, unsplice, splice, output[0], output[1], points[:, 0], points[:, 1], 30, 'triple', self.GravityModel.attention, self.nbrs, self.negs, self._origin_cache)[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, x = batch
        with time_section("forward(test)"):
            output = self(x)
        cell_info, gene_info = x[:, :3], x[:, 3:]
        points = cell_info[:, 1:]
        unsplice_idx = get_unsplice_indices(x.shape[1])
        splice_idx = get_splice_indices(x.shape[1])
        unsplice = gene_info[:, unsplice_idx]
        splice = gene_info[:, splice_idx]
        cell_index = cell_info[:, 0].to('cpu')

        if self.attention_output:
            if not os.path.exists(self.output_network_path):
                os.makedirs(self.output_network_path, exist_ok=True)
            with time_section("export_attention_batch", level=2):
                att_batch = self.GravityModel.attention.detach().to("cpu", dtype=torch.float32)
                ctype_map = self.cell_to_type or {}

                for row in range(att_batch.shape[0]):
                    celli = cell_index[row]
                    atts = att_batch[row]
                    if atts.ndim == 4:
                        mat_gc = atts.mean(dim=(0, 1))
                    elif atts.ndim == 3:
                        mat_gc = atts.mean(dim=0)
                    elif atts.ndim == 2:
                        mat_gc = atts
                    else:
                        raise ValueError(f"Unsupported attention shape per cell: {tuple(atts.shape)}")

                    ctype = ctype_map.get(str(int(celli.item())), "NA")
                    if ctype not in self.type_sums:
                        self.type_sums[ctype] = mat_gc.clone()
                    else:
                        self.type_sums[ctype].add_(mat_gc)
                    self.type_counts[ctype] += 1

                    with time_section("tf_score_one", level=3):
                        tf_scores = self.scorer.score_one(mat_gc)

                    self.builder.add_row(tf_scores.numpy())
                    self.obs_names.append(str(int(celli.item())))
                    self.obs_celltype.append(ctype)

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
        loss = velocity_calculate(self.origin_data, cell_index, unsplice, splice, output[0], output[1], points[:, 0], points[:, 1], 30, 'triple', self.GravityModel.attention, self.nbrs, self.negs, self._origin_cache)[0]
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        if hasattr(self, "trainer") and getattr(self.trainer, "global_rank", 0) != 0:
            return
        self.csv = self.csv.sort_values(by=['gene_name', 'cellIndex'])
        self.csv.reset_index(drop=True, inplace=True)
        if self.output_csv:
            self.csv.to_csv(self.output_csv)

        if self.attention_output:
            from anndata import AnnData

            X = self.builder.to_csr(n_rows=len(self.obs_names))
            adata = AnnData(X)
            adata.obs_names = pd.Index([str(x) for x in self.obs_names], dtype="string")
            adata.var_names = pd.Index([str(v) for v in self.scorer.keys], dtype="string")
            adata.obs["cell_type"] = pd.Categorical([str(x) for x in self.obs_celltype])

            out_h5ad = os.path.join(self.output_network_path, "attention_TF_scores_with_types.h5ad")
            adata.write_h5ad(out_h5ad)
            log_verbose(f"Saved TF scores to {out_h5ad} | shape {adata.shape} | nnz {X.nnz}", level=1)

            mean_dir = os.path.join(self.output_network_path, "mean_attention_by_celltype")
            os.makedirs(mean_dir, exist_ok=True)
            for ctype, sum_mat in self.type_sums.items():
                n = int(self.type_counts.get(ctype, 0))
                if n <= 0:
                    continue
                mean_mat = (sum_mat / n)
                from scipy.sparse import csr_matrix, save_npz

                M_csr = csr_matrix(mean_mat.numpy())
                safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ctype)
                save_npz(os.path.join(mean_dir, f"{safe}_mean_attention.npz"), M_csr)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
