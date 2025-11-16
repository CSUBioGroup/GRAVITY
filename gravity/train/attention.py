"""Attention utilities for GRAVITY training exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from scipy.sparse import csr_matrix

__all__ = [
    "CSRStreamBuilder",
    "TFScorer",
]


class CSRStreamBuilder:
    """Incrementally build a CSR matrix keeping top-k entries per row.

    Parameters
    ----------
    n_cols:
        Total number of columns in the target CSR matrix.
    per_row_topk:
        If > 0, keep only the largest ``k`` values per row; otherwise keep
        entries above ``value_threshold``.
    value_threshold:
        Minimum value to include when ``per_row_topk`` is disabled.
    """

    def __init__(self, n_cols: int, per_row_topk: int = 64, value_threshold: float = 0.0):
        self.n_cols = int(n_cols)
        self.per_row_topk = per_row_topk
        self.value_threshold = float(value_threshold)
        self.data: List[float] = []
        self.indices: List[int] = []
        self.indptr = [0]
        self._nnz = 0

    def add_row(self, row: np.ndarray) -> None:
        """Append one dense row to the internal CSR buffers.

        The method selects either the top-k entries or those above the
        threshold, then updates ``data``, ``indices`` and ``indptr``.
        """
        if self.per_row_topk and 0 < self.per_row_topk < row.size:
            idx = np.argpartition(row, -self.per_row_topk)[-self.per_row_topk:]
            idx = idx[np.argsort(-row[idx])]
        else:
            idx = np.flatnonzero(row > self.value_threshold)
        vals = row[idx].astype(np.float32, copy=False)
        self.data.extend(vals.tolist())
        self.indices.extend(idx.tolist())
        self._nnz += len(idx)
        self.indptr.append(self._nnz)

    def to_csr(self, n_rows: int) -> csr_matrix:
        """Materialize the accumulated rows into a SciPy CSR matrix."""
        return csr_matrix(
            (np.asarray(self.data, np.float32),
             np.asarray(self.indices, np.int32),
             np.asarray(self.indptr, np.int64)),
            shape=(n_rows, self.n_cols),
            dtype=np.float32,
        )


class TFScorer:
    """Summarise gene attention scores into transcription factor activity.

    The scorer aggregates per-gene attention into TF activity vectors using a
    TF→target mapping. To reduce compute, it restricts to candidate columns
    determined by per-gene top-k selections.

    Parameters
    ----------
    gene_list_path:
        Path to a newline-separated list of genes; defines matrix axes.
    gene_mapper_json:
        JSON mapping from TF symbol to list of target genes.
    topk_frac:
        Fraction of columns kept per gene (``k = ceil(frac * n_cells)``).
    device, dtype:
        Torch device and dtype for compute.
    """

    def __init__(
        self,
        gene_list_path: str,
        gene_mapper_json: str,
        topk_frac: float = 0.10,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.topk_frac = float(topk_frac)

        with open(gene_list_path, 'r') as fp:
            self.genes = [l.strip() for l in fp]
        self.g2i = {g: i for i, g in enumerate(self.genes)}

        with open(gene_mapper_json, 'r') as fp:
            mapper_raw = json.load(fp)

        mapper = {tf: [t for t in tg if t in self.g2i]
                  for tf, tg in mapper_raw.items() if tf in self.g2i}
        mapper = {tf: tg for tf, tg in mapper.items() if len(tg) > 0}
        self.mapper = mapper

        rev = {}
        for tf, tg in mapper.items():
            for t in tg:
                rev.setdefault(t, []).append(tf)
        self.rows_need = torch.tensor([self.g2i[g] for g in rev.keys()], dtype=torch.long)

        self.keys = sorted(self.mapper.keys(), key=lambda k: self.g2i[k])
        self.key2pos = {k: i for i, k in enumerate(self.keys)}
        self.tf_cols_all = torch.tensor([self.g2i[tf] for tf in self.mapper.keys()], dtype=torch.long)
        self.tf_name_list = list(self.mapper.keys())
        self.mapper_indices = {tf: torch.tensor([self.g2i[t] for t in tg], dtype=torch.long)
                               for tf, tg in self.mapper.items()}

    @torch.no_grad()
    def score_one(self, mat_gc: torch.Tensor) -> torch.Tensor:
        """Compute a TF activity vector from a gene×cell attention matrix.

        Parameters
        ----------
        mat_gc:
            Tensor of shape ``[G, C]`` with per-gene attention over columns.

        Returns
        -------
        torch.Tensor
            Vector of length ``len(self.keys)`` aligned to the TF list.
        """
        mat_gc = mat_gc.to(self.device, dtype=self.dtype, non_blocking=True)
        g, c = mat_gc.shape
        k = max(1, int(c * self.topk_frac))

        rows_need = self.rows_need.to(self.device)
        sub = mat_gc.index_select(0, rows_need)
        _, top_idx = torch.topk(sub, k, dim=1, largest=True, sorted=False)
        cand_cols = torch.unique(top_idx)

        tf_cols_all = self.tf_cols_all.to(self.device)
        tf_mask = torch.isin(tf_cols_all, cand_cols)
        if not tf_mask.any():
            return torch.zeros(len(self.keys), dtype=torch.float32)

        sel_tf_cols = tf_cols_all[tf_mask]
        sel_tf_names = [n for n, m in zip(self.tf_name_list, tf_mask.tolist()) if m]

        rows_all = []
        cols_all = []
        groups = []
        for i, tf in enumerate(sel_tf_names):
            tgt = self.mapper_indices[tf].to(self.device)
            rows_all.append(tgt)
            cols_all.append(sel_tf_cols[i].expand_as(tgt))
            groups.append(torch.full((tgt.numel(),), i, dtype=torch.long, device=self.device))

        rows_all = torch.cat(rows_all)
        cols_all = torch.cat(cols_all)
        groups = torch.cat(groups)
        vals = mat_gc[rows_all, cols_all]

        tf_scores_local = torch.zeros(len(sel_tf_names), dtype=self.dtype, device=self.device)
        tf_scores_local.index_add_(0, groups, vals)

        out = torch.zeros(len(self.keys), dtype=self.dtype, device=self.device)
        pos = torch.tensor([self.key2pos[n] for n in sel_tf_names], dtype=torch.long, device=self.device)
        out.index_copy_(0, pos, tf_scores_local)
        return out.float().cpu()
