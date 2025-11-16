"""Dataset implementations used by GRAVITY training stages."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils import log_verbose, resolve_path

__all__ = [
    "PreprocessDataset",
    "CustomDataset",
    "CustomDatasetGeneWise",
]


class PreprocessDataset(Dataset):
    """Transform long-format counts into the wide format used by GRAVITY."""

    def __init__(self, csv_file: str, middle_name: str):
        csv_path = resolve_path(csv_file)
        data = pd.read_csv(csv_path)
        if 'cellIndex' not in data.columns:
            data.insert(0, 'cellIndex', pd.factorize(data['cellID'])[0])
        data = data.iloc[:, [0, 1, 2, 3, -2, -1]]
        unique_indices = data.cellIndex.unique()
        records = []
        for cell_idx in tqdm(unique_indices, desc="preprocess"):
            subset = data[data.cellIndex == cell_idx]
            row = {
                'cellIndex': subset.iloc[0, 0],
                'embedding1': subset.iloc[0, -2],
                'embedding2': subset.iloc[0, -1],
            }
            for _, entry in subset.iterrows():
                gene_tup = (entry.gene_name, entry.unsplice, entry.splice)
                row[f'gene_u_s_{entry.gene_name}'] = str(gene_tup)
            records.append(row)
        combined_df = pd.DataFrame(records)
        output_path = Path(middle_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)

    def __len__(self) -> int:  # pragma: no cover - dataset placeholder
        return 0

    def __getitem__(self, index):  # pragma: no cover - dataset placeholder
        raise NotImplementedError


class CustomDataset(Dataset):
    """Dataset consumed by the cell-wise training stage."""

    def __init__(self, csv_file: str, prior: str = './prior_data/network_mouse.zip', gene_select: Optional[Sequence[str]] = None, n_pos_neighbors: int = 30, n_neg_neighbors: int = 10):
        csv_path = resolve_path(csv_file)
        data = pd.read_csv(csv_path, index_col=None)
        if gene_select is not None:
            hvgs = [g.upper() for g in gene_select]
        else:
            hvgs = [name.upper().split('_')[-1] for name in data.columns[3:]]
        hvgs = [g.upper() for g in hvgs]

        self.niche_dict: dict[str, list[str]] = {}
        self.niche_dict_reverse: dict[str, list[str]] = {}

        self.TF_from_net, self.TG_from_net = self._load_prior(prior, hvgs)
        self.hvg_full = [name.upper().split('_')[-1] for name in data.columns[3:]]
        sub_index = [0, 1, 2]
        sub_genes = [self.hvg_full.index(item) + 3 for item in hvgs]
        sub_index.extend(sub_genes)
        self.data = data.iloc[:, sub_index]
        self.hvg = hvgs
        self.nbrs, self.negs = self._build_neighbors(n_pos_neighbors, n_neg_neighbors)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor]:
        line = self.data.iloc[idx]
        value_list = list(line.values)
        value_final = value_list[:3]
        for item in value_list[3:]:
            gene_triplet = ast.literal_eval(item)
            value_final.append(gene_triplet[1])
            value_final.append(gene_triplet[2])
        return int(value_list[0]), torch.tensor(value_final, dtype=torch.float32)

    def _load_prior(self, path: Optional[str], hvgs: Sequence[str]) -> Tuple[list[str], list[str]]:
        self.niche_dict.clear()
        self.niche_dict_reverse.clear()
        if path is None:
            log_verbose("[gravity] prior network missing; assuming fully connected mask.", level=1)
            unique = list(dict.fromkeys(g.upper() for g in hvgs))
            return unique, unique

        try:
            resolved = resolve_path(path, must_exist=False)
            if not Path(resolved).exists():
                raise FileNotFoundError
            net = pd.read_csv(resolved, index_col=None, header=0)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            log_verbose(
                "[gravity] prior network not found; assuming fully connected mask.",
                level=1,
            )
            unique = list(dict.fromkeys(g.upper() for g in hvgs))
            return unique, unique

        net['from'] = net['from'].str.upper()
        net['to'] = net['to'].str.upper()
        net = net.loc[net['from'].isin(hvgs) & net['to'].isin(hvgs), :]
        net = net.drop_duplicates(subset=['from', 'to'], keep='first', inplace=False)
        if net.empty:
            log_verbose(
                "[gravity] prior network empty after filtering; assuming fully connected mask.",
                level=1,
            )
            unique = list(dict.fromkeys(g.upper() for g in hvgs))
            return unique, unique

        for _, row in net.iterrows():
            self.niche_dict.setdefault(row['from'], []).append(row['to'])
            self.niche_dict_reverse.setdefault(row['to'], []).append(row['from'])
        TFs = list(net['from'].unique())
        TGs = list(net['to'].unique())
        hvgs_clean = [g for g in hvgs if g in TFs or g in TGs]
        removed = set(hvgs) - set(hvgs_clean)
        for gene in removed:
            log_verbose(f"filtered out gene without prior support: {gene}", level=2)
        hvgs[:] = hvgs_clean
        return TFs, TGs

    def _build_neighbors(self, n_pos_neighbors: int = 30, n_neg_neighbors: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        points = self.data.iloc[:, 1:3].to_numpy(dtype=float)
        nbrs = NearestNeighbors(n_neighbors=n_pos_neighbors, algorithm='ball_tree').fit(points)
        _, indices = nbrs.kneighbors(points)

        def farthest_distance_metric(x, y):
            return -np.linalg.norm(x - y)
        neg_k = max(1, min(n_neg_neighbors, points.shape[0]))
        try:
            neg_nbrs = NearestNeighbors(
                n_neighbors=neg_k,
                algorithm='ball_tree',
                metric=farthest_distance_metric,
            ).fit(points)
            _, negs = neg_nbrs.kneighbors(points)
        except Exception as exc:  # pragma: no cover - fallback for invalid metrics
            log_verbose(
                f"[gravity] fallback to distance-based farthest neighbors (k={neg_k}): {exc}",
                level=1,
            )
            diff = points[:, None, :] - points[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2, dtype=float))
            np.fill_diagonal(dist, -np.inf)
            idx_part = np.argpartition(dist, -neg_k, axis=1)[:, -neg_k:]
            dist_part = np.take_along_axis(dist, idx_part, axis=1)
            order = np.argsort(-dist_part, axis=1)
            negs = np.take_along_axis(idx_part, order, axis=1)

        return indices, np.array(negs)


class CustomDatasetGeneWise(Dataset):
    """Dataset consumed by the gene-wise refinement stage."""

    def __init__(self, csv_file: str, prior: str = './prior_data/network_mouse.zip', gene_select: Optional[Sequence[str]] = None, future_pos: str = './final_positions_with_index_yixian.npy'):
        csv_path = resolve_path(csv_file)
        data = pd.read_csv(csv_path, index_col=None)
        if gene_select is not None:
            hvgs = [g.upper() for g in gene_select]
        else:
            hvgs = [name.upper().split('_')[-1] for name in data.columns[3:]]
        hvgs = [g.upper() for g in hvgs]

        self.niche_dict: dict[str, list[str]] = {}
        self.niche_dict_reverse: dict[str, list[str]] = {}
        self.TF_from_net, self.TG_from_net = self._load_prior(prior, hvgs)
        self.hvg_full = [name.upper().split('_')[-1] for name in data.columns[3:]]
        sub_index = [0, 1, 2]
        sub_genes = [self.hvg_full.index(item) + 3 for item in hvgs]
        sub_index.extend(sub_genes)
        self.data = data.iloc[:, sub_index]
        self.hvg = hvgs
        self.nbrs = self._load_future_neighbors(future_pos)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor]:
        line = self.data.iloc[idx]
        value_list = list(line.values)
        value_final = value_list[:3]
        for item in value_list[3:]:
            gene_triplet = ast.literal_eval(item)
            value_final.append(gene_triplet[1])
            value_final.append(gene_triplet[2])
        return int(value_list[0]), torch.tensor(value_final, dtype=torch.float32)

    def _load_prior(self, path: Optional[str], hvgs: Sequence[str]) -> Tuple[list[str], list[str]]:
        self.niche_dict.clear()
        self.niche_dict_reverse.clear()
        if path is None:
            log_verbose("[gravity] prior network missing; assuming fully connected mask.", level=1)
            return list(dict.fromkeys(g.upper() for g in hvgs)), list(dict.fromkeys(g.upper() for g in hvgs))

        try:
            resolved = resolve_path(path, must_exist=False)
            if not Path(resolved).exists():
                raise FileNotFoundError
            net = pd.read_csv(resolved, index_col=None, header=0)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            log_verbose(
                "[gravity] prior network not found; assuming fully connected mask.",
                level=1,
            )
            return list(dict.fromkeys(g.upper() for g in hvgs)), list(dict.fromkeys(g.upper() for g in hvgs))

        net['from'] = net['from'].str.upper()
        net['to'] = net['to'].str.upper()
        net = net.loc[net['from'].isin(hvgs) & net['to'].isin(hvgs), :]
        net = net.drop_duplicates(subset=['from', 'to'], keep='first', inplace=False)
        if net.empty:
            log_verbose(
                "[gravity] prior network empty after filtering; assuming fully connected mask.",
                level=1,
            )
            return list(dict.fromkeys(g.upper() for g in hvgs)), list(dict.fromkeys(g.upper() for g in hvgs))

        for _, row in net.iterrows():
            self.niche_dict.setdefault(row['from'], []).append(row['to'])
            self.niche_dict_reverse.setdefault(row['to'], []).append(row['from'])
        TFs = list(net['from'].unique())
        TGs = list(net['to'].unique())
        hvgs_clean = [g for g in hvgs if g in TFs or g in TGs]
        removed = set(hvgs) - set(hvgs_clean)
        for gene in removed:
            log_verbose(f"filtered out gene without prior support: {gene}", level=2)
        hvgs[:] = hvgs_clean
        return TFs, TGs

    def _load_future_neighbors(self, future_pos: str) -> np.ndarray:
        resolved = resolve_path(future_pos)
        future_position = np.load(resolved)
        if future_position.shape[0] != len(self.data):
            raise ValueError(
                f"Future positions shape {future_position.shape} does not match dataset length {len(self.data)}"
            )
        indices_new = future_position[:, -1].astype(int)
        return indices_new
