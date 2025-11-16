"""Utilities to interpret attention-derived TF scores.

This helper wraps the manual snippet that ranked regulatory factors (TFs) by their
differential scores across cell types.  Given the exported
``attention_TF_scores_with_types.h5ad`` (produced after stage-1 inference), the
module loads the AnnData, ensures categorical cell-type annotations, performs
``sc.tl.rank_genes_groups`` and returns a tidy table ready for inspection or
downstream filtering.

Typical usage::

    from gravity.analysis.attention_explain import rank_attention_differentials
    df = rank_attention_differentials(
        "/path/to/attention_TF_scores_with_types.h5ad",
        cluster_key="cell_type",
        method="wilcoxon",
        group="Beta",
        top_n=30,
        save_csv="beta_top_tfs.csv",
    )
    print(df)

The returned dataframe mirrors the ad-hoc analysis carried out previously while
providing a reusable API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict

import pandas as pd
import scanpy as sc
from scipy.sparse import load_npz
import numpy as np
import re

from ..utils import log_verbose, resolve_path

__all__ = ["rank_attention_differentials", "rank_tf_targets", "build_regulatory_modules"]


def _to_categorical(adata, cluster_key: str) -> None:
    if cluster_key not in adata.obs:
        raise KeyError(f"Column '{cluster_key}' not found in adata.obs")
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")


def _rank_groups(adata, cluster_key: str, method: str, key_added: str) -> None:
    sc.tl.rank_genes_groups(adata, cluster_key, method=method, key_added=key_added)


def _collect_rankings(
    adata,
    key_added: str,
    top_n: Optional[int],
) -> pd.DataFrame:
    df = sc.get.rank_genes_groups_df(adata, key=key_added, group=None)
    df = df.rename(columns={
        "names": "regulatory_factor",
        "logfoldchanges": "logfoldchange",
        "pvals": "pval",
    })
    df = df[["group", "regulatory_factor", "logfoldchange", "pval"]]
    df.sort_values(["group", "logfoldchange", "pval"], ascending=[True, False, True], inplace=True)

    if top_n is not None:
        df = df.groupby("group", as_index=False).head(top_n)
    return df.reset_index(drop=True)


def rank_attention_differentials(
    attention_path: str,
    *,
    cluster_key: str = "cell_type",
    method: str = "wilcoxon",
    group: Optional[str] = None,
    top_n: int = 30,
    save_csv: Optional[str] = None,
    key_added: str = "wilcoxon",
) -> pd.DataFrame:
    """Rank regulatory factors (TFs) from attention-derived scores across cell types.

    Parameters
    ----------
    attention_path:
        Path to ``attention_TF_scores_with_types.h5ad`` produced after stage-1
        inference.
    cluster_key:
        Column in ``adata.obs`` indicating cell-type/cluster labels.
    method:
        Differential ranking method passed to ``scanpy.tl.rank_genes_groups``.
    group:
        Optional target group. When provided, the returned dataframe is filtered
        to the specified group and sorted by descending log-fold change (ties by
        ascending ``pval``).
    top_n:
        Number of top rows to keep when ``group`` is specified.
    save_csv:
        Optional path to persist the resulting dataframe.
    key_added:
        Key used by Scanpy to store ranking results.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``group``, ``regulatory_factor``, ``logfoldchange`` and ``pval``.
    """

    path = resolve_path(attention_path)
    log_verbose(f"[attention] loading TF scores: {path}", level=1)
    adata = sc.read_h5ad(path)

    _to_categorical(adata, cluster_key)
    _rank_groups(adata, cluster_key, method, key_added)
    df = _collect_rankings(adata, key_added, top_n if group is None else None)

    if group is not None:
        log_verbose(f"[attention] extracting rankings for group '{group}'", level=1)
        df = df[df["group"] == group].copy()
        df.sort_values(["logfoldchange", "pval"], ascending=[False, True], inplace=True)
        df = df.head(top_n).reset_index(drop=True)

    if save_csv is not None:
        out_path = Path(save_csv).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log_verbose(f"[attention] saved rankings to {out_path}", level=1)

    return df


def _load_gene_list(attention_dir: Path) -> List[str]:
    genes_path = attention_dir.parent / "genes.txt"
    if not genes_path.exists():
        raise FileNotFoundError(f"Gene list not found at {genes_path}")
    genes: List[str] = []
    with genes_path.open("r") as handle:
        for line in handle:
            name = line.strip()
            if name:
                genes.append(name.upper())
    if not genes:
        raise ValueError("Gene list is empty; cannot interpret attention matrix.")
    return genes


def _sanitize_cell_type(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def rank_tf_targets(
    attention_dir: str,
    tf_gene: str,
    cell_type: str,
    *,
    top_k: int = 10,
    threshold: Optional[float] = None,
    target_gene: Optional[str] = None,
) -> pd.DataFrame | Tuple[int, float]:
    """Retrieve top target genes for a regulatory factor within a given cell type attention network.

    Parameters
    ----------
    attention_dir:
        Directory containing the ``attentions/`` artefacts (same folder that holds
        ``mean_attention_by_celltype`` and ``genes.txt``).
    tf_gene:
        Regulatory factor (TF) symbol to inspect.
    cell_type:
        Cell type/cluster name. The helper uses the same sanitisation as the
        stage-1 exporter when locating ``mean_attention_by_celltype/<cell>.npz``.
    top_k:
        Number of top targets to return when ``target_gene`` is ``None``. Defaults
        to 10.
    threshold:
        Optional minimum edge weight to keep. When ``None`` (default) no
        thresholding is applied.
    target_gene:
        When provided, the function returns a tuple ``(rank, weight)`` for the
        specified target within the ranked list. The rank is ``1``-based. When
        ``None`` (default) a dataframe of the top targets is returned.

    Returns
    -------
    pandas.DataFrame or Tuple[int, float]
        If ``target_gene`` is ``None``: dataframe with columns ``rank``,
        ``target_gene`` and ``weight``. Otherwise a tuple containing the rank and
        edge weight for the requested target (raises if the gene is absent).
    threshold:
        When ``None`` (default) no filtering is applied. If ``0 < threshold < 1``
        the value is interpreted as a quantile (e.g. ``0.99`` keeps the top 1%
        weights). Otherwise it is treated as an absolute cutoff.
    """

    base_dir = Path(resolve_path(attention_dir))
    genes = _load_gene_list(base_dir)
    gene_to_idx = {g: idx for idx, g in enumerate(genes)}

    tf_key = tf_gene.upper()
    if tf_key not in gene_to_idx:
        raise KeyError(f"TF '{tf_gene}' not found in gene list ({base_dir.parent/'genes.txt'}).")

    safe_cell = _sanitize_cell_type(cell_type)
    mean_dir = base_dir / "mean_attention_by_celltype"
    npz_path = mean_dir / f"{safe_cell}_mean_attention.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Mean attention matrix for cell type '{cell_type}' not found at {npz_path}")

    mat = load_npz(npz_path).toarray()

    tf_idx = gene_to_idx[tf_key]
    column = mat[:, tf_idx]

    order = np.argsort(-column)
    weights = column[order]

    if threshold is not None:
        if 0.0 < threshold < 1.0:
            cutoff = float(np.quantile(weights, threshold))
        else:
            cutoff = float(threshold)
        valid_mask = weights >= cutoff
        order = order[valid_mask]
        weights = weights[valid_mask]

    if target_gene is not None:
        target_key = target_gene.upper()
        if target_key not in gene_to_idx:
            raise KeyError(f"Target gene '{target_gene}' not present in gene list.")
        target_idx = gene_to_idx[target_key]
        matches = np.where(order == target_idx)[0]
        if matches.size == 0:
            raise ValueError(
                f"Target gene '{target_gene}' does not meet the threshold for TF '{tf_gene}' in cell type '{cell_type}'."
            )
        rank_pos = int(matches[0])
        return rank_pos + 1, float(weights[rank_pos])

    limit = min(top_k, len(order))
    records = []
    for rank, idx in enumerate(order[:limit], 1):
        records.append({
            "rank": rank,
            "target_gene": genes[idx],
            "weight": float(weights[rank - 1]),
        })

    return pd.DataFrame(records, columns=["rank", "target_gene", "weight"])


def build_regulatory_modules(
    attention_path: str,
    attention_dir: str,
    *,
    cluster_key: str = "cell_type",
    method: str = "wilcoxon",
    top_tf: int = 10,
    top_tg: int = 5,
    threshold: float = 0.0,
) -> Dict[str, pd.DataFrame]:
    """Construct regulatory-factor → target modules per cell type.

    Top regulatory factors are derived via :func:`rank_attention_differentials`;
    for each factor we then extract the ``top_tg`` strongest targets using
    :func:`rank_tf_targets` (respecting ``threshold``).
    """

    tf_table = rank_attention_differentials(
        attention_path,
        cluster_key=cluster_key,
        method=method,
        group=None,
        top_n=top_tf,
    )

    modules: Dict[str, pd.DataFrame] = {}
    for cell_type, subset in tf_table.groupby("group"):
        rows: List[pd.DataFrame] = []
        for factor in subset["regulatory_factor"].tolist():
            tg_df = rank_tf_targets(
                attention_dir=attention_dir,
                tf_gene=factor,
                cell_type=cell_type,
                top_k=top_tg,
                threshold=threshold,
                target_gene=None,
            )
            tg_df = tg_df.assign(regulatory_factor=factor)[["regulatory_factor", "target_gene", "weight"]]
            rows.append(tg_df)
        modules[cell_type] = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(columns=["regulatory_factor", "target_gene", "weight"])
        )

    return modules
