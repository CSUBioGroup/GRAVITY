"""Analyze transcription factor importance scores produced by GRAVITY."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import inspect
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

from ..utils import log_verbose, resolve_path

__all__ = [
    "rank_tf_scores",
]


def rank_tf_scores(
    attention_h5ad: str,
    *,
    groupby: str = 'cell_type',
    method: str = 'wilcoxon',
    key_added: str = 'tf_rankings',
    n_genes: int = 30,
    output_plot: Optional[str] = None,
    sort_group: Optional[Union[int, str]] = None,
    top_n: int = 30,
    reuse_h5ad: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Run differential ranking on TF scores aggregated per cell type.

    Parameters
    ----------
    attention_h5ad:
        Path to `attention_TF_scores_with_types.h5ad`.
    sort_group:
        Group to sort by (logFC descending, p-value ascending). Accepts index
        (int) or name (str).
    reuse_h5ad:
        If provided with an h5ad already containing `uns[key_added]`, reuse the
        stored ranking results.

    Returns
    -------
    all_rankings, top_rankings:
        DataFrame of all per-group rankings, and (if `sort_group` is given)
        the Top-N table for the selected group.
    """

    path = resolve_path(reuse_h5ad or attention_h5ad)

    adata = sc.read_h5ad(path)
    if groupby not in adata.obs:
        raise KeyError(f"column '{groupby}' not present in adata.obs")

    if reuse_h5ad is None:
        log_verbose(f"[gravity] ranking TF scores within groups '{groupby}'", level=1)
        sc.tl.rank_genes_groups(adata, groupby, method=method, key_added=key_added)

    if key_added not in adata.uns:
        raise KeyError(f"'{key_added}' not found in adata.uns; ensure ranking has been computed")

    if output_plot is not None and reuse_h5ad is None:
        fig = None
        try:
            sig = inspect.signature(sc.pl.rank_genes_groups)
            if 'return_fig' in sig.parameters:
                fig_or_axes = sc.pl.rank_genes_groups(adata, key=key_added, n_genes=n_genes, sharey=False, show=False, return_fig=True)
                if hasattr(fig_or_axes, 'savefig'):
                    fig = fig_or_axes
                elif isinstance(fig_or_axes, (list, tuple)) and len(fig_or_axes) > 0 and hasattr(fig_or_axes[0], 'figure'):
                    fig = fig_or_axes[0].figure
            else:
                axes = sc.pl.rank_genes_groups(adata, key=key_added, n_genes=n_genes, sharey=False, show=False)
                if axes is None:
                    fig = plt.gcf()
                elif isinstance(axes, (list, tuple)) and len(axes) > 0 and hasattr(axes[0], 'figure'):
                    fig = axes[0].figure
                elif hasattr(axes, 'figure'):
                    fig = axes.figure
        except Exception:
            fig = plt.gcf()

        if fig is not None:
            plot_path = Path(output_plot).expanduser().resolve()
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            log_verbose(f"[gravity] saved TF ranking plot to {plot_path}", level=2)

    names = adata.uns[key_added]['names']
    pvals = adata.uns[key_added]['pvals']
    logfold = adata.uns[key_added]['logfoldchanges']

    categories = list(adata.obs[groupby].cat.categories)
    records: List[dict] = []
    max_genes = min(n_genes, names.shape[1])
    for idx, group in enumerate(categories):
        for gene_idx in range(max_genes):
            records.append({
                'group': group,
                'rank': gene_idx + 1,
                'tf': names[idx, gene_idx],
                'logFC': logfold[idx, gene_idx],
                'pval': pvals[idx, gene_idx],
            })

    all_rankings = pd.DataFrame.from_records(records)

    top_rankings: Optional[pd.DataFrame] = None
    if sort_group is not None:
        if isinstance(sort_group, int):
            if sort_group < 0 or sort_group >= len(categories):
                raise IndexError(f"sort_group index {sort_group} out of range")
            target_group = categories[sort_group]
        else:
            target_group = str(sort_group)
            if target_group not in categories:
                raise KeyError(f"group '{target_group}' not found; available: {categories}")
        top_rankings = (all_rankings[all_rankings['group'] == target_group]
                        .sort_values(['logFC', 'pval'], ascending=[False, True])
                        .head(top_n)
                        .reset_index(drop=True))

    return all_rankings, top_rankings
