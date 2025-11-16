"""Dataset builders mirroring the original GRAVITY training scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from ..utils import log_verbose, resolve_path
from .datasets import PreprocessDataset, CustomDataset, CustomDatasetGeneWise

# Add to __all__
__all__ = [
    "preprocess_counts",
    "load_cell_stage_dataset",
    "load_gene_stage_dataset",
    "export_intermediate_from_h5ad",   # new
    "adata_to_df_with_embed",          # new (self-contained version)
]

def export_intermediate_from_h5ad(
    input_h5ad: str,
    output_csv: str,
    *,
    retain_genes: Optional[Sequence[str]] = None,
    min_shared_counts: int = 20,
    n_top_genes: int = 2000,
    n_pcs: int = 30,
    n_neighbors: int = 30,
    embed_key: str = "X_umap",
    celltype_key: str = "cell_type",
    overwrite: bool = False,
    normalized: bool = False,
) -> Path:
    """Preprocess an AnnData (.h5ad) for GRAVITY and export an intermediate CSV.

    This function performs ONLY preprocessing required by GRAVITY's intermediate
    table. It does not run RNA velocity inference, does not plot, and does not
    save a processed .h5ad.

    Pipeline
    --------
    1) Read the input AnnData (.h5ad).
    2) Optionally force-keep user-specified genes through filtering/HVG selection.
    3) Run a single scVelo preprocessing pass (`filter_and_normalize`).
    4) Compute first-order moments with `scv.pp.moments` (produces 'Mu'/'Ms').
    5) Export a CSV via the local `adata_to_df_with_embed` function, including
       Mu/Ms, the chosen 2D embedding, and cell-type labels.

    Parameters
    ----------
    input_h5ad
        Path to an AnnData file that includes 'spliced' and 'unspliced' layers.
    output_csv
        Destination CSV file path for the intermediate table.
    retain_genes
        Genes that must be retained during filtering/HVG selection (if present).
    min_shared_counts
        Minimum shared counts across cells for gene filtering (scVelo).
    n_top_genes
        Number of highly variable genes to retain (in addition to `retain_genes`).
    n_pcs
        Number of principal components used by `scv.pp.moments`.
    n_neighbors
        Neighborhood size used by `scv.pp.moments`.
    embed_key
        Key in `adata.obsm` for a 2D embedding (e.g., "X_umap").
    celltype_key
        Column in `adata.obs` that holds cell-type labels (e.g., "cell_type" or "celltype").
    overwrite
        If False and `output_csv` exists, skip work and return the existing path.

    Returns
    -------
    pathlib.Path
        The path to the generated CSV.

    Raises
    ------
    KeyError
        If `embed_key` is not found in `adata.obsm`.
    RuntimeError
        If required layers ('spliced' and 'unspliced') are missing.
    ImportError
        If `scanpy` or `scvelo` are missing.
    """
    # Lazy imports to keep the top-level module light-weight.
    try:
        import scanpy as sc
        import scvelo as scv
    except Exception as e:
        raise ImportError("`export_intermediate_from_h5ad` requires `scanpy` and `scvelo`.") from e

    input_path = resolve_path(input_h5ad)
    csv_path = Path(output_csv).resolve()

    if csv_path.exists() and not overwrite:
        log_verbose(f"[gravity] found existing intermediate CSV: {csv_path}; skip.", level=1)
        return csv_path

    log_verbose(f"[gravity] reading h5ad: {input_path}", level=1)
    adata = sc.read_h5ad(str(input_path))

    # Basic sanity checks.
    if "spliced" not in adata.layers or "unspliced" not in adata.layers:
        raise RuntimeError(
            "[gravity] required layers 'spliced' and 'unspliced' are missing in the input AnnData.\n"
            "Please provide an .h5ad that includes these layers before preprocessing."
        )
    if embed_key not in adata.obsm:
        raise KeyError(f"[gravity] embedding '{embed_key}' not found in adata.obsm.")

    # Build retain list intersected with present genes to avoid warnings.
    keep_list = None
    if retain_genes:
        present = set(map(str, adata.var_names))
        keep_list = sorted([g for g in retain_genes if g in present])
        log_verbose(f"[gravity] retain_genes: requested={len(retain_genes)}, kept={len(keep_list)}", level=2)

    # Single-pass preprocessing (avoid double normalization).
    if normalized:
        log_verbose("[gravity] input data marked as normalized; skipping normalization step.", level=1)
        scv.pp.filter_genes_dispersion(adata, n_top_genes=n_top_genes, retain_genes=keep_list)
    else:
        log_verbose(
            f"[gravity] scvelo.filter_and_normalize(min_shared_counts={min_shared_counts}, "
            f"n_top_genes={n_top_genes}, retain_genes={len(keep_list) if keep_list else 0})",
            level=1,
        )
        scv.pp.normalize_per_cell(adata)
        scv.pp.log1p(adata)
        scv.pp.filter_and_normalize(
            adata,
            min_shared_counts=min_shared_counts,
            n_top_genes=n_top_genes,
            retain_genes=keep_list,
        )
    # First-order moments (creates 'Mu' and 'Ms' in `adata.layers`).
    log_verbose(f"[gravity] scvelo.moments(n_pcs={n_pcs}, n_neighbors={n_neighbors})", level=1)
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)

    # Export to CSV via local helper; ensures no external project dependency.
    log_verbose(f"[gravity] exporting intermediate CSV → {csv_path}", level=1)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = adata_to_df_with_embed(
        adata,
        us_para=["Mu", "Ms"],
        cell_type_para=celltype_key,
        embed_para=embed_key,
        save_path=str(csv_path),
        gene_list=None,  # default: all genes present after preprocessing
    )

    # Ensure the CSV exists (the helper already writes it).
    try:
        df.to_csv(csv_path, index=False)
    except Exception:
        pass

    return csv_path


def adata_to_df_with_embed(
    adata,
    us_para: Sequence[str] = ("Mu", "Ms"),
    cell_type_para: str = "celltype",
    embed_para: str = "X_umap",
    save_path: str = "cell_type_u_s_sample_df.csv",
    gene_list: Optional[Sequence[str]] = None,
):
    """Convert an AnnData object to a long CSV with per-gene/per-cell rows and 2D embedding.

    The resulting CSV contains, for every (gene, cell) pair:
    - gene_name, unsplice, splice
    - cellID, clusters (cell-type), embedding1, embedding2

    Notes
    -----
    - The two matrices are taken from `adata.layers[us_para[0]]` (unspliced) and
      `adata.layers[us_para[1]]` (spliced). By default, that is ['Mu', 'Ms'].
    - The 2D embedding is read from `adata.obsm[embed_para]` (e.g., 'X_umap').
    - This function writes the CSV incrementally (one gene at a time) to keep
      memory usage manageable for large datasets, then appends cell metadata.

    Parameters
    ----------
    adata
        An `anndata.AnnData` object containing layers and embedding.
    us_para
        Names of the two layers for unspliced and spliced moments/counts,
        respectively; default ('Mu', 'Ms').
    cell_type_para
        Column name in `adata.obs` that holds cell-type labels (default 'celltype').
    embed_para
        Key in `adata.obsm` for 2D embedding (default 'X_umap').
    save_path
        Destination CSV file path.
    gene_list
        Specific genes to export. If None, use all genes (`adata.var.index`).

    Returns
    -------
    pandas.DataFrame
        The final DataFrame that was saved to `save_path`.
    """
    # Local imports to avoid adding hard dependencies at module import time.
    import numpy as np
    import pandas as pd

    # tqdm is optional; fall back to a no-op iterator if unavailable.
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, **kwargs):  # type: ignore
            return x

    # Helper: extract a single gene's (unspliced, spliced) vectors as a DataFrame.
    def _adata_to_raw_one_gene(_adata, _us_para, _gene) -> pd.DataFrame:
        """Return a DataFrame with columns [gene_name, unsplice, splice] for one gene."""
        data2 = _adata[:, _adata.var.index.isin([_gene])].copy()
        # Expect shapes: (n_cells, 1)
        u0 = np.asarray(data2.layers[_us_para[0]][:, 0], dtype=np.float32)
        s0 = np.asarray(data2.layers[_us_para[1]][:, 0], dtype=np.float32)
        df_one = pd.DataFrame(
            {"gene_name": _gene, "unsplice": u0, "splice": s0},
            copy=False,
        )
        return df_one

    # Determine gene list.
    if gene_list is None:
        gene_list = list(adata.var.index)

    # Stream-write per-gene blocks to CSV (header for the first gene only).
    for i, gene in enumerate(tqdm(gene_list, desc="Export genes")):
        df_g = _adata_to_raw_one_gene(adata, us_para, gene)
        if i == 0:
            df_g.to_csv(save_path, header=True, index=False)
        else:
            df_g.to_csv(save_path, mode="a", header=False, index=False)

    # Build per-cell metadata (will be repeated for every gene).
    n_genes = len(gene_list)
    cellID = pd.DataFrame({"cellID": adata.obs.index})
    # Ensure the cell-type column exists; raise a clear error if missing.
    if cell_type_para not in adata.obs:
        raise KeyError(f"[gravity] column '{cell_type_para}' not found in adata.obs.")
    celltype_meta = adata.obs[cell_type_para].reset_index(drop=True)
    celltype = pd.DataFrame({"clusters": celltype_meta})

    # Validate the embedding.
    if embed_para not in adata.obsm:
        raise KeyError(f"[gravity] embedding '{embed_para}' not found in adata.obsm.")
    if adata.obsm[embed_para].shape[1] < 2:
        raise ValueError(f"[gravity] embedding '{embed_para}' must have at least 2 columns.")
    embed_map = pd.DataFrame(
        {
            "embedding1": adata.obsm[embed_para][:, 0],
            "embedding2": adata.obsm[embed_para][:, 1],
        }
    )

    # Repeat per-cell metadata for every gene.
    embed_info = pd.concat([cellID, celltype, embed_map], axis=1)
    embed_raw = pd.concat([embed_info] * n_genes, ignore_index=True)

    # Read the just-written raw gene table and append metadata.
    raw_data = pd.read_csv(save_path)
    if len(raw_data) != len(embed_raw):
        # Defensive check to catch mismatches early.
        raise RuntimeError(
            f"[gravity] row mismatch: gene table has {len(raw_data)} rows, "
            f"but repeated cell-metadata has {len(embed_raw)}."
        )
    raw_data = pd.concat([raw_data, embed_raw], axis=1)
    raw_data.to_csv(save_path, header=True, index=False)

    return raw_data


def preprocess_counts(input_file: str, output_csv: str) -> Path:
    """Prepare the cell-wise training table from a long single-cell CSV.

    Parameters
    ----------
    input_file:
        Path to the raw long-format CSV with columns including
        `cellID`, `gene_name`, `unsplice`, `splice`, `embedding1`, `embedding2`.
    output_csv:
        Destination CSV containing one row per cell with serialized gene tuples.

    Returns
    -------
    pathlib.Path
        The path to the generated intermediate CSV.
    """

    input_path = resolve_path(input_file)
    output_path = Path(output_csv).resolve()
    if output_path.exists():
        log_verbose(f"[gravity] found existing preprocessed file: {output_path}; skip.", level=1)
        return output_path
    log_verbose(f"[gravity] preprocessing raw counts from {input_path} → {output_path}", level=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PreprocessDataset(str(input_path), str(output_path))
    return output_path


def load_cell_stage_dataset(middle_file: str, *, prior_path: str = './prior_data/network_mouse.zip', gene_list: Optional[Sequence[str]] = None, n_pos_neighbors = 30, n_neg_neighbors = 10) -> CustomDataset:
    """Instantiate the PyTorch dataset used for the cell-wise stage."""

    dataset = CustomDataset(middle_file, prior=prior_path, gene_select=gene_list, n_pos_neighbors=n_pos_neighbors, n_neg_neighbors=n_neg_neighbors)
    log_verbose(
        f"[gravity] loaded cell-wise dataset with {len(dataset)} cells and {len(dataset.hvg)} HVGs",
        level=2,
    )
    return dataset


def load_gene_stage_dataset(middle_file: str, *, prior_path: str = './prior_data/network_mouse.zip', future_positions: str = './final_positions_with_index_yixian.npy', gene_list: Optional[Sequence[str]] = None) -> CustomDatasetGeneWise:
    """Instantiate the PyTorch dataset used for the gene-wise refinement stage."""

    dataset = CustomDatasetGeneWise(middle_file, prior=prior_path, gene_select=gene_list, future_pos=future_positions)
    log_verbose(
        f"[gravity] loaded gene-wise dataset with {len(dataset)} cells and {len(dataset.hvg)} HVGs",
        level=2,
    )
    return dataset
