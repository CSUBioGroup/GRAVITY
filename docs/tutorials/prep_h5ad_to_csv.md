# Convert AnnData (.h5ad) to GRAVITY CSV

GRAVITY expects a long-format CSV containing spliced/unspliced counts, embeddings, and optional cluster labels. Use `gravity.export_intermediate_from_h5ad` to produce this CSV once per dataset.

```python
from gravity import export_intermediate_from_h5ad

export_intermediate_from_h5ad(
    input_h5ad="data/postprocessed.h5ad",
    output_csv="data/your_counts.csv",
    retain_genes=["GCG", "INS1"],
    n_top_genes=1000,
    embed_key="X_umap",
    celltype_key="celltype",
    overwrite=True,
)
```

The helper performs:

1. Reading the AnnData file and checking that `spliced`/`unspliced` layers exist.
2. Running `scvelo` preprocessing and moments.
3. Exporting a CSV with embedded coordinates plus optional cluster labels.

Keep large generated CSV files outside git and document their expected paths so subsequent pipeline runs can reuse them without recomputing AnnData steps.

## Gene Order

By default, GRAVITY preserves the gene order found in the exported CSV. This is
fine for training a new model, but pretrained checkpoints require the same gene
index order used during their original run. If you plan to reuse a checkpoint,
keep the checkpoint's `genes.txt` file and pass it later as `gene_order_path`
when running the pipeline.

For the published pancreas reference checkpoints, use:

```python
gene_order_path = "data/pancreas/reference_checkpoints/pancreas_genes.txt"
```

The gene set alone is not sufficient for checkpoint reproduction; the order
also matters because model weights and attention matrices are indexed by gene
position.
