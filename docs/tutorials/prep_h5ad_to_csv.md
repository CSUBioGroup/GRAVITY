# Convert AnnData (.h5ad) to GRAVITY CSV

GRAVITY expects a long-format CSV containing spliced/unspliced counts, embeddings, and optional cluster labels. Use `gravity.export_intermediate_from_h5ad` to produce this CSV once per dataset.

```python
from gravity import export_intermediate_from_h5ad

export_intermediate_from_h5ad(
    input_h5ad="data/postprocessed.h5ad",
    output_csv="data/pancreas_long.csv",
    retain_genes=["GCG", "INS1"],
    n_top_genes=1000,
    embed_key="X_umap",
    celltype_key="celltype",
    overwrite=True,
)
```

The helper mirrors the logic demonstrated in `gravity/smoke_test_hair.py` and performs:

1. Reading the AnnData file and checking that `spliced`/`unspliced` layers exist.
2. Running `scvelo` preprocessing and moments.
3. Exporting a CSV with embedded coordinates plus per-cell TF annotations.

Keep CSV outputs versioned under `data/` so subsequent pipeline runs can reuse them without recomputing AnnData steps.
