# Inspect Outputs and Visualizations

GRAVITY writes several artefacts under your `workdir`. Use built-in utilities to inspect them.

Keep `genes.txt` with any checkpoint or attention export you intend to reuse.
The file records the gene-index order used by the model. When re-running a
pretrained/reference checkpoint, pass that file as `gene_order_path`; otherwise
the same genes in a different CSV order can produce misaligned attention
summaries.

## Attention matrices and TF ranks

```python
from gravity.analysis.importance import rank_tf_scores

tf_summary, tf_plots = rank_tf_scores(
    attention_h5ad="gravity_outputs_pancreas/attentions/attention_TF_scores_with_types.h5ad",
    groupby="cell_type",
    method="wilcoxon",
    n_genes=30,
)
print(tf_summary.head())
```

## Velocity plots

```python
from gravity.plotting.velocity import plot_velocity_cell, plot_velocity_gene

plot_velocity_cell(
    stage2_csv="gravity_outputs_pancreas/stage2.csv",
    output_path="gravity_outputs_pancreas/velocity_cell.png",
)
plot_velocity_gene(
    stage2_csv="gravity_outputs_pancreas/stage2.csv",
    gene="INS2",
    output_path="gravity_outputs_pancreas/velocity_gene_ins2.png",
)
```

The plotting functions read the arrays produced by the pipeline, so no
re-training is required. See `gravity/smoke_test.py` for a runnable example.
