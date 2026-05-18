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

## Pathway activity from attention tensors

The pancreas reference notebook reads
`data/pancreas/reference_outputs/pancreas_insulin_signaling_attention_activity.csv`
as a precomputed table for plotting. That table is not a separate model output;
it is a downstream per-cell summary computed from raw stage-1 attention
matrices. For each cell, the insulin signaling activity score is the sum of
attention weights between insulin signaling genes, restricted to
regulator-target pairs present in the prior network. If you save raw per-cell
attention matrices during inference, use the same checkpoint gene order before
indexing genes and writing the per-cell activity table.

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
