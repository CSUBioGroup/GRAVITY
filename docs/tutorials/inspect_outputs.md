# Inspect Outputs and Visualizations

GRAVITY writes several artefacts under your `workdir`. Use built-in utilities to inspect them.

## Attention matrices and TF ranks

```python
from gravity.analysis.importance import rank_tf_scores

tf_summary, tf_plots = rank_tf_scores(
    attention_h5ad="gravity_outputs/attentions/attention_TF_scores_with_types.h5ad",
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
    stage2_csv="gravity_outputs/stage2.csv",
    output_path="gravity_outputs/velocity_cell.png",
)
plot_velocity_gene(
    stage2_csv="gravity_outputs/stage2.csv",
    gene="GCG",
    output_path="gravity_outputs/velocity_gene_gcg.png",
)
```

These helpers re-use the arrays produced by the pipeline, so no re-training is required. See `gravity/smoke_test.py` for a full scripted example.
