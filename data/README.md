# Data Directory

Large raw count tables are not stored directly in this repository. GRAVITY
expects a cellDancer-style long-format CSV with one row per cell-gene pair.
The pancreatic endocrinogenesis CSV is the real demo dataset for smoke tests
and tutorials. It is the preprocessed input file linked from cellDancer's
pancreatic endocrinogenesis case study:

```text
https://guangyuwanglab2021.github.io/cellDancer_website/notebooks/case_study_pancreas.html
```

Download the CSV archive from:

```text
https://drive.google.com/file/d/16hV9t66edOgjCmoBuEfekS3ijtL1fYNc/view?usp=sharing
```

Then place the file here as:

```text
data/PancreaticEndocrinogenesis_cell_type_u_s.csv
```

You can also point the smoke test to another compatible file:

```bash
GRAVITY_RAW_COUNTS=/data/shared/compatible_counts.csv python gravity/smoke_test.py
```

The CSV must include at least `cellID`, `gene_name`, `unsplice`, `splice`,
`embedding1`, and `embedding2`; `clusters` is optional and is used for plot
coloring and attention summaries.

## Prior Networks

The repository includes mouse and human prior network archives:

```text
prior_data/nichenet_mouse.zip
prior_data/nichenet_human.zip
```

The pancreas demo uses the mouse archive by default. For human datasets, pass
`prior_network="prior_data/nichenet_human.zip"` in `PipelineConfig` or set
`GRAVITY_PRIOR_NET=prior_data/nichenet_human.zip` when running
`gravity/smoke_test.py`.

The bundled archives follow the prior-network processing described by CEFCON.
They start from NicheNet's integrated gene interaction network, remove
cell-cell ligand-receptor interactions, use the unweighted integrated network,
and represent undirected edges as bidirectional directed edges. The human
archive keeps human gene symbols, and the mouse archive uses one-to-one ENSEMBL
ortholog mapping with ambiguous genes removed. GRAVITY stores the processed
networks as zipped edge-list CSV files with `from`, `to`, and `edge_type`
columns.

Background links:

- NicheNet: https://www.nature.com/articles/s41592-019-0667-5
- CEFCON: https://www.nature.com/articles/s41467-023-44103-3
- cellDancer input layout and pancreas demo: https://www.nature.com/articles/s41587-023-01728-5

## Pancreas Reference Files

This repository includes small pretrained checkpoints for the pancreatic
endocrinogenesis reference run:

```text
data/pancreas/reference_checkpoints/pancreas_stage1.ckpt
data/pancreas/reference_checkpoints/pancreas_stage2.ckpt
data/pancreas/reference_checkpoints/pancreas_genes.txt
```

These files can be used directly as the pancreas stage-1 and stage-2 weights.
They were converted to the current GRAVITY module names and can be loaded by
the current package directly. Use `stage1_pretrained_checkpoint` and
`stage2_pretrained_checkpoint` in `PipelineConfig`, or `pretrained_checkpoint`
in the modular stage configs, to run inference/export from these weights instead
of training new weights.

`pancreas_genes.txt` is part of the checkpoint contract. GRAVITY models and
attention matrices are aligned by gene index, so the same gene set in a
different order is not equivalent for pretrained checkpoint reproduction. When
using the pancreas reference checkpoints, pass this file as `gene_order_path`.

The corresponding reference exports are named:

```text
pancreas_stage1_reference.csv
pancreas_stage2_reference.csv
pancreas_attention_scores.h5ad
pancreas_insulin_signaling_attention_activity.csv
pancreas_mean_attention_by_celltype/Beta_mean_attention.npz
```

Those exports are large and are intentionally not tracked in git. Treat them as
pancreas reference results produced from the provided checkpoints.
The `pancreas_mean_attention_by_celltype/` matrices are cell-type averaged
stage-1 attention networks and are used by the notebook's TF target-gene
examples.
`pancreas_insulin_signaling_attention_activity.csv` is a precomputed per-cell
summary from raw stage-1 attention tensors. It is computed by summing attention
weights between insulin signaling genes, restricted to regulator-target pairs
present in the prior network. The pancreas reference notebook documents the
formula for users who save per-cell attention matrices and want to recompute the
table.
