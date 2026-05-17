# Data Directory

Large raw count tables are not stored directly in this repository. To run the
pancreatic endocrinogenesis smoke test, place the long-format CSV here as:

```text
data/PancreaticEndocrinogenesis_cell_type_u_s.csv
```

You can also point the smoke test to another compatible file:

```bash
GRAVITY_RAW_COUNTS=/path/to/your_counts.csv python gravity/smoke_test.py
```

The CSV must include at least `cellID`, `gene_name`, `unsplice`, `splice`,
`embedding1`, and `embedding2`; `clusters` is optional and is used for plot
coloring and attention summaries.

## Pancreas Reference Files

This repository includes small pretrained checkpoints for the pancreatic
endocrinogenesis reference run:

```text
data/pancreas/reference_checkpoints/pancreas_stage1.ckpt
data/pancreas/reference_checkpoints/pancreas_stage2.ckpt
data/pancreas/reference_checkpoints/pancreas_genes.txt
```

These files can be used directly as the pancreas stage-1 and stage-2 weights.
They were converted from the manuscript experiment checkpoints to the current
GRAVITY module names, so no legacy `NetVelo` key remapping is needed at use
time.

`pancreas_genes.txt` is part of the checkpoint contract. GRAVITY models and
attention matrices are aligned by gene index, so the same gene set in a
different order is not equivalent for pretrained checkpoint reproduction. When
using the pancreas reference checkpoints, pass this file as `gene_order_path`.

The corresponding reference exports are named:

```text
pancreas_stage1_reference.csv
pancreas_stage2_reference.csv
```

Those CSV exports are large and are intentionally not tracked in git. Treat
them as pancreas reference results produced from the published checkpoints.
