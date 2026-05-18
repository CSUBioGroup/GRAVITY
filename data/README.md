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
the current package directly.

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
them as pancreas reference results produced from the provided checkpoints.
