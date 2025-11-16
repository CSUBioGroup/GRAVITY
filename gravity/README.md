GRAVITY: Dynamic gene regulatory network-enhanced RNA velocity modeling for trajectory inference and biological discovery
===================================================================================

This subpackage provides a refactored, modular implementation of the GRAVITY
workflow, inspired by the usability of scVelo while retaining GRAVITY’s
regulation‑aware kinetics. The library exposes high‑level pipeline helpers and
modular building blocks for preprocessing, two‑stage training, future position
estimation, visualization, and TF importance analysis.

This subpackage focuses on a streamlined, research‑oriented interface with
clear separation between preprocessing, training, future projection, and
visualization components.

Key features
------------
- End‑to‑end pipeline with a single configuration object.
- Two‑stage training (cell‑wise, then gene‑wise) with multi‑GPU support.
- Regulation‑informed future position estimation and plotting utilities.
- Exportable TF attention scores for downstream analysis.

Installation
------------
It is recommended to use Python 3.9+ and a fresh virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you need to control dependency versions manually, refer to `pyproject.toml`.
For GPU, install an appropriate `torch` wheel first, then run `pip install -e .`.

Quickstart (end‑to‑end)
-----------------------
```python
from gravity import PipelineConfig, run_pipeline

cfg = PipelineConfig(
    raw_counts="data/pancreas_long.csv",
    workdir="gravity_outputs",
    prior_network="prior_data/network_mouse.zip",
    accelerator="gpu",
    devices=[0, 1],
    strategy="ddp",
    make_plot=True,
    plot_genes=["GCG", "INS1"],
)
outputs = run_pipeline(cfg)
print(outputs)
```

Preparing CSV from AnnData (.h5ad)
----------------------------------
If your dataset starts as an AnnData object, convert it once to the long-format
CSV that GRAVITY consumes:

```python
from gravity import export_intermediate_from_h5ad

export_intermediate_from_h5ad(
    input_h5ad="data/postprocessed.h5ad",
    output_csv="data/hair.csv",
    n_top_genes=1000,
    embed_key="X_umap",
    celltype_key="celltype",
)
```
This helper mirrors the workflow in `gravity/smoke_test_hair.py` and persists
embeddings/clusters alongside spliced/unspliced counts.

Upon completion, `workdir` contains (names configurable via `PipelineConfig`):

- `combine.csv` — preprocessed wide table
- `stage1.csv`, `stage1.ckpt` — cell‑wise stage outputs
- `future_positions.npy` — predicted future positions
- `stage2.csv`, `stage2.ckpt` — gene‑wise stage outputs
- `attentions/` — TF score matrices and cell‑type mean attention networks
- `velocity_plots/*.png` — cell‑ and gene‑level velocity plots (if enabled)

Modular usage
-------------
```python
from gravity import (
    preprocess_counts,
    CellStageConfig, train_cell_stage,
    GeneStageConfig, train_gene_stage,
)
from gravity.tools.future import estimate_future_positions
from gravity.plotting.velocity import plot_velocity_cell, plot_velocity_gene

# 1) Preprocess
middle_csv = preprocess_counts("data/pancreas_long.csv", "gravity_outputs/combine.csv")

# 2) Cell‑wise training (multi‑GPU optional)
cell_cfg = CellStageConfig(
    raw_counts="data/pancreas_long.csv",
    middle_csv=str(middle_csv),
    prior_network="prior_data/network_mouse.zip",
    output_dir="gravity_outputs",
    accelerator="gpu",
    devices=[0, 1],
    strategy="ddp",
)
stage1 = train_cell_stage(cell_cfg)

# 3) Future position estimation
estimate_future_positions(stage1["stage1_csv"], "gravity_outputs/future_positions.npy")

# 4) Gene‑wise fine‑tuning
gene_cfg = GeneStageConfig(
    raw_counts="data/pancreas_long.csv",
    middle_csv=str(middle_csv),
    stage1_checkpoint=str(stage1["checkpoint"]),
    future_positions="gravity_outputs/future_positions.npy",
    prior_network="prior_data/network_mouse.zip",
    output_dir="gravity_outputs",
    accelerator="gpu",
    devices=[0, 1],
    strategy="ddp",
)
stage2 = train_gene_stage(gene_cfg)

# 5) Visualization (cell‑ and gene‑level)
plot_velocity_cell(str(stage2["stage2_csv"]), output_path="gravity_outputs/cell_velocity.png")
plot_velocity_gene(str(stage2["stage2_csv"]), gene="GCG", output_path="gravity_outputs/gcg_velocity_expression.png")
```

Configuration highlights
-----------------------
- `PipelineConfig`
  - `gene_subset`: restrict the gene set used for training
  - `stage1_epochs` / `stage2_epochs`: number of epochs per stage
  - `val_fraction_stage1` / `val_fraction_stage2`: optional hold-out ratio (default `0.0`, meaning no validation split)
  - `future_tau`: scaling factor controlling the radius for future-neighbor search
  - `accelerator` / `devices` / `strategy`: forwarded to PyTorch Lightning (e.g., `accelerator="gpu"`, `devices=[0,1]`, `strategy="ddp"`)
  - `make_plot`, `plot_genes`: enable plotting and choose genes; `'all'` plots every gene
- `CellStageConfig`
  - `attention_output`: whether to export TF attention matrices
  - `attention_topk`: number of TFs kept per cell
- `GeneStageConfig`
  - `future_positions`: path to the `.npy` produced by future projection
  - `stage1_checkpoint`: cell‑wise checkpoint

Inputs and formats
------------------
The long‑format CSV must include at least: `cellID`, `gene_name`, `unsplice`,
`splice`, `embedding1`, `embedding2`. The optional column `clusters` is used for
coloring in plots and summary tables. Prior network archive
`prior_data/network_mouse.zip` should match the original GRAVITY prior format.

Troubleshooting
---------------
- Out‑of‑memory (OOM): reduce `batch_size` or provide a smaller `gene_subset`.
- No GPU available: Lightning falls back to CPU; training will be slower.
- Optional deps missing (e.g., SciPy): plotting/sampling may disable gracefully.
- Verbosity: use `from gravity.utils import log_verbose` or project‑level toggles.

Citing
------
If this package contributes to your research, please cite the GRAVITY paper,
“GRAVITY: Dynamic gene regulatory network-enhanced RNA velocity modeling for trajectory inference and biological discovery.” Include version, environment details, and key configuration options in your methods section when possible.

Contributing & license
----------------------
Please open issues/PRs with reproduction steps and sample commands. This
subpackage is MIT‑licensed as declared in the project’s metadata.

Chinese README
--------------
An up‑to‑date Chinese version is available as `README_zh.md` in this folder.
