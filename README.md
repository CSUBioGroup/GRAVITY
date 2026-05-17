GRAVITY predicts RNA velocity and regulatory rewiring by dynamic regulatory mechanism-enhanced deep learning
==========================================================================================================

GRAVITY is a deep learning framework that connects RNA velocity inference with
dynamic gene regulatory network modeling. It jointly models cell-state
transitions, gene-specific transcriptional kinetics, and regulatory rewiring by
combining unspliced/spliced abundances, cell embeddings, and prior gene
regulatory networks in a regulatory network-aware attention architecture.

This repository provides a research-oriented Python implementation of the
GRAVITY workflow. The pipeline first optimizes cell-level velocity and future
cell positions, then refines gene-level kinetic parameters and exports
attention-based regulatory summaries for downstream analysis.

Key features
------------
- End‑to‑end pipeline with a single configuration object.
- Dynamic regulatory network-aware velocity inference from spliced and
  unspliced counts.
- Two-stage optimization: cell-wise trajectory recovery followed by gene-wise
  kinetic refinement.
- Prior GRN-informed attention exports for regulator and module analysis.
- Velocity plotting utilities for cell-level trajectories and selected genes.

Installation
------------
It is recommended to use Python 3.10 or 3.11 and a fresh virtual environment.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you need to control dependency versions manually, refer to `pyproject.toml`.
For GPU, install an appropriate `torch` wheel first, then run `pip install -e .`.
For example, CUDA 11.7 systems can use:

```bash
pip install --index-url https://download.pytorch.org/whl/cu117 "torch==2.0.1+cu117"
pip install -e .
```

Quickstart (end‑to‑end)
-----------------------
```python
from gravity import PipelineConfig, run_pipeline

cfg = PipelineConfig(
    raw_counts="path/to/your_counts.csv",
    workdir="gravity_outputs",
    prior_network="prior_data/network_mouse.zip",
    accelerator="gpu",
    devices=1,
    batch_size=16,
    stage1_lr=1e-6,
    stage2_lr=1e-4,
    make_plot=True,
    plot_genes=["Rfx6", "Rbfox3"],
)
outputs = run_pipeline(cfg)
print(outputs)
```

The unsupervised and contrastive objectives are moderately learning-rate
sensitive. For reference-style runs, start with `stage1_lr < 1e-5` and tune
`stage2_lr` within `1e-3` to `1e-5`.

Preparing CSV from AnnData (.h5ad)
----------------------------------
If your dataset starts as an AnnData object, convert it once to the long-format
CSV that GRAVITY consumes:

```python
from gravity import export_intermediate_from_h5ad

export_intermediate_from_h5ad(
    input_h5ad="data/postprocessed.h5ad",
    output_csv="data/your_counts.csv",
    n_top_genes=1000,
    embed_key="X_umap",
    celltype_key="celltype",
)
```
This helper validates the required spliced/unspliced layers and persists
embeddings and cluster labels alongside the long-format count table.

Upon completion, `workdir` contains (names configurable via `PipelineConfig`):

- `combine.csv` — preprocessed wide table
- `stage1.csv`, `stage1.ckpt` — cell‑wise stage outputs
- `future_positions.npy` — predicted future positions
- `stage2.csv`, `stage2.ckpt` — gene‑wise stage outputs
- `attentions/` — TF score matrices and cell‑type mean attention networks
- `velocity_plots/*.png` — cell‑ and gene‑level velocity plots (if enabled)

Pancreatic endocrinogenesis reference checkpoints are provided under
`data/pancreas/reference_checkpoints/`. The matching reference exports are
named `pancreas_stage1_reference.csv` and `pancreas_stage2_reference.csv`; they
are large pancreas reference results and are not tracked in git.
When reproducing the published pancreas checkpoints, also pass
`gene_order_path="data/pancreas/reference_checkpoints/pancreas_genes.txt"` so
the model and attention tensors use the checkpoint-matching gene index order.

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
middle_csv = preprocess_counts("path/to/your_counts.csv", "gravity_outputs/combine.csv")

# 2) Cell‑wise training (multi‑GPU optional)
cell_cfg = CellStageConfig(
    raw_counts="path/to/your_counts.csv",
    middle_csv=str(middle_csv),
    prior_network="prior_data/network_mouse.zip",
    output_dir="gravity_outputs",
    accelerator="gpu",
    devices=1,
    batch_size=16,
)
stage1 = train_cell_stage(cell_cfg)

# 3) Future position estimation
estimate_future_positions(stage1["stage1_csv"], "gravity_outputs/future_positions.npy")

# 4) Gene‑wise fine‑tuning
gene_cfg = GeneStageConfig(
    raw_counts="path/to/your_counts.csv",
    middle_csv=str(middle_csv),
    stage1_checkpoint=str(stage1["checkpoint"]),
    future_positions="gravity_outputs/future_positions.npy",
    prior_network="prior_data/network_mouse.zip",
    output_dir="gravity_outputs",
    accelerator="gpu",
    devices=1,
    batch_size=16,
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
  - `gene_order_path`: load a newline-delimited gene order file; use this for pretrained/reference checkpoints because tensors are gene-index aligned
  - `stage1_epochs` / `stage2_epochs`: number of epochs per stage
  - `val_fraction_stage1` / `val_fraction_stage2`: optional hold-out ratio (default `0.0`, meaning no validation split)
  - `future_tau`: scaling factor controlling the radius for future-neighbor search
  - `accelerator` / `devices` / `strategy`: forwarded to PyTorch Lightning (e.g., `accelerator="gpu"`, `devices=1`; use `devices=[0,1]`, `strategy="ddp"` for multi-GPU runs)
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
Large raw count tables are kept out of the repository; see `data/README.md` for
the expected path used by the pancreatic endocrinogenesis smoke test.

Troubleshooting
---------------
- Out‑of‑memory (OOM): reduce `batch_size` or provide a smaller `gene_subset`.
- No GPU available: Lightning falls back to CPU; training will be slower.
- Optional deps missing (e.g., SciPy): plotting/sampling may disable gracefully.
- Verbosity: use `from gravity.utils import log_verbose` or project‑level toggles.

Citing
------
If this package contributes to your research, please cite the GRAVITY paper:
“GRAVITY predicts RNA velocity and regulatory rewiring by dynamic regulatory
mechanism-enhanced deep learning.” Include version, environment details, and key
configuration options in your methods section when possible.

Contributing & license
----------------------
Please open issues/PRs with reproduction steps and sample commands. This
package is MIT‑licensed as declared in the project’s metadata.

Chinese README
--------------
An up‑to‑date Chinese version is available as `README_zh.md` in this folder.
