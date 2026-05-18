# Run the GRAVITY Pipeline

The pancreatic endocrinogenesis CSV is the real demo dataset used by the smoke
test and tutorials. It is the preprocessed input file linked from cellDancer's
pancreatic endocrinogenesis case study:

```text
https://guangyuwanglab2021.github.io/cellDancer_website/notebooks/case_study_pancreas.html
```

Download the CSV archive from:

```text
https://drive.google.com/file/d/16hV9t66edOgjCmoBuEfekS3ijtL1fYNc/view?usp=sharing
```

Save it as `data/PancreaticEndocrinogenesis_cell_type_u_s.csv`, then run the
two-stage pipeline via `PipelineConfig` and `run_pipeline`. The example below
uses the same pancreas reference layout as the smoke test.

```python
from gravity import PipelineConfig, run_pipeline

cfg = PipelineConfig(
    raw_counts="data/PancreaticEndocrinogenesis_cell_type_u_s.csv",
    workdir="gravity_outputs_pancreas",
    prior_network="prior_data/network_mouse.zip",
    gene_order_path="data/pancreas/reference_checkpoints/pancreas_genes.txt",
    accelerator="gpu",
    devices=1,
    batch_size=16,
    stage1_lr=1e-6,
    stage2_lr=1e-4,
    make_plot=True,
    plot_genes=["GCG", "INS2"],
    stage1_epochs=6,
    stage2_epochs=4,
)
outputs = run_pipeline(cfg)
print(outputs)
```

The expected output is a dictionary containing paths to `combine.csv`, stage
checkpoints, `future_positions.npy`, stage CSV files, attention exports, and
velocity plots for the selected genes. The scripted smoke test uses the same
layout:

```bash
python gravity/smoke_test.py
```

Expected runtime depends on the GPU and epoch settings; the default pancreas
demo is intended for a CUDA workstation and typically runs on the order of tens
of minutes. For a shorter check, reduce `GRAVITY_STAGE1_EPOCHS` and
`GRAVITY_STAGE2_EPOCHS`.

Key tips:

- Use unique `workdir` names per experiment to avoid overwriting checkpoints.
- Set `devices` and `strategy` to match your hardware. Start with `devices=1`; use `devices=[0,1]`, `strategy="ddp"` only for multi-GPU runs.
- Reduce `batch_size` or provide `gene_subset` when GPU memory is limited.
- Pass `gene_order_path` when using pretrained/reference checkpoints; checkpoint tensors are aligned by gene index, not only by gene name.
- The unsupervised and contrastive objectives are learning-rate sensitive. For reference-style runs, use `stage1_lr < 1e-5` and tune `stage2_lr` within `1e-3` to `1e-5`.

The resulting dictionary contains paths to `combine.csv`, stage checkpoints, `future_positions.npy`, and attention exports.

For pancreatic endocrinogenesis examples, pretrained reference weights are
available in `data/pancreas/reference_checkpoints/`. The matching large
reference exports are named `pancreas_stage1_reference.csv` and
`pancreas_stage2_reference.csv`; keep them outside git or distribute them
separately. Use
`gene_order_path="data/pancreas/reference_checkpoints/pancreas_genes.txt"` for
the provided pancreas checkpoint reproduction.
