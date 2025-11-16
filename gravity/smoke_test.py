"""Smoke-test GRAVITY pipeline with plotting options.

This script runs two consecutive pipeline executions to exercise:
1. Full training + expression-space gene plots.
2. Reuse existing outputs + embedding-space cell plot (no gene plots).
"""

from pprint import pprint

from gravity import PipelineConfig, run_pipeline


RAW_COUNTS = "/media/asus/data16t/miaozy/PancreaticEndocrinogenesis_cell_type_u_s.csv"
PRIOR_NET = "/home/sda1/miaozy/cellDancer-main/prior_data/network_mouse.zip"
WORKDIR = "gravity_outputs_new"
GENE_LIST = ["RFX6", "RBFOX3"]

def run_and_report(cfg: PipelineConfig, label: str) -> None:
    print(f"\n===== {label} =====")
    outputs = run_pipeline(cfg)
    pprint(outputs)


# First run: enable plotting in expression space for selected genes
cfg_pipe = PipelineConfig(
    raw_counts=RAW_COUNTS,
    workdir=WORKDIR,
    prior_network=PRIOR_NET,
    accelerator="gpu",
    devices=2,
    make_plot=True,
    plot_genes=GENE_LIST,
    batch_size=32,
    strategy='ddp',
    stage1_epochs=6,
    stage2_epochs=6,
    stage1_lr=4e-5,
    stage2_lr=1e-4
)
run_and_report(cfg_pipe, f"Pipeline run (embedding plot & gene plots for {GENE_LIST})")
