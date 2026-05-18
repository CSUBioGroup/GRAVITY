"""Smoke-test GRAVITY pipeline with plotting options.

This script runs two consecutive pipeline executions to exercise:
1. Full training + expression-space gene plots.
2. Reuse existing outputs + embedding-space cell plot (no gene plots).
"""

import os
from pathlib import Path
from pprint import pprint

from gravity import PipelineConfig, run_pipeline


RAW_COUNTS = os.environ.get("GRAVITY_RAW_COUNTS", "data/PancreaticEndocrinogenesis_cell_type_u_s.csv")
PRIOR_NET = os.environ.get("GRAVITY_PRIOR_NET", "prior_data/nichenet_mouse.zip")
WORKDIR = os.environ.get("GRAVITY_WORKDIR", "gravity_outputs_pancreas")
GENE_LIST = [
    gene.strip()
    for gene in os.environ.get("GRAVITY_PLOT_GENES", "GCG,INS2").split(",")
    if gene.strip()
]
BATCH_SIZE = int(os.environ.get("GRAVITY_BATCH_SIZE", "16"))
DEVICES = int(os.environ.get("GRAVITY_DEVICES", "1"))
STRATEGY = os.environ.get("GRAVITY_STRATEGY", "") or None
STAGE1_EPOCHS = int(os.environ.get("GRAVITY_STAGE1_EPOCHS", "6"))
STAGE2_EPOCHS = int(os.environ.get("GRAVITY_STAGE2_EPOCHS", "4"))


def require_file(path: str, env_name: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path!r} does not exist. Place the required file there or set "
            f"{env_name} to a compatible path."
        )


def run_and_report(cfg: PipelineConfig, label: str) -> None:
    print(f"\n===== {label} =====")
    outputs = run_pipeline(cfg)
    pprint(outputs)


require_file(RAW_COUNTS, "GRAVITY_RAW_COUNTS")
require_file(PRIOR_NET, "GRAVITY_PRIOR_NET")

# First run: enable plotting in expression space for selected genes
cfg_pipe = PipelineConfig(
    raw_counts=RAW_COUNTS,
    workdir=WORKDIR,
    prior_network=PRIOR_NET,
    accelerator="gpu",
    devices=DEVICES,
    make_plot=True,
    plot_genes=GENE_LIST,
    batch_size=BATCH_SIZE,
    strategy=STRATEGY,
    stage1_epochs=STAGE1_EPOCHS,
    stage2_epochs=STAGE2_EPOCHS,
    stage1_lr=1e-5,
    stage2_lr=1e-4,
)
run_and_report(cfg_pipe, f"Pipeline run (embedding plot & gene plots for {GENE_LIST})")
