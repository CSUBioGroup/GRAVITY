"""User-friendly entry points for the GRAVITY library."""

from .pipeline import PipelineConfig, run_pipeline
from .data import preprocess_counts, export_intermediate_from_h5ad
from .train import (
    CellStageConfig,
    GeneStageConfig,
    FullModelCellWise,
    FullModelGeneWise,
    train_cell_stage,
    train_gene_stage,
)
from .tools.future import estimate_future_positions
from .plotting.velocity import plot_velocity_gene, plot_velocity_cell
from .analysis.importance import rank_tf_scores
from .analysis.batc import compute_batc
from .utils import set_verbose, get_verbose, log_verbose

__all__ = [
    "PipelineConfig",
    "run_pipeline",
    "preprocess_counts",
    "export_intermediate_from_h5ad",
    "CellStageConfig",
    "train_cell_stage",
    "GeneStageConfig",
    "train_gene_stage",
    "FullModelCellWise",
    "FullModelGeneWise",
    "estimate_future_positions",
    "plot_velocity_gene",
    "plot_velocity_cell",
    "rank_tf_scores",
    "compute_batc",
    "set_verbose",
    "get_verbose",
    "log_verbose",
]
