"""Top-level orchestration helpers for the GRAVITY workflow.

This module exposes a high-level configuration dataclass and a single
entry-point function that coordinates the complete GRAVITY pipeline:

- preprocess long-format counts into a wide table used by subsequent stages,
- train the cell-wise stage (stage 1) and export attention matrices if enabled,
- estimate future positions from the stage-1 outputs,
- train the gene-wise stage (stage 2),
- optionally render velocity visualizations at the cell and gene level.

The design aims for a clean, user-friendly workflow while preserving
GRAVITY's regulation-aware kinetics. Multi-GPU/
distributed execution is handled by PyTorch Lightning inside each training
stage; this module orchestrates the outer sequencing.

References
----------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import pandas as pd
import re
import time

from .data import preprocess_counts, export_intermediate_from_h5ad
from .train import CellStageConfig, GeneStageConfig, train_cell_stage, train_gene_stage
from .tools.future import estimate_future_positions
from .plotting.velocity import plot_velocity_gene, plot_velocity_cell
from .utils import log_verbose, resolve_path

__all__ = [
    "PipelineConfig",
    "run_pipeline",
]


@dataclass
class PipelineConfig:
    """Configuration for the full GRAVITY pipeline.

    Parameters
    ----------
    raw_counts:
        Path to the input long-format CSV (must include at least
        ``cellID, gene_name, unsplice, splice, embedding1, embedding2``).
    workdir:
        Output directory where intermediate and final artifacts are written.
    prior_network:
        Path to the prior TF–target network archive used by GRAVITY.
    gene_subset:
        Optional list of genes to restrict training and evaluation.
    batch_size:
        Mini-batch size used by both training stages.
    stage1_epochs, stage2_epochs:
        Number of epochs per stage.
    stage1_lr, stage2_lr:
        Learning rates per stage.
    val_fraction_stage1, val_fraction_stage2:
        Fraction of data reserved for validation in each stage.
    accelerator, devices, strategy, precision, gradient_clip_val, num_workers:
        Forwarded to PyTorch Lightning trainers to control device placement,
        distribution strategy and numerical precision.
    future_tau:
        Scaling factor governing the radius used in future-neighbor search.
    log_every_n_steps, progress_bar:
        Logging and progress display controls.
    make_plot, plot_gene, plot_color, plot_genes, arrow_grid, arrow_scale:
        Plotting options for optional velocity visualization.
    middle_csv_name, stage*_csv_name, future_positions_name, stage*_checkpoint_name:
        Filenames for artifacts written under ``workdir``.
    """

    raw_counts: str
    workdir: str = 'gravity_outputs'
    prior_network: Optional[str] = './prior_data/network_mouse.zip'
    gene_subset: Optional[Sequence[str]] = None
    batch_size: int = 16
    n_pos_neighbors: int = 30
    n_neg_neighbors: int = 10
    stage1_epochs: int = 6
    stage2_epochs: int = 6
    stage1_lr: float = 1e-5
    stage2_lr: float = 1e-2
    embedding_size: int = 16
    model_dimension: int = 16
    ffn_dimension: int = 16
    val_fraction_stage1: float = 0.0
    val_fraction_stage2: float = 0.0
    accelerator: str = 'auto'
    devices: Optional[Union[int, Sequence[int]]] = None
    num_workers: int = 8
    strategy: Optional[str] = None
    precision: Optional[Union[int, str]] = None
    gradient_clip_val: Optional[float] = None
    future_tau: float = 0.5
    log_every_n_steps: int = 50
    progress_bar: bool = True
    make_plot: bool = False
    plot_gene: Optional[str] = None
    plot_color: Optional[str] = 'clusters'
    plot_genes: Optional[Union[str, Sequence[str]]] = None
    arrow_grid: Tuple[int, int] = (20, 20)
    arrow_scale: float = 1.0
    middle_csv_name: str = 'combine.csv'
    stage1_csv_name: str = 'stage1.csv'
    stage2_csv_name: str = 'stage2.csv'
    future_positions_name: str = 'future_positions.npy'
    stage1_checkpoint_name: str = 'stage1.ckpt'
    stage2_checkpoint_name: str = 'stage2.ckpt'


def run_pipeline(config: PipelineConfig) -> Dict[str, Path]:
    """Execute preprocessing, two training stages, future projection, and optional plotting.

    Parameters
    ----------
    config:
        :class:`PipelineConfig` instance that specifies inputs, training and
        plotting options, and output filenames.

    Returns
    -------
    Dict[str, pathlib.Path]
        Mapping of artifact names to absolute paths under ``workdir``. Keys
        include ``middle_csv``, ``stage1_csv``, ``stage1_checkpoint``,
        ``attention_dir``, ``future_positions``, ``stage2_csv``, and
        ``stage2_checkpoint``. If plotting is enabled, additional keys may be
        present for generated figures.

    Notes
    -----
    Multi-GPU controls in ``config`` are forwarded to PyTorch Lightning inside
    the stage trainers. Preprocessing, future projection and plotting always run
    in the main process. When using DDP with ``strategy='ddp'`` (spawn), child
    processes may re-import and execute the entry script; to avoid duplicated
    orchestration, this function performs an early return on non-zero ranks.
    """

    # Guard: if called from DDP worker, skip orchestration.
    import os
    try:
        global_rank = int(os.environ.get("PL_GLOBAL_RANK", os.environ.get("RANK", "0")))
    except Exception:
        global_rank = 0
    if global_rank != 0:
        log_verbose("[gravity] run_pipeline invoked on non-zero rank; skipping in worker.", level=1)
        workdir = Path(config.workdir).resolve()
        return {
            'middle_csv': workdir / config.middle_csv_name,
            'stage1_csv': workdir / config.stage1_csv_name,
            'stage1_checkpoint': workdir / config.stage1_checkpoint_name,
            'attention_dir': workdir / 'attentions',
            'future_positions': workdir / config.future_positions_name,
            'stage2_csv': workdir / config.stage2_csv_name,
            'stage2_checkpoint': workdir / config.stage2_checkpoint_name,
        }
    workdir = Path(config.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    middle_csv_path = workdir / config.middle_csv_name
    raw_counts_path = resolve_path(config.raw_counts)
    if config.prior_network is None:
        prior_network_path = None
    else:
        prior_network_path = resolve_path(config.prior_network)
    preprocess_counts(str(raw_counts_path), str(middle_csv_path))

    cell_cfg = CellStageConfig(
        raw_counts=str(raw_counts_path),
        middle_csv=str(middle_csv_path),
        prior_network=prior_network_path,
        output_dir=str(workdir),
        stage1_csv=config.stage1_csv_name,
        checkpoint_name=config.stage1_checkpoint_name,
        attention_dir='attentions',
        gene_subset=config.gene_subset,
        n_pos_neighbors = config.n_pos_neighbors,
        n_neg_neighbors = config.n_neg_neighbors,
        batch_size=config.batch_size,
        epochs=config.stage1_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy,
        num_workers=config.num_workers,
        val_fraction=config.val_fraction_stage1,
        attention_topk=64,
        attention_output=True,
        log_every_n_steps=config.log_every_n_steps,
        progress_bar=config.progress_bar,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        learning_rate=config.stage1_lr,
        embedding_size = config.embedding_size,
        model_dimension = config.model_dimension,
        ffn_dimension = config.ffn_dimension,
    )

    stage1_outputs = train_cell_stage(cell_cfg)

    # Ensure stage1 CSV is written before proceeding
    stage1_csv_path = Path(stage1_outputs['stage1_csv']).resolve()
    if not stage1_csv_path.exists():
        log_verbose(f"[gravity] waiting for stage1 CSV to appear: {stage1_csv_path}", level=1)
        waited = 0
        while waited < 900:  # wait up to 15 minutes for slow filesystems
            if stage1_csv_path.exists():
                break
            time.sleep(1.0)
            waited += 1
    if not stage1_csv_path.exists():
        raise FileNotFoundError(f"stage1 CSV not found after training: {stage1_csv_path}")

    plots_dir: Optional[Path] = None
    plot_results: Dict[str, object] = {}
    if config.make_plot:
        plots_dir = workdir / 'velocity_plots'
        plots_dir.mkdir(parents=True, exist_ok=True)

    future_positions_path = workdir / config.future_positions_name
    future_plot_path: Optional[Path] = None
    if plots_dir is not None:
        future_plot_path = plots_dir / 'future_projection_embedding.png'

    estimate_future_positions(
        str(stage1_csv_path),
        str(future_positions_path),
        tau=config.future_tau,
        show_plot=False,
        plot_path=str(future_plot_path) if future_plot_path else None,
    )

    if future_plot_path is not None and future_plot_path.exists():
        plot_results['future_projection_plot'] = future_plot_path

    gene_cfg = GeneStageConfig(
        raw_counts=str(raw_counts_path),
        middle_csv=str(middle_csv_path),
        stage1_checkpoint=str(stage1_outputs['checkpoint']),
        future_positions=str(future_positions_path),
        prior_network=prior_network_path,
        output_dir=str(workdir),
        stage2_csv=config.stage2_csv_name,
        checkpoint_name=config.stage2_checkpoint_name,
        gene_subset=config.gene_subset,
        batch_size=config.batch_size,
        epochs=config.stage2_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy,
        num_workers=config.num_workers,
        val_fraction=config.val_fraction_stage2,
        log_every_n_steps=config.log_every_n_steps,
        progress_bar=config.progress_bar,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        learning_rate=config.stage2_lr,
        embedding_size=config.embedding_size,
        model_dimension=config.model_dimension,
        ffn_dimension=config.ffn_dimension,
    )

    stage2_outputs = train_gene_stage(gene_cfg)

    if config.make_plot and plots_dir is not None:
        def _safe_plot(stage_csv: str, output_name: str) -> Path | None:
            plot_path = plots_dir / output_name
            try:
                plot_velocity_cell(
                    stage_csv,
                    gene=config.plot_gene,
                    color_by=config.plot_color,
                    arrow_grid=config.arrow_grid,
                    arrow_scale=config.arrow_scale,
                    output_path=str(plot_path),
                    show=False,
                )
                return plot_path
            except Exception as exc:  # pragma: no cover - plotting optional
                log_verbose(f"[gravity] cell-level plotting failed ({output_name}): {exc}", level=1)
                return None

        stage1_plot = _safe_plot(str(stage1_outputs['stage1_csv']), 'cell_velocity_stage1_embedding.png')
        if stage1_plot is not None:
            plot_results['cell_velocity_plot_stage1'] = stage1_plot

        stage2_plot = _safe_plot(str(stage2_outputs['stage2_csv']), 'cell_velocity_stage2_embedding.png')
        if stage2_plot is not None:
            plot_results['cell_velocity_plot_stage2'] = stage2_plot

        gene_plots: list[Path] = []
        genes_to_plot: list[str] = []
        if config.plot_genes:
            if isinstance(config.plot_genes, str):
                if config.plot_genes.lower() == 'all':
                    stage2_df = pd.read_csv(stage2_outputs['stage2_csv'])
                    genes_to_plot = stage2_df['gene_name'].dropna().astype(str).unique().tolist()
                else:
                    genes_to_plot = [config.plot_genes]
            else:
                genes_to_plot = list(config.plot_genes)

        for gene_name in genes_to_plot:
            gene_path = plots_dir / f"gene_{_sanitize_name(gene_name)}_expression.png"
            try:
                plot_velocity_gene(
                    str(stage2_outputs['stage2_csv']),
                    gene=gene_name,
                    color_by=config.plot_color,
                    arrow_grid=config.arrow_grid,
                    arrow_scale=config.arrow_scale,
                    output_path=str(gene_path),
                    show=False,
                )
                gene_plots.append(gene_path)
            except Exception as exc:
                log_verbose(f"[gravity] gene-level plotting failed for {gene_name}: {exc}", level=1)

        if gene_plots:
            plot_results['gene_velocity_plots'] = gene_plots

    outputs = {
        'middle_csv': middle_csv_path,
        'stage1_csv': stage1_outputs['stage1_csv'],
        'stage1_checkpoint': stage1_outputs['checkpoint'],
        'attention_dir': stage1_outputs['attention_dir'],
        'future_positions': future_positions_path,
        'stage2_csv': stage2_outputs['stage2_csv'],
        'stage2_checkpoint': stage2_outputs['checkpoint'],
    }
    if plot_results:
        outputs.update(plot_results)

    return outputs


def _sanitize_name(name: str) -> str:
    """Return a filesystem-friendly version of ``name``.

    Only ASCII letters, digits and the characters ``.``, ``_`` and ``-`` are
    retained; all other runs are replaced by underscores.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)
