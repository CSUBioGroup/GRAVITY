"""High-level helpers to train the GRAVITY cell-wise stage.

The functions in this module prepare datasets, configure PyTorch Lightning
trainers, and export artefacts (CSV predictions, TF scores, priors) expected by
downstream steps. They do not change the model's inner computations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

try:  # Lightning 2.x
    from pytorch_lightning.callbacks import TQDMProgressBar
except ImportError:  # pragma: no cover - Lightning 1.x fallback path
    TQDMProgressBar = None

from ..data.preprocessing import load_cell_stage_dataset
from .cell_model import FullModelCellWise
from ..utils import log_verbose, resolve_path
import time

__all__ = [
    "CellStageConfig",
    "train_cell_stage",
]


class ExportProgressBar(TQDMProgressBar if TQDMProgressBar is not None else object):
    """Progress bar with custom description for export/inference passes."""

    def __init__(self, description: str, refresh_rate: int = 1):
        if TQDMProgressBar is None:  # pragma: no cover - fallback when class unavailable
            self._description = description
            return
        super().__init__(refresh_rate=refresh_rate)
        self._description = description

    def init_test_tqdm(self):  # pragma: no cover - Lightning handles display
        if TQDMProgressBar is None:
            return None
        bar = super().init_test_tqdm()
        if bar is not None:
            bar.set_description(self._description)
        return bar


@dataclass
class CellStageConfig:
    """Configuration bundle for the cell-wise training stage.

    See also the top-level :class:`gravity.pipeline.PipelineConfig` for how
    device/distribution options propagate.
    """

    raw_counts: str
    middle_csv: str
    prior_network: Optional[str] = './prior_data/network_mouse.zip'
    output_dir: str = 'gravity_outputs'
    stage1_csv: str = 'stage1.csv'
    checkpoint_name: str = 'stage1.ckpt'
    attention_dir: str = 'attentions'
    gene_subset: Optional[Sequence[str]] = None
    batch_size: int = 32
    epochs: int = 6
    accelerator: str = 'auto'
    devices: Optional[Union[int, Sequence[int]]] = None
    num_workers: int = 0
    val_fraction: float = 0.0
    attention_topk: int = 64
    attention_output: bool = True
    precision: Optional[Union[int, str]] = None
    gradient_clip_val: Optional[float] = None
    strategy: Optional[str] = None
    seed: int = 42
    log_every_n_steps: int = 50
    progress_bar: bool = True
    learning_rate: float = 1e-5
    embedding_size: int = 16
    model_dimension: int = 16
    ffn_dimension: int = 16
    n_pos_neighbors: int = 30
    n_neg_neighbors:int = 10


def _split_for_validation(dataset, val_fraction: float, seed: int):
    """Split a dataset deterministically into train/val subsets if requested."""
    if not 0.0 < val_fraction < 1.0:
        return dataset, None
    total = len(dataset)
    if total <= 1:
        return dataset, None
    val_size = max(1, int(total * val_fraction))
    train_size = total - val_size
    if train_size <= 0:
        train_size = total - 1
        val_size = 1
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def train_cell_stage(config: CellStageConfig) -> Dict[str, Path]:
    """Run the GRAVITY cell-wise stage and export key artefacts.

    Returns
    -------
    Dict[str, Path]
        Paths to ``stage1_csv``, ``checkpoint``, attention outputs (optional),
        and the generated gene files used by later steps.
    """

    pl.seed_everything(config.seed, workers=True)

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    attention_dir = output_dir / config.attention_dir
    if config.attention_output:
        attention_dir.mkdir(parents=True, exist_ok=True)

    if config.prior_network is None:
        prior_network_path = None
    else:
        prior_network_path = resolve_path(config.prior_network)

    # Skip if outputs already exist
    stage1_csv_path = output_dir / config.stage1_csv
    checkpoint_path = output_dir / config.checkpoint_name
    attn_h5ad = attention_dir / "attention_TF_scores_with_types.h5ad"
    if stage1_csv_path.exists() and checkpoint_path.exists() and (not config.attention_output or attn_h5ad.exists()):
        log_verbose("[gravity] stage1 outputs detected (csv/ckpt/attn); skipping training.", level=1)
        return {
            'stage1_csv': stage1_csv_path,
            'checkpoint': checkpoint_path,
            'attention_dir': attention_dir if config.attention_output else None,
            'genes_path': output_dir / 'genes.txt',
            'gene_map': output_dir / 'genemap.json',
        }

    dataset = load_cell_stage_dataset(
        config.middle_csv,
        prior_path=prior_network_path,
        gene_list=config.gene_subset,
        n_pos_neighbors=config.n_pos_neighbors,
        n_neg_neighbors=config.n_neg_neighbors
    )

    total_cells = len(dataset)
    log_verbose(
        f"[gravity] stage1 dataset loaded: {total_cells} cells; val_fraction={config.val_fraction}",
        level=1,
    )

    hvgs = dataset.hvg
    stage1_csv_path = output_dir / config.stage1_csv
    checkpoint_path = output_dir / config.checkpoint_name
    attn_h5ad = attention_dir / "attention_TF_scores_with_types.h5ad"

    skip_stage1 = False
    if stage1_csv_path.exists() and checkpoint_path.exists() and (not config.attention_output or attn_h5ad.exists()):
        try:
            df_stats = pd.read_csv(stage1_csv_path, usecols=['cellIndex', 'alpha', 'beta'])
            existing_cells = df_stats['cellIndex'].nunique()
            has_complete_rates = df_stats[['alpha', 'beta']].notna().all().all()
        except Exception as exc:
            log_verbose(f"[gravity] failed to inspect existing stage1 CSV ({exc}); retraining.", level=1)
            existing_cells = -1
            has_complete_rates = False
        if existing_cells == len(dataset) and has_complete_rates:
            skip_stage1 = True
            log_verbose("[gravity] stage1 outputs match current dataset; reusing and skipping training.", level=1)
        else:
            log_verbose(
                f"[gravity] existing stage1 outputs mismatch dataset (cells: {existing_cells} vs {len(dataset)}; complete rates: {has_complete_rates}); retraining.",
                level=1,
            )

    raw_counts_path = resolve_path(config.raw_counts)
    raw_df = pd.read_csv(raw_counts_path)
    if 'cellIndex' not in raw_df.columns:
        raw_df.insert(0, 'cellIndex', pd.factorize(raw_df['cellID'])[0])
    raw_df['cellIndex'] = raw_df['cellIndex'].astype(int)

    cell_to_type = dict(zip(raw_df['cellIndex'].astype(str), raw_df['clusters'].astype(str))) if 'clusters' in raw_df.columns else {}
    if 'clusters' in raw_df.columns:
        clusters_series = raw_df['clusters']
    else:
        clusters_series = pd.Series(['NA'] * len(raw_df), index=raw_df.index)
    embedding_map = dict(zip(raw_df['cellIndex'], clusters_series))
    id_map = dict(zip(raw_df['cellIndex'], raw_df['cellID'])) if 'cellID' in raw_df.columns else {idx: str(idx) for idx in raw_df['cellIndex']}

    genes_path = output_dir / 'genes.txt'
    with genes_path.open('w') as fp:
        for gene in hvgs:
            fp.write(f"{gene}\n")
    log_verbose(f"[gravity] wrote {len(hvgs)} gene identifiers to {genes_path}", level=2)

    import json
    mapper_path = output_dir / 'genemap.json'
    with mapper_path.open('w') as fp:
        json.dump(dataset.niche_dict, fp, indent=2)
    log_verbose(f"[gravity] stored TF→target prior map at {mapper_path}", level=2)

    train_subset, val_subset = _split_for_validation(dataset, config.val_fraction, config.seed)
    if val_subset is None:
        log_verbose("[gravity] stage1 training uses all cells; no validation split.", level=1)
    else:
        log_verbose(
            f"[gravity] stage1 split → train: {len(train_subset)} cells, val: {len(val_subset)} cells",
            level=1,
        )
    train_dataset = train_subset if val_subset is not None else dataset
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.num_workers)
    val_loader = None
    if val_subset is not None:
        val_loader = DataLoader(val_subset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.num_workers)

    if skip_stage1:
        return {
            'stage1_csv': stage1_csv_path,
            'checkpoint': checkpoint_path,
            'attention_dir': attention_dir if config.attention_output else None,
            'genes_path': genes_path,
            'gene_map': mapper_path,
        }

    model_kwargs = dict(
        input_dim=len(hvgs) * 2 + 3,
        gene_list=hvgs,
        TF_list=dataset.TF_from_net,
        TG_list=dataset.TG_from_net,
        TFTG_map=dataset.niche_dict,
        TFTG_map_reverse=dataset.niche_dict_reverse,
        output_dim_trans=len(hvgs),
        embedding_map=embedding_map,
        id_map=id_map,
        gene_select=hvgs,
        nbrs=dataset.nbrs,
        origin_data=dataset.data,
        negs=dataset.negs,
        output_csv=str(stage1_csv_path),
        attention_output=config.attention_output,
        output_network_path=str(attention_dir),
        cell_to_type=cell_to_type,
        gene_list_path=str(genes_path),
        gene_mapper_path=str(mapper_path),
        csr_topk=config.attention_topk,
        learning_rate=config.learning_rate,
        embedding_size=config.embedding_size,
        model_dimension=config.model_dimension,
        ffn_dimension=config.ffn_dimension,
    )

    model = FullModelCellWise(**model_kwargs)

    devices = config.devices if config.devices is not None else 1
    trainer_kwargs = dict(
        accelerator=config.accelerator,
        devices=devices,
        max_epochs=config.epochs,
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=config.log_every_n_steps,
        enable_progress_bar=config.progress_bar,
        default_root_dir=str(output_dir),
    )
    if config.precision is not None:
        trainer_kwargs['precision'] = config.precision
    if config.gradient_clip_val is not None:
        trainer_kwargs['gradient_clip_val'] = config.gradient_clip_val
    if config.strategy is not None:
        trainer_kwargs['strategy'] = config.strategy

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader)

    # Only rank 0 performs export/saving
    if getattr(trainer, "global_rank", 0) == 0:
        single_model = FullModelCellWise(**model_kwargs)
        single_model.load_state_dict(model.state_dict())

        single_test_loader = DataLoader(dataset,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=config.num_workers)

        log_verbose(
            "[gravity] stage1 infer/export: computing velocities & attention matrices...",
            level=1,
        )

        export_callbacks = []
        if config.progress_bar and TQDMProgressBar is not None:
            export_callbacks.append(ExportProgressBar("Stage1 infer/export"))

        single_tester = pl.Trainer(
            accelerator='auto', devices=1, logger=False, enable_checkpointing=False,
            enable_progress_bar=config.progress_bar, default_root_dir=str(output_dir),
            callbacks=export_callbacks,
        )
        single_tester.test(single_model, dataloaders=single_test_loader)
        log_verbose("[gravity] stage1 infer/export finished; synchronising files...", level=1)

        trainer.save_checkpoint(str(checkpoint_path))

        for _ in range(60):
            if stage1_csv_path.exists():
                break
            time.sleep(1.0)
        if not stage1_csv_path.exists():
            raise RuntimeError(f"Stage1 CSV was not written: {stage1_csv_path}")

    # ensure all ranks wait until exports finish
    # Synchronize ranks if running distributed
    world_size = 1
    if hasattr(trainer, "strategy") and hasattr(trainer.strategy, "world_size"):
        world_size = trainer.strategy.world_size
    elif hasattr(trainer, "training_type_plugin") and hasattr(trainer.training_type_plugin, "world_size"):
        world_size = trainer.training_type_plugin.world_size
    if world_size and world_size > 1:
        barrier_fn = None
        if hasattr(trainer, "strategy") and hasattr(trainer.strategy, "barrier"):
            barrier_fn = trainer.strategy.barrier
        elif hasattr(trainer, "training_type_plugin") and hasattr(trainer.training_type_plugin, "barrier"):
            barrier_fn = trainer.training_type_plugin.barrier
        if barrier_fn is not None:
            barrier_fn()

    return {
        'stage1_csv': stage1_csv_path,
        'checkpoint': checkpoint_path,
        'attention_dir': attention_dir if config.attention_output else None,
        'genes_path': genes_path,
        'gene_map': mapper_path,
    }
