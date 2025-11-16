"""Utility to project GRAVITY velocities towards future positions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import log_verbose, resolve_path
from ..velocity import compute_cell_velocity_, extract_from_df

__all__ = [
    "estimate_future_positions",
]


def estimate_future_positions(
    stage1_csv: str,
    output_path: str,
    *,
    tau: float = 0.5,
    show_plot: bool = False,
    plot_path: Optional[str] = None,
    projection_neighbor_choice: str = 'embedding',
    projection_neighbor_size: int = 200,
    expression_scale: Optional[str] = 'power10',
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Estimate future embeddings using the learned cell-wise velocities.

    Parameters
    ----------
    stage1_csv:
        Output CSV from the cell-wise stage (stage1).
    output_path:
        Destination ``.npy`` file to store the nearest-neighbour anchors.
    tau:
        Scaling factor that shrinks the velocity vectors when forming the search radius.
    show_plot:
        Whether to display a matplotlib window with the quiver plot.
    plot_path:
        Optional path to save the figure instead of (or in addition to) showing it.

    Returns
    -------
    final_positions, cell_df:
        Tuple of the ``n × 3`` array with neighbour coordinates + indices, and the
        augmented dataframe returned by :func:`compute_cell_velocity_`.
    """

    stage1_path = resolve_path(stage1_csv)
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stage1_df = pd.read_csv(stage1_path, index_col=0)

    existing_future = None
    if out_path.exists():
        try:
            existing_future = np.load(str(out_path))
            expected_cells = stage1_df['cellIndex'].nunique()
            if existing_future.shape[0] == expected_cells:
                log_verbose(f"[gravity] found existing future positions: {out_path}; skip.", level=1)
                return existing_future, stage1_df
            else:
                log_verbose(
                    f"[gravity] existing future positions mismatch dataset (rows: {existing_future.shape[0]} vs {expected_cells}); recomputing.",
                    level=1,
                )
                existing_future = None
        except Exception as exc:
            log_verbose(f"[gravity] failed to load existing future positions ({exc}); recomputing.", level=1)
            existing_future = None

    log_verbose(f"[gravity] computing projected velocities from {stage1_path}", level=1)

    cell_df, velocity_embedding = compute_cell_velocity_(
        stage1_df,
        projection_neighbor_choice=projection_neighbor_choice,
        expression_scale=expression_scale,
        projection_neighbor_size=projection_neighbor_size,
        speed_up=None,
    )

    embeddings = extract_from_df(cell_df, ['embedding1', 'embedding2'], None)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 2)
    directions = velocity_embedding
    new_positions = embeddings + directions

    radius = np.linalg.norm(directions, axis=1) * tau
    final_positions = np.zeros((new_positions.shape[0], 3), dtype=float)

    hits = 0
    for idx in range(new_positions.shape[0]):
        distances = np.linalg.norm(embeddings - new_positions[idx], axis=1)
        neighbours = np.where(distances < radius[idx])[0]
        if neighbours.size == 0:
            final_positions[idx, :2] = embeddings[idx]
            final_positions[idx, 2] = idx
            continue
        hits += 1
        closest = neighbours[np.argmin(distances[neighbours])]
        final_positions[idx, :2] = embeddings[closest]
        final_positions[idx, 2] = closest

    log_verbose(f"[gravity] neighbours found within radius for {hits} cells", level=1)
    np.save(str(out_path), final_positions)
    log_verbose(f"[gravity] saved neighbour anchors to {out_path}", level=2)

    if show_plot or plot_path is not None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            s=8,
            color="#4A90E2",
            alpha=0.45,
            label="Current positions",
            edgecolor="none",
        )
        ax.quiver(
            embeddings[:, 0],
            embeddings[:, 1],
            directions[:, 0],
            directions[:, 1],
            angles='xy',
            scale_units='xy',
            scale=1,
            color="#FF5A5F",
            alpha=0.6,
            linewidth=0.3,
            label="Velocity",
        )
        ax.scatter(
            new_positions[:, 0],
            new_positions[:, 1],
            s=8,
            color="#37B26C",
            alpha=0.45,
            label="Projected positions",
            edgecolor="none",
        )
        ax.scatter(
            final_positions[:, 0],
            final_positions[:, 1],
            s=8,
            color="#F7C948",
            alpha=0.6,
            label="Anchor neighbors",
            edgecolor="none",
        )
        ax.set_xlabel('Embedding 1')
        ax.set_ylabel('Embedding 2')
        ax.legend(frameon=False, loc='upper right')
        ax.set_title('Future neighbor projection')
        ax.grid(True, linewidth=0.3, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')
        plt.tight_layout()

        if plot_path is not None:
            plot_path_resolved = Path(plot_path).expanduser().resolve()
            plot_path_resolved.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path_resolved, dpi=300, bbox_inches='tight')
            log_verbose(f"[gravity] saved future-position plot to {plot_path_resolved}", level=2)
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return final_positions, cell_df
