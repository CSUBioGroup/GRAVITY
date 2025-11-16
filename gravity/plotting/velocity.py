# -*- coding: utf-8 -*-
"""Plotting helpers for GRAVITY velocity visualisations.

This module preserves the legacy visual behavior while exposing a clearer API.
The cell-level
arrows are generated via a legacy grid-curve procedure (two-end grids, two KNN
passes, Gaussian weights, absolute ``min_mass`` threshold, Bezier curves, and
uniform arrow length).

Entrypoints
-----------
- :func:`plot_velocity_cell` – cell-level embedding plot with velocity arrows.
- :func:`plot_velocity_gene` – gene-level phase portrait with projected arrows.
- :func:`scatter_cell` – lower-level helper used internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Mapping, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_hex, hsv_to_rgb
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal
import bezier

from ..utils import log_verbose, resolve_path, build_colormap_for_categories, build_colormap, map_colors_auto
from ..velocity import compute_cell_velocity_, sampling_neighbors


__all__ = ["plot_velocity_cell", "plot_velocity_gene", "scatter_cell", "build_colormap_for_categories", "build_colormap"]


# -----------------------------------------------------------------------------
# DataFrame column extraction (legacy behavior: filter by gene and dropna)
# -----------------------------------------------------------------------------
def _extract_from_df(df: pd.DataFrame, attrs: Sequence[str] | str, gene: Optional[str] = None) -> np.ndarray:
    """Return selected columns for a single gene, dropping rows with NaNs."""
    if gene is None:
        gene = df["gene_name"].iloc[0]
    if isinstance(attrs, str):
        attrs = [attrs]
    sub = df.loc[df["gene_name"] == gene, list(attrs)].dropna()
    arr = sub.to_numpy()
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr


# -----------------------------------------------------------------------------
# Legacy grid-curve implementation for cell-level arrows
# -----------------------------------------------------------------------------
def _find_nn_neighbors(data: np.ndarray, queries: np.ndarray, n_neighbors: int):
    nn = NearestNeighbors(n_neighbors=max(1, int(n_neighbors)))
    nn.fit(np.asarray(data))
    dists, idxs = nn.kneighbors(np.asarray(queries), return_distance=True)
    return dists, idxs


def _grid_curve_legacy(ax,
                       embedding_ds: np.ndarray,
                       velocity_embedding: np.ndarray,
                       arrow_grid: Tuple[int, int],
                       min_mass: float) -> None:
    """Legacy curve-based arrow rendering on a rectangular grid.

    Pipeline: two-end grids → KNN smoothing (Gaussian kernel) → absolute mass
    threshold → Bezier curve evaluation → normalized tail arrows.
    """

    def _calculate_two_end_grid(emb, vel, smooth: float, steps: Tuple[int, int], min_mass: float):
        # Grid with slight offset and small padding
        grs = []
        for dim in range(emb.shape[1]):
            m, M = np.min(emb[:, dim]) - 0.2, np.max(emb[:, dim]) - 0.2
            m = m - 0.025 * np.abs(M - m)
            M = M + 0.025 * np.abs(M - m)
            grs.append(np.linspace(m, M, int(steps[dim])))

        mesh = np.meshgrid(*grs)
        XY = np.vstack([axis.flat for axis in mesh]).T

        # Number of neighbors = n/3
        k = max(1, int(vel.shape[0] / 3))

        # Two-end KNN (head/tail)
        d_head, ix_head = _find_nn_neighbors(emb, XY, k)
        d_tail, ix_tail = _find_nn_neighbors(emb + vel, XY, k)

        # Gaussian kernel (bandwidth = smooth * average grid spacing)
        std = float(np.mean([(g[1] - g[0]) for g in grs]))
        gw_head = normal.pdf(x=d_head, loc=0, scale=smooth * std)
        mass_head = gw_head.sum(1)
        gw_tail = normal.pdf(x=d_tail, loc=0, scale=smooth * std)
        mass_tail = gw_tail.sum(1)

        # Weighted averages
        UZ_head = (vel[ix_head] * gw_head[:, :, None]).sum(1) / np.maximum(1, mass_head)[:, None]
        UZ_tail = (vel[ix_tail] * gw_tail[:, :, None]).sum(1) / np.maximum(1, mass_tail)[:, None]

        # Second KNN pass (after displacement)
        d_head2, ix_head2 = _find_nn_neighbors(emb, XY + UZ_head, k)
        d_tail2, ix_tail2 = _find_nn_neighbors(emb, XY - UZ_tail, k)
        gw_head2 = normal.pdf(x=d_head2, loc=0, scale=smooth * std)
        mass_head2 = gw_head2.sum(1)
        gw_tail2 = normal.pdf(x=d_tail2, loc=0, scale=smooth * std)
        mass_tail2 = gw_tail2.sum(1)

        UZ_head2 = (vel[ix_head2] * gw_head2[:, :, None]).sum(1) / np.maximum(1, mass_head2)[:, None]
        UZ_tail2 = (vel[ix_tail2] * gw_tail2[:, :, None]).sum(1) / np.maximum(1, mass_tail2)[:, None]

        keep = mass_head >= float(min_mass)  # Absolute threshold
        return (XY[keep], UZ_head[keep], UZ_tail[keep], UZ_head2[keep], UZ_tail2[keep], grs)

    curve_smooth = 0.8  # Fixed as in the legacy implementation
    XY, UH, UT, UH2, UT2, grs = _calculate_two_end_grid(
        embedding_ds, velocity_embedding, smooth=curve_smooth, steps=arrow_grid, min_mass=min_mass
    )

    n_curves = XY.shape[0]
    s_vals = np.linspace(0.0, 1.5, 15)

    def _norm_ratio(XY, UT, UH, UT2, UH2, grs, s_vals):
        def _seg_len(x, y):
            return np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

        max_len = 0.0
        for i in range(n_curves):
            nodes = np.asfortranarray([
                [XY[i,0]-UT[i,0]-UT2[i,0], XY[i,0]-UT[i,0], XY[i,0], XY[i,0]+UH[i,0], XY[i,0]+UH[i,0]+UH2[i,0]],
                [XY[i,1]-UT[i,1]-UT2[i,1], XY[i,1]-UT[i,1], XY[i,1], XY[i,1]+UH[i,1], XY[i,1]+UH[i,1]+UH2[i,1]],
            ])
            curve = bezier.Curve(nodes, degree=4)
            dots = curve.evaluate_multi(s_vals)
            max_len = max(max_len, float(np.sum(_seg_len(dots[0], dots[1]))))

        grid_step = (abs(grs[0][1]-grs[0][0]) + abs(grs[1][1]-grs[1][0])) / 2.0
        return (grid_step / max_len) if max_len > 0 else 1.0

    ratio = _norm_ratio(XY, UT, UH, UT2, UH2, grs, s_vals)
    UT, UH, UT2, UH2 = UT * ratio, UH * ratio, UT2 * ratio, UH2 * ratio

    # Draw Bezier curves and normalized tail arrows
    for i in range(n_curves):
        nodes = np.asfortranarray([
            [XY[i,0]-UT[i,0]-UT2[i,0], XY[i,0]-UT[i,0], XY[i,0], XY[i,0]+UH[i,0], XY[i,0]+UH[i,0]+UH2[i,0]],
            [XY[i,1]-UT[i,1]-UT2[i,1], XY[i,1]-UT[i,1], XY[i,1], XY[i,1]+UH[i,1], XY[i,1]+UH[i,1]+UH2[i,1]],
        ])
        curve = bezier.Curve(nodes, degree=4)
        dots = curve.evaluate_multi(s_vals)

        ax.plot(dots[0], dots[1], linewidth=0.5, color='black', alpha=1.0)

        U = dots[0][-1] - dots[0][-2]
        V = dots[1][-1] - dots[1][-2]
        N = (U**2 + V**2) ** 0.5 + 1e-12
        U1, V1 = (U/N) * 0.5, (V/N) * 0.5  # 固定 0.5
        ax.quiver(dots[0][-2], dots[1][-2], U1, V1,
                  units='xy', angles='xy', scale=1, linewidth=0,
                  color='black', alpha=1.0, minlength=0, width=0.1)


# -----------------------------------------------------------------------------
# Cell-level API compatible with the legacy interface
# -----------------------------------------------------------------------------
def scatter_cell(
    ax,
    cellDancer_df: pd.DataFrame,
    colors=None,
    custom_xlim: Optional[Tuple[float, float]] = None,
    custom_ylim: Optional[Tuple[float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 0.5,
    s: float = 200,
    gene: Optional[str] = None,
    velocity: bool = False,
    legend: str = 'off',
    colorbar: str = 'on',
    min_mass: float = 0.5,
    arrow_grid: Tuple[int, int] = (20, 20)
):
    """Render a cell-level scatter plot and optional velocity arrows.

    This mirrors the legacy ``cdplt.scatter_cell`` routine while accepting
    higher-level color specifications typical for modern plotting APIs.
    """

    # --- Color handling (fully compatible with legacy parameter semantics) ---
    if isinstance(colors, list):
        colors = build_colormap(colors)

    if isinstance(colors, dict):
        # Cluster mapping
        cluster_vals = _extract_from_df(cellDancer_df, 'clusters', gene)
        c = np.vectorize(colors.get)(pd.Series(cluster_vals, dtype=str)).tolist()
        cmap = ListedColormap(list(colors.values()))
        if legend != 'off':
            handles = [Patch(facecolor=colors[k], edgecolor="none", label=k) for k in colors]
            lgd = ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left')
    elif isinstance(colors, str):
        # Continuous variable: use the standard 'viridis' colormap
        attr = colors
        assert gene, "Error! gene is required!"
        cmap = 'viridis'
        c = _extract_from_df(cellDancer_df, attr, gene)
    else:
        cmap = None
        c = 'Grey'

    # --- Scatter ---
    emb = _extract_from_df(cellDancer_df, ['embedding1', 'embedding2'], gene)
    n_cells = emb.shape[0]

    im = ax.scatter(emb[:, 0], emb[:, 1], c=c, cmap=cmap, s=s,
                    vmin=vmin, vmax=vmax, alpha=alpha, edgecolor="none")

    if colorbar == 'on' and isinstance(colors, str):
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("top", size="5%", pad="-5%")
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal", shrink=0.1)
        cbar.set_ticks([])

    # --- Legacy arrows ---
    if velocity:
        # Only draw for sampled cells with velocity
        sample_cells = cellDancer_df['velocity1'][:n_cells].dropna().index
        emb_ds = emb[sample_cells]
        vel = _extract_from_df(cellDancer_df, ['velocity1', 'velocity2'], gene)  # aligned with emb_ds after dropna
        _grid_curve_legacy(ax, emb_ds, vel, arrow_grid, min_mass)

    if custom_xlim is not None:
        ax.set_xlim(*custom_xlim)
    if custom_ylim is not None:
        ax.set_ylim(*custom_ylim)

    return ax


# -----------------------------------------------------------------------------
# High-level entry: plot_velocity_* helpers (embedding uses grid-curve arrows)
# -----------------------------------------------------------------------------
def plot_velocity_cell(
    stage_csv: str,
    *,
    gene: Optional[str] = None,
    x: str = "splice",
    y: str = "unsplice",
    color_by: Optional[str] = "clusters",
    palette: Optional[Mapping[str, str]] = None,
    categories: Optional[Sequence[str]] = None,
    cmap: str = "viridis",
    point_size: float = 200.0,
    alpha: float = 0.5,
    arrow_grid: Tuple[int, int] = (20, 20),
    arrow_scale: float = 1.0,
    arrow_color: str = "k",
    projection_neighbor_choice: str = "embedding",
    expression_scale: Optional[str] = "power10",
    projection_neighbor_size: int = 200,
    min_mass: float = 0.5,                    # Absolute threshold (legacy)
    axis_off: bool = True,
    output_path: Optional[str] = None,
    show: bool = False,
) -> plt.Axes:
    """Plot cell-level velocities in the embedding space.

    The function computes velocity projections via :func:`gravity.velocity.compute_cell_velocity_`
    and overlays legacy grid-curved arrows. Color handling supports both
    discrete palettes and continuous scalars.
    """

    path = resolve_path(stage_csv)
    log_verbose(f"[gravity] plotting velocities from {path}", level=1)
    df = pd.read_csv(path)
    gene_to_plot = gene

    if gene is None:
        gene = df["gene_name"].iloc[0]

    # Use a large figure size to mirror legacy scripts
    fig, ax = plt.subplots(figsize=(20, 20))

    # 1) Compute cell-level velocity (legacy logic)
    res = compute_cell_velocity_(
        df,
        projection_neighbor_choice=projection_neighbor_choice,
        expression_scale=expression_scale,
        projection_neighbor_size=projection_neighbor_size,
    )
    cell_df = res[0] if isinstance(res, tuple) else res

    # 2) Translate color_by to the form required by scatter_cell(colors=...)
    #    - Discrete column (e.g. 'clusters'): pass a dict (palette or auto)
    #    - Continuous column ('alpha'/'beta'/'gamma'/'splice'/'unsplice'/'pseudotime'): pass the column name
    #    - Other discrete column: temporarily map it to 'clusters' in a copy
    colors_arg = None
    plot_df = cell_df.copy()

    if color_by is None:
        colors_arg = None
    elif color_by in ('alpha', 'beta', 'gamma', 'splice', 'unsplice', 'pseudotime'):
        # scatter_cell uses the 'viridis' continuous colormap
        colors_arg = color_by
    else:
        # Treat as discrete labels
        if color_by != 'clusters':
            # Copy the requested discrete column to 'clusters' (plot-local)
            plot_df.loc[:, 'clusters'] = plot_df[color_by].astype(str)
        # Use provided palette; otherwise auto-generate one
        if palette is not None:
            colors_arg = dict(palette)
        else:
            # Derive categories from the current gene only
            gene_mask = (plot_df['gene_name'] == gene)
            cats = list(pd.unique(plot_df.loc[gene_mask, 'clusters'].astype(str)))
            colors_arg = build_colormap_for_categories(cats)

    # 3) Delegate to scatter_cell (grid-curve arrows)
    scatter_cell(
        ax=ax,
        cellDancer_df=plot_df,
        colors=colors_arg,
        alpha=alpha,
        s=point_size,             # Keep point size consistent with legacy
        gene=gene,
        velocity=True,            # Draw cell-level arrows
        legend='off',             # Enable if a legend is desired
        colorbar='on' if isinstance(colors_arg, str) else 'off',
        min_mass=min_mass,
        arrow_grid=arrow_grid,
    )

    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.set_title("GRAVITY cell velocities (embedding)")

    if axis_off:
        ax.axis("off")
    ax.grid(False)

    if output_path is not None:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        log_verbose(f"[gravity] saved velocity plot to {out}", level=2)

    if show:
        plt.show()

    return ax

def plot_velocity_gene(
    stage_csv: str,
    *,
    gene: Optional[str] = None,
    x: str = "splice",
    y: str = "unsplice",
    color_by: Optional[str] = "clusters",
    palette: Optional[Mapping[str, str]] = None,
    categories: Optional[Sequence[str]] = None,
    cmap: str = "viridis",
    point_size: float = 200.0,
    alpha: float = 0.5,
    arrow_grid: Tuple[int, int] = (20, 20),
    arrow_scale: float = 1.0,
    arrow_color: str = "k",
    projection_neighbor_choice: str = "embedding",
    expression_scale: Optional[str] = "power10",
    projection_neighbor_size: int = 200,
    min_mass: float = 0.5,                    # 绝对阈值（与原版一致）
    axis_off: bool = True,
    output_path: Optional[str] = None,
    show: bool = False,
) -> plt.Axes:
    """Plot gene-level phase portraits with projected velocity arrows.

    The arrows connect current to predicted expression for a sampled subset of
    cells, following the legacy straight-arrow strategy.
    """

    path = resolve_path(stage_csv)
    log_verbose(f"[gravity] plotting velocities from {path}", level=1)
    df = pd.read_csv(path)
    gene_to_plot = gene

    if gene is None:
        gene = df["gene_name"].iloc[0]

    # Use a large figure size to mirror legacy scripts
    fig, ax = plt.subplots(figsize=(20, 20))

    if gene_to_plot:
        # ====== Gene-level straight-arrow implementation (legacy) ======
        if x not in {"splice", "unsplice"} or y not in {"splice", "unsplice"}:
            raise ValueError("For gene phase portrait mode, x and y must be 'splice' or 'unsplice'.")
        gene_df = df[df["gene_name"] == gene_to_plot].copy()

        # Colors: discrete → palette/autogenerated; continuous → colormap
        mapped, meta = map_colors_auto(
            gene_df,
            color_by=color_by,
            palette=palette,
            categories=categories,
            cmap_continuous=cmap,
        )
        scatter_kwargs = dict(s=point_size, alpha=alpha, edgecolor="none")
        coords = gene_df[[x, y]].to_numpy()

        if isinstance(meta, tuple) and meta[0] == "discrete":
            _, handles = meta
            ax.scatter(coords[:, 0], coords[:, 1], color=mapped, **scatter_kwargs)
            if handles:
                ax.legend(handles=handles, title=color_by, loc="best", frameon=False)
        elif isinstance(meta, tuple) and meta[0] == "continuous":
            _, cm_name = meta
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=mapped, cmap=cm_name, **scatter_kwargs)
            plt.colorbar(sc, ax=ax, label=color_by)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], color=mapped, **scatter_kwargs)

        # Straight arrows (legacy strategy)
        u_s = gene_df[["unsplice", "splice", "unsplice_predict", "splice_predict"]].to_numpy()
        idx = np.asarray(sampling_neighbors(u_s[:, 0:2], step=arrow_grid, percentile=15))
        idx = idx[idx < u_s.shape[0]]
        U = u_s[idx, :]

        if x == "splice" and y == "unsplice":
            P_x, P_y = U[:, 1], U[:, 0]
            dX, dY = (U[:, 3] - U[:, 1]), (U[:, 2] - U[:, 0])
        elif x == "unsplice" and y == "splice":
            P_x, P_y = U[:, 0], U[:, 1]
            dX, dY = (U[:, 2] - U[:, 0]), (U[:, 3] - U[:, 1])
        else:
            P_x, P_y = U[:, 1], U[:, 0]
            dX, dY = (U[:, 3] - U[:, 1]), (U[:, 2] - U[:, 0])

        if P_x.size > 0:
            ax.scatter(P_x, P_y, facecolors="none", edgecolors=arrow_color, s=point_size * 1.2)
            ax.quiver(P_x, P_y, dX * arrow_scale, dY * arrow_scale,
                      angles="xy", color=arrow_color, alpha=0.8)

        ax.set_xlabel("Splice" if x == "splice" else "Unsplice")
        ax.set_ylabel("Splice" if y == "splice" else "Unsplice")
        ax.set_title(f"GRAVITY gene velocities (expression) – {gene} [{x} vs {y}]")

        if axis_off:
            ax.axis("off")
        ax.grid(False)

        if output_path is not None:
            out = Path(output_path).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=300, bbox_inches="tight")
            log_verbose(f"[gravity] saved velocity plot to {out}", level=2)

        if show:
            plt.show()

    return ax
