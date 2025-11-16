# gravity/plotting/colors.py
from __future__ import annotations

from typing import Optional, Sequence, Mapping, Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import to_hex, to_rgb, rgb_to_hsv, hsv_to_rgb
from matplotlib.lines import Line2D
# Additional dtype helpers from pandas
from pandas.api.types import (
    is_categorical_dtype,
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)


__all__ = [
    "BASE_PALETTE_DEFAULT",
    "complete_palette",
    "build_colormap_for_categories",
    "build_colormap",
    "map_colors_auto",
]

# ===== Base palette (anchor colors) =====
BASE_PALETTE_DEFAULT: Dict[str, str] = {
    "Ductal": "#7c9895",
    "Ngn3 low EP": "#92a5d1",
    "Ngn3 high EP": "#d9b9d4",
    "Pre-endocrine": "#EC6E66",
    "Alpha": "#fbbf45",
    "Beta": "#B5CE4E",
    "Delta": "#bd7795",
    "Epsilon": "#DAA87C",
}

# ===== Internal helpers =====
def _unique_in_order(seq) -> list[str]:
    seen, out = set(), []
    for x in seq:
        s = str(x)
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def _hue_rotate(hex_color: str, delta: float) -> str:
    r, g, b = to_rgb(hex_color)
    h, s, v = rgb_to_hsv([r, g, b])
    h = (h + float(delta)) % 1.0
    return to_hex(hsv_to_rgb([h, s, v]))

def _lighten(hex_color: str, amount: float = 0.12) -> str:
    r, g, b = to_rgb(hex_color)
    h, s, v = rgb_to_hsv([r, g, b])
    v = np.clip(v + amount, 0, 1)
    s = np.clip(s * (1 - amount * 0.5), 0, 1)
    return to_hex(hsv_to_rgb([h, s, v]))

def _darken(hex_color: str, amount: float = 0.12) -> str:
    r, g, b = to_rgb(hex_color)
    h, s, v = rgb_to_hsv([r, g, b])
    v = np.clip(v - amount, 0, 1)
    s = np.clip(min(1, s * (1 + amount * 0.5)), 0, 1)
    return to_hex(hsv_to_rgb([h, s, v]))

def _expand_base_colors(base_colors: Sequence[str], target: int) -> list[str]:
    """Expand anchor colors to a stable set (≈12) via hue/brightness tweaks."""
    base_colors = list(base_colors)
    pool = list(base_colors)
    if len(pool) >= target:
        return pool[:target]

    i = 0
    while len(pool) < target:
        c = base_colors[i % len(base_colors)]
        ring = (i // len(base_colors)) % 3
        if ring == 0:
            new = _hue_rotate(c, +0.08 if ((i // len(base_colors)) % 2 == 0) else -0.08)
        elif ring == 1:
            new = _lighten(c, 0.12)
        else:
            new = _darken(c, 0.12)
        if new not in pool:
            pool.append(new)
        i += 1
        if i > 1000:
            break
    return pool[:target]

# ===== Core: complete palette generation =====
def complete_palette(
    categories: Sequence[str],
    base_palette: Optional[Mapping[str, str]] = None,
    seed: Optional[int] = 0,
) -> Dict[str, str]:
    """Deterministic palette builder.

    Order-preserving and reproducible:
      1) Use provided base palette (override/extend anchors).
      2) If needed, expand anchors to ≈12 colors.
      3) If still short, fallback to tab20/20b/20c.
      4) Finally, fill remaining via evenly spaced HSV.
    """
    cats = _unique_in_order(categories)

    # 1) Merge base anchors
    base = dict(BASE_PALETTE_DEFAULT)
    if base_palette:
        base.update({str(k): str(v) for k, v in dict(base_palette).items()})

    anchors = [c for c in cats if c in base]
    anchor_colors = [base[c] for c in anchors]

    # 2) Expand anchors to ≈12
    DERIVED_TARGET = max(12, len(anchor_colors))
    derived_colors = _expand_base_colors(anchor_colors or list(base.values()), DERIVED_TARGET)

    pal: Dict[str, str] = {}
    for c in anchors:
        pal[c] = base[c]

    remaining = [c for c in cats if c not in pal]
    used = set(pal.values())

    # Prefer expanded colors first
    for col in derived_colors:
        if not remaining:
            break
        if col in used:
            continue
        pal[remaining.pop(0)] = col
        used.add(col)

    # 3) Then try tab20/20b/20c
    if remaining:
        fallback_list = []
        for name in ("tab20", "tab20b", "tab20c"):
            cmap_m = cm.get_cmap(name, 20)
            fallback_list.extend([to_hex(cmap_m(i)) for i in range(cmap_m.N)])
        for col in fallback_list:
            if not remaining:
                break
            if col in used:
                continue
            pal[remaining.pop(0)] = col
            used.add(col)

    # 4) Even HSV spacing
    if remaining:
        rng = np.random.RandomState(int(seed)) if seed is not None else np.random
        H = np.linspace(0, 1, len(remaining), endpoint=False)
        rng.shuffle(H)
        for h in H:
            col = to_hex(hsv_to_rgb([h, 0.65, 0.85]))
            if col in used:
                col = to_hex(hsv_to_rgb([(h + 0.03) % 1.0, 0.65, 0.85]))
            pal[remaining.pop(0)] = col

    return pal

# ===== Legacy-compatible wrappers =====
def build_colormap_for_categories(categories: Sequence[str], *, seed: Optional[int] = 0) -> Dict[str, str]:
    return complete_palette(categories, base_palette=None, seed=seed)

def build_colormap(clusters: Sequence[str], base_palette: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    cats = _unique_in_order(clusters)
    return complete_palette(cats, base_palette=base_palette)

# ===== Automatic mapping for DataFrame (discrete-first version) =====
def map_colors_auto(
    df: pd.DataFrame,
    *,
    color_by: Optional[str],
    palette: Optional[Mapping[str, str]] = None,
    categories: Optional[Sequence[str]] = None,
    cmap_continuous: str = "viridis",
    discrete_unique_threshold: int = 32,   # Treat as discrete if unique ≤ 32
):
    """Return (mapped, meta) with discrete-first heuristics.

    - Discrete: mapped = list of colors, meta = ("discrete", handles)
    - Continuous: mapped = numeric array, meta = ("continuous", cmap_name)
    - None: mapped = "#95D9EF", meta = None
    Rules (fixes 0/1/2/3 misclassification):
      1) Given ``categories`` → discrete (in the given order)
      2) dtype is category/object/bool/integer → discrete
      3) numeric with few unique values (≤ ``discrete_unique_threshold``) → discrete
      4) otherwise continuous
    """
    if color_by is None or color_by not in df.columns:
        return "#95D9EF", None

    s = df[color_by]

    # 1) Explicit order → discrete
    if categories is not None and len(categories) > 0:
        cats = [str(c) for c in categories]
        pal = complete_palette(cats, base_palette=palette)
        mapped = [pal.get(str(v), "#999999") for v in s.astype(str)]
        handles = [
            Line2D([0], [0], color='w', marker='o', label=c,
                   markerfacecolor=pal[c], markeredgewidth=0, markersize=5)
            for c in cats
        ]
        return mapped, ("discrete", handles)

    # 2) Type priority: string/category/bool/integer → discrete
    if is_categorical_dtype(s) or s.dtype == object or is_bool_dtype(s) or is_integer_dtype(s):
        cats = _unique_in_order(s.astype(str))
        pal = complete_palette(cats, base_palette=palette)
        mapped = [pal.get(str(v), "#999999") for v in s.astype(str)]
        handles = [
            Line2D([0], [0], color='w', marker='o', label=c,
                   markerfacecolor=pal[c], markeredgewidth=0, markersize=5)
            for c in cats
        ]
        return mapped, ("discrete", handles)

    # 3) Numeric column → check unique count
    if is_numeric_dtype(s):
        nuniq = int(pd.unique(s.astype(float)).size)
        if nuniq <= discrete_unique_threshold:
            cats = _unique_in_order(s.astype(str))
            pal = complete_palette(cats, base_palette=palette)
            mapped = [pal.get(str(v), "#999999") for v in s.astype(str)]
            handles = [
                Line2D([0], [0], color='w', marker='o', label=c,
                       markerfacecolor=pal[c], markeredgewidth=0, markersize=5)
                for c in cats
            ]
            return mapped, ("discrete", handles)
        else:
            numeric = s.astype(float).to_numpy()
            return numeric, ("continuous", cmap_continuous)

    # 4) Fallback → discrete
    cats = _unique_in_order(s.astype(str))
    pal = complete_palette(cats, base_palette=palette)
    mapped = [pal.get(str(v), "#999999") for v in s.astype(str)]
    handles = [
        Line2D([0], [0], color='w', marker='o', label=c,
               markerfacecolor=pal[c], markeredgewidth=0, markersize=5)
        for c in cats
    ]
    return mapped, ("discrete", handles)
