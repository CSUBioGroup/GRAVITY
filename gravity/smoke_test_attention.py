"""Smoke test for attention interpretation helpers.

Adjust the paths and parameters as needed before running the script. The
default paths expect the standard Stage-1 outputs written under
``gravity_outputs_new/`` (relative to the project root)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad

from gravity.analysis.attention_explain import (
    rank_attention_differentials,
    rank_tf_targets,
)
from gravity.utils import log_verbose


# ------------------------------
# Adjust these values to match your setup
# ------------------------------
ATTENTION_DIR = Path("gravity_outputs_new/attentions")
REGULATORY_FACTOR = "PDX1"
CELL_TYPE = "Beta"
TOP_FACTORS = 10
TOP_TARGETS = 5
THRESHOLD = None  # e.g. 0.99 keeps top 1% edges; None disables filtering
TARGET_GENE = None  # e.g. "CXCL8" to inspect a specific target gene


def _select_cell_type(h5ad_path: Path, preferred: str) -> str:
    adata = ad.read_h5ad(h5ad_path)
    if "cell_type" not in adata.obs:
        raise KeyError("Column 'cell_type' not found in attention h5ad.")

    counts = adata.obs["cell_type"].astype(str).value_counts()
    if counts.empty:
        raise ValueError("No cell types found in attention h5ad.")
    if preferred in counts.index:
        return preferred

    fallback = str(counts.index[0])
    log_verbose(
        f"[smoke] cell type '{preferred}' not found; using '{fallback}' instead.",
        level=1,
    )
    return fallback


def _load_genes(attention_dir: Path) -> list[str]:
    genes_path = attention_dir.parent / "genes.txt"
    if not genes_path.exists():
        raise FileNotFoundError(f"Gene list not found at {genes_path}")
    genes = [line.strip().upper() for line in genes_path.read_text().splitlines() if line.strip()]
    if not genes:
        raise ValueError(f"Gene list is empty: {genes_path}")
    return genes


def _select_regulatory_factor(
    attention_dir: Path,
    preferred: str,
    ranked_factors,
) -> str:
    genes = set(_load_genes(attention_dir))
    preferred_key = preferred.upper()
    if preferred_key in genes:
        return preferred_key

    for factor in ranked_factors:
        factor_key = str(factor).upper()
        if factor_key in genes:
            log_verbose(
                f"[smoke] regulatory factor '{preferred}' not found; using '{factor_key}' instead.",
                level=1,
            )
            return factor_key

    fallback = sorted(genes)[0]
    log_verbose(
        f"[smoke] regulatory factor '{preferred}' not found; using '{fallback}' instead.",
        level=1,
    )
    return fallback


def main() -> None:
    if not ATTENTION_DIR.exists():
        raise FileNotFoundError(f"Attention directory not found: {ATTENTION_DIR}")

    h5ad_path = ATTENTION_DIR / "attention_TF_scores_with_types.h5ad"
    cell_type = CELL_TYPE
    ranked_factors: list[str] = []
    if h5ad_path.exists():
        cell_type = _select_cell_type(h5ad_path, CELL_TYPE)
        log_verbose(f"[smoke] running differential ranking on {h5ad_path}", level=1)
        diff_df = rank_attention_differentials(
            str(h5ad_path),
            cluster_key="cell_type",
            method="wilcoxon",
            group=cell_type,
            top_n=TOP_FACTORS,
        )
        ranked_factors = diff_df["regulatory_factor"].tolist()
        print("\nDifferential ranking (per-cell type):")
        print(diff_df.to_string(index=False))

        print("\nRegulatory modules (top factors & targets):")
        for factor in ranked_factors:
            tg_df = rank_tf_targets(
                attention_dir=str(ATTENTION_DIR),
                tf_gene=factor,
                cell_type=cell_type,
                top_k=TOP_TARGETS,
                threshold=THRESHOLD,
            )
            print(f"\nRegulatory factor: {factor}")
            print(tg_df.to_string(index=False))
    else:
        log_verbose(f"[smoke] consolidated h5ad not found at {h5ad_path}; skipping differential ranking.", level=1)

    regulatory_factor = _select_regulatory_factor(ATTENTION_DIR, REGULATORY_FACTOR, ranked_factors)
    print(f"Inspecting regulatory factor '{regulatory_factor}' in cell type '{cell_type}'")
    results = rank_tf_targets(
        attention_dir=str(ATTENTION_DIR),
        tf_gene=regulatory_factor,
        cell_type=cell_type,
        top_k=TOP_TARGETS,
        threshold=THRESHOLD,
        target_gene=TARGET_GENE,
    )

    if TARGET_GENE is not None:
        rank_pos, weight = results  # type: ignore[assignment]
        print(f"Target '{TARGET_GENE}' rank: #{rank_pos} (weight={weight:.4f})")
    else:
        print(results.to_string(index=False))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
