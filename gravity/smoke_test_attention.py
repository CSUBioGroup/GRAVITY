"""Smoke test for attention interpretation helpers.

Adjust the paths and parameters as needed before running the script. The
default placeholders expect the standard Stage-1 outputs written under
``gravity_outputs/`` (relative to the project root)."""

from __future__ import annotations

from pathlib import Path

from gravity.analysis.attention_explain import (
    rank_attention_differentials,
    rank_tf_targets,
)
from gravity.utils import log_verbose


# ------------------------------
# Adjust these values to match your setup
# ------------------------------
ATTENTION_DIR = Path("/home/sda1/miaozy/cellDancer-main/src/celldancer/重构代码/gravity/gravity/gravity_outputs_new/attentions/")
REGULATORY_FACTOR = "PDX1"
CELL_TYPE = "Beta"
TOP_FACTORS = 10
TOP_TARGETS = 5
THRESHOLD = None  # e.g. 0.99 keeps top 1% edges; None disables filtering
TARGET_GENE = None  # e.g. "CXCL8" to inspect a specific target gene


def main() -> None:
    if not ATTENTION_DIR.exists():
        raise FileNotFoundError(f"Attention directory not found: {ATTENTION_DIR}")

    h5ad_path = ATTENTION_DIR / "attention_TF_scores_with_types.h5ad"
    if h5ad_path.exists():
        log_verbose(f"[smoke] running differential ranking on {h5ad_path}", level=1)
        diff_df = rank_attention_differentials(
            str(h5ad_path),
            cluster_key="cell_type",
            method="wilcoxon",
            group=CELL_TYPE,
            top_n=TOP_FACTORS,
        )
        print("\nDifferential ranking (per-cell type):")
        print(diff_df.to_string(index=False))

        print("\nRegulatory modules (top factors & targets):")
        for factor in diff_df["regulatory_factor"].tolist():
            tg_df = rank_tf_targets(
                attention_dir=str(ATTENTION_DIR),
                tf_gene=factor,
                cell_type=CELL_TYPE,
                top_k=TOP_TARGETS,
                threshold=THRESHOLD,
            )
            print(f"\nRegulatory factor: {factor}")
            print(tg_df.to_string(index=False))
    else:
        log_verbose(f"[smoke] consolidated h5ad not found at {h5ad_path}; skipping differential ranking.", level=1)

    print(f"Inspecting regulatory factor '{REGULATORY_FACTOR}' in cell type '{CELL_TYPE}'")
    results = rank_tf_targets(
        attention_dir=str(ATTENTION_DIR),
        tf_gene=REGULATORY_FACTOR,
        cell_type=CELL_TYPE,
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
