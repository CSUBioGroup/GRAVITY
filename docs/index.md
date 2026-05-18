# GRAVITY predicts RNA velocity and regulatory rewiring by dynamic regulatory mechanism-enhanced deep learning

Welcome to the GRAVITY documentation site.

These docs cover project goals, pipeline concepts, tutorials, and API usage so you can run GRAVITY for RNA velocity inference, dynamic regulatory rewiring analysis, attention-based regulator summaries, and downstream velocity visualization. GRAVITY uses a cellDancer-style long-format count table as its user-facing input format, then builds the internal wide `combine.csv` used by the two-stage model.

![GRAVITY method overview](assets/gravity_method_overview.png)

```{toctree}
:maxdepth: 2
:hidden:

tutorials/index
api/index
```

## Getting Started

- Clone the repository and follow the installation steps in the README.
- Create a Python 3.10 or 3.11 virtual environment, then install the package in editable mode: `pip install -e .[plots]`.
- Place the pancreas example CSV at `data/PancreaticEndocrinogenesis_cell_type_u_s.csv`, or set `GRAVITY_RAW_COUNTS` to another compatible cellDancer-style CSV.
- Use `prior_data/nichenet_mouse.zip` for the pancreas or other mouse examples; switch to `prior_data/nichenet_human.zip` for human datasets.
- Run one of the smoke tests under `gravity/smoke_test_*.py` to ensure your environment is GPU-ready.

## Tutorials

Head over to the [Tutorials](tutorials/index.md) section for:

- preparing long-format CSV files from AnnData sources,
- configuring and running the two-stage GRAVITY pipeline,
- preserving checkpoint-compatible gene order with `gene_order_path`,
- reproducing the pancreas reference run from provided stage checkpoints,
- interpreting outputs such as TF attention matrices, pathway activity
  summaries, and velocity plots.

## API Reference

The [API Reference](api/index.md) lists major modules (`gravity.pipeline`, `gravity.train`, `gravity.tools`, etc.) with autodoc-generated signatures so you can quickly locate configuration options.

## Development Transparency

Codex was used as an engineering assistant to help reorganize this repository
into a reusable tool package, update documentation, and run implementation-level
checks. The GRAVITY model design, biological analysis strategy, and computational
methodology were developed by the authors; tool-assisted changes were checked
and tested by the authors before release.

## Contributing

If you improve the docs, please keep Markdown concise and link code-first examples back to the repository so they stay in sync with the source.
