# GRAVITY Documentation

Welcome to the documentation site for **GRAVITY: Dynamic gene regulatory network-enhanced RNA velocity modeling for trajectory inference and biological discovery**.

These docs cover project goals, pipeline concepts, tutorials, and API usage so you can run the full workflow or embed GRAVITY into your own notebooks and services.

```{toctree}
:maxdepth: 2
:hidden:

tutorials/index
api/index
```

## Getting Started

- Clone the repository and follow the installation steps in the README.
- Create a Python ≥3.9 virtual environment, then install the package in editable mode: `pip install -e .[plots]`.
- Run one of the smoke tests under `gravity/smoke_test_*.py` to ensure your environment is GPU-ready.

## Tutorials

Head over to the [Tutorials](tutorials/index.md) section for:

- preparing long-format CSV files from AnnData sources,
- configuring and running the two-stage GRAVITY pipeline,
- interpreting outputs such as TF attention matrices and velocity plots.

## API Reference

The [API Reference](api/index.md) lists major modules (`gravity.pipeline`, `gravity.train`, `gravity.tools`, etc.) with autodoc-generated signatures so you can quickly locate configuration options.

## Contributing

If you improve the docs, please keep Markdown concise and link code-first examples back to the repository so they stay in sync with the source.
