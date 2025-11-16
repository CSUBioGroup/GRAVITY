# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `gravity/`. Key subpackages: `analysis/` (research helpers), `data/` (preprocessing + loaders), `models/` and `train/` (stage-specific Lightning modules), `tools/` (future projection, attention exports), and `plotting/` (velocity figures). Pipeline entry points sit in `pipeline.py` and `velocity.py`, while smoke tests (`gravity/smoke_test*.py`) demonstrate realistic input/output layouts under `gravity_outputs/`. Top-level `pyproject.toml` and `setup.py` define packaging metadata.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated Python ≥3.9 env.
- `pip install -e .[plots]`: editable install with optional plotting extras from `pyproject.toml`.
- `python - <<'PY' ... PY`: copy the snippet in `gravity/README.md` to run `PipelineConfig` experiments pointing at your `data/` CSVs and `prior_data/` network.
- `python gravity/smoke_test.py` (or another `smoke_test_*.py`): end-to-end regression check after updating the constants at the top to match accessible files.

## Coding Style & Naming Conventions
Use 4-space indentation, Black-compatible formatting, and keep imports ordered (stdlib → third-party → local). Favor dataclasses with explicit type hints for configs (`PipelineConfig`, `CellStageConfig`, `GeneStageConfig`) and snake_case module/function names; reserve PascalCase for classes. Prefer `pathlib.Path` over raw strings when handling filesystem paths, and preserve the existing keyword ordering for Lightning trainers so configs remain diff-friendly.

## Testing Guidelines
Pytest is not wired up yet; rely on smoke tests plus lightweight notebooks under `analysis/`. Each smoke script should print the artifact map returned by `run_pipeline`; treat missing keys or raised exceptions as failures. When adding new coverage, drop sample CSVs under `gravity/data/`, guard GPU-only logic with `if accelerator == "gpu"`, and name new scripts `smoke_test_<focus>.py` for discoverability.

## Commit & Pull Request Guidelines
History currently shows short, capitalized summaries (e.g., “Initial commit”), so keep following that tone: single-line imperative subject (<70 chars) plus optional wrapped body describing motivation, risk, and testing. Pull requests should call out touched modules, required data migrations, reproducible commands (`python gravity/smoke_test_toggle_new.py`, etc.), and screenshots whenever plot output changes; link issues with `Fixes #ID` when relevant.

## Environment & Configuration Tips
`PipelineConfig` drives device placement (`accelerator`, `devices`, `strategy`) and artifact naming; set a unique `workdir` per experiment to avoid overwriting checkpoints and plots. Store reusable CSVs and priors under versioned folders (`data/`, `prior_data/`) and keep credentials out of the repo. When working on multi-GPU features, ensure `strategy='ddp'` and watch for duplicate orchestration—`run_pipeline` skips non-zero ranks, so log at debug level if behavior differs.
