# Repository Guidelines

## Project Structure & Module Organization
`llava/` contains the main VidKV/LLaVA code:
- `llava/model/` for model architecture and quantized cache integration
- `llava/train/` for training entry points
- `llava/eval/` for evaluation scripts
- `llava/serve/` for serving/CLI utilities

`transformers/` is a vendored Transformers tree used for VidKV-specific cache behavior; keep edits scoped and intentional. `figures/` stores paper assets. Top-level scripts include `demo.py` (inference example) and `env_setup.sh` (environment bootstrap). Dependencies are defined in `pyproject.toml` and `requirements.txt`.

## Build, Test, and Development Commands
- `conda create -n vidkv python=3.10 -y && conda activate vidkv`: create the recommended environment.
- `pip install -e ".[train]"`: install this package in editable mode with training extras.
- `(cd transformers && pip install .)`: install the local patched Transformers build.
- `python demo.py`: run the demo (set a real video path first).
- `python -m pytest transformers/tests -k <pattern>`: run targeted tests.
- `make -C transformers quality` and `make -C transformers test`: lint/format checks and full test pass for `transformers/` changes.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and clear, small functions. Follow naming conventions: `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants. Root formatting is configured with Black (`line-length = 240` in `pyproject.toml`). For `transformers/` edits, use Ruff/format targets from its `Makefile` instead of ad-hoc style changes.

## Testing Guidelines
Testing uses `pytest` (primarily under `transformers/tests/`). Add tests near the affected component and use `test_*.py` filenames. Prefer deterministic unit tests; isolate GPU-heavy checks so contributors can still run fast local validation. Before opening a PR, run at least targeted tests for modified modules and include exact commands used.

## Commit & Pull Request Guidelines
Current history uses very short subjects (for example, `update`, `first commit`). For new work, prefer concise imperative commits with scope, e.g., `llava: add mixed-precision cache config validation`. Keep PRs focused and include:
- what changed and why
- related issue/paper reference
- reproducible validation steps (commands, model, CUDA/PyTorch versions)
- benchmark or qualitative output when behavior/performance changes
