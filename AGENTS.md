# Repository Guidelines

## Project Structure & Module Organization

RTR-GS combines inverse-rendering Gaussian splatting with SGS equirectangular training. Core entry points live at the repository root: `train.py` for training, `baking.py` for occlusion baking, and `eval_relighting_*.py` for evaluation. Rendering code is in `gaussian_renderer/`; scene loading is in `scene/`; PBR utilities are in `pbr/`; shared helpers are in `utils/` and `arguments/`. CUDA extensions and imported code are under `submodules/`. Run scripts are in `script/` and `scripts/`; configs are in `configs/`; environment maps are in `env_maps/`; formal tests should live in `tests/`, with extension-specific tests in `pbr/renderutils/tests/` and fixtures in `test_data/`. Technical notes are in `doc/` and `docs/`.

## Build, Test, and Development Commands

Activate the project environment before running commands:

```bash
conda activate odgs-rtr
```

Create the environment with `conda env create -f environment.yml`, then compile CUDA extensions with the editable install steps in `README.md`.

Common workflows:

```bash
sh script/run_synthetic.sh     # perspective synthetic training
sh script/run_real_scene.sh    # perspective real-scene training
sh script/run_sgs_rtr.sh       # SGS-to-RTR-GS equirect pipeline
python baking.py ...           # bake occlusion volumes
python viewer_clean.py ...     # inspect trained outputs
```

Run formal tests with `pytest tests` when a top-level suite is present. Use `pytest pbr/renderutils/tests` for renderutils checks. Existing `scripts/test_*.py` files are historical ad hoc checks; avoid adding new formal tests there.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and `snake_case` for functions, variables, files, and CLI flags. Keep module names descriptive, matching patterns such as `render_equirect.py`, `point_light_shadow.py`, and `run_real_scene.sh`. Preserve coordinate-system comments and be explicit when converting between COLMAP, reflvec, and equirect spaces. Avoid broad refactors inside `submodules/` unless the change targets that extension.

## Testing Guidelines

Prefer focused tests for rendering math, coordinate conversions, and CUDA wrapper behavior. Name new tests `test_<behavior>.py` and place general tests under `tests/`; use `pbr/renderutils/tests/` only for renderutils coverage. For training or relighting changes, include a small reproducing command or config path, and note GPU/CUDA assumptions.

## Commit & Pull Request Guidelines

Recent history uses conventional-style prefixes such as `feat:`, `fix:`, `docs:`, `chore:`, and `refactor:`. Keep commit messages imperative and scoped, for example `fix: align equirect baking normals`. Pull requests should summarize the affected pipeline, list verification commands, mention dataset or checkpoint requirements, and include rendered comparisons for visual changes.

## Agent-Specific Instructions

Read `CLAUDE.md` before algorithmic changes. Coordinate conventions are a common failure point; verify transformations against the relevant docs in `doc/RTR-GS/` before editing renderer, baking, or normal code.
