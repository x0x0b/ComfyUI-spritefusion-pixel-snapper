# Repository Guidelines

Guide for contributing to the ComfyUI Sprite Fusion Pixel Snapper node while keeping changes reviewable.

## Project Structure & Module Organization
- `pixel_snapper_node.py` — core node logic, NumPy/torch helpers, `Config` dataclass defaults, and `PixelSnapperError` for user-facing failures.
- `__init__.py` — re-exports `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS` so ComfyUI can discover the node.
- `README.md` — usage notes; `pyproject.toml` — metadata/dependencies; `LICENSE*` — MIT licenses for this port and the upstream algorithm.
- No assets or tests yet; keep additions scoped to this folder so ComfyUI auto-loads correctly.

## Development Setup & Commands
- Python 3.10+ with ComfyUI’s bundled `torch` and `numpy`.
- Editable install (optional for IDEs): `python -m pip install -e .`
- Quick syntax check: `python -m compileall pixel_snapper_node.py`
- Manual smoke test inside ComfyUI: place this folder under `ComfyUI/custom_nodes/`, restart ComfyUI, run a simple workflow that feeds an IMAGE into **SpriteFusion Pixel Snapper** with `k_colors=16` and verify the output grid/palette look clean.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents; favor explicit type hints and dataclasses for configuration-style structs.
- Prefer pure functions for image transforms and keep tensor↔NumPy conversions isolated in helpers.
- Raise `PixelSnapperError` for any validation issue that should be surfaced to the user; keep messages specific (e.g., dimension, dtype, or palette errors).
- Use deterministic RNG seeds (`k_seed`) when adding randomized steps.

## Testing Guidelines
- No automated test suite yet; add targeted unit tests when changing the algorithm (`pytest` is fine).
- Manual checks: use a 16×16 or 32×32 sprite and confirm palette size matches `k_colors`, grid lines stay stable after `output_scale > 1`, and batched IMAGE tensors shaped `(B,C,H,W)` or `(B,H,W,C)` run without errors.

## Commit & Pull Request Guidelines
- Commits: imperative mood and scoped prefixes help (e.g., `fix: clamp palette distances`, `chore: document node inputs`).
- Pull requests: include a short description, a before/after image or workflow JSON for visual changes, and the manual test steps you ran. Link related issues.
- Split algorithm tweaks, docs, and UI default changes into separate PRs.

## Security & Performance Notes
- Avoid heavy or networked dependencies; keep everything CPU-safe and deterministic.
- Validate image shapes early (`validate_image_dimensions`) to prevent oversized tensors from exhausting memory.
- When adding parameters, document defaults in both `Config` and the node input metadata so UI tooltips stay accurate.
