# ComfyUI Sprite Fusion Pixel Snapper

This custom node is a Python port of the Sprite Fusion **Pixel Snapper** tool
that fixes messy AI-generated pixel art by snapping it to a clean grid and
quantized palette.

## Install
Place this folder inside `ComfyUI/custom_nodes/` and restart ComfyUI.
Dependencies are already bundled with ComfyUI (PyTorch + NumPy); no extra
packages are required.

## Node
- **Name:** `SpriteFusion Pixel Snapper`
- **Category:** `image/processing`

### Inputs
- `image` – ComfyUI `IMAGE` tensor.
- `k_colors` (int) – Palette size (default 16).
- `k_seed` (int) – RNG seed for palette init (default 42).
- `output_scale` (int) – Optional integer upscaling after snapping (nearest-neighbor, default 1).

Advanced parameters mirror the original Rust defaults and can be tweaked:
`max_kmeans_iterations`, `peak_threshold_multiplier`, `peak_distance_filter`,
`walker_search_window_ratio`, `walker_min_search_window`,
`walker_strength_threshold`, `min_cuts_per_axis`,
`fallback_target_segments`, `max_step_ratio`.

### Output
- `IMAGE` – pixel-snapped result, one frame per input frame.

## License
- This repository: MIT (see [LICENSE](LICENSE), © 2025 x0x0b).
- Upstream algorithm: MIT (see [LICENSE-spritefusion-pixel-snapper](LICENSE-spritefusion-pixel-snapper), © 2025 Hugo Duprez).
