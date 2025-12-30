# ComfyUI Sprite Fusion Pixel Snapper

![Sprite Fusion Pixel Snapper icon](icon.png)

This custom node is a Python port of the [Sprite Fusion Pixel Snapper](https://github.com/Hugo-Dz/spritefusion-pixel-snapper) tool
that fixes messy AI-generated pixel art by snapping it to a clean grid and
quantized palette.

## Install
Place this folder inside `ComfyUI/custom_nodes/` and restart ComfyUI.
Dependencies are already bundled with ComfyUI (PyTorch + NumPy); no extra
packages are required.

## Node
- **Names:** `Sprite Fusion Pixel Snapper`, `Sprite Fusion Pixel Snapper (List)`
- **Category:** `image/transform`

### Inputs
- `image` – ComfyUI `IMAGE` tensor (single-image input, B=1).
- `k_colors` (int) – Palette size (default 16).
- `k_seed` (int) – RNG seed for palette init (default 42).
- `output_scale` (int) – Optional integer upscaling after snapping (nearest-neighbor, default 1, max 16).

Advanced parameters mirror the original Rust defaults and can be tweaked:
`max_kmeans_iterations`, `peak_threshold_multiplier`, `peak_distance_filter`,
`walker_search_window_ratio`, `walker_min_search_window`,
`walker_strength_threshold`, `min_cuts_per_axis`,
`fallback_target_segments`, `max_step_ratio`.

### Output
- `IMAGE` – pixel-snapped result, one frame per input frame.
- `IMAGE (List)` – per-frame outputs as a list, preserving each frame's size.

### Batch Handling
- `Sprite Fusion Pixel Snapper` expects a single image (B=1).
- `Sprite Fusion Pixel Snapper (List)` handles batches and preserves per-frame sizes.

## Credits
- Upstream repository: https://github.com/Hugo-Dz/spritefusion-pixel-snapper

## License
- This repository: MIT (see [LICENSE](LICENSE), © 2025 x0x0b).
- Upstream repository: MIT (see [LICENSE-spritefusion-pixel-snapper](LICENSE-spritefusion-pixel-snapper), © 2025 Hugo Duprez).
