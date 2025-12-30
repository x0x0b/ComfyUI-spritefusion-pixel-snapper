"""
ComfyUI custom node port of Sprite Fusion Pixel Snapper (original Rust).
The node snaps AI‑generated pixel art to a consistent grid and palette.

Original project (MIT) © 2025 Hugo Duprez:
https://github.com/Hugo-Dz/spritefusion-pixel-snapper
Ported to Python for ComfyUI custom_nodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class Config:
    k_colors: int = 16
    k_seed: int = 42
    max_kmeans_iterations: int = 15
    peak_threshold_multiplier: float = 0.2
    peak_distance_filter: int = 4
    walker_search_window_ratio: float = 0.35
    walker_min_search_window: float = 2.0
    walker_strength_threshold: float = 0.5
    min_cuts_per_axis: int = 4
    fallback_target_segments: int = 64
    max_step_ratio: float = 1.8  # Lowered from 3.0 to catch more skew cases


class PixelSnapperError(ValueError):
    """Raised when invalid input or processing failure occurs."""


# --- Utility: tensor <-> numpy ------------------------------------------------

def tensor_to_numpy_batch(image: torch.Tensor) -> np.ndarray:
    """
    Convert a ComfyUI IMAGE tensor to numpy uint8 in channel-last layout (B, H, W, 3).
    Accepts both ComfyUI standard (B, H, W, C) and channel-first (B, C, H, W) tensors.
    If C == 4, the alpha channel is dropped.
    """
    t = image.detach().cpu().clamp(0.0, 1.0)

    if t.ndim == 3:
        # Single image without batch
        if t.shape[0] in (3, 4):  # C, H, W
            t = t.unsqueeze(0).movedim(1, -1)
        elif t.shape[-1] in (3, 4):  # H, W, C
            t = t.unsqueeze(0)
        else:
            raise PixelSnapperError(
                f"Expected IMAGE tensor with 3 or 4 channels, got shape {tuple(t.shape)}."
            )
    elif t.ndim == 4:
        if t.shape[1] in (3, 4):  # B, C, H, W
            t = t.movedim(1, -1)
        elif t.shape[-1] in (3, 4):  # B, H, W, C
            pass
        else:
            raise PixelSnapperError(
                f"Expected IMAGE tensor with 3 or 4 channels, got shape {tuple(t.shape)}."
            )
    else:
        raise PixelSnapperError("Expected IMAGE tensor with shape (B,H,W,C) or (B,C,H,W).")

    if t.shape[-1] == 4:
        t = t[..., :3]

    np_imgs = (t.numpy() * 255.0).round().astype(np.uint8)
    return np_imgs


def numpy_batch_to_tensor(images: np.ndarray) -> torch.Tensor:
    """
    Convert numpy batch (B, H, W, 3) uint8 to ComfyUI IMAGE tensor (B, H, W, 3) float in [0,1].
    """
    if images.ndim != 4 or images.shape[-1] != 3:
        raise PixelSnapperError("Expected numpy batch with shape (B, H, W, 3).")
    tensor = torch.from_numpy(images.astype(np.float32) / 255.0)
    return tensor


def dominant_color(img: np.ndarray) -> np.ndarray:
    """
    Return the most frequent RGB color in a uint8 image.
    """
    if img.size == 0:
        raise PixelSnapperError("Cannot compute dominant color of empty image.")
    pixels = img.reshape(-1, 3)
    values, counts = np.unique(pixels, axis=0, return_counts=True)
    return values[int(np.argmax(counts))]


def pad_image_to_shape(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Pad image to target shape using its dominant color (top-left anchored).
    """
    h, w, c = img.shape
    if c != 3:
        raise PixelSnapperError("Expected RGB image when padding batch outputs.")
    if h > target_h or w > target_w:
        raise PixelSnapperError(
            f"Cannot pad image from {(h, w)} to smaller target {(target_h, target_w)}."
        )
    if h == target_h and w == target_w:
        return img
    fill = dominant_color(img)
    padded = np.empty((target_h, target_w, 3), dtype=np.uint8)
    padded[:] = fill
    padded[:h, :w] = img
    return padded


def upscale_nearest(img: np.ndarray, scale: int) -> np.ndarray:
    """
    Pixel-art-safe integer upscaling by nearest neighbor.
    img: (H, W, 3) uint8
    scale: int >= 1
    """
    if scale <= 1:
        return img
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


# --- Core algorithm (ported from Rust) ---------------------------------------

def validate_image_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise PixelSnapperError("Image dimensions cannot be zero")
    if width > 10_000 or height > 10_000:
        raise PixelSnapperError("Image dimensions too large (max 10000x10000)")


def quantize_image(img: np.ndarray, config: Config) -> np.ndarray:
    """
    Simple K-Means color quantization using the Rust defaults.
    img: (H, W, 3) uint8
    """
    if config.k_colors <= 0:
        raise PixelSnapperError("Number of colors must be greater than 0")

    pixels = img.reshape(-1, 3).astype(np.float32)
    n_pixels = pixels.shape[0]
    if n_pixels == 0:
        raise PixelSnapperError("Image has no pixels")

    k = min(config.k_colors, n_pixels)
    rng = np.random.default_rng(config.k_seed)

    def dist_sq(points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        diff = points - centroid
        return np.sum(diff * diff, axis=1)

    first_idx = rng.integers(0, n_pixels)
    centroids = [pixels[first_idx]]
    distances = np.full(n_pixels, np.inf, dtype=np.float32)

    # KMeans++ style init (mirroring Rust logic)
    for _ in range(1, k):
        last_c = centroids[-1]
        d_sq = dist_sq(pixels, last_c)
        distances = np.minimum(distances, d_sq)
        sum_sq = float(np.sum(distances))
        if sum_sq <= 0.0 or not math.isfinite(sum_sq):
            idx = rng.integers(0, n_pixels)
        else:
            probs = distances / sum_sq
            idx = rng.choice(n_pixels, p=probs)
        centroids.append(pixels[idx])

    centroids = np.stack(centroids, axis=0)
    prev_centroids = centroids.copy()

    for iteration in range(config.max_kmeans_iterations):
        # Compute distances to centroids
        diff = pixels[:, None, :] - centroids[None, :, :]
        dists = np.sum(diff * diff, axis=2)
        labels = np.argmin(dists, axis=1)

        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, pixels)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        counts_expanded = counts[:, None]
        nonzero = counts_expanded[:, 0] > 0
        centroids[nonzero] = sums[nonzero] / counts_expanded[nonzero]

        if iteration > 0:
            movement = np.max(np.sum((centroids - prev_centroids) ** 2, axis=1))
            if movement < 0.01:
                break
        prev_centroids = centroids.copy()

    best = centroids[labels].round().clip(0, 255).astype(np.uint8)
    return best.reshape(img.shape)


def compute_profiles(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = img.shape
    if w < 3 or h < 3:
        raise PixelSnapperError("Image too small (minimum 3x3)")

    gray = (
        0.299 * img[:, :, 0].astype(np.float64)
        + 0.587 * img[:, :, 1].astype(np.float64)
        + 0.114 * img[:, :, 2].astype(np.float64)
    )

    col_proj = np.zeros(w, dtype=np.float64)
    row_proj = np.zeros(h, dtype=np.float64)

    col_proj[1:-1] = np.sum(np.abs(gray[:, 2:] - gray[:, :-2]), axis=0)
    row_proj[1:-1] = np.sum(np.abs(gray[2:, :] - gray[:-2, :]), axis=1)

    return col_proj, row_proj


def estimate_step_size(profile: Sequence[float], config: Config) -> Optional[float]:
    if len(profile) == 0:
        return None
    max_val = float(np.max(profile))
    if max_val == 0.0:
        return None
    threshold = max_val * config.peak_threshold_multiplier

    peaks: List[int] = []
    for i in range(1, len(profile) - 1):
        if (
            profile[i] > threshold
            and profile[i] > profile[i - 1]
            and profile[i] > profile[i + 1]
        ):
            peaks.append(i)
    if len(peaks) < 2:
        return None

    clean_peaks = [peaks[0]]
    for p in peaks[1:]:
        if p - clean_peaks[-1] > (config.peak_distance_filter - 1):
            clean_peaks.append(p)
    if len(clean_peaks) < 2:
        return None

    diffs = np.diff(clean_peaks)
    diffs.sort()
    return float(diffs[len(diffs) // 2])


def resolve_step_sizes(
    step_x_opt: Optional[float],
    step_y_opt: Optional[float],
    width: int,
    height: int,
    config: Config,
) -> Tuple[float, float]:
    if step_x_opt is not None and step_y_opt is not None:
        sx, sy = step_x_opt, step_y_opt
        ratio = sx / sy if sx > sy else sy / sx
        if ratio > config.max_step_ratio:
            smaller = min(sx, sy)
            return smaller, smaller
        avg = (sx + sy) / 2.0
        return avg, avg

    if step_x_opt is not None:
        return step_x_opt, step_x_opt
    if step_y_opt is not None:
        return step_y_opt, step_y_opt

    fallback = (min(width, height) / float(config.fallback_target_segments)) if config.fallback_target_segments else 1.0
    return max(fallback, 1.0), max(fallback, 1.0)


def walk(profile: Sequence[float], step_size: float, limit: int, config: Config) -> List[int]:
    if len(profile) == 0:
        raise PixelSnapperError("Cannot walk on empty profile")

    cuts = [0]
    current_pos = 0.0
    search_window = max(step_size * config.walker_search_window_ratio, config.walker_min_search_window)
    mean_val = float(np.mean(profile))

    while current_pos < limit:
        target = current_pos + step_size
        if target >= limit:
            cuts.append(limit)
            break

        start_search = max(int(target - search_window), int(current_pos + 1))
        end_search = min(int(target + search_window), limit)

        if end_search <= start_search:
            current_pos = target
            continue

        segment = np.asarray(profile[start_search:end_search])
        max_rel = int(np.argmax(segment))
        max_val = float(segment[max_rel]) if segment.size else -1.0
        max_idx = start_search + max_rel

        if max_val > mean_val * config.walker_strength_threshold:
            cuts.append(max_idx)
            current_pos = float(max_idx)
        else:
            cuts.append(int(target))
            current_pos = target
    return cuts


def sanitize_cuts(cuts: List[int], limit: int) -> List[int]:
    if limit == 0:
        return [0]

    normalized = []
    has_zero = False
    has_limit = False
    for v in cuts:
        v = 0 if v < 0 else v
        if v == 0:
            has_zero = True
        if v >= limit:
            v = limit
            has_limit = True
        normalized.append(v)
    if not has_zero:
        normalized.append(0)
    if not has_limit:
        normalized.append(limit)

    normalized = sorted(set(normalized))
    return normalized


def snap_uniform_cuts(
    profile: Sequence[float],
    limit: int,
    target_step: float,
    config: Config,
    min_required: int,
) -> List[int]:
    if limit == 0:
        return [0]
    if limit == 1:
        return [0, 1]

    desired_cells = int(round(limit / target_step)) if target_step > 0 and math.isfinite(target_step) else 0
    desired_cells = max(desired_cells, min_required - 1, 1)
    desired_cells = min(desired_cells, limit)

    cell_width = limit / float(desired_cells)
    search_window = max(cell_width * config.walker_search_window_ratio, config.walker_min_search_window)
    mean_val = float(np.mean(profile)) if len(profile) else 0.0

    cuts: List[int] = [0]
    for idx in range(1, desired_cells):
        target = cell_width * idx
        prev = cuts[-1]
        if prev + 1 >= limit:
            break

        start = int(math.floor(target - search_window))
        start = max(start, prev + 1, 0)
        end = int(math.ceil(target + search_window))
        end = min(end, limit - 1)
        if end < start:
            start = prev + 1
            end = start

        segment = np.asarray(profile[start : min(end + 1, len(profile))])
        if segment.size:
            best_rel = int(np.argmax(segment))
            best_val = float(segment[best_rel])
            best_idx = start + best_rel
        else:
            best_val = -1.0
            best_idx = start

        strength_threshold = mean_val * config.walker_strength_threshold
        if best_val < strength_threshold:
            fallback_idx = int(round(target))
            if fallback_idx <= prev:
                fallback_idx = prev + 1
            if fallback_idx >= limit:
                fallback_idx = max(limit - 1, prev + 1)
            best_idx = fallback_idx

        cuts.append(best_idx)

    if cuts[-1] != limit:
        cuts.append(limit)

    return sanitize_cuts(cuts, limit)


def stabilize_cuts(
    profile: Sequence[float],
    cuts: List[int],
    limit: int,
    sibling_cuts: Sequence[int],
    sibling_limit: int,
    config: Config,
) -> List[int]:
    if limit == 0:
        return [0]

    cuts = sanitize_cuts(cuts, limit)
    min_required = max(config.min_cuts_per_axis, 2)
    min_required = min(min_required, limit + 1)

    axis_cells = max(len(cuts) - 1, 0)
    sibling_cells = max(len(sibling_cuts) - 1, 0)
    sibling_has_grid = (
        sibling_limit > 0
        and sibling_cells >= (min_required - 1)
        and sibling_cells > 0
    )
    steps_skewed = False
    if sibling_has_grid and axis_cells > 0:
        axis_step = limit / float(axis_cells)
        sibling_step = sibling_limit / float(sibling_cells)
        step_ratio = axis_step / sibling_step
        steps_skewed = step_ratio > config.max_step_ratio or step_ratio < 1.0 / config.max_step_ratio

    has_enough = len(cuts) >= min_required
    if has_enough and not steps_skewed:
        return cuts

    if sibling_has_grid:
        target_step = sibling_limit / float(sibling_cells)
    elif config.fallback_target_segments > 1:
        target_step = limit / float(config.fallback_target_segments)
    elif axis_cells > 0:
        target_step = limit / float(axis_cells)
    else:
        target_step = float(limit)

    if not math.isfinite(target_step) or target_step <= 0.0:
        target_step = 1.0

    return snap_uniform_cuts(profile, limit, target_step, config, min_required)


def stabilize_both_axes(
    profile_x: Sequence[float],
    profile_y: Sequence[float],
    raw_col_cuts: List[int],
    raw_row_cuts: List[int],
    width: int,
    height: int,
    config: Config,
) -> Tuple[List[int], List[int]]:
    col_cuts_pass1 = stabilize_cuts(
        profile_x, list(raw_col_cuts), width, raw_row_cuts, height, config
    )
    row_cuts_pass1 = stabilize_cuts(
        profile_y, list(raw_row_cuts), height, raw_col_cuts, width, config
    )

    col_cells = max(len(col_cuts_pass1) - 1, 1)
    row_cells = max(len(row_cuts_pass1) - 1, 1)
    col_step = width / float(col_cells)
    row_step = height / float(row_cells)
    step_ratio = col_step / row_step if col_step > row_step else row_step / col_step

    if step_ratio > config.max_step_ratio:
        target_step = min(col_step, row_step)
        if col_step > target_step * 1.2:
            final_cols = snap_uniform_cuts(
                profile_x, width, target_step, config, config.min_cuts_per_axis
            )
        else:
            final_cols = col_cuts_pass1

        if row_step > target_step * 1.2:
            final_rows = snap_uniform_cuts(
                profile_y, height, target_step, config, config.min_cuts_per_axis
            )
        else:
            final_rows = row_cuts_pass1
        return final_cols, final_rows

    return col_cuts_pass1, row_cuts_pass1


def resample(img: np.ndarray, cols: Sequence[int], rows: Sequence[int]) -> np.ndarray:
    if len(cols) < 2 or len(rows) < 2:
        raise PixelSnapperError("Insufficient grid cuts for resampling")

    out_w = max(len(cols) - 1, 1)
    out_h = max(len(rows) - 1, 1)
    final_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for y_i, (ys, ye) in enumerate(zip(rows[:-1], rows[1:])):
        for x_i, (xs, xe) in enumerate(zip(cols[:-1], cols[1:])):
            if xe <= xs or ye <= ys:
                continue
            cell = img[ys:ye, xs:xe]
            if cell.size == 0:
                continue
            pixels = cell.reshape(-1, 3)
            values, counts = np.unique(pixels, axis=0, return_counts=True)
            best_idx = int(np.argmax(counts))
            final_img[y_i, x_i] = values[best_idx]

    return final_img


def process_image_array(img: np.ndarray, config: Config) -> np.ndarray:
    """
    Full pipeline on a single RGB uint8 numpy image (H, W, 3).
    Returns processed uint8 numpy image.
    """
    h, w, _ = img.shape

    # Guard against NaN / Inf inputs (can appear with half-precision pipelines)
    if not np.isfinite(img).all():
        raise PixelSnapperError("Input image contains NaN or Inf values.")

    validate_image_dimensions(w, h)

    quantized = quantize_image(img, config)
    profile_x, profile_y = compute_profiles(quantized)

    step_x_opt = estimate_step_size(profile_x, config)
    step_y_opt = estimate_step_size(profile_y, config)
    step_x, step_y = resolve_step_sizes(step_x_opt, step_y_opt, w, h, config)

    raw_col_cuts = walk(profile_x, step_x, w, config)
    raw_row_cuts = walk(profile_y, step_y, h, config)

    col_cuts, row_cuts = stabilize_both_axes(
        profile_x, profile_y, raw_col_cuts, raw_row_cuts, w, h, config
    )

    return resample(quantized, col_cuts, row_cuts)


# --- ComfyUI node ------------------------------------------------------------


class PixelSnapperNode:
    """
    Exposes the Pixel Snapper algorithm as a ComfyUI custom node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image tensor (B,3,H,W) to be snapped"}),
                "k_colors": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 512,
                        "tooltip": "Palette size for color quantization (K-Means)",
                    },
                ),
                "k_seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "Random seed used when seeding K-Means++ centroids",
                    },
                ),
            },
            "optional": {
                "output_scale": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "tooltip": "Integer upscaling factor (nearest-neighbor) applied after snapping",
                    },
                ),
                "max_kmeans_iterations": (
                    "INT",
                    {
                        "default": 15,
                        "min": 1,
                        "max": 200,
                        "tooltip": "Upper limit on K-Means iterations while learning the palette",
                    },
                ),
                "peak_threshold_multiplier": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Fraction of max gradient used as threshold to keep profile peaks",
                    },
                ),
                "peak_distance_filter": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 512,
                        "tooltip": "Minimum pixel spacing between retained peaks in the profile",
                    },
                ),
                "walker_search_window_ratio": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.01,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Search window size as a fraction of estimated step when walking cuts",
                    },
                ),
                "walker_min_search_window": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 32.0,
                        "step": 0.1,
                        "tooltip": "Minimum search window (pixels) around each target cut",
                    },
                ),
                "walker_strength_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Peak must exceed mean*threshold; otherwise fallback to uniform cut",
                    },
                ),
                "min_cuts_per_axis": (
                    "INT",
                    {
                        "default": 4,
                        "min": 2,
                        "max": 512,
                        "tooltip": "Lowest number of cut positions per axis (including ends)",
                    },
                ),
                "fallback_target_segments": (
                    "INT",
                    {
                        "default": 64,
                        "min": 1,
                        "max": 2048,
                        "tooltip": "Target number of cells when step detection fails; derives fallback step",
                    },
                ),
                "max_step_ratio": (
                    "FLOAT",
                    {
                        "default": 1.8,
                        "min": 1.0,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "Max allowed X/Y step ratio before snapping to a uniform grid",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "snap"
    CATEGORY = "image/transform"

    def snap(
        self,
        image: torch.Tensor,
        k_colors: int,
        k_seed: int,
        output_scale: int = 1,
        max_kmeans_iterations: int = 15,
        peak_threshold_multiplier: float = 0.2,
        peak_distance_filter: int = 4,
        walker_search_window_ratio: float = 0.35,
        walker_min_search_window: float = 2.0,
        walker_strength_threshold: float = 0.5,
        min_cuts_per_axis: int = 4,
        fallback_target_segments: int = 64,
        max_step_ratio: float = 1.8,
    ):
        config = Config(
            k_colors=k_colors,
            k_seed=k_seed,
            max_kmeans_iterations=max_kmeans_iterations,
            peak_threshold_multiplier=peak_threshold_multiplier,
            peak_distance_filter=peak_distance_filter,
            walker_search_window_ratio=walker_search_window_ratio,
            walker_min_search_window=walker_min_search_window,
            walker_strength_threshold=walker_strength_threshold,
            min_cuts_per_axis=min_cuts_per_axis,
            fallback_target_segments=fallback_target_segments,
            max_step_ratio=max_step_ratio,
        )

        np_batch = tensor_to_numpy_batch(image)
        if np_batch.shape[0] == 0:
            raise PixelSnapperError("Input batch is empty.")
        outputs: List[np.ndarray] = []
        for img_np in np_batch:
            try:
                processed = process_image_array(img_np, config)
                if output_scale > 1:
                    processed = upscale_nearest(processed, output_scale)
                outputs.append(processed)
            except PixelSnapperError as exc:
                raise PixelSnapperError(f"PixelSnapper failed on one frame: {exc}") from exc

        heights = [img.shape[0] for img in outputs]
        widths = [img.shape[1] for img in outputs]
        target_h = max(heights)
        target_w = max(widths)
        if any(h != target_h or w != target_w for h, w in zip(heights, widths)):
            outputs = [pad_image_to_shape(img, target_h, target_w) for img in outputs]

        out_tensor = numpy_batch_to_tensor(np.stack(outputs, axis=0))
        return (out_tensor,)


NODE_CLASS_MAPPINGS = {
    "PixelSnapper": PixelSnapperNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelSnapper": "Sprite Fusion Pixel Snapper",
}

# Friendly name for ComfyUI extension listing
__all__ = ["PixelSnapperNode", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
