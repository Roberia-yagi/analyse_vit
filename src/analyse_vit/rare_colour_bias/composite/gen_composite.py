from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import numpy as np

from analyse_vit.rare_colour_bias.composite.gen_mask import _compute_bbox
from analyse_vit.rare_colour_bias.generation.gen_types import CompositeParams, TransformInfo


def _compute_transform(
    mask: np.ndarray,
    bg_size: Tuple[int, int],
    composite: CompositeParams,
    rng: Optional[random.Random] = None,
    *,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> TransformInfo:
    bg_w, bg_h = bg_size
    if bbox is None:
        bbox = _compute_bbox(mask)
    min_x, min_y, max_x, max_y = bbox
    obj_h = max_y - min_y + 1
    obj_w = max_x - min_x + 1

    scale = 1.0
    scaled_w = max(1, int(round(obj_w)))
    scaled_h = max(1, int(round(obj_h)))

    if composite.placement_mode == "random":
        if rng is None:
            raise ValueError("Random placement requested but RNG is missing.")
        max_x0 = max(bg_w - scaled_w, 0)
        max_y0 = max(bg_h - scaled_h, 0)
        paste_x = rng.randint(0, max_x0) if max_x0 > 0 else 0
        paste_y = rng.randint(0, max_y0) if max_y0 > 0 else 0
        anchor_x = paste_x + scaled_w / 2.0
        anchor_y = paste_y + scaled_h
    elif composite.placement_mode == "center":
        paste_x = max(int(round((bg_w - scaled_w) / 2.0)), 0)
        paste_y = max(int(round((bg_h - scaled_h) / 2.0)), 0)
        anchor_x = paste_x + scaled_w / 2.0
        anchor_y = paste_y + scaled_h / 2.0
    else:
        anchor_x = composite.anchor_x * bg_w
        anchor_y = composite.anchor_y * bg_h
        paste_x = int(round(anchor_x - scaled_w / 2.0))
        paste_y = int(round(anchor_y - scaled_h))
        paste_x = min(max(paste_x, 0), max(bg_w - scaled_w, 0))
        paste_y = min(max(paste_y, 0), max(bg_h - scaled_h, 0))
        anchor_x = paste_x + scaled_w / 2.0
        anchor_y = paste_y + scaled_h

    return TransformInfo(
        bbox=(min_x, min_y, max_x, max_y),
        scale=scale,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        paste_x=paste_x,
        paste_y=paste_y,
        scaled_size=(scaled_w, scaled_h),
    )


def _compute_place_slices(transform: TransformInfo, bg_w: int, bg_h: int) -> Optional[Tuple[slice, slice, slice, slice]]:
    x0 = transform.paste_x
    y0 = transform.paste_y
    scaled_w, scaled_h = transform.scaled_size
    x1 = x0 + scaled_w
    y1 = y0 + scaled_h

    ix0 = max(x0, 0)
    iy0 = max(y0, 0)
    ix1 = min(x1, bg_w)
    iy1 = min(y1, bg_h)

    if ix0 >= ix1 or iy0 >= iy1:
        return None

    sx0 = ix0 - x0
    sy0 = iy0 - y0
    sx1 = sx0 + (ix1 - ix0)
    sy1 = sy0 + (iy1 - iy0)

    return slice(iy0, iy1), slice(ix0, ix1), slice(sy0, sy1), slice(sx0, sx1)


def _crop_and_resize_2d(arr: np.ndarray, bbox: Tuple[int, int, int, int], size: Tuple[int, int]) -> np.ndarray:
    import cv2

    min_x, min_y, max_x, max_y = bbox
    crop = arr[min_y : max_y + 1, min_x : max_x + 1]
    w, h = size
    if w <= 0 or h <= 0:
        raise ValueError("Invalid resize size.")
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def _crop_and_resize_rgb(arr: np.ndarray, bbox: Tuple[int, int, int, int], size: Tuple[int, int]) -> np.ndarray:
    import cv2

    min_x, min_y, max_x, max_y = bbox
    crop = arr[min_y : max_y + 1, min_x : max_x + 1, :]
    w, h = size
    if w <= 0 or h <= 0:
        raise ValueError("Invalid resize size.")
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def _composite_variants(
    bg_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    transform: TransformInfo,
    alpha_full: np.ndarray,
    rgb_variants_full: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    bg_h, bg_w = bg_rgb.shape[:2]
    slices = _compute_place_slices(transform, bg_w, bg_h)
    if slices is None:
        empty_mask = np.zeros((bg_h, bg_w), dtype=np.float32)
        return {k: bg_rgb.copy() for k in rgb_variants_full}, empty_mask

    dst_y, dst_x, src_y, src_x = slices

    alpha_scaled = _crop_and_resize_2d(alpha_full, bbox, transform.scaled_size)
    alpha_canvas = np.zeros((bg_h, bg_w), dtype=np.float32)
    alpha_canvas[dst_y, dst_x] = alpha_scaled[src_y, src_x]

    out: Dict[str, np.ndarray] = {}
    a = alpha_scaled[src_y, src_x][..., None]
    bg_roi = bg_rgb[dst_y, dst_x, :]

    for name, rgb_full in rgb_variants_full.items():
        rgb_scaled = _crop_and_resize_rgb(rgb_full, bbox, transform.scaled_size)
        fg_roi = rgb_scaled[src_y, src_x, :]
        blended = fg_roi * a + bg_roi * (1.0 - a)
        canvas = bg_rgb.copy()
        canvas[dst_y, dst_x, :] = blended
        out[name] = np.clip(canvas, 0.0, 1.0)

    return out, alpha_canvas


def _compute_outside_diff(img_a: np.ndarray, img_b: np.ndarray, mask: np.ndarray) -> float:
    outside = mask <= 0.5
    if not np.any(outside):
        return 0.0
    diff = np.abs(img_a - img_b)
    return float(diff[outside].max())
