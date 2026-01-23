from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def _save_mask(mask: np.ndarray, path) -> None:
    mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(path)


def _compute_alpha(mask: np.ndarray, feather_radius_px: int) -> np.ndarray:
    import cv2

    mask_f = mask.astype(np.float32)
    if feather_radius_px <= 0:
        return mask_f
    k = feather_radius_px * 2 + 1
    alpha = cv2.GaussianBlur(mask_f, (k, k), 0)
    return np.clip(alpha, 0.0, 1.0)


def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    import cv2

    num_labels, labels = cv2.connectedComponents(mask_u8)
    if num_labels <= 1:
        return mask_u8
    areas = np.bincount(labels.flatten())
    areas[0] = 0
    largest = int(np.argmax(areas))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    import cv2

    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask_u8, flood_inv)


def _area_frac_u8(mask_u8: np.ndarray) -> float:
    return float(np.mean(mask_u8 > 0))


def _postprocess_mask(mask: np.ndarray, morph_kernel_px: int) -> np.ndarray:
    import cv2

    max_fill_increase = 0.10
    max_close_increase = 0.10

    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    mask_u8 = _largest_component(mask_u8)
    filled = _fill_holes(mask_u8)
    if _area_frac_u8(filled) <= _area_frac_u8(mask_u8) * (1.0 + max_fill_increase):
        mask_u8 = filled
    if morph_kernel_px > 1:
        kernel = np.ones((morph_kernel_px, morph_kernel_px), np.uint8)
        closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        if _area_frac_u8(closed) <= _area_frac_u8(mask_u8) * (1.0 + max_close_increase):
            mask_u8 = closed
    return (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _postprocess_mask_debug(mask: np.ndarray, morph_kernel_px: int) -> dict[str, np.ndarray]:
    import cv2

    max_fill_increase = 0.10
    max_close_increase = 0.10

    binary = (mask > 0.5).astype(np.uint8) * 255
    largest = _largest_component(binary)
    filled_raw = _fill_holes(largest)
    if _area_frac_u8(filled_raw) <= _area_frac_u8(largest) * (1.0 + max_fill_increase):
        filled = filled_raw
    else:
        filled = largest
    closed_raw = filled
    if morph_kernel_px > 1:
        kernel = np.ones((morph_kernel_px, morph_kernel_px), np.uint8)
        closed_raw = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
    if _area_frac_u8(closed_raw) <= _area_frac_u8(filled) * (1.0 + max_close_increase):
        closed = closed_raw
    else:
        closed = filled
    post = (closed.astype(np.float32) / 255.0).clip(0.0, 1.0)
    return {
        "binary": binary.astype(np.float32) / 255.0,
        "largest": largest.astype(np.float32) / 255.0,
        "filled": filled.astype(np.float32) / 255.0,
        "closed": closed.astype(np.float32) / 255.0,
        "post": post,
    }


def _mask_area_frac(mask: np.ndarray) -> float:
    return float(np.mean(mask > 0.5))


def _select_mask(masks: np.ndarray, center_xy: Tuple[int, int]) -> np.ndarray | None:
    if masks.ndim == 2:
        return masks
    if masks.shape[0] == 0:
        return None
    cx, cy = center_xy
    candidates = []
    for idx in range(masks.shape[0]):
        mask = masks[idx]
        if mask[cy, cx] > 0.5:
            candidates.append((float(mask.sum()), idx))
    if candidates:
        _, best_idx = max(candidates, key=lambda item: item[0])
        return masks[best_idx]
    if masks.shape[0] == 0:
        return None
    areas = [float(mask.sum()) for mask in masks]
    best_idx = int(np.argmax(areas))
    return masks[best_idx]


def _select_mask_by_score(
    masks: np.ndarray,
    scores: np.ndarray | None,
    center_xy: Tuple[int, int],
) -> np.ndarray | None:
    if masks.ndim == 2:
        return masks
    if masks.shape[0] == 0:
        return None
    if scores is None:
        return _select_mask(masks, center_xy)
    scores = np.asarray(scores)
    if scores.ndim == 0 or len(scores) != masks.shape[0]:
        return _select_mask(masks, center_xy)
    cx, cy = center_xy
    center_hits = [masks[i, cy, cx] > 0.5 for i in range(masks.shape[0])]
    if any(center_hits):
        best_idx = int(np.argmax([scores[i] if center_hits[i] else -1.0 for i in range(len(scores))]))
        return masks[best_idx]
    best_idx = int(np.argmax(scores))
    return masks[best_idx]


def _compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask is empty; cannot compute bbox.")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
