from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from ..generation.gen_types import ColorTransform, _COLOR_ALIASES, _COLOR_TRANSFORMS_CANONICAL


def _resolve_color_transform(color_name: str, fallback_hue_deg: Optional[float]) -> ColorTransform:
    name = color_name.strip().lower()
    if not name:
        raise ValueError("Color name must be non-empty.")

    name = _COLOR_ALIASES.get(name, name)

    transform = _COLOR_TRANSFORMS_CANONICAL.get(name)
    if transform is not None:
        return transform

    try:
        hue = float(name)
    except ValueError:
        hue = None

    if hue is not None:
        return ColorTransform(hue_deg=hue, desaturate=False, value_scale=None, value_lift=None)

    if fallback_hue_deg is not None:
        return ColorTransform(hue_deg=fallback_hue_deg, desaturate=False, value_scale=None, value_lift=None)

    raise ValueError(f"Unknown color name '{color_name}'. Use a supported name or a numeric hue in degrees.")


def _pil_to_np_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _np_to_pil(array: np.ndarray, mode: str) -> Image.Image:
    array_u8 = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array_u8, mode=mode)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc == 0, 0.0, delta / np.maximum(maxc, 1e-8))

    h = np.zeros_like(maxc)
    mask = delta > 1e-8
    rc = np.where(mask, (maxc - r) / np.maximum(delta, 1e-8), 0.0)
    gc = np.where(mask, (maxc - g) / np.maximum(delta, 1e-8), 0.0)
    bc = np.where(mask, (maxc - b) / np.maximum(delta, 1e-8), 0.0)

    h = np.where(mask & (r == maxc), (bc - gc), h)
    h = np.where(mask & (g == maxc), 2.0 + (rc - bc), h)
    h = np.where(mask & (b == maxc), 4.0 + (gc - rc), h)
    h = (h / 6.0) % 1.0

    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    i = np.floor(h * 6.0).astype(np.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    r = np.where(i_mod == 0, v, r)
    g = np.where(i_mod == 0, t, g)
    b = np.where(i_mod == 0, p, b)

    r = np.where(i_mod == 1, q, r)
    g = np.where(i_mod == 1, v, g)
    b = np.where(i_mod == 1, p, b)

    r = np.where(i_mod == 2, p, r)
    g = np.where(i_mod == 2, v, g)
    b = np.where(i_mod == 2, t, b)

    r = np.where(i_mod == 3, p, r)
    g = np.where(i_mod == 3, q, g)
    b = np.where(i_mod == 3, v, b)

    r = np.where(i_mod == 4, t, r)
    g = np.where(i_mod == 4, p, g)
    b = np.where(i_mod == 4, v, b)

    r = np.where(i_mod == 5, v, r)
    g = np.where(i_mod == 5, p, g)
    b = np.where(i_mod == 5, q, b)

    return np.stack([r, g, b], axis=-1)


def _apply_color_transform(
    rgb: np.ndarray,
    mask: np.ndarray,
    transform: ColorTransform,
    min_saturation: float,
) -> np.ndarray:
    hsv = _rgb_to_hsv(rgb)
    mask_bool = mask > 0.5
    hsv_h = hsv[..., 0]
    hsv_s = hsv[..., 1]
    hsv_v = hsv[..., 2]

    if transform.hue_deg is not None:
        target_h = (transform.hue_deg % 360.0) / 360.0
        hsv_h = np.where(mask_bool, target_h, hsv_h)
        hsv_s = np.where(mask_bool, np.maximum(hsv_s, min_saturation), hsv_s)

    if transform.desaturate:
        hsv_s = np.where(mask_bool, 0.0, hsv_s)

    if transform.value_scale is not None:
        hsv_v = np.where(mask_bool, np.clip(hsv_v * transform.value_scale, 0.0, 1.0), hsv_v)

    if transform.value_lift is not None:
        hsv_v = np.where(
            mask_bool,
            np.clip(hsv_v * (1.0 - transform.value_lift) + transform.value_lift, 0.0, 1.0),
            hsv_v,
        )

    hsv_out = np.stack([hsv_h, hsv_s, hsv_v], axis=-1)
    rgb_out = _hsv_to_rgb(hsv_out)
    rgb_out = np.where(mask_bool[..., None], rgb_out, rgb)
    return np.clip(rgb_out, 0.0, 1.0)
