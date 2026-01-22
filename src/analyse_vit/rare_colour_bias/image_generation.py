from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class FluxParams:
    pipeline: str
    model_id: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    max_sequence_length: int
    torch_dtype: str
    local_files_only: bool


@dataclass
class SamParams:
    checkpoint: Path
    model_type: str
    prompt_mode: str
    box_ratio_w: float
    box_ratio_h: float
    feather_radius_px: int
    mask_selection_rule: str
    morph_kernel_px: int
    min_area_frac: float
    max_area_frac: float
    lang_sam_box_threshold: float
    lang_sam_text_threshold: float


@dataclass
class CompositeParams:
    anchor_x: float
    anchor_y: float
    target_obj_height_ratio: float
    scale_multiplier: float
    placement_mode: str


@dataclass
class ColorParams:
    target_hue_deg: Optional[float]
    min_saturation: float


@dataclass
class RunConfig:
    output_dir: Path
    background_prompt: str
    base_prompt: str
    paired_prompt: str
    object_name: Optional[str]
    base_prompt_elements_path: Optional[Path]
    paired_prompt_elements_path: Optional[Path]
    base_prompt_elements: Optional[Dict[str, str]]
    paired_prompt_elements: Optional[Dict[str, str]]
    dominant_color: str
    rare_color: str
    seed_bg: int
    seed_real: int
    seed_toy: int
    run_index: int
    seed_offset: int
    flux: FluxParams
    sam: SamParams
    composite: CompositeParams
    color: ColorParams
    device: str
    hf_token: Optional[str]


@dataclass(frozen=True)
class RunDirs:
    root: Path
    inputs: Path
    masks: Path
    outputs: Path
    meta: Path


@dataclass
class TransformInfo:
    bbox: Tuple[int, int, int, int]
    scale: float
    anchor_x: float
    anchor_y: float
    paste_x: int
    paste_y: int
    scaled_size: Tuple[int, int]


@dataclass(frozen=True)
class ColorTransform:
    hue_deg: Optional[float]
    desaturate: bool
    value_scale: Optional[float]
    value_lift: Optional[float]


_COLOR_TRANSFORMS: Dict[str, ColorTransform] = {
    "red": ColorTransform(hue_deg=0.0, desaturate=False, value_scale=None, value_lift=None),
    "orange": ColorTransform(hue_deg=30.0, desaturate=False, value_scale=None, value_lift=None),
    "yellow": ColorTransform(hue_deg=60.0, desaturate=False, value_scale=None, value_lift=None),
    "green": ColorTransform(hue_deg=120.0, desaturate=False, value_scale=None, value_lift=None),
    "cyan": ColorTransform(hue_deg=180.0, desaturate=False, value_scale=None, value_lift=None),
    "blue": ColorTransform(hue_deg=210.0, desaturate=False, value_scale=None, value_lift=None),
    "purple": ColorTransform(hue_deg=270.0, desaturate=False, value_scale=None, value_lift=None),
    "magenta": ColorTransform(hue_deg=300.0, desaturate=False, value_scale=None, value_lift=None),
    "pink": ColorTransform(hue_deg=330.0, desaturate=False, value_scale=None, value_lift=None),
    "brown": ColorTransform(hue_deg=30.0, desaturate=False, value_scale=None, value_lift=None),
    "gray": ColorTransform(hue_deg=None, desaturate=True, value_scale=0.6, value_lift=None),
    "grey": ColorTransform(hue_deg=None, desaturate=True, value_scale=0.6, value_lift=None),
    "black": ColorTransform(hue_deg=None, desaturate=True, value_scale=0.25, value_lift=None),
    "white": ColorTransform(hue_deg=None, desaturate=True, value_scale=None, value_lift=0.25),
}

_PIPELINE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "flux": {"num_inference_steps": 50, "guidance_scale": 3.5},
    "sd3": {"num_inference_steps": 30, "guidance_scale": 5.0},
    "qwen": {"num_inference_steps": 50, "guidance_scale": 4.0},
}

_MODEL_DEFAULTS: Dict[str, Dict[str, Dict[str, float]]] = {
    "flux": {
        "black-forest-labs/FLUX.1-dev": {"num_inference_steps": 50, "guidance_scale": 3.5},
    },
    "sd3": {
        "stabilityai/stable-diffusion-3.5-large": {"num_inference_steps": 30, "guidance_scale": 5.0},
    },
    "qwen": {
        "Qwen/Qwen-Image-2512": {"num_inference_steps": 50, "guidance_scale": 4.0},
    },
}


def _resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _load_prompt_elements(path: Path) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Prompt elements JSON must be a non-empty object: {path}")
    keys: list[str] = []
    groups: list[Tuple[str, ...]] = []
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"Prompt elements JSON keys must be strings: {path}")
        if not isinstance(value, list) or not value:
            raise ValueError(f"Prompt elements JSON values must be non-empty lists: {path}")
        if not all(isinstance(item, str) and item for item in value):
            raise ValueError(f"Prompt elements JSON lists must contain non-empty strings: {path}")
        keys.append(key)
        groups.append(tuple(value))
    return tuple(keys), tuple(groups)


def _count_combinations(groups: Tuple[Tuple[str, ...], ...]) -> int:
    total = 1
    for group in groups:
        total *= len(group)
    return total


def _prompt_from_index(
    keys: Tuple[str, ...],
    groups: Tuple[Tuple[str, ...], ...],
    index: int,
    joiner: str,
) -> Tuple[str, Dict[str, str]]:
    lengths = [len(group) for group in groups]
    selections: list[str] = ["" for _ in lengths]
    for pos in range(len(groups) - 1, -1, -1):
        size = lengths[pos]
        selections[pos] = groups[pos][index % size]
        index //= size
    prompt = joiner.join(selections)
    chosen = {keys[i]: selections[i] for i in range(len(keys))}
    return prompt, chosen


def _generate_prompt_set(
    keys: Tuple[str, ...],
    groups: Tuple[Tuple[str, ...], ...],
    num_runs: int,
    base_seed: int,
    salt: int,
    joiner: str,
    label: str,
) -> Tuple[list[str], list[Dict[str, str]]]:
    total = _count_combinations(groups)
    if num_runs > total:
        raise ValueError(
            f"{label} prompts require {num_runs} unique combinations, but only {total} are available."
        )
    rng = random.Random(_derive_run_seed(base_seed, 0, salt))
    indices = rng.sample(range(total), num_runs)
    prompts: list[str] = []
    selections: list[Dict[str, str]] = []
    for index in indices:
        prompt, chosen = _prompt_from_index(keys, groups, index, joiner)
        prompts.append(prompt)
        selections.append(chosen)
    return prompts, selections


def _find_repo_root(start: Path) -> Optional[Path]:
    for parent in (start, *start.parents):
        if (parent / "AGENTS.md").exists():
            return parent
    return None


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _get_hf_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token:
        return explicit_token
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        token = os.getenv(env_name)
        if token:
            return token
    return None


def _prepare_run_dirs(output_dir: Path) -> RunDirs:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = output_dir / "inputs"
    masks_dir = output_dir / "masks"
    outputs_dir = output_dir / "outputs"
    meta_dir = output_dir / "meta"
    for path in (inputs_dir, masks_dir, outputs_dir, meta_dir):
        path.mkdir(parents=True, exist_ok=True)
    return RunDirs(
        root=output_dir,
        inputs=inputs_dir,
        masks=masks_dir,
        outputs=outputs_dir,
        meta=meta_dir,
    )


def _setup_logger(run_dir: Path, log_dir: Optional[Path] = None) -> logging.Logger:
    log_dir = log_dir or run_dir
    logger = logging.getLogger(f"flux_sam_composite.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _get_version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _get_cv2_version() -> Optional[str]:
    try:
        import cv2
    except Exception:
        return None
    return getattr(cv2, "__version__", None)


def _pil_to_np_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _np_to_pil_rgb(array: np.ndarray) -> Image.Image:
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _np_to_pil_rgba(array: np.ndarray) -> Image.Image:
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGBA")


def _save_image(image: Image.Image, path: Path) -> None:
    image.save(path)


def _load_required_rgb_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing generated image: {path}")
    return Image.open(path).convert("RGB")


def _resolve_run_input(run_dirs: RunDirs, filename: str) -> Path:
    candidate = run_dirs.inputs / filename
    if candidate.exists():
        return candidate
    fallback = run_dirs.root / filename
    if fallback.exists():
        return fallback
    return candidate


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in sig.parameters}


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

    h = np.where((mask) & (r == maxc), (bc - gc), h)
    h = np.where((mask) & (g == maxc), 2.0 + (rc - bc), h)
    h = np.where((mask) & (b == maxc), 4.0 + (gc - rc), h)
    h = (h / 6.0) % 1.0

    hsv = np.stack([h, s, v], axis=-1)
    return hsv


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


def _apply_hue_replace(
    rgb: np.ndarray,
    mask: np.ndarray,
    target_hue_deg: float,
    min_saturation: float,
) -> np.ndarray:
    hsv = _rgb_to_hsv(rgb)
    target_h = (target_hue_deg % 360.0) / 360.0

    mask_bool = mask > 0.5
    hsv_h = hsv[..., 0]
    hsv_s = hsv[..., 1]
    hsv_v = hsv[..., 2]

    hsv_h = np.where(mask_bool, target_h, hsv_h)
    hsv_s = np.where(mask_bool, np.maximum(hsv_s, min_saturation), hsv_s)

    hsv_out = np.stack([hsv_h, hsv_s, hsv_v], axis=-1)
    rgb_out = _hsv_to_rgb(hsv_out)

    rgb_out = np.where(mask_bool[..., None], rgb_out, rgb)
    return np.clip(rgb_out, 0.0, 1.0)


def _resolve_color_transform(color_name: str, fallback_hue_deg: Optional[float]) -> ColorTransform:
    name = color_name.strip().lower()
    if not name:
        raise ValueError("Color name must be non-empty.")
    transform = _COLOR_TRANSFORMS.get(name)
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
    raise ValueError(
        f"Unknown color name '{color_name}'. Use a supported name or a numeric hue in degrees."
    )


def _resolve_generation_defaults(pipeline: str, model_id: str) -> Tuple[int, float]:
    model_defaults = _MODEL_DEFAULTS.get(pipeline, {}).get(model_id)
    if model_defaults is not None:
        return int(model_defaults["num_inference_steps"]), float(model_defaults["guidance_scale"])
    pipe_defaults = _PIPELINE_DEFAULTS.get(pipeline)
    if pipe_defaults is None:
        raise ValueError(f"Unknown pipeline type: {pipeline}")
    return int(pipe_defaults["num_inference_steps"]), float(pipe_defaults["guidance_scale"])


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
    filled = cv2.bitwise_or(mask_u8, flood_inv)
    return filled


def _postprocess_mask(mask: np.ndarray, morph_kernel_px: int) -> np.ndarray:
    import cv2

    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    mask_u8 = _largest_component(mask_u8)
    mask_u8 = _fill_holes(mask_u8)
    if morph_kernel_px > 1:
        kernel = np.ones((morph_kernel_px, morph_kernel_px), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _mask_area_frac(mask: np.ndarray) -> float:
    return float(np.mean(mask > 0.5))


def _select_mask(masks: np.ndarray, center_xy: Tuple[int, int]) -> np.ndarray:
    if masks.ndim == 2:
        return masks
    cx, cy = center_xy
    candidates = []
    for idx in range(masks.shape[0]):
        mask = masks[idx]
        if mask[cy, cx] > 0.5:
            area = mask.sum()
            candidates.append((area, idx))
    if candidates:
        _, best_idx = max(candidates, key=lambda item: item[0])
        return masks[best_idx]
    areas = [mask.sum() for mask in masks]
    best_idx = int(np.argmax(areas))
    return masks[best_idx]


def _compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask is empty; cannot compute bbox.")
    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())
    return min_x, min_y, max_x, max_y


def _compute_transform(
    mask: np.ndarray,
    bg_size: Tuple[int, int],
    composite: CompositeParams,
    rng: Optional[random.Random] = None,
) -> TransformInfo:
    bg_w, bg_h = bg_size
    min_x, min_y, max_x, max_y = _compute_bbox(mask)
    obj_h = max_y - min_y + 1
    obj_w = max_x - min_x + 1

    target_h = composite.target_obj_height_ratio * bg_h
    scale = (target_h / max(obj_h, 1)) * composite.scale_multiplier
    max_scale = min(bg_w / max(obj_w, 1), bg_h / max(obj_h, 1))
    if scale > max_scale:
        scale = max_scale
    scaled_w = max(1, int(round(obj_w * scale)))
    scaled_h = max(1, int(round(obj_h * scale)))

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


def _apply_transform_and_composite(
    bg_rgb: np.ndarray,
    rgba: np.ndarray,
    transform: TransformInfo,
) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    min_x, min_y, max_x, max_y = transform.bbox
    crop = rgba[min_y : max_y + 1, min_x : max_x + 1, :]

    scaled_w, scaled_h = transform.scaled_size
    if scaled_w <= 0 or scaled_h <= 0:
        raise ValueError("Invalid scaled size.")

    rgb = crop[..., :3]
    alpha = crop[..., 3]

    rgb_scaled = cv2.resize(rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
    alpha_scaled = cv2.resize(alpha, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)

    bg_h, bg_w = bg_rgb.shape[:2]
    fg_canvas_rgb = np.zeros_like(bg_rgb)
    fg_canvas_a = np.zeros((bg_h, bg_w), dtype=np.float32)

    x0 = transform.paste_x
    y0 = transform.paste_y
    x1 = x0 + scaled_w
    y1 = y0 + scaled_h

    ix0 = max(x0, 0)
    iy0 = max(y0, 0)
    ix1 = min(x1, bg_w)
    iy1 = min(y1, bg_h)

    if ix0 >= ix1 or iy0 >= iy1:
        return bg_rgb.copy(), fg_canvas_a

    sx0 = ix0 - x0
    sy0 = iy0 - y0
    sx1 = sx0 + (ix1 - ix0)
    sy1 = sy0 + (iy1 - iy0)

    fg_canvas_rgb[iy0:iy1, ix0:ix1, :] = rgb_scaled[sy0:sy1, sx0:sx1, :]
    fg_canvas_a[iy0:iy1, ix0:ix1] = alpha_scaled[sy0:sy1, sx0:sx1]

    fg_a = fg_canvas_a[..., None]
    out_rgb = fg_canvas_rgb * fg_a + bg_rgb * (1.0 - fg_a)

    return out_rgb, fg_canvas_a


def _compute_outside_diff(img_a: np.ndarray, img_b: np.ndarray, mask: np.ndarray) -> float:
    outside = mask <= 0.5
    if not np.any(outside):
        return 0.0
    diff = np.abs(img_a - img_b)
    return float(diff[outside].max())


def _save_mask(mask: np.ndarray, path: Path) -> None:
    mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(path)


def _resolve_torch_dtype(name: str):
    import torch

    if name == "bf8":
        for attr in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"):
            dtype = getattr(torch, attr, None)
            if dtype is None:
                continue
            orig_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(dtype)
            except Exception:
                try:
                    torch.set_default_dtype(orig_dtype)
                except Exception:
                    pass
                continue
            else:
                try:
                    torch.set_default_dtype(orig_dtype)
                except Exception:
                    pass
            try:
                torch.empty(1, dtype=dtype)
            except Exception:
                continue
            return dtype
        logging.getLogger(__name__).warning(
            "bf8 requested but float8 is unsupported in this torch build; falling back to bfloat16."
        )
        return torch.bfloat16
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(name, torch.bfloat16)


def _release_torch_cuda() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _load_text2image_pipeline(
    pipeline_type: str,
    model_id: str,
    torch_dtype: str,
    device: str,
    token: Optional[str],
    local_files_only: bool,
):
    import torch

    if pipeline_type == "flux":
        from diffusers import FluxPipeline as PipelineClass
    elif pipeline_type == "sd3":
        from diffusers import StableDiffusion3Pipeline as PipelineClass
    elif pipeline_type == "qwen":
        from diffusers import DiffusionPipeline as PipelineClass
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    dtype = _resolve_torch_dtype(torch_dtype)
    if device == "cpu":
        dtype = torch.float32

    sig = inspect.signature(PipelineClass.from_pretrained)
    kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if "local_files_only" in sig.parameters:
        kwargs["local_files_only"] = local_files_only
    if token:
        if "token" in sig.parameters:
            kwargs["token"] = token
        elif "use_auth_token" in sig.parameters:
            kwargs["use_auth_token"] = token
        else:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    try:
        pipe = PipelineClass.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        hints = [f"Failed to load pipeline ({pipeline_type})."]
        hints.append(f"Original error: {type(exc).__name__}: {exc}")
        if token is None:
            hints.append(
                "If the model is gated, pass --hf-token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
            )
        if local_files_only:
            hints.append(
                "local_files_only is enabled; ensure the model is cached or omit --flux-local-files-only."
            )
        if torch_dtype == "bf8":
            hints.append("bf8 was requested; if float8 is unsupported, try --torch-dtype bfloat16.")
        hints.append("You can also point --model-id to a local path or a public model.")
        raise RuntimeError(" ".join(hints)) from exc
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _generate_flux_image(
    pipe,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    max_sequence_length: int,
    device: str,
) -> Image.Image:
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    call_kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "max_sequence_length": max_sequence_length,
        "generator": generator,
    }
    sig = inspect.signature(pipe.__call__)
    if "true_cfg_scale" in sig.parameters:
        call_kwargs["true_cfg_scale"] = guidance_scale
        if guidance_scale > 1.0 and "negative_prompt" in sig.parameters:
            call_kwargs.setdefault("negative_prompt", "")
    call_kwargs = _filter_kwargs_for_callable(pipe.__call__, call_kwargs)
    result = pipe(**call_kwargs)
    return result.images[0]


def _generate_flux_images_batch(
    pipe,
    prompts: Iterable[str],
    seeds: Iterable[int],
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    max_sequence_length: int,
    device: str,
) -> list[Image.Image]:
    import torch

    prompt_list = list(prompts)
    seed_list = list(seeds)
    if len(prompt_list) != len(seed_list):
        raise ValueError("Prompt and seed counts must match for batch generation.")
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seed_list]
    call_kwargs = {
        "prompt": prompt_list,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "max_sequence_length": max_sequence_length,
        "generator": generators,
    }
    sig = inspect.signature(pipe.__call__)
    if "true_cfg_scale" in sig.parameters:
        call_kwargs["true_cfg_scale"] = guidance_scale
        if guidance_scale > 1.0 and "negative_prompt" in sig.parameters:
            call_kwargs.setdefault("negative_prompt", [""] * len(prompt_list))
    call_kwargs = _filter_kwargs_for_callable(pipe.__call__, call_kwargs)
    result = pipe(**call_kwargs)
    images = list(result.images)
    if len(images) != len(prompt_list):
        raise RuntimeError("Batch generation returned an unexpected number of images.")
    return images


def _load_sam_predictor(checkpoint: Path, model_type: str, device: str):
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamPredictor(sam)


def _load_lang_sam_model(device: str):
    from lang_sam import LangSAM

    sig = inspect.signature(LangSAM)
    if "device" in sig.parameters:
        return LangSAM(device=device)
    return LangSAM()


def _coerce_numpy(array_like) -> Optional[np.ndarray]:
    if array_like is None:
        return None
    if hasattr(array_like, "detach"):
        array_like = array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _select_lang_sam_mask(
    masks: np.ndarray,
    scores: Optional[np.ndarray],
    center_xy: Tuple[int, int],
) -> Optional[np.ndarray]:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        return None

    if scores is not None:
        scores = np.asarray(scores)
        if scores.ndim == 0:
            scores = None
    if scores is not None and len(scores) == masks.shape[0]:
        cx, cy = center_xy
        center_hits = [masks[i, cy, cx] > 0.5 for i in range(masks.shape[0])]
        if any(center_hits):
            idx = int(np.argmax([scores[i] if center_hits[i] else -1.0 for i in range(len(scores))]))
            return masks[idx]
        idx = int(np.argmax(scores))
        return masks[idx]

    return _select_mask(masks, center_xy)


def _predict_mask_lang_sam(
    model,
    image: Image.Image,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Optional[np.ndarray]:
    if not hasattr(model, "predict"):
        return None
    sig = inspect.signature(model.predict)
    kwargs: dict[str, Any] = {}
    if "box_threshold" in sig.parameters:
        kwargs["box_threshold"] = box_threshold
    if "text_threshold" in sig.parameters:
        kwargs["text_threshold"] = text_threshold
    images_arg: Any = image
    texts_arg: Any = text_prompt
    if "images_pil" in sig.parameters or "texts_prompt" in sig.parameters:
        images_arg = [image]
        texts_arg = [text_prompt]
    output = model.predict(images_arg, texts_arg, **kwargs)

    masks = None
    scores = None
    if isinstance(output, dict):
        masks = output.get("masks") if output.get("masks") is not None else output.get("mask")
        scores = output.get("scores") if output.get("scores") is not None else output.get("logits")
    elif isinstance(output, list) and output and isinstance(output[0], dict):
        first = output[0]
        masks = first.get("masks") if first.get("masks") is not None else first.get("mask")
        scores = (
            first.get("mask_scores")
            if first.get("mask_scores") is not None
            else first.get("scores")
            if first.get("scores") is not None
            else first.get("logits")
        )
    elif isinstance(output, (list, tuple)):
        for item in output:
            if masks is None:
                arr = _coerce_numpy(item)
                if arr is not None and arr.ndim >= 2:
                    masks = arr
            if scores is None and isinstance(item, (list, tuple)):
                scores = item

    masks_np = _coerce_numpy(masks)
    scores_np = _coerce_numpy(scores)
    if masks_np is None:
        return None
    if masks_np.ndim == 2 and masks_np.shape != (image.height, image.width):
        return None
    if masks_np.ndim >= 3 and masks_np.shape[-2:] != (image.height, image.width):
        return None
    center_xy = (image.width // 2, image.height // 2)
    return _select_lang_sam_mask(masks_np, scores_np, center_xy)


def _predict_mask(
    predictor,
    image: Image.Image,
    prompt_mode: str,
    box_ratio_w: float,
    box_ratio_h: float,
) -> np.ndarray:
    image_np = np.asarray(image.convert("RGB"))
    h, w, _ = image_np.shape
    predictor.set_image(image_np)
    cx = int(round(w / 2.0))
    cy = int(round(h / 2.0))

    if prompt_mode == "points":
        point_coords = np.array(
            [
                [cx, cy],
                [0, 0],
                [w - 1, 0],
                [0, h - 1],
                [w - 1, h - 1],
            ],
            dtype=np.float32,
        )
        point_labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
    elif prompt_mode == "box":
        bw = int(round(w * box_ratio_w))
        bh = int(round(h * box_ratio_h))
        x0 = int(round((w - bw) / 2.0))
        y0 = int(round((h - bh) / 2.0))
        x1 = x0 + bw
        y1 = y0 + bh
        box = np.array([x0, y0, x1, y1])
        masks, _, _ = predictor.predict(
            box=box[None, :],
            multimask_output=True,
        )
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    return _select_mask(masks, (cx, cy))


def _run_generation_stage(config: RunConfig, pipe=None):
    output_dir = config.output_dir
    run_dirs = _prepare_run_dirs(output_dir)
    logger = _setup_logger(output_dir, run_dirs.meta)

    logger.info("Starting run %03d (seed offset=%d).", config.run_index, config.seed_offset)

    if pipe is None:
        logger.info("Loading text-to-image pipeline (%s).", config.flux.pipeline)
        pipe = _load_text2image_pipeline(
            config.flux.pipeline,
            config.flux.model_id,
            config.flux.torch_dtype,
            config.device,
            config.hf_token,
            config.flux.local_files_only,
        )

    logger.info("Generating background image.")
    bg_image = _generate_flux_image(
        pipe,
        config.background_prompt,
        config.seed_bg,
        config.flux.width,
        config.flux.height,
        config.flux.num_inference_steps,
        config.flux.guidance_scale,
        config.flux.max_sequence_length,
        config.device,
    )
    _save_image(bg_image, run_dirs.inputs / "bg.png")

    prompt_real = config.base_prompt
    prompt_toy = config.paired_prompt
    logger.info("Using base prompt: %s", prompt_real)
    logger.info("Using paired prompt: %s", prompt_toy)

    logger.info("Generating anchor real object image.")
    real_anchor_image = _generate_flux_image(
        pipe,
        prompt_real,
        config.seed_real,
        config.flux.width,
        config.flux.height,
        config.flux.num_inference_steps,
        config.flux.guidance_scale,
        config.flux.max_sequence_length,
        config.device,
    )
    _save_image(real_anchor_image, run_dirs.inputs / "real_anchor.png")

    logger.info("Generating anchor toy object image.")
    toy_anchor_image = _generate_flux_image(
        pipe,
        prompt_toy,
        config.seed_toy,
        config.flux.width,
        config.flux.height,
        config.flux.num_inference_steps,
        config.flux.guidance_scale,
        config.flux.max_sequence_length,
        config.device,
    )
    _save_image(toy_anchor_image, run_dirs.inputs / "toy_anchor.png")

    return pipe


def _ensure_batch_compatible(run_configs: list[RunConfig]) -> RunConfig:
    if not run_configs:
        raise ValueError("No run configs provided for batch generation.")
    base = run_configs[0]
    for cfg in run_configs[1:]:
        if cfg.device != base.device:
            raise ValueError("Batch generation requires the same device for all runs.")
        if cfg.flux != base.flux:
            raise ValueError("Batch generation requires identical flux parameters for all runs.")
    return base


def _run_generation_stage_batch(run_configs: list[RunConfig], pipe=None):
    base = _ensure_batch_compatible(run_configs)
    if pipe is None:
        base_dirs = _prepare_run_dirs(base.output_dir)
        logger = _setup_logger(base.output_dir, base_dirs.meta)
        logger.info("Loading text-to-image pipeline (%s).", base.flux.pipeline)
        pipe = _load_text2image_pipeline(
            base.flux.pipeline,
            base.flux.model_id,
            base.flux.torch_dtype,
            base.device,
            base.hf_token,
            base.flux.local_files_only,
        )

    run_dirs_list: list[RunDirs] = []
    for cfg in run_configs:
        run_dirs = _prepare_run_dirs(cfg.output_dir)
        run_dirs_list.append(run_dirs)
        logger = _setup_logger(cfg.output_dir, run_dirs.meta)
        logger.info(
            "Starting run %03d (seed offset=%d) using batched generation.",
            cfg.run_index,
            cfg.seed_offset,
        )

    bg_prompts = [cfg.background_prompt for cfg in run_configs]
    bg_seeds = [cfg.seed_bg for cfg in run_configs]
    bg_images = _generate_flux_images_batch(
        pipe,
        bg_prompts,
        bg_seeds,
        base.flux.width,
        base.flux.height,
        base.flux.num_inference_steps,
        base.flux.guidance_scale,
        base.flux.max_sequence_length,
        base.device,
    )
    for run_dirs, image in zip(run_dirs_list, bg_images):
        _save_image(image, run_dirs.inputs / "bg.png")

    real_prompts = [cfg.base_prompt for cfg in run_configs]
    real_seeds = [cfg.seed_real for cfg in run_configs]
    real_images = _generate_flux_images_batch(
        pipe,
        real_prompts,
        real_seeds,
        base.flux.width,
        base.flux.height,
        base.flux.num_inference_steps,
        base.flux.guidance_scale,
        base.flux.max_sequence_length,
        base.device,
    )
    for run_dirs, image in zip(run_dirs_list, real_images):
        _save_image(image, run_dirs.inputs / "real_anchor.png")

    toy_prompts = [cfg.paired_prompt for cfg in run_configs]
    toy_seeds = [cfg.seed_toy for cfg in run_configs]
    toy_images = _generate_flux_images_batch(
        pipe,
        toy_prompts,
        toy_seeds,
        base.flux.width,
        base.flux.height,
        base.flux.num_inference_steps,
        base.flux.guidance_scale,
        base.flux.max_sequence_length,
        base.device,
    )
    for run_dirs, image in zip(run_dirs_list, toy_images):
        _save_image(image, run_dirs.inputs / "toy_anchor.png")

    return pipe


def _run_sam_stage(config: RunConfig, predictor=None, lang_sam_model=None) -> None:
    output_dir = config.output_dir
    run_dirs = _prepare_run_dirs(output_dir)
    logger = _setup_logger(output_dir, run_dirs.meta)

    logger.info("Starting SAM/composite stage for run %03d.", config.run_index)

    bg_image = _load_required_rgb_image(_resolve_run_input(run_dirs, "bg.png"))
    real_anchor_image = _load_required_rgb_image(_resolve_run_input(run_dirs, "real_anchor.png"))
    toy_anchor_image = _load_required_rgb_image(_resolve_run_input(run_dirs, "toy_anchor.png"))

    prompt_real = config.base_prompt
    prompt_toy = config.paired_prompt
    logger.info("Using base prompt: %s", prompt_real)
    logger.info("Using paired prompt: %s", prompt_toy)
    if config.sam.prompt_mode == "grounding":
        if not config.object_name:
            raise ValueError("--object-name is required when --sam-prompt-mode grounding is used.")
        logger.info("Using object name for grounding SAM: %s", config.object_name)

    dominant_transform = _resolve_color_transform(config.dominant_color, None)
    rare_transform = _resolve_color_transform(config.rare_color, config.color.target_hue_deg)

    if config.sam.prompt_mode == "grounding":
        if lang_sam_model is None:
            logger.info("Loading LangSAM model for text-guided masks.")
            lang_sam_model = _load_lang_sam_model(config.device)
    elif predictor is None:
        logger.info("Loading SAM predictor.")
        predictor = _load_sam_predictor(config.sam.checkpoint, config.sam.model_type, config.device)

    logger.info("Predicting real mask.")
    if config.sam.prompt_mode == "grounding":
        real_mask_raw = _predict_mask_lang_sam(
            lang_sam_model,
            real_anchor_image,
            config.object_name,
            config.sam.lang_sam_box_threshold,
            config.sam.lang_sam_text_threshold,
        )
        if real_mask_raw is None:
            raise RuntimeError("LangSAM did not return a mask for the real prompt.")
    else:
        real_mask_raw = _predict_mask(
            predictor,
            real_anchor_image,
            config.sam.prompt_mode,
            config.sam.box_ratio_w,
            config.sam.box_ratio_h,
        )
    real_mask = _postprocess_mask(real_mask_raw, config.sam.morph_kernel_px)

    logger.info("Predicting toy mask.")
    if config.sam.prompt_mode == "grounding":
        toy_mask_raw = _predict_mask_lang_sam(
            lang_sam_model,
            toy_anchor_image,
            config.object_name,
            config.sam.lang_sam_box_threshold,
            config.sam.lang_sam_text_threshold,
        )
        if toy_mask_raw is None:
            raise RuntimeError("LangSAM did not return a mask for the toy prompt.")
    else:
        toy_mask_raw = _predict_mask(
            predictor,
            toy_anchor_image,
            config.sam.prompt_mode,
            config.sam.box_ratio_w,
            config.sam.box_ratio_h,
        )
    toy_mask = _postprocess_mask(toy_mask_raw, config.sam.morph_kernel_px)

    _save_mask(real_mask, run_dirs.masks / "real_mask.png")
    _save_mask(toy_mask, run_dirs.masks / "toy_mask.png")

    sanity_checks: Dict[str, Any] = {}
    real_area = _mask_area_frac(real_mask)
    toy_area = _mask_area_frac(toy_mask)
    sanity_checks["real_mask_area_frac"] = real_area
    sanity_checks["toy_mask_area_frac"] = toy_area

    if real_area < config.sam.min_area_frac or real_area > config.sam.max_area_frac:
        logger.error("Real mask area out of bounds: %.4f", real_area)
        sanity_checks["real_mask_area_ok"] = False
    else:
        sanity_checks["real_mask_area_ok"] = True

    if toy_area < config.sam.min_area_frac or toy_area > config.sam.max_area_frac:
        logger.error("Toy mask area out of bounds: %.4f", toy_area)
        sanity_checks["toy_mask_area_ok"] = False
    else:
        sanity_checks["toy_mask_area_ok"] = True

    if real_area == 0.0 or toy_area == 0.0:
        raise RuntimeError("Mask is empty; aborting run.")

    logger.info("Computing alpha mattes.")
    real_alpha = _compute_alpha(real_mask, config.sam.feather_radius_px)
    toy_alpha = _compute_alpha(toy_mask, config.sam.feather_radius_px)

    real_anchor_rgb = _pil_to_np_rgb(real_anchor_image)
    toy_anchor_rgb = _pil_to_np_rgb(toy_anchor_image)

    logger.info("Applying dominant color transformation.")
    real_dom_rgb = _apply_color_transform(
        real_anchor_rgb,
        real_mask,
        dominant_transform,
        config.color.min_saturation,
    )
    toy_dom_rgb = _apply_color_transform(
        toy_anchor_rgb,
        toy_mask,
        dominant_transform,
        config.color.min_saturation,
    )

    _save_image(_np_to_pil_rgb(real_dom_rgb), run_dirs.outputs / "real_dom.png")
    _save_image(_np_to_pil_rgb(toy_dom_rgb), run_dirs.outputs / "toy_dom.png")

    logger.info("Applying rare color transformation.")
    real_rare_rgb = _apply_color_transform(
        real_anchor_rgb,
        real_mask,
        rare_transform,
        config.color.min_saturation,
    )
    toy_rare_rgb = _apply_color_transform(
        toy_anchor_rgb,
        toy_mask,
        rare_transform,
        config.color.min_saturation,
    )

    _save_image(_np_to_pil_rgb(real_rare_rgb), run_dirs.outputs / "real_rare.png")
    _save_image(_np_to_pil_rgb(toy_rare_rgb), run_dirs.outputs / "toy_rare.png")

    real_dom_rgba = np.dstack([real_dom_rgb, real_alpha])
    real_rare_rgba = np.dstack([real_rare_rgb, real_alpha])
    toy_dom_rgba = np.dstack([toy_dom_rgb, toy_alpha])
    toy_rare_rgba = np.dstack([toy_rare_rgb, toy_alpha])

    _save_image(_np_to_pil_rgba(real_dom_rgba), run_dirs.outputs / "real_dom_rgba.png")
    _save_image(_np_to_pil_rgba(real_rare_rgba), run_dirs.outputs / "real_rare_rgba.png")
    _save_image(_np_to_pil_rgba(toy_dom_rgba), run_dirs.outputs / "toy_dom_rgba.png")
    _save_image(_np_to_pil_rgba(toy_rare_rgba), run_dirs.outputs / "toy_rare_rgba.png")

    dom_rare_diff_real = _compute_outside_diff(real_dom_rgb, real_rare_rgb, real_mask)
    dom_rare_diff_toy = _compute_outside_diff(toy_dom_rgb, toy_rare_rgb, toy_mask)
    sanity_checks["real_dom_rare_outside_max_diff"] = dom_rare_diff_real
    sanity_checks["toy_dom_rare_outside_max_diff"] = dom_rare_diff_toy

    if dom_rare_diff_real != 0.0:
        logger.error("Real dom/rare differ outside mask: %.6f", dom_rare_diff_real)
    if dom_rare_diff_toy != 0.0:
        logger.error("Toy dom/rare differ outside mask: %.6f", dom_rare_diff_toy)

    bg_rgb = _pil_to_np_rgb(bg_image)

    logger.info("Compositing real images.")
    real_place_rng = random.Random(_derive_run_seed(config.seed_real, config.run_index, 0xC0A7E1))
    real_transform = _compute_transform(
        real_mask,
        (bg_rgb.shape[1], bg_rgb.shape[0]),
        config.composite,
        rng=real_place_rng,
    )
    scene_real_dom, real_obj_mask = _apply_transform_and_composite(bg_rgb, real_dom_rgba, real_transform)
    scene_real_rare, real_obj_mask_rare = _apply_transform_and_composite(bg_rgb, real_rare_rgba, real_transform)

    logger.info("Compositing toy images.")
    toy_place_rng = random.Random(_derive_run_seed(config.seed_toy, config.run_index, 0xC0A7E2))
    toy_transform = _compute_transform(
        toy_mask,
        (bg_rgb.shape[1], bg_rgb.shape[0]),
        config.composite,
        rng=toy_place_rng,
    )
    scene_toy_dom, toy_obj_mask = _apply_transform_and_composite(bg_rgb, toy_dom_rgba, toy_transform)
    scene_toy_rare, toy_obj_mask_rare = _apply_transform_and_composite(bg_rgb, toy_rare_rgba, toy_transform)

    _save_image(_np_to_pil_rgb(scene_real_dom), run_dirs.outputs / "scene_real_dom.png")
    _save_image(_np_to_pil_rgb(scene_real_rare), run_dirs.outputs / "scene_real_rare.png")
    _save_image(_np_to_pil_rgb(scene_toy_dom), run_dirs.outputs / "scene_toy_dom.png")
    _save_image(_np_to_pil_rgb(scene_toy_rare), run_dirs.outputs / "scene_toy_rare.png")

    scene_diff_real = _compute_outside_diff(scene_real_dom, scene_real_rare, real_obj_mask)
    scene_diff_toy = _compute_outside_diff(scene_toy_dom, scene_toy_rare, toy_obj_mask)
    sanity_checks["scene_real_outside_max_diff"] = scene_diff_real
    sanity_checks["scene_toy_outside_max_diff"] = scene_diff_toy
    sanity_checks["scene_real_mask_max_diff"] = float(np.max(np.abs(real_obj_mask - real_obj_mask_rare)))
    sanity_checks["scene_toy_mask_max_diff"] = float(np.max(np.abs(toy_obj_mask - toy_obj_mask_rare)))

    if scene_diff_real != 0.0:
        logger.error("Scene real dom/rare differ outside object: %.6f", scene_diff_real)
    if scene_diff_toy != 0.0:
        logger.error("Scene toy dom/rare differ outside object: %.6f", scene_diff_toy)
    if sanity_checks["scene_real_mask_max_diff"] != 0.0:
        logger.error("Scene real object masks differ between dom/rare: %.6f", sanity_checks["scene_real_mask_max_diff"])
    if sanity_checks["scene_toy_mask_max_diff"] != 0.0:
        logger.error("Scene toy object masks differ between dom/rare: %.6f", sanity_checks["scene_toy_mask_max_diff"])

    meta: Dict[str, Any] = {
        "run_index": config.run_index,
        "seed_offset": config.seed_offset,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "background_prompt": config.background_prompt,
        "base_prompt": config.base_prompt,
        "paired_prompt": config.paired_prompt,
        "object_name": config.object_name,
        "base_prompt_elements_json": str(config.base_prompt_elements_path)
        if config.base_prompt_elements_path
        else None,
        "paired_prompt_elements_json": str(config.paired_prompt_elements_path)
        if config.paired_prompt_elements_path
        else None,
        "prompt_elements": {
            "base_prompt": config.base_prompt_elements,
            "paired_prompt": config.paired_prompt_elements,
        },
        "dominant_color": config.dominant_color,
        "rare_color": config.rare_color,
        "prompts": {
            "real_dom": prompt_real,
            "toy_dom": prompt_toy,
        },
        "seeds": {
            "seed_bg": config.seed_bg,
            "seed_real": config.seed_real,
            "seed_toy": config.seed_toy,
        },
        "flux": asdict(config.flux),
        "sam": {
            "checkpoint": str(config.sam.checkpoint),
            "model_type": config.sam.model_type,
            "prompt_mode": config.sam.prompt_mode,
            "box_ratio_w": config.sam.box_ratio_w,
            "box_ratio_h": config.sam.box_ratio_h,
            "mask_selection_rule": config.sam.mask_selection_rule,
            "feather_radius_px": config.sam.feather_radius_px,
            "morph_kernel_px": config.sam.morph_kernel_px,
            "min_area_frac": config.sam.min_area_frac,
            "max_area_frac": config.sam.max_area_frac,
            "lang_sam_box_threshold": config.sam.lang_sam_box_threshold,
            "lang_sam_text_threshold": config.sam.lang_sam_text_threshold,
        },
        "color": asdict(config.color),
        "color_transforms": {
            "dominant": asdict(dominant_transform),
            "rare": asdict(rare_transform),
            "min_saturation": config.color.min_saturation,
        },
        "composite": asdict(config.composite),
        "transforms": {
            "real": asdict(real_transform),
            "toy": asdict(toy_transform),
        },
        "sanity_checks": sanity_checks,
        "library_versions": {
            "torch": _get_version("torch"),
            "diffusers": _get_version("diffusers"),
            "segment_anything": _get_version("segment-anything"),
            "segment_anything_alt": _get_version("segment-anything-py"),
            "opencv": _get_cv2_version() or _get_version("opencv-python"),
            "numpy": _get_version("numpy"),
            "Pillow": _get_version("Pillow"),
        },
    }

    with (run_dirs.meta / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)

    logger.info("Run complete.")


def run_pipeline(config: RunConfig, pipe=None, predictor=None, lang_sam_model=None) -> None:
    pipe = _run_generation_stage(config, pipe=pipe)
    if config.sam.prompt_mode == "grounding":
        _run_sam_stage(config, predictor=None, lang_sam_model=lang_sam_model)
    else:
        if predictor is None:
            predictor = _load_sam_predictor(config.sam.checkpoint, config.sam.model_type, config.device)
        _run_sam_stage(config, predictor=predictor)


def _create_timestamp_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = base_dir / timestamp
    suffix = 1
    while run_root.exists():
        run_root = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 30) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 31) & 0xFFFFFFFFFFFFFFFF
    return x


def _derive_run_seed(base_seed: int, run_index: int, salt: int) -> int:
    mixed = (base_seed & 0xFFFFFFFFFFFFFFFF) ^ ((run_index + 1) * 0x9E3779B97F4A7C15) ^ salt
    return int(_splitmix64(mixed) & 0x7FFFFFFFFFFFFFFF)


def _parse_args(
    argv: Optional[Iterable[str]] = None,
) -> Tuple[RunConfig, int, int, Optional[int], Optional[Path]]:
    parser = argparse.ArgumentParser(description="Text-to-image + SAM composite pipeline")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--background-prompt", required=True)
    parser.add_argument("--base-prompt", default=None)
    parser.add_argument("--paired-prompt", default=None)
    parser.add_argument(
        "--object-name",
        default=None,
        help="Object name to use for grounding SAM masks (required for grounding prompt mode).",
    )
    parser.add_argument(
        "--base-prompt-elements-json",
        default=None,
        help="JSON file of prompt elements used to build a unique base prompt per run.",
    )
    parser.add_argument(
        "--paired-prompt-elements-json",
        default=None,
        help="JSON file of prompt elements used to build a unique paired prompt per run.",
    )
    parser.add_argument("--dominant-color", required=True)
    parser.add_argument("--rare-color", required=True)
    parser.add_argument("--seed-bg", type=int, required=True)
    parser.add_argument("--seed-real", type=int, required=True)
    parser.add_argument("--seed-toy", type=int, required=True)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--run-start", type=int, default=0)
    parser.add_argument("--run-count", type=int, default=None)
    parser.add_argument(
        "--run-root",
        default=None,
        help="Optional run root directory (skips timestamp creation).",
    )

    parser.add_argument("--pipeline", default="flux", choices=["flux", "sd3", "qwen"])
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--flux-model-id", default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--torch-dtype",
        default=None,
        choices=["bfloat16", "float16", "float32", "bf8"],
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token (or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN).",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Optional .env file to load (defaults to repo root .env if present).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load weights only from local cache (no network).",
    )
    parser.add_argument(
        "--flux-local-files-only",
        action="store_true",
        help="(deprecated) Use --local-files-only instead.",
    )

    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument(
        "--sam-prompt-mode",
        default="grounding",
        choices=["points", "box", "grounding"],
    )
    parser.add_argument("--sam-box-ratio-w", type=float, default=0.8)
    parser.add_argument("--sam-box-ratio-h", type=float, default=0.85)
    parser.add_argument("--feather-radius-px", type=int, default=3)
    parser.add_argument("--mask-selection-rule", default="center_included_max_area")
    parser.add_argument("--morph-kernel-px", type=int, default=3)
    parser.add_argument("--mask-min-area-frac", type=float, default=0.01)
    parser.add_argument("--mask-max-area-frac", type=float, default=0.9)
    parser.add_argument("--lang-sam-box-threshold", type=float, default=0.3)
    parser.add_argument("--lang-sam-text-threshold", type=float, default=0.25)

    parser.add_argument("--anchor-x", type=float, default=0.5)
    parser.add_argument("--anchor-y", type=float, default=0.9)
    parser.add_argument("--target-obj-height-ratio", type=float, default=0.65)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    parser.add_argument("--placement-mode", default="center", choices=["random", "anchor", "center"])

    parser.add_argument("--target-hue-deg", type=float, default=None)
    parser.add_argument("--min-saturation", type=float, default=0.25)

    parser.add_argument("--device", default="cuda")

    args = parser.parse_args(argv)

    output_dir = _resolve_path(args.output_dir)
    run_root = _resolve_path(args.run_root) if args.run_root else None
    sam_checkpoint = _resolve_path(args.sam_checkpoint)
    base_prompt_elements_path = (
        _resolve_path(args.base_prompt_elements_json) if args.base_prompt_elements_json else None
    )
    paired_prompt_elements_path = (
        _resolve_path(args.paired_prompt_elements_json) if args.paired_prompt_elements_json else None
    )
    base_prompt = args.base_prompt
    paired_prompt = args.paired_prompt
    if base_prompt_elements_path is None and base_prompt is None:
        raise ValueError("--base-prompt or --base-prompt-elements-json must be provided.")
    if paired_prompt_elements_path is None and paired_prompt is None:
        raise ValueError("--paired-prompt or --paired-prompt-elements-json must be provided.")
    if base_prompt is None:
        base_prompt = ""
    if paired_prompt is None:
        paired_prompt = ""
    if not (0.0 <= args.anchor_x <= 1.0 and 0.0 <= args.anchor_y <= 1.0):
        raise ValueError("--anchor-x/--anchor-y must be in [0, 1].")
    if args.target_obj_height_ratio <= 0.0:
        raise ValueError("--target-obj-height-ratio must be > 0.")
    if args.scale_multiplier <= 0.0:
        raise ValueError("--scale-multiplier must be > 0.")
    if args.mask_selection_rule != "center_included_max_area":
        raise ValueError("--mask-selection-rule must be center_included_max_area for determinism.")
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be > 0.")
    if args.run_start < 0:
        raise ValueError("--run-start must be >= 0.")
    if args.run_start >= args.num_runs:
        raise ValueError("--run-start must be < --num-runs.")
    if args.run_count is not None and args.run_count <= 0:
        raise ValueError("--run-count must be > 0 when provided.")
    if args.run_count is not None and args.run_start + args.run_count > args.num_runs:
        raise ValueError("--run-start + --run-count must be <= --num-runs.")
    if args.model_id and args.flux_model_id and args.model_id != args.flux_model_id:
        raise ValueError("--model-id and --flux-model-id must match if both are set.")
    if args.sam_prompt_mode == "grounding" and not args.object_name:
        raise ValueError("--object-name is required when --sam-prompt-mode grounding is used.")
    if args.target_hue_deg is None:
        _resolve_color_transform(args.rare_color, None)

    env_file = _resolve_path(args.env_file) if args.env_file else None
    if env_file is None:
        repo_root = _find_repo_root(Path(__file__).resolve())
        if repo_root is not None:
            env_file = repo_root / ".env"
    if env_file is not None:
        _load_dotenv(env_file)

    hf_token = _get_hf_token(args.hf_token)
    default_model_id = "black-forest-labs/FLUX.1-dev"
    if args.pipeline == "sd3":
        default_model_id = "stabilityai/stable-diffusion-3.5-large"
    elif args.pipeline == "qwen":
        default_model_id = "Qwen/Qwen-Image-2512"
    model_id = args.model_id or args.flux_model_id or default_model_id
    local_files_only = args.local_files_only or args.flux_local_files_only
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    default_steps, default_guidance = _resolve_generation_defaults(args.pipeline, model_id)
    if num_inference_steps is None:
        num_inference_steps = default_steps
    if guidance_scale is None:
        guidance_scale = default_guidance
    torch_dtype = args.torch_dtype
    if torch_dtype is None:
        torch_dtype = "bf8" if args.pipeline == "qwen" else "bfloat16"

    flux = FluxParams(
        pipeline=args.pipeline,
        model_id=model_id,
        width=args.width,
        height=args.height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=args.max_sequence_length,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )

    sam = SamParams(
        checkpoint=sam_checkpoint,
        model_type=args.sam_model_type,
        prompt_mode=args.sam_prompt_mode,
        box_ratio_w=args.sam_box_ratio_w,
        box_ratio_h=args.sam_box_ratio_h,
        feather_radius_px=args.feather_radius_px,
        mask_selection_rule=args.mask_selection_rule,
        morph_kernel_px=args.morph_kernel_px,
        min_area_frac=args.mask_min_area_frac,
        max_area_frac=args.mask_max_area_frac,
        lang_sam_box_threshold=args.lang_sam_box_threshold,
        lang_sam_text_threshold=args.lang_sam_text_threshold,
    )

    composite = CompositeParams(
        anchor_x=args.anchor_x,
        anchor_y=args.anchor_y,
        target_obj_height_ratio=args.target_obj_height_ratio,
        scale_multiplier=args.scale_multiplier,
        placement_mode=args.placement_mode,
    )

    color = ColorParams(
        target_hue_deg=args.target_hue_deg,
        min_saturation=args.min_saturation,
    )

    return RunConfig(
        output_dir=output_dir,
        background_prompt=args.background_prompt,
        base_prompt=base_prompt,
        paired_prompt=paired_prompt,
        object_name=args.object_name,
        base_prompt_elements_path=base_prompt_elements_path,
        paired_prompt_elements_path=paired_prompt_elements_path,
        base_prompt_elements=None,
        paired_prompt_elements=None,
        dominant_color=args.dominant_color,
        rare_color=args.rare_color,
        seed_bg=args.seed_bg,
        seed_real=args.seed_real,
        seed_toy=args.seed_toy,
        run_index=0,
        seed_offset=0,
        flux=flux,
        sam=sam,
        composite=composite,
        color=color,
        device=args.device,
        hf_token=hf_token,
    ), args.num_runs, args.run_start, args.run_count, run_root


def main(argv: Optional[Iterable[str]] = None) -> None:
    config, num_runs, run_start, run_count, run_root_arg = _parse_args(argv)
    if run_root_arg is None:
        run_root = _create_timestamp_dir(config.output_dir)
    else:
        run_root = run_root_arg
        run_root.mkdir(parents=True, exist_ok=True)

    joiner = ", "
    base_prompts: Optional[list[str]] = None
    base_prompt_elements: Optional[list[Dict[str, str]]] = None
    if config.base_prompt_elements_path is not None:
        base_keys, base_groups = _load_prompt_elements(config.base_prompt_elements_path)
        base_prompts, base_prompt_elements = _generate_prompt_set(
            base_keys,
            base_groups,
            num_runs,
            config.seed_real,
            0xBADDCAFE,
            joiner,
            "Base prompt",
        )

    paired_prompts: Optional[list[str]] = None
    paired_prompt_elements: Optional[list[Dict[str, str]]] = None
    if config.paired_prompt_elements_path is not None:
        paired_keys, paired_groups = _load_prompt_elements(config.paired_prompt_elements_path)
        paired_prompts, paired_prompt_elements = _generate_prompt_set(
            paired_keys,
            paired_groups,
            num_runs,
            config.seed_toy,
            0xDEADBEEF,
            joiner,
            "Paired prompt",
        )

    run_configs: list[RunConfig] = []
    if run_count is None:
        run_count = num_runs - run_start
    run_indices = range(run_start, run_start + run_count)
    for run_index in run_indices:
        run_dir = run_root / f"run_{run_index:02d}"
        seed_bg = _derive_run_seed(config.seed_bg, run_index, 0xA5A5A5A5)
        seed_real = _derive_run_seed(config.seed_real, run_index, 0x5A5A5A5A)
        seed_toy = _derive_run_seed(config.seed_toy, run_index, 0x12345678)
        base_prompt = base_prompts[run_index] if base_prompts is not None else config.base_prompt
        paired_prompt = paired_prompts[run_index] if paired_prompts is not None else config.paired_prompt
        base_elements = base_prompt_elements[run_index] if base_prompt_elements is not None else None
        paired_elements = paired_prompt_elements[run_index] if paired_prompt_elements is not None else None
        run_config = replace(
            config,
            output_dir=run_dir,
            seed_bg=seed_bg,
            seed_real=seed_real,
            seed_toy=seed_toy,
            base_prompt=base_prompt,
            paired_prompt=paired_prompt,
            base_prompt_elements=base_elements,
            paired_prompt_elements=paired_elements,
            run_index=run_index,
            seed_offset=run_index,
        )
        run_configs.append(run_config)

    pipe = _load_text2image_pipeline(
        config.flux.pipeline,
        config.flux.model_id,
        config.flux.torch_dtype,
        config.device,
        config.hf_token,
        config.flux.local_files_only,
    )
    if len(run_configs) == 1:
        _run_generation_stage(run_configs[0], pipe=pipe)
    else:
        _run_generation_stage_batch(run_configs, pipe=pipe)

    try:
        pipe.to("cpu")
    except Exception:
        pass
    del pipe
    gc.collect()
    if config.device != "cpu":
        _release_torch_cuda()

    if config.sam.prompt_mode == "grounding":
        lang_sam_model = _load_lang_sam_model(config.device)
        for run_config in run_configs:
            _run_sam_stage(run_config, predictor=None, lang_sam_model=lang_sam_model)
    else:
        predictor = _load_sam_predictor(config.sam.checkpoint, config.sam.model_type, config.device)
        for run_config in run_configs:
            _run_sam_stage(run_config, predictor=predictor)


if __name__ == "__main__":
    main()
