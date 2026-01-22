from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class ModelParams:
    model_id: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    torch_dtype: str
    local_files_only: bool


@dataclass
class SamParams:
    checkpoint: Path
    model_type: str
    prompt_mode: str
    mask_mode: str
    box_ratio_w: float
    box_ratio_h: float
    morph_kernel_px: int
    mask_min_area_frac: float
    mask_max_area_frac: float
    mask_blur_radius_px: int
    lang_sam_box_threshold: float
    lang_sam_text_threshold: float
    save_mask_overlay: bool


@dataclass
class RunConfig:
    output_dir: Path
    background_prompt: str
    object_prompt_template: str
    categories: Tuple[str, ...]
    scale_ratios: Tuple[float, ...]
    seed_bg: int
    seed_object: int
    model: ModelParams
    sam: SamParams
    device: str
    hf_token: Optional[str]


DEFAULT_BG_PROMPT = "a simple studio background, plain backdrop, soft gradient, even lighting"
DEFAULT_OBJECT_PROMPT_TEMPLATE = (
    "a photo of a single {category}, centered, isolated on a plain background, studio lighting"
)
DEFAULT_CATEGORIES = (
    "dog",
    "cat",
    "horse",
    "eagle",
    "shark",
    "rose",
    "sunflower",
    "cactus",
    "oak tree",
    "person",
)


def _resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


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


def _setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"sd35_sam_scale_sweep.{output_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


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


def _get_version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _filter_kwargs_for_callable(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in sig.parameters}


def _pil_to_np_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _np_to_pil_rgb(array: np.ndarray) -> Image.Image:
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _save_image(image: Image.Image, path: Path) -> None:
    image.save(path)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(path)


def _load_text2image_pipeline(
    model_id: str,
    torch_dtype: str,
    device: str,
    token: Optional[str],
    local_files_only: bool,
):
    import torch
    from diffusers import StableDiffusion3Pipeline

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)
    if device == "cpu":
        dtype = torch.float32

    sig = inspect.signature(StableDiffusion3Pipeline.from_pretrained)
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
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
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        hints = ["Failed to load Stable Diffusion 3.5 pipeline."]
        if token is None:
            hints.append("If the model is gated, pass --hf-token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN.")
        if local_files_only:
            hints.append("local_files_only is enabled; ensure the model is cached or omit --local-files-only.")
        hints.append("You can also point --model-id to a local path or a public model.")
        raise RuntimeError(" ".join(hints)) from exc
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _generate_image(
    pipe,
    prompt: str,
    seed: int,
    params: ModelParams,
    device: str,
) -> Image.Image:
    import torch

    generator_device = "cpu" if device == "cpu" else device
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    call_kwargs = {
        "prompt": prompt,
        "width": params.width,
        "height": params.height,
        "num_inference_steps": params.num_inference_steps,
        "guidance_scale": params.guidance_scale,
        "generator": generator,
    }
    call_kwargs = _filter_kwargs_for_callable(pipe.__call__, call_kwargs)
    result = pipe(**call_kwargs)
    return result.images[0]


def _load_sam_predictor(checkpoint: Path, model_type: str, device: str):
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam = sam.float()  # avoid bfloat16 outputs that break numpy conversion in SAM predictor
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


def _predict_mask(
    predictor,
    image: Image.Image,
    prompt_mode: str,
    box_ratio_w: float,
    box_ratio_h: float,
) -> np.ndarray:
    import torch

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
        coords = predictor.transform.apply_coords(point_coords, predictor.original_size)
        coords_torch = torch.as_tensor(coords, dtype=torch.float, device=predictor.device)[None, :, :]
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=predictor.device)[None, :]
        with torch.no_grad():
            masks_torch, _, _ = predictor.predict_torch(
                coords_torch,
                labels_torch,
                boxes=None,
                mask_input=None,
                multimask_output=True,
                return_logits=False,
            )
    elif prompt_mode == "box":
        bw = int(round(w * box_ratio_w))
        bh = int(round(h * box_ratio_h))
        x0 = int(round((w - bw) / 2.0))
        y0 = int(round((h - bh) / 2.0))
        x1 = x0 + bw
        y1 = y0 + bh
        box = np.array([x0, y0, x1, y1])
        box = predictor.transform.apply_boxes(box[None, :], predictor.original_size)
        box_torch = torch.as_tensor(box, dtype=torch.float, device=predictor.device)
        with torch.no_grad():
            masks_torch, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=box_torch,
                mask_input=None,
                multimask_output=True,
                return_logits=False,
            )
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    masks = masks_torch[0].float().cpu().numpy()
    return _select_mask(masks, (cx, cy))


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
    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    try:
        import cv2
    except Exception:
        return (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    mask_u8 = _largest_component(mask_u8)
    mask_u8 = _fill_holes(mask_u8)
    if morph_kernel_px > 1:
        kernel = np.ones((morph_kernel_px, morph_kernel_px), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _mask_area_frac(mask: np.ndarray) -> float:
    return float(np.mean(mask > 0.5))


def _compute_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Mask is empty; cannot compute bbox.")
    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())
    return min_x, min_y, max_x, max_y


def _compute_alpha(mask: np.ndarray, blur_radius_px: int) -> np.ndarray:
    mask_img = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    if blur_radius_px > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius_px))
    alpha = np.asarray(mask_img, dtype=np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def _score_mask(mask: np.ndarray, center_xy: Tuple[int, int], min_area: float, max_area: float) -> float:
    area = _mask_area_frac(mask)
    cx, cy = center_xy
    center_hit = mask[cy, cx] > 0.5
    score = 0.0
    if min_area <= area <= max_area:
        score += 2.0
    if center_hit:
        score += 1.0
    score += min(area, max_area)
    return score


def _save_mask_overlay(image: Image.Image, mask: np.ndarray, path: Path) -> None:
    base = image.convert("RGBA")
    mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha = (mask_u8 * 0.45).astype(np.uint8)
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    overlay.putalpha(Image.fromarray(alpha, mode="L"))
    Image.alpha_composite(base, overlay).convert("RGB").save(path)


def _sanitize_name(name: str) -> str:
    stripped = name.strip().lower()
    if not stripped:
        return "item"
    cleaned = []
    for ch in stripped:
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in (" ", "-", "_"):
            cleaned.append("_")
    output = "".join(cleaned).strip("_")
    while "__" in output:
        output = output.replace("__", "_")
    return output or "item"


def _resize_rgb_and_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    rgb_img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    mask_img = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    rgb_img = rgb_img.resize(size, resample=Image.BICUBIC)
    mask_img = mask_img.resize(size, resample=Image.NEAREST)
    rgb_resized = np.asarray(rgb_img, dtype=np.float32) / 255.0
    mask_resized = np.asarray(mask_img, dtype=np.float32) / 255.0
    return rgb_resized, mask_resized


def _composite_center(
    bg_rgb: np.ndarray,
    fg_rgb: np.ndarray,
    fg_alpha: np.ndarray,
    center_xy: Tuple[int, int],
) -> np.ndarray:
    bg_h, bg_w, _ = bg_rgb.shape
    fg_h, fg_w, _ = fg_rgb.shape
    cx, cy = center_xy
    x0 = int(round(cx - fg_w / 2.0))
    y0 = int(round(cy - fg_h / 2.0))
    x1 = x0 + fg_w
    y1 = y0 + fg_h

    ox0 = max(0, x0)
    oy0 = max(0, y0)
    ox1 = min(bg_w, x1)
    oy1 = min(bg_h, y1)
    if ox0 >= ox1 or oy0 >= oy1:
        return bg_rgb.copy()

    fx0 = ox0 - x0
    fy0 = oy0 - y0
    fx1 = fx0 + (ox1 - ox0)
    fy1 = fy0 + (oy1 - oy0)

    out = bg_rgb.copy()
    bg_region = out[oy0:oy1, ox0:ox1, :]
    fg_region = fg_rgb[fy0:fy1, fx0:fx1, :]
    alpha_region = fg_alpha[fy0:fy1, fx0:fx1][:, :, None]
    blended = fg_region * alpha_region + bg_region * (1.0 - alpha_region)
    out[oy0:oy1, ox0:ox1, :] = blended
    return out


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 30) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 31) & 0xFFFFFFFFFFFFFFFF
    return x


def _seed_for_category(base_seed: int, category: str) -> int:
    digest = hashlib.sha256(category.encode("utf-8")).digest()
    category_int = int.from_bytes(digest[:8], "little")
    mixed = (base_seed & 0xFFFFFFFFFFFFFFFF) ^ category_int
    return int(_splitmix64(mixed) & 0x7FFFFFFFFFFFFFFF)


def _parse_scale_ratios(
    scale_ratios: Optional[str],
    scale_start: float,
    scale_end: float,
    scale_step: float,
) -> Tuple[float, ...]:
    if scale_ratios:
        values = [float(item.strip()) for item in scale_ratios.split(",") if item.strip()]
    else:
        if scale_step <= 0:
            raise ValueError("--scale-step must be > 0.")
        if scale_end < scale_start:
            raise ValueError("--scale-end must be >= --scale-start.")
        values = []
        current = scale_start
        while current <= scale_end + 1e-9:
            values.append(round(current, 4))
            current += scale_step
    cleaned = []
    for value in values:
        if value <= 0:
            raise ValueError("Scale ratios must be > 0.")
        cleaned.append(value)
    return tuple(cleaned)


def _parse_categories(
    categories: Optional[str],
    categories_file: Optional[Path],
) -> Tuple[str, ...]:
    items: list[str]
    if categories_file is not None:
        lines = categories_file.read_text(encoding="utf-8").splitlines()
        items = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    elif categories:
        items = [part.strip() for part in categories.split(",") if part.strip()]
    else:
        items = list(DEFAULT_CATEGORIES)
    if not items:
        raise ValueError("No categories provided.")
    return tuple(items)


def run_pipeline(config: RunConfig, pipe=None, predictor=None) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(output_dir)

    logger.info("Starting SD3.5 + SAM scale sweep.")

    if pipe is None:
        logger.info("Loading Stable Diffusion 3.5 pipeline.")
        pipe = _load_text2image_pipeline(
            config.model.model_id,
            config.model.torch_dtype,
            config.device,
            config.hf_token,
            config.model.local_files_only,
        )

    logger.info("Generating background image.")
    bg_image = _generate_image(
        pipe,
        config.background_prompt,
        config.seed_bg,
        config.model,
        config.device,
    )
    _save_image(bg_image, output_dir / "background.png")
    bg_rgb = _pil_to_np_rgb(bg_image)
    bg_h, bg_w, _ = bg_rgb.shape
    center_xy = (int(round(bg_w * 0.5)), int(round(bg_h * 0.5)))

    if predictor is None:
        logger.info("Loading SAM predictor.")
        predictor = _load_sam_predictor(config.sam.checkpoint, config.sam.model_type, config.device)

    lang_sam_model = None
    if config.sam.mask_mode in ("lang_sam", "auto"):
        try:
            logger.info("Loading LangSAM model for text-guided masks.")
            lang_sam_model = _load_lang_sam_model(config.device)
        except Exception as exc:
            if config.sam.mask_mode == "lang_sam":
                raise RuntimeError("Failed to load LangSAM for text-guided masks.") from exc
            logger.warning("LangSAM unavailable; falling back to SAM masks. (%s)", exc)

    images_meta: list[dict[str, Any]] = []
    category_mask_info: dict[str, dict[str, Any]] = {}
    categories_root = output_dir / "categories"
    categories_root.mkdir(parents=True, exist_ok=True)

    for category in config.categories:
        safe_name = _sanitize_name(category)
        cat_dir = categories_root / safe_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        prompt = config.object_prompt_template.format(category=category)
        seed = _seed_for_category(config.seed_object, category)

        logger.info("Generating object image for category '%s'.", category)
        obj_image = _generate_image(
            pipe,
            prompt,
            seed,
            config.model,
            config.device,
        )
        _save_image(obj_image, cat_dir / "object.png")
        obj_rgb = _pil_to_np_rgb(obj_image)

        logger.info("Predicting SAM mask for category '%s'.", category)
        sam_mask_raw = _predict_mask(
            predictor,
            obj_image,
            config.sam.prompt_mode,
            config.sam.box_ratio_w,
            config.sam.box_ratio_h,
        )
        sam_mask = _postprocess_mask(sam_mask_raw, config.sam.morph_kernel_px)
        _save_mask(sam_mask, cat_dir / "mask_sam.png")

        lang_mask = None
        if config.sam.mask_mode in ("lang_sam", "auto") and lang_sam_model is not None:
            logger.info("Predicting LangSAM mask for category '%s'.", category)
            lang_mask_raw = _predict_mask_lang_sam(
                lang_sam_model,
                obj_image,
                category,
                config.sam.lang_sam_box_threshold,
                config.sam.lang_sam_text_threshold,
            )
            if lang_mask_raw is not None:
                lang_mask = _postprocess_mask(lang_mask_raw, config.sam.morph_kernel_px)
                _save_mask(lang_mask, cat_dir / "mask_lang_sam.png")

        obj_center_xy = (obj_image.width // 2, obj_image.height // 2)
        selected_mask = sam_mask
        selected_method = "sam"
        score_sam = _score_mask(
            sam_mask,
            obj_center_xy,
            config.sam.mask_min_area_frac,
            config.sam.mask_max_area_frac,
        )
        score_lang = None
        if config.sam.mask_mode == "lang_sam":
            if lang_mask is None:
                raise RuntimeError(f"LangSAM did not return a mask for '{category}'.")
            selected_mask = lang_mask
            selected_method = "lang_sam"
            score_lang = _score_mask(
                lang_mask,
                obj_center_xy,
                config.sam.mask_min_area_frac,
                config.sam.mask_max_area_frac,
            )
        elif config.sam.mask_mode == "auto" and lang_mask is not None:
            score_lang = _score_mask(
                lang_mask,
                obj_center_xy,
                config.sam.mask_min_area_frac,
                config.sam.mask_max_area_frac,
            )
            if score_lang > score_sam:
                selected_mask = lang_mask
                selected_method = "lang_sam"

        _save_mask(selected_mask, cat_dir / "mask.png")
        if score_lang is None:
            logger.info(
                "Selected mask for '%s': %s (sam_score=%.3f).",
                category,
                selected_method,
                score_sam,
            )
        else:
            logger.info(
                "Selected mask for '%s': %s (sam_score=%.3f, lang_sam_score=%.3f).",
                category,
                selected_method,
                score_sam,
                score_lang,
            )
        if config.sam.save_mask_overlay:
            _save_mask_overlay(obj_image, selected_mask, cat_dir / "mask_overlay.png")
            _save_mask_overlay(obj_image, sam_mask, cat_dir / "mask_overlay_sam.png")
            if lang_mask is not None:
                _save_mask_overlay(obj_image, lang_mask, cat_dir / "mask_overlay_lang_sam.png")

        area = _mask_area_frac(selected_mask)
        if area < config.sam.mask_min_area_frac or area > config.sam.mask_max_area_frac:
            logger.warning(
                "Mask area for '%s' is out of bounds: %.4f",
                category,
                area,
            )
        bbox = _compute_bbox(selected_mask)
        x0, y0, x1, y1 = bbox
        crop_rgb = obj_rgb[y0 : y1 + 1, x0 : x1 + 1, :]
        crop_mask = selected_mask[y0 : y1 + 1, x0 : x1 + 1]
        crop_h, crop_w, _ = crop_rgb.shape

        category_mask_info[category] = {
            "selected_method": selected_method,
            "sam_score": score_sam,
            "lang_sam_score": score_lang,
            "sam_area": _mask_area_frac(sam_mask),
            "lang_sam_area": _mask_area_frac(lang_mask) if lang_mask is not None else None,
        }

        for scale_ratio in config.scale_ratios:
            target_h = max(1, int(round(bg_h * scale_ratio)))
            scale = target_h / float(crop_h)
            new_w = max(1, int(round(crop_w * scale)))
            new_h = max(1, int(round(crop_h * scale)))
            fg_rgb, fg_mask = _resize_rgb_and_mask(crop_rgb, crop_mask, (new_w, new_h))
            fg_alpha = _compute_alpha(fg_mask, config.sam.mask_blur_radius_px)
            composite = _composite_center(bg_rgb, fg_rgb, fg_alpha, center_xy)

            filename = f"composite_scale_{scale_ratio:.2f}.png"
            out_path = cat_dir / filename
            _save_image(_np_to_pil_rgb(composite), out_path)

            images_meta.append(
                {
                    "category": category,
                    "prompt": prompt,
                    "seed": seed,
                    "scale_ratio": scale_ratio,
                    "output_file": str(out_path.relative_to(output_dir)),
                    "bbox": [x0, y0, x1, y1],
                    "scaled_size": [new_w, new_h],
                    "center_xy": list(center_xy),
                    "mask_method": selected_method,
                }
            )

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "background_prompt": config.background_prompt,
        "object_prompt_template": config.object_prompt_template,
        "categories": list(config.categories),
        "scale_ratios": list(config.scale_ratios),
        "seeds": {"background": config.seed_bg, "object_base": config.seed_object},
        "model": asdict(config.model),
        "sam": {
            "checkpoint": str(config.sam.checkpoint),
            "model_type": config.sam.model_type,
            "prompt_mode": config.sam.prompt_mode,
            "mask_mode": config.sam.mask_mode,
            "box_ratio_w": config.sam.box_ratio_w,
            "box_ratio_h": config.sam.box_ratio_h,
            "morph_kernel_px": config.sam.morph_kernel_px,
            "mask_min_area_frac": config.sam.mask_min_area_frac,
            "mask_max_area_frac": config.sam.mask_max_area_frac,
            "mask_blur_radius_px": config.sam.mask_blur_radius_px,
            "lang_sam_box_threshold": config.sam.lang_sam_box_threshold,
            "lang_sam_text_threshold": config.sam.lang_sam_text_threshold,
            "save_mask_overlay": config.sam.save_mask_overlay,
        },
        "device": config.device,
        "mask_selection": category_mask_info,
        "versions": {
            "torch": _get_version("torch"),
            "diffusers": _get_version("diffusers"),
            "transformers": _get_version("transformers"),
            "segment_anything": _get_version("segment_anything"),
            "numpy": _get_version("numpy"),
            "pillow": _get_version("Pillow"),
        },
        "images": images_meta,
    }

    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)

    logger.info("Run complete.")


def _parse_args(argv: Optional[Iterable[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="SD3.5 + SAM scale sweep composite pipeline")
    parser.add_argument("--output-dir", default="results/raw/sd35_sam_scale_sweep")
    parser.add_argument("--background-prompt", default=DEFAULT_BG_PROMPT)
    parser.add_argument("--object-prompt-template", default=DEFAULT_OBJECT_PROMPT_TEMPLATE)
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated list of categories (e.g. 'dog,cat,rose').",
    )
    parser.add_argument(
        "--categories-file",
        default=None,
        help="Text file with one category per line (comments with # are ignored).",
    )

    parser.add_argument("--scale-ratios", default=None, help="Comma-separated ratios (e.g. 0.1,0.2,0.3).")
    parser.add_argument("--scale-start", type=float, default=0.1)
    parser.add_argument("--scale-end", type=float, default=1.0)
    parser.add_argument("--scale-step", type=float, default=0.1)

    parser.add_argument("--seed-bg", type=int, default=0)
    parser.add_argument("--seed-object", type=int, default=1)

    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
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
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--sam-checkpoint", default="models/sam/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--sam-prompt-mode", default="points", choices=["points", "box"])
    parser.add_argument(
        "--mask-mode",
        default="auto",
        choices=["sam", "lang_sam", "auto"],
        help="Mask selection mode (sam, lang_sam, or auto).",
    )
    parser.add_argument("--sam-box-ratio-w", type=float, default=0.8)
    parser.add_argument("--sam-box-ratio-h", type=float, default=0.85)
    parser.add_argument("--morph-kernel-px", type=int, default=3)
    parser.add_argument("--mask-min-area-frac", type=float, default=0.01)
    parser.add_argument("--mask-max-area-frac", type=float, default=0.9)
    parser.add_argument("--mask-blur-radius-px", type=int, default=3)
    parser.add_argument("--lang-sam-box-threshold", type=float, default=0.3)
    parser.add_argument("--lang-sam-text-threshold", type=float, default=0.25)
    parser.add_argument("--save-mask-overlay", action="store_true")
    parser.add_argument("--no-save-mask-overlay", action="store_true")

    parser.add_argument("--device", default="cuda")

    args = parser.parse_args(argv)

    output_dir = _resolve_path(args.output_dir)
    categories_file = _resolve_path(args.categories_file) if args.categories_file else None
    scale_ratios = _parse_scale_ratios(args.scale_ratios, args.scale_start, args.scale_end, args.scale_step)
    categories = _parse_categories(args.categories, categories_file)

    env_file = _resolve_path(args.env_file) if args.env_file else None
    if env_file is None:
        repo_root = _find_repo_root(Path(__file__).resolve())
        if repo_root is not None:
            env_file = repo_root / ".env"
    if env_file is not None:
        _load_dotenv(env_file)

    hf_token = _get_hf_token(args.hf_token)

    model = ModelParams(
        model_id=args.model_id,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
    )

    save_mask_overlay = True
    if args.no_save_mask_overlay:
        save_mask_overlay = False
    elif args.save_mask_overlay:
        save_mask_overlay = True

    sam = SamParams(
        checkpoint=_resolve_path(args.sam_checkpoint),
        model_type=args.sam_model_type,
        prompt_mode=args.sam_prompt_mode,
        mask_mode=args.mask_mode,
        box_ratio_w=args.sam_box_ratio_w,
        box_ratio_h=args.sam_box_ratio_h,
        morph_kernel_px=args.morph_kernel_px,
        mask_min_area_frac=args.mask_min_area_frac,
        mask_max_area_frac=args.mask_max_area_frac,
        mask_blur_radius_px=args.mask_blur_radius_px,
        lang_sam_box_threshold=args.lang_sam_box_threshold,
        lang_sam_text_threshold=args.lang_sam_text_threshold,
        save_mask_overlay=save_mask_overlay,
    )

    return RunConfig(
        output_dir=output_dir,
        background_prompt=args.background_prompt,
        object_prompt_template=args.object_prompt_template,
        categories=categories,
        scale_ratios=scale_ratios,
        seed_bg=args.seed_bg,
        seed_object=args.seed_object,
        model=model,
        sam=sam,
        device=args.device,
        hf_token=hf_token,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    config = _parse_args(argv)
    run_root = _create_timestamp_dir(config.output_dir)
    config = replace(config, output_dir=run_root)

    pipe = _load_text2image_pipeline(
        config.model.model_id,
        config.model.torch_dtype,
        config.device,
        config.hf_token,
        config.model.local_files_only,
    )
    predictor = _load_sam_predictor(config.sam.checkpoint, config.sam.model_type, config.device)

    run_pipeline(config, pipe=pipe, predictor=predictor)


if __name__ == "__main__":
    main()
