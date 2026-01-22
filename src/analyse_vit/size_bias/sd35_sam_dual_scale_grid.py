from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
from PIL import Image

from analyse_vit.clip_oscope.path_utils import resolve_path
from analyse_vit.size_bias.sd35_sam_scale_sweep import (
    ModelParams,
    SamParams,
    _compute_alpha,
    _compute_bbox,
    _create_timestamp_dir,
    _generate_image,
    _get_hf_token,
    _get_version,
    _load_dotenv,
    _load_lang_sam_model,
    _load_sam_predictor,
    _load_text2image_pipeline,
    _mask_area_frac,
    _parse_scale_ratios,
    _pil_to_np_rgb,
    _postprocess_mask,
    _predict_mask,
    _predict_mask_lang_sam,
    _resize_rgb_and_mask,
    _np_to_pil_rgb,
    _composite_center,
    _save_image,
    _save_mask,
    _save_mask_overlay,
    _sanitize_name,
    _score_mask,
    _seed_for_category,
)

@dataclass
class RunConfig:
    output_dir: Path
    background_prompt: str
    left_object_prompt_template: str
    right_object_prompt_template: str
    pairs: Tuple[Tuple[str, str], ...]
    scale_ratios: Tuple[float, ...]
    seed_bg: int
    seed_object: int
    model: ModelParams
    sam: SamParams
    device: str
    hf_token: Optional[str]
    left_center_x_ratio: float
    right_center_x_ratio: float
    center_y_ratio: float


DEFAULT_BG_PROMPT = "a simple studio background, plain backdrop, soft gradient, even lighting"
DEFAULT_OBJECT_PROMPT_TEMPLATE = (
    "a photo of a single {category}, centered, isolated on a plain background, studio lighting"
)
DEFAULT_LEFT_CATEGORY = "dog"
DEFAULT_RIGHT_CATEGORY = "cat"


def _setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"sd35_sam_dual_scale_grid.{output_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _find_repo_root(start: Path) -> Optional[Path]:
    for parent in (start, *start.parents):
        if (parent / "AGENTS.md").exists():
            return parent
    return None


def _parse_pairs(
    left_category: Optional[str],
    right_category: Optional[str],
    left_categories: Optional[str],
    right_categories: Optional[str],
    pairs_file: Optional[Path],
) -> Tuple[Tuple[str, str], ...]:
    if pairs_file is not None:
        lines = pairs_file.read_text(encoding="utf-8").splitlines()
        pairs: list[Tuple[str, str]] = []
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            if len(parts) < 2:
                raise ValueError(f"Invalid pair line (expected 'left,right'): {line}")
            pairs.append((parts[0], parts[1]))
        if not pairs:
            raise ValueError("pairs file did not include any valid entries.")
        return tuple(pairs)

    def _split_list(value: Optional[str]) -> Tuple[str, ...]:
        if not value:
            return tuple()
        items = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(items)

    if left_category:
        left_list = (left_category,)
    else:
        left_list = _split_list(left_categories)

    if right_category:
        right_list = (right_category,)
    else:
        right_list = _split_list(right_categories)

    if not left_list:
        left_list = (DEFAULT_LEFT_CATEGORY,)
    if not right_list:
        right_list = (DEFAULT_RIGHT_CATEGORY,)

    if len(left_list) == 1 and len(right_list) > 1:
        left_list = left_list * len(right_list)
    elif len(right_list) == 1 and len(left_list) > 1:
        right_list = right_list * len(left_list)
    elif len(left_list) != len(right_list):
        raise ValueError("left/right category counts must match (or one side must have 1 item).")

    return tuple(zip(left_list, right_list))


def _center_from_ratios(width: int, height: int, x_ratio: float, y_ratio: float) -> Tuple[int, int]:
    cx = int(round(width * x_ratio))
    cy = int(round(height * y_ratio))
    return cx, cy


def _select_mask_for_object(
    predictor,
    lang_sam_model,
    image: Image.Image,
    category: str,
    sam: SamParams,
    logger: logging.Logger,
) -> Tuple[np.ndarray, dict[str, Any]]:
    logger.info("Predicting SAM mask for '%s'.", category)
    sam_mask_raw = _predict_mask(
        predictor,
        image,
        sam.prompt_mode,
        sam.box_ratio_w,
        sam.box_ratio_h,
    )
    sam_mask = _postprocess_mask(sam_mask_raw, sam.morph_kernel_px)

    lang_mask = None
    if sam.mask_mode in ("lang_sam", "auto") and lang_sam_model is not None:
        logger.info("Predicting LangSAM mask for '%s'.", category)
        lang_mask_raw = _predict_mask_lang_sam(
            lang_sam_model,
            image,
            category,
            sam.lang_sam_box_threshold,
            sam.lang_sam_text_threshold,
        )
        if lang_mask_raw is not None:
            lang_mask = _postprocess_mask(lang_mask_raw, sam.morph_kernel_px)

    center_xy = (image.width // 2, image.height // 2)
    selected_mask = sam_mask
    selected_method = "sam"
    score_sam = _score_mask(
        sam_mask,
        center_xy,
        sam.mask_min_area_frac,
        sam.mask_max_area_frac,
    )
    score_lang = None
    if sam.mask_mode == "lang_sam":
        if lang_mask is None:
            raise RuntimeError(f"LangSAM did not return a mask for '{category}'.")
        selected_mask = lang_mask
        selected_method = "lang_sam"
        score_lang = _score_mask(
            lang_mask,
            center_xy,
            sam.mask_min_area_frac,
            sam.mask_max_area_frac,
        )
    elif sam.mask_mode == "auto" and lang_mask is not None:
        score_lang = _score_mask(
            lang_mask,
            center_xy,
            sam.mask_min_area_frac,
            sam.mask_max_area_frac,
        )
        if score_lang > score_sam:
            selected_mask = lang_mask
            selected_method = "lang_sam"

    info = {
        "selected_method": selected_method,
        "sam_score": score_sam,
        "lang_sam_score": score_lang,
        "sam_area": _mask_area_frac(sam_mask),
        "lang_sam_area": _mask_area_frac(lang_mask) if lang_mask is not None else None,
    }

    return selected_mask, info


def _prepare_object_assets(
    *,
    side_label: str,
    category: str,
    prompt_template: str,
    output_dir: Path,
    pipe,
    predictor,
    lang_sam_model,
    config: RunConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    safe_side = _sanitize_name(side_label)
    side_dir = output_dir / safe_side
    side_dir.mkdir(parents=True, exist_ok=True)

    prompt = prompt_template.format(category=category)
    seed = _seed_for_category(config.seed_object, f"{side_label}:{category}")
    logger.info("Generating %s object image for '%s'.", side_label, category)
    obj_image = _generate_image(
        pipe,
        prompt,
        seed,
        config.model,
        config.device,
    )
    obj_path = side_dir / "object.png"
    _save_image(obj_image, obj_path)

    selected_mask, mask_info = _select_mask_for_object(
        predictor,
        lang_sam_model,
        obj_image,
        category,
        config.sam,
        logger,
    )
    _save_mask(selected_mask, side_dir / "mask.png")

    if config.sam.save_mask_overlay:
        _save_mask_overlay(obj_image, selected_mask, side_dir / "mask_overlay.png")

    area = _mask_area_frac(selected_mask)
    if area < config.sam.mask_min_area_frac or area > config.sam.mask_max_area_frac:
        logger.warning(
            "Mask area for '%s' (%s) is out of bounds: %.4f",
            category,
            side_label,
            area,
        )

    bbox = _compute_bbox(selected_mask)
    x0, y0, x1, y1 = bbox
    obj_rgb = _pil_to_np_rgb(obj_image)
    crop_rgb = obj_rgb[y0 : y1 + 1, x0 : x1 + 1, :]
    crop_mask = selected_mask[y0 : y1 + 1, x0 : x1 + 1]

    return {
        "category": category,
        "prompt": prompt,
        "seed": seed,
        "object_path": obj_path,
        "crop_rgb": crop_rgb,
        "crop_mask": crop_mask,
        "bbox": bbox,
        "mask_info": mask_info,
        "side_dir": side_dir,
    }


def _prepare_scaled_variants(
    *,
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    scales: Tuple[float, ...],
    bg_h: int,
    blur_radius_px: int,
) -> dict[float, dict[str, Any]]:
    variants: dict[float, dict[str, Any]] = {}
    crop_h, crop_w, _ = crop_rgb.shape
    for scale_ratio in scales:
        target_h = max(1, int(round(bg_h * scale_ratio)))
        scale = target_h / float(crop_h)
        new_w = max(1, int(round(crop_w * scale)))
        new_h = max(1, int(round(crop_h * scale)))
        fg_rgb, fg_mask = _resize_rgb_and_mask(crop_rgb, crop_mask, (new_w, new_h))
        fg_alpha = _compute_alpha(fg_mask, blur_radius_px)
        variants[scale_ratio] = {
            "fg_rgb": fg_rgb,
            "fg_alpha": fg_alpha,
            "scaled_size": [new_w, new_h],
        }
    return variants


def _composite_single(
    bg_rgb: np.ndarray,
    fg_rgb: np.ndarray,
    fg_alpha: np.ndarray,
    center_xy: Tuple[int, int],
) -> np.ndarray:
    return _composite_center(bg_rgb, fg_rgb, fg_alpha, center_xy)


def run_pipeline(config: RunConfig, pipe=None, predictor=None) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(output_dir)

    logger.info("Starting SD3.5 + SAM dual scale grid.")

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

    left_center = _center_from_ratios(
        bg_w,
        bg_h,
        config.left_center_x_ratio,
        config.center_y_ratio,
    )
    right_center = _center_from_ratios(
        bg_w,
        bg_h,
        config.right_center_x_ratio,
        config.center_y_ratio,
    )
    single_center = _center_from_ratios(
        bg_w,
        bg_h,
        0.5,
        config.center_y_ratio,
    )

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

    pairs_root = output_dir / "pairs"
    pairs_root.mkdir(parents=True, exist_ok=True)

    pairs_meta: list[dict[str, Any]] = []

    for left_category, right_category in config.pairs:
        pair_key = f"{_sanitize_name(left_category)}__{_sanitize_name(right_category)}"
        pair_dir = pairs_root / pair_key
        pair_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Processing pair: %s (left=%s, right=%s)", pair_key, left_category, right_category)
        if left_category.strip().lower() == right_category.strip().lower():
            logger.warning("Left/right categories are identical; results may be less informative.")

        left_assets = _prepare_object_assets(
            side_label="left",
            category=left_category,
            prompt_template=config.left_object_prompt_template,
            output_dir=pair_dir,
            pipe=pipe,
            predictor=predictor,
            lang_sam_model=lang_sam_model,
            config=config,
            logger=logger,
        )
        right_assets = _prepare_object_assets(
            side_label="right",
            category=right_category,
            prompt_template=config.right_object_prompt_template,
            output_dir=pair_dir,
            pipe=pipe,
            predictor=predictor,
            lang_sam_model=lang_sam_model,
            config=config,
            logger=logger,
        )

        left_variants = _prepare_scaled_variants(
            crop_rgb=left_assets["crop_rgb"],
            crop_mask=left_assets["crop_mask"],
            scales=config.scale_ratios,
            bg_h=bg_h,
            blur_radius_px=config.sam.mask_blur_radius_px,
        )
        right_variants = _prepare_scaled_variants(
            crop_rgb=right_assets["crop_rgb"],
            crop_mask=right_assets["crop_mask"],
            scales=config.scale_ratios,
            bg_h=bg_h,
            blur_radius_px=config.sam.mask_blur_radius_px,
        )

        singles_dir = pair_dir / "singles"
        composites_dir = pair_dir / "composites"
        singles_dir.mkdir(parents=True, exist_ok=True)
        composites_dir.mkdir(parents=True, exist_ok=True)

        single_left_meta: list[dict[str, Any]] = []
        single_right_meta: list[dict[str, Any]] = []
        composite_meta: list[dict[str, Any]] = []

    for scale_ratio, variant in left_variants.items():
        composite = _composite_single(
            bg_rgb,
            variant["fg_rgb"],
            variant["fg_alpha"],
            single_center,
        )
        filename = f"single_left_scale_{scale_ratio:.2f}.png"
        out_path = singles_dir / filename
        _save_image(_np_to_pil_rgb(composite), out_path)
        single_left_meta.append(
            {
                "scale_ratio": scale_ratio,
                "output_file": str(out_path.relative_to(output_dir)),
                "scaled_size": variant["scaled_size"],
                "center_xy": list(single_center),
            }
        )

    for scale_ratio, variant in right_variants.items():
        composite = _composite_single(
            bg_rgb,
            variant["fg_rgb"],
            variant["fg_alpha"],
            single_center,
        )
        filename = f"single_right_scale_{scale_ratio:.2f}.png"
        out_path = singles_dir / filename
        _save_image(_np_to_pil_rgb(composite), out_path)
        single_right_meta.append(
            {
                "scale_ratio": scale_ratio,
                "output_file": str(out_path.relative_to(output_dir)),
                "scaled_size": variant["scaled_size"],
                "center_xy": list(single_center),
            }
        )

    for left_scale, left_variant in left_variants.items():
        for right_scale, right_variant in right_variants.items():
            composite = _composite_single(
                bg_rgb,
                left_variant["fg_rgb"],
                left_variant["fg_alpha"],
                left_center,
            )
            composite = _composite_single(
                composite,
                right_variant["fg_rgb"],
                right_variant["fg_alpha"],
                right_center,
            )
            filename = f"composite_L{left_scale:.2f}_R{right_scale:.2f}.png"
            out_path = composites_dir / filename
            _save_image(_np_to_pil_rgb(composite), out_path)

            left_ref = next(item for item in single_left_meta if item["scale_ratio"] == left_scale)
            right_ref = next(item for item in single_right_meta if item["scale_ratio"] == right_scale)

            composite_meta.append(
                {
                    "left_scale_ratio": left_scale,
                    "right_scale_ratio": right_scale,
                    "output_file": str(out_path.relative_to(output_dir)),
                    "left_only_file": left_ref["output_file"],
                    "right_only_file": right_ref["output_file"],
                    "left_scaled_size": left_variant["scaled_size"],
                    "right_scaled_size": right_variant["scaled_size"],
                    "left_center_xy": list(left_center),
                    "right_center_xy": list(right_center),
                }
            )

        pairs_meta.append(
            {
                "pair_key": pair_key,
                "left_category": left_category,
                "right_category": right_category,
                "left_prompt": left_assets["prompt"],
                "right_prompt": right_assets["prompt"],
                "left_seed": left_assets["seed"],
                "right_seed": right_assets["seed"],
                "left_bbox": list(left_assets["bbox"]),
                "right_bbox": list(right_assets["bbox"]),
                "left_object_file": str(left_assets["object_path"].relative_to(output_dir)),
                "right_object_file": str(right_assets["object_path"].relative_to(output_dir)),
                "mask_selection": {
                    "left": left_assets["mask_info"],
                    "right": right_assets["mask_info"],
                },
                "images": {
                    "single_left": single_left_meta,
                    "single_right": single_right_meta,
                    "composites": composite_meta,
                },
            }
        )

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "background_prompt": config.background_prompt,
        "background_file": "background.png",
        "left_object_prompt_template": config.left_object_prompt_template,
        "right_object_prompt_template": config.right_object_prompt_template,
        "pairs": pairs_meta,
        "scale_ratios": list(config.scale_ratios),
        "seeds": {
            "background": config.seed_bg,
            "object_base": config.seed_object,
        },
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
        "positions": {
            "left_center_x_ratio": config.left_center_x_ratio,
            "right_center_x_ratio": config.right_center_x_ratio,
            "center_y_ratio": config.center_y_ratio,
            "left_center_xy": list(left_center),
            "right_center_xy": list(right_center),
        },
        "device": config.device,
        "versions": {
            "torch": _get_version("torch"),
            "diffusers": _get_version("diffusers"),
            "transformers": _get_version("transformers"),
            "segment_anything": _get_version("segment_anything"),
            "numpy": _get_version("numpy"),
            "pillow": _get_version("Pillow"),
        },
    }

    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)

    logger.info("Run complete.")


def _parse_args(argv: Optional[Iterable[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="SD3.5 + SAM dual object size grid")
    parser.add_argument("--output-dir", default="results/raw/sd35_sam_dual_scale_grid")
    parser.add_argument("--background-prompt", default=DEFAULT_BG_PROMPT)
    parser.add_argument("--left-object-prompt-template", default=DEFAULT_OBJECT_PROMPT_TEMPLATE)
    parser.add_argument("--right-object-prompt-template", default=DEFAULT_OBJECT_PROMPT_TEMPLATE)

    parser.add_argument("--left-category", default=None, help="Single left category (overrides list).")
    parser.add_argument("--right-category", default=None, help="Single right category (overrides list).")
    parser.add_argument("--left-categories", default=None, help="Comma-separated left categories.")
    parser.add_argument("--right-categories", default=None, help="Comma-separated right categories.")
    parser.add_argument("--pairs-file", default=None, help="CSV file with left,right per line.")

    parser.add_argument("--scale-ratios", default=None, help="Comma-separated ratios (e.g. 0.1,0.2,0.3).")
    parser.add_argument("--scale-start", type=float, default=0.1)
    parser.add_argument("--scale-end", type=float, default=0.5)
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

    parser.add_argument("--left-center-x-ratio", type=float, default=0.25)
    parser.add_argument("--right-center-x-ratio", type=float, default=0.75)
    parser.add_argument("--center-y-ratio", type=float, default=0.5)

    parser.add_argument("--device", default="cuda")

    args = parser.parse_args(argv)

    output_dir = resolve_path(args.output_dir)
    pairs_file = resolve_path(args.pairs_file) if args.pairs_file else None
    scale_ratios = _parse_scale_ratios(args.scale_ratios, args.scale_start, args.scale_end, args.scale_step)
    pairs = _parse_pairs(
        args.left_category,
        args.right_category,
        args.left_categories,
        args.right_categories,
        pairs_file,
    )

    env_file = resolve_path(args.env_file) if args.env_file else None
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
        checkpoint=resolve_path(args.sam_checkpoint),
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
        left_object_prompt_template=args.left_object_prompt_template,
        right_object_prompt_template=args.right_object_prompt_template,
        pairs=pairs,
        scale_ratios=scale_ratios,
        seed_bg=args.seed_bg,
        seed_object=args.seed_object,
        model=model,
        sam=sam,
        device=args.device,
        hf_token=hf_token,
        left_center_x_ratio=args.left_center_x_ratio,
        right_center_x_ratio=args.right_center_x_ratio,
        center_y_ratio=args.center_y_ratio,
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
