from __future__ import annotations

import json
import logging
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .gen_color import _apply_color_transform, _np_to_pil, _pil_to_np_rgb, _resolve_color_transform
from .gen_composite import _composite_variants, _compute_outside_diff, _compute_transform
from .gen_mask import _compute_alpha, _compute_bbox, _mask_area_frac, _postprocess_mask, _save_mask
from .gen_pipeline import _generate_images, _load_text2image_pipeline
from .gen_sam import _load_lang_sam_model, _predict_mask_lang_sam
from .gen_types import RunConfig, RunDirs, TransformInfo
from .gen_utils import (
    _derive_run_seed,
    _get_cv2_version,
    _get_version,
    _load_required_rgb_image,
    _prepare_run_dirs,
    _resolve_run_input,
    _setup_logger,
)


def _ensure_batch_compatible(run_configs: list[RunConfig]) -> RunConfig:
    if not run_configs:
        raise ValueError("No run configs provided for generation.")
    base = run_configs[0]
    for cfg in run_configs[1:]:
        if cfg.device != base.device:
            raise ValueError("Batch generation requires the same device for all runs.")
        if cfg.flux != base.flux:
            raise ValueError("Batch generation requires identical flux parameters for all runs.")
        if cfg.background_prompt != base.background_prompt:
            raise ValueError("Batch generation requires identical background prompt for all runs.")
    return base


def _run_generation(run_configs: list[RunConfig], pipe=None):
    base = _ensure_batch_compatible(run_configs)
    run_states: list[Tuple[RunConfig, RunDirs, logging.Logger]] = []

    for cfg in run_configs:
        run_dirs = _prepare_run_dirs(cfg.output_dir)
        logger = _setup_logger(run_dirs)
        logger.info("Starting run %03d (seed offset=%d).", cfg.run_index, cfg.seed_offset)
        logger.info(
            "Generation settings: size=%dx%d steps=%d guidance=%.2f max_seq=%d dtype=%s device=%s",
            cfg.flux.width,
            cfg.flux.height,
            cfg.flux.num_inference_steps,
            cfg.flux.guidance_scale,
            cfg.flux.max_sequence_length,
            cfg.flux.torch_dtype,
            cfg.device,
        )
        logger.info("Seeds: bg=%d real=%d toy=%d", cfg.seed_bg, cfg.seed_real, cfg.seed_toy)
        run_states.append((cfg, run_dirs, logger))

    if pipe is None:
        # Use base run's meta logger for pipeline loading logs
        base_dirs = _prepare_run_dirs(base.output_dir)
        base_logger = _setup_logger(base_dirs, name_suffix=".base")
        base_logger.info("Loading text-to-image pipeline (%s).", base.flux.pipeline)
        pipe = _load_text2image_pipeline(
            base.flux.pipeline,
            base.flux.model_id,
            base.flux.torch_dtype,
            base.device,
            base.hf_token,
            base.flux.local_files_only,
        )

    negative_prompt = base.negative_prompt
    if negative_prompt is not None and not negative_prompt.strip():
        negative_prompt = None

    if base.flux.pipeline in {"flux", "sd3", "qwen"}:
        guidance_scale = base.flux.guidance_scale if negative_prompt else None
    else:
        guidance_scale = base.flux.guidance_scale

    has_background_prompt = bool(base.background_prompt)

    # Background generation: preserve prior behavior
    shared_bg: Optional[Image.Image] = None
    if has_background_prompt and len(run_states) > 1:
        _, _, base_logger = run_states[0]
        base_logger.info("Generating shared background image once (single forward).")
        base_logger.info("Waiting for diffusion inference (this can take a while)...")
        start = time.perf_counter()
        shared_bg = _generate_images(
            pipe,
            [base.background_prompt],
            [base.seed_bg],
            base.flux.width,
            base.flux.height,
            base.flux.num_inference_steps,
            guidance_scale,
            base.flux.max_sequence_length,
            base.device,
        )[0]
        base_logger.info("Shared background generation finished in %.2fs.", time.perf_counter() - start)

    for cfg, run_dirs, logger in run_states:
        bg_image: Optional[Image.Image] = None
        if has_background_prompt:
            if shared_bg is None:
                logger.info("Generating background image (single forward).")
                logger.info("Waiting for diffusion inference (this can take a while)...")
                start = time.perf_counter()
                bg_image = _generate_images(
                    pipe,
                    [cfg.background_prompt],
                    [cfg.seed_bg],
                    cfg.flux.width,
                    cfg.flux.height,
                    cfg.flux.num_inference_steps,
                    guidance_scale,
                    cfg.flux.max_sequence_length,
                    cfg.device,
                    negative_prompt,
                )[0]
                logger.info("Background generation finished in %.2fs.", time.perf_counter() - start)
            else:
                bg_image = shared_bg
                logger.info("Reusing shared background image for this run.")
        else:
            logger.info("Skipping background generation because --background-prompt was not provided.")

        if cfg.flux.pipeline == "qwen":
            logger.info("Generating anchor images sequentially (qwen uses more VRAM per sample).")
            logger.info("Waiting for diffusion inference (real anchor)...")
            start = time.perf_counter()
            real_anchor_image = _generate_images(
                pipe,
                [cfg.base_prompt],
                [cfg.seed_real],
                cfg.flux.width,
                cfg.flux.height,
                cfg.flux.num_inference_steps,
                guidance_scale,
                cfg.flux.max_sequence_length,
                cfg.device,
                negative_prompt,
            )[0]
            logger.info("Real anchor finished in %.2fs.", time.perf_counter() - start)

            logger.info("Waiting for diffusion inference (toy anchor)...")
            start = time.perf_counter()
            toy_anchor_image = _generate_images(
                pipe,
                [cfg.paired_prompt],
                [cfg.seed_toy],
                cfg.flux.width,
                cfg.flux.height,
                cfg.flux.num_inference_steps,
                guidance_scale,
                cfg.flux.max_sequence_length,
                cfg.device,
                negative_prompt,
            )[0]
            logger.info("Toy anchor finished in %.2fs.", time.perf_counter() - start)
        else:
            logger.info("Generating anchor images (batched: real + toy).")
            logger.info("Waiting for diffusion inference (this can take a while)...")
            start = time.perf_counter()
            real_anchor_image, toy_anchor_image = _generate_images(
                pipe,
                [cfg.base_prompt, cfg.paired_prompt],
                [cfg.seed_real, cfg.seed_toy],
                cfg.flux.width,
                cfg.flux.height,
                cfg.flux.num_inference_steps,
                guidance_scale,
                cfg.flux.max_sequence_length,
                cfg.device,
                negative_prompt,
            )
            logger.info("Anchor generation finished in %.2fs.", time.perf_counter() - start)

        logger.info("Saving generated inputs...")
        if bg_image is not None:
            bg_image.save(run_dirs.inputs / "bg.png")
        real_anchor_image.save(run_dirs.inputs / "real_anchor.png")
        toy_anchor_image.save(run_dirs.inputs / "toy_anchor.png")
        logger.info("Saved generated inputs to %s", run_dirs.inputs)

    return pipe


# ----------------------------
# SAM stage
# ----------------------------


def _run_sam_stage(config: RunConfig, predictor=None, lang_sam_model=None) -> None:
    run_dirs = _prepare_run_dirs(config.output_dir)
    logger = _setup_logger(run_dirs)

    logger.info("Starting SAM/composite stage for run %03d.", config.run_index)

    bg_image = _load_required_rgb_image(_resolve_run_input(run_dirs, "bg.png"))
    anchor_images: Dict[str, Image.Image] = {
        "real": _load_required_rgb_image(_resolve_run_input(run_dirs, "real_anchor.png")),
        "toy": _load_required_rgb_image(_resolve_run_input(run_dirs, "toy_anchor.png")),
    }

    if config.sam.prompt_mode == "grounding":
        if not config.object_name_real or not config.object_name_toy:
            raise ValueError("--object-name-real/--object-name-toy is required when --sam-prompt-mode grounding is used.")
        logger.info(
            "Using object names for grounding SAM (real=%s, toy=%s)",
            config.object_name_real,
            config.object_name_toy,
        )
        if lang_sam_model is None:
            logger.info("Loading SAM3 model for text-guided masks.")
            lang_sam_model = _load_lang_sam_model(config.device)
    else:
        raise ValueError("grounding 以外の SAM プロンプトは現在無効です。")

    dominant_transform = _resolve_color_transform(config.dominant_color, None)
    rare_transform = _resolve_color_transform(config.rare_color, config.color.target_hue_deg)

    # Predict masks (loop)
    masks_raw: Dict[str, np.ndarray] = {}
    for kind in ("real", "toy"):
        logger.info("Predicting %s mask.", kind)
        if config.sam.prompt_mode == "grounding":
            prompt = config.object_name_real if kind == "real" else config.object_name_toy
            m = _predict_mask_lang_sam(
                lang_sam_model,
                anchor_images[kind],
                prompt or "",
                config.sam.lang_sam_box_threshold,
                config.sam.lang_sam_text_threshold,
                config.sam.mask_selection_rule,
            )
            if m is None:
                logger.warning("SAM3 did not return a mask for the %s prompt; skipping this run.", kind)
                return
            masks_raw[kind] = m

    multi_mask_mode = any(raw.ndim == 3 for raw in masks_raw.values())
    masks: Dict[str, list[np.ndarray]] = {"real": [], "toy": []}
    alphas: Dict[str, list[np.ndarray]] = {"real": [], "toy": []}

    sanity_checks: Dict[str, Any] = {}

    def _mask_suffix(mask_index: int) -> str:
        return f"_m{mask_index:02d}" if multi_mask_mode else ""

    for kind in ("real", "toy"):
        raw = masks_raw[kind]
        if multi_mask_mode:
            if raw.ndim == 2:
                mask_list = [raw]
            else:
                mask_list = [raw[i] for i in range(raw.shape[0])]
        else:
            mask_list = [raw]

        if not mask_list:
            logger.warning("No masks returned for %s; skipping this run.", kind)
            return

        for idx, mask_raw in enumerate(mask_list):
            mask = _postprocess_mask(mask_raw, config.sam.morph_kernel_px)
            suffix = _mask_suffix(idx)
            _save_mask(mask, run_dirs.masks / f"{kind}_mask{suffix}.png")
            masks[kind].append(mask)

            area = _mask_area_frac(mask)
            sanity_checks[f"{kind}_mask_area_frac{suffix}"] = area
            ok = config.sam.min_area_frac <= area <= config.sam.max_area_frac
            sanity_checks[f"{kind}_mask_area_ok{suffix}"] = ok
            if not ok:
                logger.error("%s mask area out of bounds%s: %.4f", kind.capitalize(), suffix, area)
            if area == 0.0:
                logger.warning("Mask is empty; skipping this run.")
                return

            alphas[kind].append(_compute_alpha(mask, config.sam.feather_radius_px))

    # Color transforms (loop)
    anchor_rgb: Dict[str, np.ndarray] = {k: _pil_to_np_rgb(anchor_images[k]) for k in ("real", "toy")}

    variants = {
        "dom": dominant_transform,
        "rare": rare_transform,
    }

    transformed_rgb: Dict[str, list[Dict[str, np.ndarray]]] = {"real": [], "toy": []}

    for kind in ("real", "toy"):
        for idx, mask in enumerate(masks[kind]):
            suffix = _mask_suffix(idx)
            rgb_variants: Dict[str, np.ndarray] = {}

            for var_name, transform in variants.items():
                logger.info("Applying %s color transformation to %s%s.", var_name, kind, suffix)
                rgb_out = _apply_color_transform(anchor_rgb[kind], mask, transform, config.color.min_saturation)
                rgb_variants[var_name] = rgb_out
                _np_to_pil(rgb_out, "RGB").save(run_dirs.outputs / f"{kind}_{var_name}{suffix}.png")

            transformed_rgb[kind].append(rgb_variants)

            # Save RGBA variants (shared alpha)
            for var_name in variants:
                rgba = np.dstack([rgb_variants[var_name], alphas[kind][idx]])
                _np_to_pil(rgba, "RGBA").save(run_dirs.outputs / f"{kind}_{var_name}_rgba{suffix}.png")

            # Sanity: dom/rare differ only inside mask
            diff_outside = _compute_outside_diff(rgb_variants["dom"], rgb_variants["rare"], mask)
            sanity_checks[f"{kind}_dom_rare_outside_max_diff{suffix}"] = diff_outside
            if diff_outside != 0.0:
                logger.error("%s dom/rare differ outside mask%s: %.6f", kind.capitalize(), suffix, diff_outside)

    # Composite on background (alpha placement reused across variants)
    bg_rgb = _pil_to_np_rgb(bg_image)
    bg_size = (bg_rgb.shape[1], bg_rgb.shape[0])

    transforms: Dict[str, list[TransformInfo]] = {"real": [], "toy": []}
    obj_masks_canvas: Dict[str, list[np.ndarray]] = {"real": [], "toy": []}

    for kind in ("real", "toy"):
        logger.info("Compositing %s images.", kind)

        for idx, mask in enumerate(masks[kind]):
            suffix = _mask_suffix(idx)
            bbox = _compute_bbox(mask)
            salt = (0xC0A7E1 if kind == "real" else 0xC0A7E2) + idx
            seed_for_place = config.seed_real if kind == "real" else config.seed_toy
            place_rng = random.Random(_derive_run_seed(seed_for_place, config.run_index, salt))

            transform = _compute_transform(mask, bg_size, config.composite, rng=place_rng, bbox=bbox)
            transforms[kind].append(transform)

            rgb_variants_full = transformed_rgb[kind][idx]
            out_scenes, alpha_canvas = _composite_variants(
                bg_rgb,
                bbox=bbox,
                transform=transform,
                alpha_full=alphas[kind][idx],
                rgb_variants_full=rgb_variants_full,
            )
            obj_masks_canvas[kind].append(alpha_canvas)

            _np_to_pil(out_scenes["dom"], "RGB").save(run_dirs.outputs / f"scene_{kind}_dom{suffix}.png")
            _np_to_pil(out_scenes["rare"], "RGB").save(run_dirs.outputs / f"scene_{kind}_rare{suffix}.png")

            scene_diff_outside = _compute_outside_diff(out_scenes["dom"], out_scenes["rare"], alpha_canvas)
            sanity_checks[f"scene_{kind}_outside_max_diff{suffix}"] = scene_diff_outside
            if scene_diff_outside != 0.0:
                logger.error("Scene %s dom/rare differ outside object%s: %.6f", kind, suffix, scene_diff_outside)

            # Alpha placement is shared for dom/rare by construction.
            sanity_checks[f"scene_{kind}_mask_max_diff{suffix}"] = 0.0

    transforms_meta: Dict[str, Any]
    if multi_mask_mode:
        transforms_meta = {
            "real": [asdict(t) for t in transforms["real"]],
            "toy": [asdict(t) for t in transforms["toy"]],
        }
    else:
        transforms_meta = {
            "real": asdict(transforms["real"][0]),
            "toy": asdict(transforms["toy"][0]),
        }

    meta: Dict[str, Any] = {
        "run_index": config.run_index,
        "seed_offset": config.seed_offset,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "background_prompt": config.background_prompt,
        "base_prompt": config.base_prompt,
        "paired_prompt": config.paired_prompt,
        "negative_prompt": config.negative_prompt,
        "object_name_real": config.object_name_real,
        "object_name_toy": config.object_name_toy,
        "object_name": (
            config.object_name_real
            if config.object_name_real == config.object_name_toy
            else f"real={config.object_name_real}, toy={config.object_name_toy}"
        ),
        "base_prompt_elements_json": str(config.base_prompt_elements_path) if config.base_prompt_elements_path else None,
        "paired_prompt_elements_json": str(config.paired_prompt_elements_path) if config.paired_prompt_elements_path else None,
        "prompt_elements": {"base_prompt": config.base_prompt_elements, "paired_prompt": config.paired_prompt_elements},
        "dominant_color": config.dominant_color,
        "rare_color": config.rare_color,
        "seeds": {"seed_bg": config.seed_bg, "seed_real": config.seed_real, "seed_toy": config.seed_toy},
        "flux": asdict(config.flux),
        "sam": {
            **{k: v for k, v in asdict(config.sam).items() if k != "checkpoint"},
            "checkpoint": str(config.sam.checkpoint),
        },
        "color": asdict(config.color),
        "color_transforms": {
            "dominant": asdict(dominant_transform),
            "rare": asdict(rare_transform),
            "min_saturation": config.color.min_saturation,
        },
        "composite": asdict(config.composite),
        "transforms": transforms_meta,
        "multi_mask_mode": multi_mask_mode,
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
