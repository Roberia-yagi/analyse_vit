from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import re

import numpy as np
from PIL import Image

from .gen_color import _apply_color_transform, _np_to_pil, _pil_to_np_rgb, _resolve_color_transform
from .gen_composite import _composite_variants, _compute_transform
from .gen_mask import _compute_alpha, _compute_bbox, _mask_area_frac, _postprocess_mask, _save_mask
from ..generation.gen_pipeline import _generate_images, _load_text2image_pipeline
from .gen_sam import _load_lang_sam_model, _predict_mask_lang_sam
from ..generation.gen_types import RunConfig, RunDirs, TransformInfo
from ..generation.gen_utils import (
    _derive_run_seed,
    _json_safe,
    _get_cv2_version,
    _get_version,
    _collect_runtime_info,
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
    runtime_info = _collect_runtime_info(run_configs[0].output_dir if run_configs else None)

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
        logger.info("Prompt: %s", cfg.prompt)
        logger.info("Negative prompt: %s", cfg.negative_prompt)
        logger.info("Background prompt: %s", cfg.background_prompt)
        logger.info("Prompt elements json: %s", cfg.prompt_elements_path)
        logger.info("Prompt elements: %s", cfg.prompt_elements)
        logger.info("Seeds: bg=%d primary=%d", cfg.seed_bg, cfg.seed)
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
        timings: Dict[str, Optional[float]] = {
            "background_seconds": None,
            "anchor_seconds": None,
        }
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
                timings["background_seconds"] = time.perf_counter() - start
                logger.info("Background generation finished in %.2fs.", timings["background_seconds"])
            else:
                bg_image = shared_bg
                logger.info("Reusing shared background image for this run.")
        else:
            logger.info("Skipping background generation because --background-prompt was not provided.")

        if cfg.flux.pipeline == "qwen":
            logger.info("Generating anchor image sequentially (qwen uses more VRAM per sample).")
            logger.info("Waiting for diffusion inference (base anchor)...")
            start = time.perf_counter()
            anchor_image = _generate_images(
                pipe,
                [cfg.prompt],
                [cfg.seed],
                cfg.flux.width,
                cfg.flux.height,
                cfg.flux.num_inference_steps,
                guidance_scale,
                cfg.flux.max_sequence_length,
                cfg.device,
                negative_prompt,
            )[0]
            timings["anchor_seconds"] = time.perf_counter() - start
            logger.info("Primary anchor finished in %.2fs.", timings["anchor_seconds"])
        else:
            logger.info("Generating anchor image (single prompt).")
            logger.info("Waiting for diffusion inference (this can take a while)...")
            start = time.perf_counter()
            anchor_image = _generate_images(
                pipe,
                [cfg.prompt],
                [cfg.seed],
                cfg.flux.width,
                cfg.flux.height,
                cfg.flux.num_inference_steps,
                guidance_scale,
                cfg.flux.max_sequence_length,
                cfg.device,
                negative_prompt,
            )[0]
            timings["anchor_seconds"] = time.perf_counter() - start
            logger.info("Anchor generation finished in %.2fs.", timings["anchor_seconds"])

        logger.info("Saving generated inputs...")
        if bg_image is not None:
            bg_image.save(run_dirs.inputs / "bg.png")
        anchor_image.save(run_dirs.inputs / "anchor.png")
        logger.info("Saved generated inputs to %s", run_dirs.inputs)

        generation_meta: Dict[str, Any] = {
            "stage": "generation",
            "run_index": cfg.run_index,
            "seed_offset": cfg.seed_offset,
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "prompt": cfg.prompt,
            "negative_prompt": cfg.negative_prompt,
            "background_prompt": cfg.background_prompt,
            "prompt_elements_json": str(cfg.prompt_elements_path) if cfg.prompt_elements_path else None,
            "prompt_elements": cfg.prompt_elements,
            "color": cfg.color_name,
            "seeds": {"seed_bg": cfg.seed_bg, "seed": cfg.seed},
            "flux": asdict(cfg.flux),
            "effective_guidance_scale": guidance_scale,
            "sam": {
                **{k: v for k, v in asdict(cfg.sam).items() if k != "checkpoint"},
                "checkpoint": str(cfg.sam.checkpoint),
            },
            "composite": asdict(cfg.composite),
            "color": asdict(cfg.color_params),
            "device": cfg.device,
            "hf_token_present": bool(cfg.hf_token),
            "timings": timings,
            "outputs": {
                "root": str(run_dirs.root),
                "inputs": {
                    "bg": str(run_dirs.inputs / "bg.png") if bg_image is not None else None,
                    "anchor": str(run_dirs.inputs / "anchor.png"),
                },
            },
            "runtime": runtime_info,
        }

        with (run_dirs.meta / "generation_meta.json").open("w", encoding="utf-8") as f:
            json.dump(_json_safe(generation_meta), f, indent=2, ensure_ascii=True)

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
        "primary": _load_required_rgb_image(_resolve_run_input(run_dirs, "anchor.png")),
    }

    if config.sam.prompt_mode == "grounding":
        if not config.object_name:
            raise ValueError("--object-name is required when --sam-prompt-mode grounding is used.")
        logger.info("Using object name for grounding SAM (%s)", config.object_name)
        if lang_sam_model is None:
            logger.info("Loading SAM3 model for text-guided masks.")
            lang_sam_model = _load_lang_sam_model(config.device)
    else:
        raise ValueError("grounding 以外の SAM プロンプトは現在無効です。")

    color_transform = _resolve_color_transform(config.color_name, config.color_params.target_hue_deg)

    # Predict masks (loop)
    masks_raw: Dict[str, np.ndarray] = {}
    for kind in ("primary",):
        logger.info("Predicting %s mask.", kind)
        if config.sam.prompt_mode == "grounding":
            prompt = config.object_name
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
    masks: Dict[str, list[np.ndarray]] = {"primary": []}
    alphas: Dict[str, list[np.ndarray]] = {"primary": []}

    sanity_checks: Dict[str, Any] = {}

    def _mask_suffix(mask_index: int) -> str:
        return f"_m{mask_index:02d}" if multi_mask_mode else ""

    for kind in ("primary",):
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
            _save_mask(mask, run_dirs.masks / f"mask{suffix}.png")
            masks[kind].append(mask)

            area = _mask_area_frac(mask)
            sanity_checks[f"mask_area_frac{suffix}"] = area
            ok = config.sam.min_area_frac <= area <= config.sam.max_area_frac
            sanity_checks[f"mask_area_ok{suffix}"] = ok
            if not ok:
                logger.error("Mask area out of bounds%s: %.4f", suffix, area)
            if area == 0.0:
                logger.warning("Mask is empty; skipping this run.")
                return

            alphas[kind].append(_compute_alpha(mask, config.sam.feather_radius_px))

    # Color transforms (loop)
    anchor_rgb: Dict[str, np.ndarray] = {k: _pil_to_np_rgb(anchor_images[k]) for k in ("primary",)}

    def _sanitize_tag(tag: str) -> str:
        cleaned = re.sub(r"[^a-z0-9._-]+", "_", tag.strip().lower())
        return cleaned or "color"

    output_tag = _sanitize_tag(config.color_name)

    variants = {
        output_tag: color_transform,
    }

    transformed_rgb: Dict[str, list[Dict[str, np.ndarray]]] = {"primary": []}

    for kind in ("primary",):
        for idx, mask in enumerate(masks[kind]):
            suffix = _mask_suffix(idx)
            rgb_variants: Dict[str, np.ndarray] = {}

            for var_name, transform in variants.items():
                logger.info("Applying %s color transformation to %s%s.", var_name, kind, suffix)
                rgb_out = _apply_color_transform(
                    anchor_rgb[kind],
                    mask,
                    transform,
                    config.color_params.min_saturation,
                )
                rgb_variants[var_name] = rgb_out
                _np_to_pil(rgb_out, "RGB").save(run_dirs.outputs / f"anchor_{var_name}{suffix}.png")

            transformed_rgb[kind].append(rgb_variants)

            # Save RGBA variants (shared alpha)
            for var_name in variants:
                rgba = np.dstack([rgb_variants[var_name], alphas[kind][idx]])
                _np_to_pil(rgba, "RGBA").save(run_dirs.outputs / f"anchor_{var_name}_rgba{suffix}.png")

            # Single-color pipeline: no cross-variant sanity check.

    # Composite on background (alpha placement reused across variants)
    bg_rgb = _pil_to_np_rgb(bg_image)
    bg_size = (bg_rgb.shape[1], bg_rgb.shape[0])

    transforms: Dict[str, list[TransformInfo]] = {"primary": []}
    obj_masks_canvas: Dict[str, list[np.ndarray]] = {"primary": []}

    for kind in ("primary",):
        logger.info("Compositing anchor images.")

        for idx, mask in enumerate(masks[kind]):
            suffix = _mask_suffix(idx)
            bbox = _compute_bbox(mask)
            salt = 0xC0A7E1 + idx
            seed_for_place = config.seed
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

            _np_to_pil(out_scenes[output_tag], "RGB").save(run_dirs.outputs / f"scene_{output_tag}{suffix}.png")

    transforms_meta: Dict[str, Any]
    if multi_mask_mode:
        transforms_meta = {
            "anchor": [asdict(t) for t in transforms["primary"]],
        }
    else:
        transforms_meta = {
            "anchor": asdict(transforms["primary"][0]),
        }

    meta: Dict[str, Any] = {
        "run_index": config.run_index,
        "seed_offset": config.seed_offset,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "background_prompt": config.background_prompt,
        "prompt": config.prompt,
        "negative_prompt": config.negative_prompt,
        "object_name": config.object_name,
        "prompt_elements_json": str(config.prompt_elements_path) if config.prompt_elements_path else None,
        "prompt_elements": config.prompt_elements,
        "color": config.color_name,
        "seeds": {"seed_bg": config.seed_bg, "seed": config.seed},
        "flux": asdict(config.flux),
        "sam": {
            **{k: v for k, v in asdict(config.sam).items() if k != "checkpoint"},
            "checkpoint": str(config.sam.checkpoint),
        },
        "color": asdict(config.color_params),
        "color_transform": {
            **asdict(color_transform),
            "min_saturation": config.color_params.min_saturation,
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
