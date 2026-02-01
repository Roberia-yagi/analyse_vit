from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence
import re

import numpy as np
from PIL import Image

from analyse_vit.rare_colour_bias.composite.gen_color import _apply_color_transform, _np_to_pil, _pil_to_np_rgb, _resolve_color_transform
from analyse_vit.rare_colour_bias.composite.gen_mask import _mask_area_frac, _postprocess_mask, _save_mask
from analyse_vit.rare_colour_bias.composite.gen_sam import _load_lang_sam_model, _predict_mask_lang_sam
from analyse_vit.rare_colour_bias.generation.gen_types import ColorTransform
from analyse_vit.rare_colour_bias.generation.gen_utils import (
    _get_cv2_version,
    _get_version,
    _load_required_rgb_image,
    _prepare_run_dirs,
    _resolve_path,
    _resolve_run_input,
    _setup_logger,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


@dataclass
class RecolorConfig:
    run_root: Optional[Path]
    anchors_root: Optional[Path]
    output_root: Optional[Path]
    composite_subdir: Optional[str]
    gen_model: Optional[str]
    animals: Optional[list[str]]
    object_name: str
    colors: list[str]
    color_hue_deg: Optional[float]
    min_saturation: float
    mask_selection_rule: str
    morph_kernel_px: int
    dilate_kernel_px: int
    mask_min_area_frac: float
    mask_max_area_frac: float
    lang_sam_box_threshold: float
    lang_sam_text_threshold: float
    device: str
    hf_token: Optional[str]
    save_rgba: bool
    num_shards: int
    shard_index: int


def _get_stdout_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def _parse_args(argv: Optional[list[str]] = None) -> RecolorConfig:
    parser = argparse.ArgumentParser(description="SAM3 mask + HSV recolor for anchor images (no composite).")
    parser.add_argument("--run-root", default=None, help="Run root (timestamp dir or a single run_* dir).")
    parser.add_argument(
        "--anchors-root",
        default=None,
        help="Structured anchors root (e.g., results/selected/anchors/without_composite).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Structured root to write colour/masks (default: anchors-root/..).",
    )
    parser.add_argument(
        "--composite-subdir",
        default=None,
        help="Composite subdir name to write under colour/masks (e.g., with_composite).",
    )
    parser.add_argument("--gen-model", default=None, help="Filter generation model (flux/qwen/sd3.5).")
    parser.add_argument(
        "--animals",
        default=None,
        help="Comma-separated list of animals to process (optional).",
    )
    parser.add_argument("--object-name", required=True, help="Object name for anchor grounding.")
    parser.add_argument("--color", default="", help="Target color name or numeric hue degrees.")
    parser.add_argument(
        "--colors",
        default=None,
        help="Comma-separated list of target colors (overrides --color).",
    )
    parser.add_argument("--color-hue-deg", type=float, default=None)
    parser.add_argument(
        "--target-hue-deg",
        type=float,
        default=None,
        help="Deprecated: use --color-hue-deg.",
    )
    parser.add_argument("--min-saturation", type=float, default=0.25)

    parser.add_argument(
        "--mask-selection-rule",
        default="center_included_max_area",
        choices=["center_included_max_area", "center_included_max_score", "all_masks"],
    )
    parser.add_argument("--morph-kernel-px", type=int, default=3)
    parser.add_argument("--mask-dilate-px", type=int, default=3)
    parser.add_argument("--mask-min-area-frac", type=float, default=0.01)
    parser.add_argument("--mask-max-area-frac", type=float, default=0.9)
    parser.add_argument("--lang-sam-box-threshold", type=float, default=0.25)
    parser.add_argument("--lang-sam-text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--save-rgba", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args(argv)

    anchors_root = _resolve_path(args.anchors_root) if args.anchors_root else None
    output_root = _resolve_path(args.output_root) if args.output_root else None
    composite_subdir = args.composite_subdir.strip() if isinstance(args.composite_subdir, str) else None
    if anchors_root is not None:
        if not anchors_root.is_dir():
            raise FileNotFoundError(f"--anchors-root does not exist: {anchors_root}")
        if composite_subdir is None:
            anchor_tail = anchors_root.name
            if anchor_tail in ("with_composite", "without_composite"):
                composite_subdir = anchor_tail
        if composite_subdir is None:
            raise ValueError("--composite-subdir is required when using --anchors-root.")
        if output_root is None:
            output_root = anchors_root.parent
        run_root = None
    else:
        if not args.run_root:
            raise ValueError("--run-root is required unless --anchors-root is provided.")
        run_root = _resolve_path(args.run_root)
        if not run_root.exists():
            raise FileNotFoundError(f"--run-root does not exist: {run_root}")

    animals = None
    if args.animals:
        animals = [item for item in (a.strip() for a in args.animals.split(",")) if item]

    object_name = args.object_name.strip()
    if not object_name:
        raise ValueError("--object-name must be non-empty.")
    if args.color_hue_deg is not None and args.target_hue_deg is not None:
        if args.color_hue_deg != args.target_hue_deg:
            raise ValueError("--color-hue-deg and --target-hue-deg must match when both are set.")

    colors_arg = args.colors.strip() if isinstance(args.colors, str) else None
    if colors_arg:
        colors = [item for item in (c.strip() for c in colors_arg.split(",")) if item]
        if not colors:
            raise ValueError("--colors must contain at least one color.")
    else:
        color = args.color.strip()
        if not color:
            raise ValueError("--color must be non-empty.")
        colors = [color]

    color_hue_deg = args.color_hue_deg if args.color_hue_deg is not None else args.target_hue_deg

    if args.mask_min_area_frac <= 0.0 or args.mask_max_area_frac <= 0.0:
        raise ValueError("--mask-min-area-frac/--mask-max-area-frac must be > 0.")
    if args.mask_min_area_frac >= args.mask_max_area_frac:
        raise ValueError("--mask-min-area-frac must be < --mask-max-area-frac.")
    if args.mask_dilate_px < 0:
        raise ValueError("--mask-dilate-px must be >= 0.")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be within [0, num-shards).")

    return RecolorConfig(
        run_root=run_root,
        anchors_root=anchors_root,
        output_root=output_root,
        composite_subdir=composite_subdir,
        gen_model=args.gen_model.strip() if isinstance(args.gen_model, str) and args.gen_model else None,
        animals=animals,
        object_name=object_name,
        colors=colors,
        color_hue_deg=color_hue_deg,
        min_saturation=float(args.min_saturation),
        mask_selection_rule=str(args.mask_selection_rule),
        morph_kernel_px=int(args.morph_kernel_px),
        dilate_kernel_px=int(args.mask_dilate_px),
        mask_min_area_frac=float(args.mask_min_area_frac),
        mask_max_area_frac=float(args.mask_max_area_frac),
        lang_sam_box_threshold=float(args.lang_sam_box_threshold),
        lang_sam_text_threshold=float(args.lang_sam_text_threshold),
        device=str(args.device),
        hf_token=args.hf_token,
        save_rgba=bool(args.save_rgba),
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
    )


def _sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^a-z0-9._-]+", "_", tag.strip().lower())
    return cleaned or "recolor"


def _collect_run_dirs(run_root: Path) -> list[Path]:
    if (run_root / "inputs").exists() or (run_root / "anchor.png").exists():
        return [run_root]
    run_dirs = sorted(
        [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.name,
    )
    if run_dirs:
        return run_dirs
    return [run_root]


def _apply_recolor_to_anchor(
    rgb: np.ndarray,
    transform: ColorTransform,
    mask: np.ndarray,
    mask_suffix: str,
    output_tag: str,
    *,
    kind: str,
    run_dirs,
    logger,
    config: RecolorConfig,
) -> None:
    rgb_out = _apply_color_transform(rgb, mask, transform, config.min_saturation)
    _np_to_pil(rgb_out, "RGB").save(run_dirs.outputs / f"anchor_{output_tag}{mask_suffix}.png")

    if config.save_rgba:
        rgba = np.dstack([rgb_out, mask])
        _np_to_pil(rgba, "RGBA").save(run_dirs.outputs / f"anchor_{output_tag}_rgba{mask_suffix}.png")


def _run_recolor_on_dir(run_dir: Path, config: RecolorConfig, lang_sam_model) -> None:
    run_dirs = _prepare_run_dirs(run_dir)
    logger = _setup_logger(run_dirs)
    logger.info("Starting anchor recolor stage in %s.", run_dir)

    output_tags = [_sanitize_tag(color) for color in config.colors]
    already_done = [
        tag
        for tag in output_tags
        if any(run_dirs.outputs.glob(f"anchor_{tag}*.png"))
    ]
    if len(already_done) == len(output_tags):
        logger.info("Recolor outputs already exist; skipping SAM mask for %s.", run_dir)
        return

    anchor_images = {
        "primary": _load_required_rgb_image(_resolve_run_input(run_dirs, "anchor.png")),
    }
    anchor_rgbs = {kind: _pil_to_np_rgb(img) for kind, img in anchor_images.items()}

    color_transforms = {
        color: _resolve_color_transform(color, config.color_hue_deg) for color in config.colors
    }

    masks_raw: dict[str, np.ndarray] = {}
    for kind in ("primary",):
        prompt = config.object_name
        logger.info("Predicting SAM3 mask for %s (%s).", kind, prompt)
        mask_raw = _predict_mask_lang_sam(
            lang_sam_model,
            anchor_images[kind],
            prompt,
            config.lang_sam_box_threshold,
            config.lang_sam_text_threshold,
            config.mask_selection_rule,
        )
        if mask_raw is None:
            logger.warning("SAM3 did not return a mask for %s; skipping this run.", kind)
            return
        masks_raw[kind] = mask_raw

    multi_mask_mode = any(raw.ndim == 3 for raw in masks_raw.values())
    mask_stats: dict[str, list[float]] = {"primary": []}

    def _mask_suffix(mask_index: int) -> str:
        return f"_m{mask_index:02d}" if multi_mask_mode else ""

    for kind in ("primary",):
        raw = masks_raw[kind]
        mask_list = [raw] if raw.ndim == 2 else [raw[i] for i in range(raw.shape[0])]
        if not mask_list:
            logger.warning("No masks returned for %s; skipping this run.", kind)
            return
        for idx, mask_raw in enumerate(mask_list):
            mask_suffix = _mask_suffix(idx)
            mask = _postprocess_mask(
                mask_raw,
                config.morph_kernel_px,
                dilate_kernel_px=config.dilate_kernel_px,
            )
            _save_mask(mask, run_dirs.masks / f"mask{mask_suffix}.png")

            area = _mask_area_frac(mask)
            mask_stats[kind].append(area)
            if not (config.mask_min_area_frac <= area <= config.mask_max_area_frac):
                logger.error("Mask area out of bounds%s: %.4f", mask_suffix, area)
            if area == 0.0:
                logger.warning("Mask is empty; skipping%s.", mask_suffix)
                continue

            for color, transform in color_transforms.items():
                output_tag = _sanitize_tag(color)
                _apply_recolor_to_anchor(
                    anchor_rgbs[kind],
                    transform,
                    mask,
                    mask_suffix,
                    output_tag,
                    kind=kind,
                    run_dirs=run_dirs,
                    logger=logger,
                    config=config,
                )

    meta = {
        "argv": sys.argv,
        "run_root": str(run_dir),
        "object_name": config.object_name,
        "colors": config.colors,
        "color_hue_deg": config.color_hue_deg,
        "min_saturation": config.min_saturation,
        "mask_selection_rule": config.mask_selection_rule,
        "morph_kernel_px": config.morph_kernel_px,
        "mask_dilate_px": config.dilate_kernel_px,
        "mask_min_area_frac": config.mask_min_area_frac,
        "mask_max_area_frac": config.mask_max_area_frac,
        "lang_sam_box_threshold": config.lang_sam_box_threshold,
        "lang_sam_text_threshold": config.lang_sam_text_threshold,
        "device": config.device,
        "color_transforms": {name: asdict(transform) for name, transform in color_transforms.items()},
        "mask_area_frac": mask_stats,
        "multi_mask_mode": multi_mask_mode,
        "num_shards": config.num_shards,
        "shard_index": config.shard_index,
        "library_versions": {
            "numpy": _get_version("numpy"),
            "Pillow": _get_version("Pillow"),
            "torch": _get_version("torch"),
            "opencv": _get_cv2_version() or _get_version("opencv-python"),
        },
    }

    meta_path = run_dirs.meta / "recolor_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)

    logger.info("Recolor complete for %s.", run_dir)


def _extract_run_tag(path: Path) -> str:
    match = re.search(r"run[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        return f"run_{match.group(1)}"
    return path.stem


def _iter_anchor_images(
    anchors_root: Path,
    gen_model: Optional[str],
    animals: Optional[Sequence[str]],
) -> List[Tuple[Path, str, str]]:
    items: List[Tuple[Path, str, str]] = []
    for animal_dir in sorted(anchors_root.iterdir()):
        if not animal_dir.is_dir():
            continue
        animal = animal_dir.name
        if animals and animal not in animals:
            continue
        for model_dir in sorted(animal_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if gen_model and model != gen_model:
                continue
            images_dir = model_dir / "images"
            search_dir = images_dir if images_dir.is_dir() else model_dir
            for path in sorted(search_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    items.append((path, animal, model))
    return items


def _run_recolor_on_image(
    image_path: Path,
    animal: str,
    gen_model: str,
    config: RecolorConfig,
    object_name: str,
    lang_sam_model,
) -> None:
    logger = _get_stdout_logger("analyse_vit.structured_recolor")
    anchor_image = Image.open(image_path.resolve()).convert("RGB")
    anchor_rgb = _pil_to_np_rgb(anchor_image)
    output_tags = [_sanitize_tag(color) for color in config.colors]
    color_transforms = {
        color: _resolve_color_transform(color, config.color_hue_deg) for color in config.colors
    }

    mask_raw = _predict_mask_lang_sam(
        lang_sam_model,
        anchor_image,
        object_name,
        config.lang_sam_box_threshold,
        config.lang_sam_text_threshold,
        config.mask_selection_rule,
    )
    if mask_raw is None:
        logger.warning("SAM3 did not return a mask for %s; skipping.", image_path)
        return

    multi_mask_mode = mask_raw.ndim == 3
    mask_list = [mask_raw] if mask_raw.ndim == 2 else [mask_raw[i] for i in range(mask_raw.shape[0])]
    if not mask_list:
        logger.warning("No masks returned for %s; skipping.", image_path)
        return

    run_tag = _extract_run_tag(image_path)
    if config.composite_subdir is None:
        raise SystemExit("--composite-subdir is required when using --anchors-root.")

    if config.anchors_root is None:
        raise SystemExit("--anchors-root is required when using --composite-subdir.")

    outputs_ready = True
    for output_tag in output_tags:
        out_dir = (
            config.output_root
            / "colour"
            / config.composite_subdir
            / animal
            / output_tag
            / gen_model
            / "images"
        )
        if not any(out_dir.glob(f"{animal}_{output_tag}_{run_tag}*.png")):
            outputs_ready = False
            break
    if outputs_ready:
        logger.info("Outputs already exist; skipping SAM prediction: %s", image_path)
        return

    masks_root = config.anchors_root / animal / gen_model / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)

    def _mask_suffix(mask_index: int) -> str:
        return f"_m{mask_index:02d}" if multi_mask_mode else ""

    for idx, mask_raw_item in enumerate(mask_list):
        mask_suffix = _mask_suffix(idx)
        mask = _postprocess_mask(
            mask_raw_item,
            config.morph_kernel_px,
            dilate_kernel_px=config.dilate_kernel_px,
        )
        _save_mask(mask, masks_root / f"mask_{run_tag}{mask_suffix}.png")

        area = _mask_area_frac(mask)
        if not (config.mask_min_area_frac <= area <= config.mask_max_area_frac):
            logger.error("Mask area out of bounds%s: %.4f (%s)", mask_suffix, area, image_path)
        if area == 0.0:
            logger.warning("Mask is empty; skipping%s (%s).", mask_suffix, image_path)
            continue

        for color, transform in color_transforms.items():
            output_tag = _sanitize_tag(color)
            rgb = _apply_color_transform(anchor_rgb, mask, transform, config.min_saturation)
            out_dir = (
                config.output_root
                / "colour"
                / config.composite_subdir
                / animal
                / output_tag
                / gen_model
                / "images"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{animal}_{output_tag}_{run_tag}{mask_suffix}.png"
            out_path = out_dir / out_name
            if out_path.exists():
                logger.info("Output already exists; skipping save: %s", out_path)
            else:
                _np_to_pil(rgb, "RGB").save(out_path)
            if config.save_rgba:
                rgba = np.dstack([rgb, mask])
                rgba_name = f"{animal}_{output_tag}_{run_tag}_rgba{mask_suffix}.png"
                rgba_path = out_dir / rgba_name
                if rgba_path.exists():
                    logger.info("Output already exists; skipping save: %s", rgba_path)
                else:
                    _np_to_pil(rgba, "RGBA").save(rgba_path)

    logger.info("Recolor complete for %s.", image_path)


def main_recolor(argv: Optional[list[str]] = None) -> None:
    config = _parse_args(argv)
    lang_sam_model = _load_lang_sam_model(config.device, config.hf_token)
    if config.anchors_root is not None and config.output_root is not None:
        anchor_items = _iter_anchor_images(config.anchors_root, config.gen_model, config.animals)
        if not anchor_items:
            raise SystemExit(f"No anchor images found under {config.anchors_root}")
        if config.num_shards > 1:
            anchor_items = anchor_items[config.shard_index :: config.num_shards]
            if not anchor_items:
                raise SystemExit("No anchor images assigned to this shard.")
        for path, animal, gen_model in anchor_items:
            prompt_name = animal if config.object_name == "auto" else config.object_name
            _run_recolor_on_image(path, animal, gen_model, config, prompt_name, lang_sam_model)
        return

    if config.run_root is None:
        raise SystemExit("No run root provided.")
    run_dirs = _collect_run_dirs(config.run_root)
    for run_dir in run_dirs:
        _run_recolor_on_dir(run_dir, config, lang_sam_model)
