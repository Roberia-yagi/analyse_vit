from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .gen_color import _apply_color_transform, _np_to_pil, _pil_to_np_rgb, _resolve_color_transform
from .gen_mask import _mask_area_frac, _postprocess_mask, _save_mask
from .gen_sam import _load_lang_sam_model, _predict_mask_lang_sam
from .gen_types import ColorTransform
from .gen_utils import (
    _get_cv2_version,
    _get_version,
    _load_required_rgb_image,
    _prepare_run_dirs,
    _resolve_path,
    _resolve_run_input,
    _setup_logger,
)


@dataclass
class RecolorConfig:
    run_root: Path
    object_name_real: str
    object_name_toy: str
    normal_color: str
    atypical_color: str
    atypical_hue_deg: Optional[float]
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


def _parse_args(argv: Optional[list[str]] = None) -> RecolorConfig:
    parser = argparse.ArgumentParser(description="SAM3 mask + HSV recolor for anchor images (no composite).")
    parser.add_argument("--run-root", required=True, help="Run root (timestamp dir or a single run_* dir).")
    parser.add_argument("--object-name-real", required=True, help="Object name for real anchor grounding.")
    parser.add_argument(
        "--object-name-toy",
        default=None,
        help="Object name for toy anchor grounding (defaults to --object-name-real).",
    )
    parser.add_argument("--normal-color", default=None, help="Normal (dominant) color name or numeric hue degrees.")
    parser.add_argument("--atypical-color", default=None, help="Atypical (rare) color name or numeric hue degrees.")
    parser.add_argument("--atypical-hue-deg", type=float, default=None)
    parser.add_argument(
        "--target-color",
        default=None,
        help="Deprecated: use --atypical-color.",
    )
    parser.add_argument(
        "--target-hue-deg",
        type=float,
        default=None,
        help="Deprecated: use --atypical-hue-deg.",
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

    args = parser.parse_args(argv)

    run_root = _resolve_path(args.run_root)
    if not run_root.exists():
        raise FileNotFoundError(f"--run-root does not exist: {run_root}")

    object_name_real = args.object_name_real.strip()
    if not object_name_real:
        raise ValueError("--object-name-real must be non-empty.")
    object_name_toy = (args.object_name_toy or object_name_real).strip()
    if not object_name_toy:
        raise ValueError("--object-name-toy must be non-empty.")

    if args.target_color and args.atypical_color and args.target_color != args.atypical_color:
        raise ValueError("--target-color and --atypical-color must match when both are set.")
    if args.target_hue_deg is not None and args.atypical_hue_deg is not None and args.target_hue_deg != args.atypical_hue_deg:
        raise ValueError("--target-hue-deg and --atypical-hue-deg must match when both are set.")

    normal_color = (args.normal_color or "brown").strip()
    if not normal_color:
        raise ValueError("--normal-color must be non-empty.")

    atypical_color = (args.atypical_color or args.target_color or "pink").strip()
    if not atypical_color:
        raise ValueError("--atypical-color must be non-empty.")

    atypical_hue_deg = args.atypical_hue_deg if args.atypical_hue_deg is not None else args.target_hue_deg

    if args.mask_min_area_frac <= 0.0 or args.mask_max_area_frac <= 0.0:
        raise ValueError("--mask-min-area-frac/--mask-max-area-frac must be > 0.")
    if args.mask_min_area_frac >= args.mask_max_area_frac:
        raise ValueError("--mask-min-area-frac must be < --mask-max-area-frac.")
    if args.mask_dilate_px < 0:
        raise ValueError("--mask-dilate-px must be >= 0.")

    return RecolorConfig(
        run_root=run_root,
        object_name_real=object_name_real,
        object_name_toy=object_name_toy,
        normal_color=normal_color,
        atypical_color=atypical_color,
        atypical_hue_deg=atypical_hue_deg,
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
    )


def _collect_run_dirs(run_root: Path) -> list[Path]:
    if (run_root / "inputs").exists() or (run_root / "real_anchor.png").exists():
        return [run_root]
    run_dirs = sorted(
        [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.name,
    )
    if run_dirs:
        return run_dirs
    return [run_root]


def _apply_recolor_to_anchor(
    image: Image.Image,
    mask_raw: np.ndarray,
    transform: ColorTransform,
    mask_suffix: str,
    output_tag: str,
    *,
    kind: str,
    run_dirs,
    logger,
    config: RecolorConfig,
    mask_stats: dict[str, list[float]],
) -> None:
    mask = _postprocess_mask(
        mask_raw,
        config.morph_kernel_px,
        dilate_kernel_px=config.dilate_kernel_px,
    )
    _save_mask(mask, run_dirs.masks / f"{kind}_mask{mask_suffix}.png")

    area = _mask_area_frac(mask)
    mask_stats[kind].append(area)
    if not (config.mask_min_area_frac <= area <= config.mask_max_area_frac):
        logger.error("%s mask area out of bounds%s: %.4f", kind.capitalize(), mask_suffix, area)
    if area == 0.0:
        logger.warning("Mask is empty; skipping %s%s.", kind, mask_suffix)
        return

    rgb = _pil_to_np_rgb(image)
    rgb_out = _apply_color_transform(rgb, mask, transform, config.min_saturation)
    _np_to_pil(rgb_out, "RGB").save(run_dirs.outputs / f"{kind}_{output_tag}{mask_suffix}.png")

    if config.save_rgba:
        rgba = np.dstack([rgb_out, mask])
        _np_to_pil(rgba, "RGBA").save(run_dirs.outputs / f"{kind}_{output_tag}_rgba{mask_suffix}.png")


def _run_recolor_on_dir(run_dir: Path, config: RecolorConfig, lang_sam_model) -> None:
    run_dirs = _prepare_run_dirs(run_dir)
    logger = _setup_logger(run_dirs)
    logger.info("Starting anchor recolor stage in %s.", run_dir)

    anchor_images = {
        "real": _load_required_rgb_image(_resolve_run_input(run_dirs, "real_anchor.png")),
        "toy": _load_required_rgb_image(_resolve_run_input(run_dirs, "toy_anchor.png")),
    }

    normal_transform = _resolve_color_transform(config.normal_color, None)
    atypical_transform = _resolve_color_transform(config.atypical_color, config.atypical_hue_deg)

    masks_raw: dict[str, np.ndarray] = {}
    for kind in ("real", "toy"):
        prompt = config.object_name_real if kind == "real" else config.object_name_toy
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
    mask_stats: dict[str, list[float]] = {"real": [], "toy": []}

    def _mask_suffix(mask_index: int) -> str:
        return f"_m{mask_index:02d}" if multi_mask_mode else ""

    for kind in ("real", "toy"):
        raw = masks_raw[kind]
        mask_list = [raw] if raw.ndim == 2 else [raw[i] for i in range(raw.shape[0])]
        if not mask_list:
            logger.warning("No masks returned for %s; skipping this run.", kind)
            return
        for idx, mask_raw in enumerate(mask_list):
            mask_suffix = _mask_suffix(idx)
            for output_tag, transform in (
                ("normal", normal_transform),
                ("atypical", atypical_transform),
            ):
                _apply_recolor_to_anchor(
                    anchor_images[kind],
                    mask_raw,
                    transform,
                    mask_suffix,
                    output_tag,
                    kind=kind,
                    run_dirs=run_dirs,
                    logger=logger,
                    config=config,
                    mask_stats=mask_stats,
                )

    meta = {
        "argv": sys.argv,
        "run_root": str(run_dir),
        "object_name_real": config.object_name_real,
        "object_name_toy": config.object_name_toy,
        "normal_color": config.normal_color,
        "atypical_color": config.atypical_color,
        "atypical_hue_deg": config.atypical_hue_deg,
        "min_saturation": config.min_saturation,
        "mask_selection_rule": config.mask_selection_rule,
        "morph_kernel_px": config.morph_kernel_px,
        "mask_dilate_px": config.dilate_kernel_px,
        "mask_min_area_frac": config.mask_min_area_frac,
        "mask_max_area_frac": config.mask_max_area_frac,
        "lang_sam_box_threshold": config.lang_sam_box_threshold,
        "lang_sam_text_threshold": config.lang_sam_text_threshold,
        "device": config.device,
        "color_transforms": {
            "normal": asdict(normal_transform),
            "atypical": asdict(atypical_transform),
        },
        "mask_area_frac": mask_stats,
        "multi_mask_mode": multi_mask_mode,
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


def main_recolor(argv: Optional[list[str]] = None) -> None:
    config = _parse_args(argv)
    run_dirs = _collect_run_dirs(config.run_root)
    lang_sam_model = _load_lang_sam_model(config.device, config.hf_token)
    for run_dir in run_dirs:
        _run_recolor_on_dir(run_dir, config, lang_sam_model)
