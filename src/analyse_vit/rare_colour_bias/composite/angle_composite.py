from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image

from analyse_vit.rare_colour_bias.generation.gen_utils import _find_repo_root, _resolve_path
from analyse_vit.rare_colour_bias.composite.gen_color import _np_to_pil, _pil_to_np_rgb
from analyse_vit.rare_colour_bias.composite.gen_mask import _compute_alpha, _mask_area_frac, _postprocess_mask, _save_mask
from analyse_vit.rare_colour_bias.composite.gen_sam import _load_lang_sam_model, _predict_mask_lang_sam

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


@dataclass
class AngleCompositeConfig:
    selected_root: Path
    angles_root: Path
    output_root: Path
    masks_root: Path
    background_root: Path
    animals: Optional[Sequence[str]]
    angles: Optional[Sequence[str]]
    models: Optional[Sequence[str]]
    object_name: str
    mask_selection_rule: str
    morph_kernel_px: int
    dilate_kernel_px: int
    mask_min_area_frac: float
    mask_max_area_frac: float
    feather_radius_px: int
    lang_sam_box_threshold: float
    lang_sam_text_threshold: float
    device: str
    hf_token: Optional[str]
    overwrite: bool


def _parse_list(raw: Optional[str]) -> Optional[list[str]]:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or None


def _load_mask(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode in {"RGBA", "LA"}:
        img = img.split()[-1]
    else:
        img = img.convert("L")
    return np.asarray(img).astype(np.float32) / 255.0


def _parse_args(argv: Optional[list[str]] = None) -> AngleCompositeConfig:
    parser = argparse.ArgumentParser(
        description="Extract SAM3 masks for angle images and composite onto model backgrounds."
    )
    parser.add_argument(
        "--selected-root",
        default=None,
        help="Root directory containing angles/background (default: repo_root/results/selected).",
    )
    parser.add_argument(
        "--angles-root",
        default=None,
        help="Angles root (default: selected_root/angles/without_composite).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root for composites (default: selected_root/angles/with_composite).",
    )
    parser.add_argument(
        "--masks-root",
        default=None,
        help="Output root for masks (default: selected_root/masks/angles).",
    )
    parser.add_argument(
        "--background-root",
        default=None,
        help="Background root (default: selected_root/background).",
    )
    parser.add_argument("--animals", default=None, help="Comma-separated animal list (optional).")
    parser.add_argument("--angles", default=None, help="Comma-separated angle list (optional).")
    parser.add_argument("--models", default=None, help="Comma-separated model list (optional).")
    parser.add_argument(
        "--object-name",
        default="auto",
        help="Object prompt for SAM (default: auto -> use animal name).",
    )
    parser.add_argument(
        "--mask-selection-rule",
        default="center_included_max_area",
        choices=["center_included_max_area", "center_included_max_score", "all_masks"],
    )
    parser.add_argument("--morph-kernel-px", type=int, default=3)
    parser.add_argument("--mask-dilate-px", type=int, default=3)
    parser.add_argument("--mask-min-area-frac", type=float, default=0.01)
    parser.add_argument("--mask-max-area-frac", type=float, default=0.9)
    parser.add_argument("--feather-radius-px", type=int, default=3)
    parser.add_argument("--lang-sam-box-threshold", type=float, default=0.25)
    parser.add_argument("--lang-sam-text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.mask_min_area_frac <= 0.0 or args.mask_max_area_frac <= 0.0:
        raise ValueError("--mask-min-area-frac/--mask-max-area-frac must be > 0.")
    if args.mask_min_area_frac >= args.mask_max_area_frac:
        raise ValueError("--mask-min-area-frac must be < --mask-max-area-frac.")
    if args.mask_dilate_px < 0:
        raise ValueError("--mask-dilate-px must be >= 0.")

    selected_root = _resolve_path(args.selected_root) if args.selected_root else None
    if selected_root is None:
        repo_root = _find_repo_root(Path(__file__).resolve())
        if repo_root is None:
            raise SystemExit("Could not find repo root; pass --selected-root explicitly.")
        selected_root = (repo_root / "results" / "selected").resolve()

    angles_root = _resolve_path(args.angles_root) if args.angles_root else selected_root / "angles" / "without_composite"
    output_root = _resolve_path(args.output_root) if args.output_root else selected_root / "angles" / "with_composite"
    masks_root = _resolve_path(args.masks_root) if args.masks_root else selected_root / "masks" / "angles"
    background_root = _resolve_path(args.background_root) if args.background_root else selected_root / "background"

    if not angles_root.is_dir():
        raise FileNotFoundError(f"Angles root not found: {angles_root}")
    if not background_root.is_dir():
        raise FileNotFoundError(f"Background root not found: {background_root}")

    object_name = str(args.object_name).strip()
    if not object_name:
        raise ValueError("--object-name must be non-empty.")

    return AngleCompositeConfig(
        selected_root=selected_root,
        angles_root=angles_root,
        output_root=output_root,
        masks_root=masks_root,
        background_root=background_root,
        animals=_parse_list(args.animals),
        angles=_parse_list(args.angles),
        models=_parse_list(args.models),
        object_name=object_name,
        mask_selection_rule=str(args.mask_selection_rule),
        morph_kernel_px=int(args.morph_kernel_px),
        dilate_kernel_px=int(args.mask_dilate_px),
        mask_min_area_frac=float(args.mask_min_area_frac),
        mask_max_area_frac=float(args.mask_max_area_frac),
        feather_radius_px=int(args.feather_radius_px),
        lang_sam_box_threshold=float(args.lang_sam_box_threshold),
        lang_sam_text_threshold=float(args.lang_sam_text_threshold),
        device=str(args.device),
        hf_token=args.hf_token,
        overwrite=bool(args.overwrite),
    )


def _iter_angle_images(
    angles_root: Path,
    animals: Optional[Sequence[str]],
    angles: Optional[Sequence[str]],
    models: Optional[Sequence[str]],
) -> Iterable[tuple[Path, str, str, str]]:
    for animal_dir in sorted(angles_root.iterdir()):
        if not animal_dir.is_dir():
            continue
        animal = animal_dir.name
        if animals and animal not in animals:
            continue
        for angle_dir in sorted(animal_dir.iterdir()):
            if not angle_dir.is_dir():
                continue
            angle = angle_dir.name
            if angles and angle not in angles:
                continue
            for model_dir in sorted(angle_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                if models and model not in models:
                    continue
                images_dir = model_dir / "images"
                if not images_dir.is_dir():
                    raise FileNotFoundError(f"Images directory not found: {images_dir}")
                for path in sorted(images_dir.iterdir()):
                    if not path.is_file():
                        continue
                    if path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    if not path.stem.startswith("run_"):
                        continue
                    yield path, animal, angle, model


def _extract_run_id(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("run_"):
        raise ValueError(f"Unexpected image filename (expected run_XX): {path.name}")
    run_id = stem.split("run_", 1)[1]
    if not run_id:
        raise ValueError(f"Missing run id in filename: {path.name}")
    return run_id


def _load_background(background_root: Path, model: str, size: tuple[int, int]) -> np.ndarray:
    bg_path = background_root / model / "bg.png"
    if not bg_path.exists():
        raise FileNotFoundError(f"Background not found for model '{model}': {bg_path}")
    bg_img = Image.open(bg_path).convert("RGB")
    if bg_img.size != size:
        raise ValueError(
            f"Background size mismatch for model '{model}': {bg_img.size} vs {size} "
            "(sizes must match; no fallback resizing)."
        )
    return _pil_to_np_rgb(bg_img)


def _composite_on_background(fg_rgb: np.ndarray, bg_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if fg_rgb.shape[:2] != bg_rgb.shape[:2]:
        raise ValueError("Foreground/background size mismatch.")
    if alpha.shape != fg_rgb.shape[:2]:
        raise ValueError("Alpha mask size mismatch.")
    a = alpha[..., None]
    return np.clip(fg_rgb * a + bg_rgb * (1.0 - a), 0.0, 1.0)


def _run_angle_composite(
    image_path: Path,
    animal: str,
    angle: str,
    model: str,
    config: AngleCompositeConfig,
    lang_sam_model,
) -> None:
    logger = logging.getLogger("analyse_vit.angle_composite")
    prompt_name = animal if config.object_name == "auto" else config.object_name

    run_id = _extract_run_id(image_path)
    masks_root = config.masks_root / animal / angle / model
    masks_root.mkdir(parents=True, exist_ok=True)

    mask_path = masks_root / f"mask_run_{run_id}.png"
    output_dir = config.output_root / animal / angle / model / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"run_{run_id}.png"

    if not config.overwrite and output_path.exists():
        if mask_path.exists():
            logger.info("Outputs already exist; skipping %s.", image_path)
        else:
            logger.warning(
                "Composite exists but mask is missing; skipping %s (mask=%s).",
                image_path,
                mask_path,
            )
        return
    fg_image = Image.open(image_path.resolve()).convert("RGB")
    if not config.overwrite and mask_path.exists():
        logger.info("Mask already exists; skipping SAM prediction for %s.", image_path)
        mask = _load_mask(mask_path)
    else:
        mask_raw = _predict_mask_lang_sam(
            lang_sam_model,
            fg_image,
            prompt_name,
            config.lang_sam_box_threshold,
            config.lang_sam_text_threshold,
            config.mask_selection_rule,
        )
        if mask_raw is None:
            logger.warning("SAM3 did not return a mask for %s; skipping.", image_path)
            return

        mask_list = [mask_raw] if mask_raw.ndim == 2 else [mask_raw[i] for i in range(mask_raw.shape[0])]
        if not mask_list:
            logger.warning("No masks returned for %s; skipping.", image_path)
            return

        if len(mask_list) > 1:
            raise RuntimeError(f"Multiple masks returned for {image_path}; expected a single mask.")

        mask = _postprocess_mask(
            mask_list[0],
            config.morph_kernel_px,
            dilate_kernel_px=config.dilate_kernel_px,
        )
        _save_mask(mask, mask_path)

    area = _mask_area_frac(mask)
    if not (config.mask_min_area_frac <= area <= config.mask_max_area_frac):
        raise RuntimeError(f"Mask area out of bounds: {area:.4f} ({image_path})")
    if area == 0.0:
        raise RuntimeError(f"Mask is empty: {image_path}")

    fg_rgb = _pil_to_np_rgb(fg_image)
    bg_rgb = _load_background(config.background_root, model, fg_image.size)
    alpha = _compute_alpha(mask, config.feather_radius_px)
    composite = _composite_on_background(fg_rgb, bg_rgb, alpha)
    _np_to_pil(composite, "RGB").save(output_path)
    logger.info("Saved mask and composite for %s.", image_path)


def main(argv: Optional[list[str]] = None) -> None:
    config = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lang_sam_model = _load_lang_sam_model(config.device, config.hf_token)
    items = list(_iter_angle_images(config.angles_root, config.animals, config.angles, config.models))
    if not items:
        raise SystemExit(f"No angle images found under {config.angles_root}")
    for path, animal, angle, model in items:
        _run_angle_composite(path, animal, angle, model, config, lang_sam_model)


if __name__ == "__main__":
    main()
