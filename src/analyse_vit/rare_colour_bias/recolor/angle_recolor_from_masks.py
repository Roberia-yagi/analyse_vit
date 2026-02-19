from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image

from analyse_vit.rare_colour_bias.composite.gen_color import (
    _apply_color_transform,
    _np_to_pil,
    _pil_to_np_rgb,
    _resolve_color_transform,
)
from analyse_vit.rare_colour_bias.generation.gen_utils import _find_repo_root, _resolve_path

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _parse_list(raw: Optional[str]) -> Optional[list[str]]:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or None


def _parse_colors(colors_raw: Optional[str], color_raw: Optional[str]) -> list[str]:
    if colors_raw:
        colors = [item.strip() for item in colors_raw.split(",") if item.strip()]
        if not colors:
            raise ValueError("--colors must contain at least one color.")
        return colors
    if color_raw:
        color = color_raw.strip()
        if color:
            return [color]
    raise ValueError("Either --colors or --color must be provided.")


def _extract_run_id(path: Path) -> str:
    stem = path.stem
    match = re.search(r"run[_-]?(\d+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    raise ValueError(f"Could not parse run id from filename: {path.name}")


def _load_mask(mask_path: Path) -> np.ndarray:
    mask_img = Image.open(mask_path)
    if mask_img.mode in {"RGBA", "LA"}:
        mask_img = mask_img.split()[-1]
    else:
        mask_img = mask_img.convert("L")
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    return np.clip(mask, 0.0, 1.0)


def _iter_angle_images(
    angles_root: Path,
    *,
    animals: Optional[Sequence[str]],
    angles: Optional[Sequence[str]],
    gen_model: Optional[str],
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
                if gen_model and model != gen_model:
                    continue
                images_dir = model_dir / "images"
                if not images_dir.is_dir():
                    raise FileNotFoundError(f"Angle images dir not found: {images_dir}")
                image_paths = [
                    p
                    for p in sorted(images_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and p.stem.startswith("run_")
                ]
                for image_path in image_paths:
                    yield image_path, animal, angle, model


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply colour transforms to angle images using precomputed masks."
    )
    parser.add_argument(
        "--selected-root",
        default=None,
        help="Selected root (default: repo_root/results/selected).",
    )
    parser.add_argument(
        "--angles-root",
        default=None,
        help="Angle images root (default: <selected-root>/angles/with_composite).",
    )
    parser.add_argument(
        "--masks-root",
        default=None,
        help="Angle masks root (default: <selected-root>/masks/angles).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root (default: <selected-root>/colour_angles/with_composite).",
    )
    parser.add_argument("--animals", default=None, help="Comma-separated animal filter.")
    parser.add_argument("--angles", default=None, help="Comma-separated angle filter.")
    parser.add_argument("--gen-model", default=None, help="Generation model filter (flux/qwen/sd3.5).")
    parser.add_argument("--color", default=None, help="Single target color.")
    parser.add_argument("--colors", default=None, help="Comma-separated target colors.")
    parser.add_argument("--color-hue-deg", type=float, default=None)
    parser.add_argument("--target-hue-deg", type=float, default=None, help="Deprecated alias of --color-hue-deg.")
    parser.add_argument("--min-saturation", type=float, default=0.25)
    parser.add_argument("--save-rgba", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if args.color_hue_deg is not None and args.target_hue_deg is not None:
        if args.color_hue_deg != args.target_hue_deg:
            raise ValueError("--color-hue-deg and --target-hue-deg must match when both are set.")
    color_hue_deg = args.color_hue_deg if args.color_hue_deg is not None else args.target_hue_deg

    repo_root = _find_repo_root(Path(__file__).resolve())
    if args.selected_root is None:
        if repo_root is None:
            raise SystemExit("Could not find repo root; pass --selected-root explicitly.")
        selected_root = (repo_root / "results" / "selected").resolve()
    else:
        selected_root = _resolve_path(args.selected_root)

    angles_root = _resolve_path(args.angles_root) if args.angles_root else selected_root / "angles" / "with_composite"
    masks_root = _resolve_path(args.masks_root) if args.masks_root else selected_root / "masks" / "angles"
    output_root = _resolve_path(args.output_root) if args.output_root else selected_root / "colour_angles" / "with_composite"

    if not selected_root.is_dir():
        raise FileNotFoundError(f"Selected root not found: {selected_root}")
    if not angles_root.is_dir():
        raise FileNotFoundError(f"Angles root not found: {angles_root}")
    if not masks_root.is_dir():
        raise FileNotFoundError(f"Angle masks root not found: {masks_root}")

    animals = _parse_list(args.animals)
    angles = _parse_list(args.angles)
    colors = _parse_colors(args.colors, args.color)

    color_transforms = {
        color: _resolve_color_transform(color, color_hue_deg) for color in colors
    }

    items = list(
        _iter_angle_images(
            angles_root,
            animals=animals,
            angles=angles,
            gen_model=args.gen_model.strip() if isinstance(args.gen_model, str) and args.gen_model else None,
        )
    )
    if not items:
        raise SystemExit(f"No angle images found under {angles_root}")

    total_images = len(items)
    total_colors = len(colors)
    total_tasks = total_images * total_colors
    completed_tasks = 0
    written_tasks = 0
    skipped_tasks = 0
    print(
        f"[progress] start images={total_images} colors={total_colors} "
        f"tasks={total_tasks} overwrite={int(bool(args.overwrite))}",
        flush=True,
    )

    for image_index, (image_path, animal, angle, model) in enumerate(items, start=1):
        run_id = _extract_run_id(image_path)
        mask_path = masks_root / animal / angle / model / f"mask_run_{run_id}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(
                "Angle mask not found (no fallback): "
                f"{mask_path} for image {image_path}"
            )

        image = Image.open(image_path.resolve()).convert("RGB")
        rgb = _pil_to_np_rgb(image)
        mask = _load_mask(mask_path)

        if rgb.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image/mask size mismatch: {image_path} vs {mask_path}")
        if not np.any(mask > 0.5):
            raise ValueError(f"Mask has no foreground pixels: {mask_path}")

        for color, transform in color_transforms.items():
            output_tag = re.sub(r"[^a-z0-9._-]+", "_", color.strip().lower()) or "recolor"
            out_dir = output_root / animal / angle / output_tag / model / "images"
            out_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{animal}_{angle}_{output_tag}_run_{run_id}.png"
            out_path = out_dir / out_name
            if out_path.exists() and not args.overwrite:
                completed_tasks += 1
                skipped_tasks += 1
                continue

            recolored = _apply_color_transform(rgb, mask, transform, float(args.min_saturation))
            _np_to_pil(recolored, "RGB").save(out_path)
            completed_tasks += 1
            written_tasks += 1

            if args.save_rgba:
                rgba = np.dstack([recolored, mask])
                rgba_name = f"{animal}_{angle}_{output_tag}_run_{run_id}_rgba.png"
                rgba_path = out_dir / rgba_name
                if rgba_path.exists() and not args.overwrite:
                    continue
                _np_to_pil(rgba, "RGBA").save(rgba_path)

        progress_pct = 100.0 * float(completed_tasks) / float(total_tasks) if total_tasks > 0 else 100.0
        print(
            f"[progress] image={image_index}/{total_images} "
            f"tasks={completed_tasks}/{total_tasks} ({progress_pct:.1f}%) "
            f"written={written_tasks} skipped={skipped_tasks} "
            f"last={animal}/{angle}/{model}/run_{run_id}",
            flush=True,
        )

    print(
        f"[progress] done tasks={completed_tasks}/{total_tasks} "
        f"written={written_tasks} skipped={skipped_tasks}",
        flush=True,
    )


if __name__ == "__main__":
    main()
