from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from analyse_vit.rare_colour_bias.composite.size_composite import (
    _apply_feather,
    _apply_morph,
    _bbox_from_alpha,
    _build_background_path,
    _composite,
    _load_mask,
    _load_rgb,
    _placement,
    _resize_array,
    _save_image,
    _stable_seed,
)
from analyse_vit.rare_colour_bias.generation.gen_utils import _find_repo_root, _resolve_path

PlacementMode = str
TargetKind = str
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _parse_sizes(raw: str) -> List[int]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("--sizes must include at least one value.")
    values: List[int] = []
    for item in items:
        try:
            size = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid size value: {item}") from exc
        if size <= 0:
            raise ValueError("--sizes values must be > 0.")
        values.append(size)
    return values


def _parse_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or None


def _extract_run_id(path: Path) -> str:
    match = re.search(r"run[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    raise ValueError(f"Unexpected image filename (run id not found): {path}")


def _iter_anchor_colour_images(
    input_root: Path,
    *,
    animals: Optional[Sequence[str]],
    models: Optional[Sequence[str]],
    colours: Optional[Sequence[str]],
) -> Iterable[Tuple[Path, str, str, str]]:
    for animal_dir in sorted(input_root.iterdir()):
        if not animal_dir.is_dir():
            continue
        animal = animal_dir.name
        if animals and animal not in animals:
            continue
        for colour_dir in sorted(animal_dir.iterdir()):
            if not colour_dir.is_dir():
                continue
            colour = colour_dir.name
            if colours and colour not in colours:
                continue
            for model_dir in sorted(colour_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                if models and model not in models:
                    continue
                images_dir = model_dir / "images"
                if not images_dir.is_dir():
                    raise FileNotFoundError(f"Images dir not found: {images_dir}")
                for image_path in sorted(images_dir.iterdir()):
                    if not image_path.is_file():
                        continue
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    yield image_path, animal, colour, model


def _iter_angle_colour_images(
    input_root: Path,
    *,
    animals: Optional[Sequence[str]],
    angles: Optional[Sequence[str]],
    models: Optional[Sequence[str]],
    colours: Optional[Sequence[str]],
) -> Iterable[Tuple[Path, str, str, str, str]]:
    for animal_dir in sorted(input_root.iterdir()):
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
            for colour_dir in sorted(angle_dir.iterdir()):
                if not colour_dir.is_dir():
                    continue
                colour = colour_dir.name
                if colours and colour not in colours:
                    continue
                for model_dir in sorted(colour_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    model = model_dir.name
                    if models and model not in models:
                        continue
                    images_dir = model_dir / "images"
                    if not images_dir.is_dir():
                        raise FileNotFoundError(f"Images dir not found: {images_dir}")
                    for image_path in sorted(images_dir.iterdir()):
                        if not image_path.is_file():
                            continue
                        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                            continue
                        yield image_path, animal, angle, colour, model


def _iter_angle_images_raw(
    input_root: Path,
    *,
    animals: Optional[Sequence[str]],
    angles: Optional[Sequence[str]],
    models: Optional[Sequence[str]],
) -> Iterable[Tuple[Path, str, str, str]]:
    for animal_dir in sorted(input_root.iterdir()):
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
                    raise FileNotFoundError(f"Images dir not found: {images_dir}")
                for image_path in sorted(images_dir.iterdir()):
                    if not image_path.is_file():
                        continue
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    if not image_path.stem.startswith("run_"):
                        continue
                    yield image_path, animal, angle, model


def _compose_sizes_for_image(
    *,
    image_path: Path,
    mask_path: Path,
    background_path: Path,
    output_root: Path,
    sizes: Sequence[int],
    placement_mode: PlacementMode,
    anchor_xy: Tuple[float, float],
    bottom_margin_ratio: float,
    seed: int,
    mask_threshold: float,
    mask_invert: bool,
    feather_radius: float,
    morph_kernel_px: int,
    morph_mode: str,
    overwrite: bool,
) -> Tuple[int, int]:
    rgb = _load_rgb(image_path)
    mask = _load_mask(mask_path)
    if rgb.shape[0] != mask.shape[0] or rgb.shape[1] != mask.shape[1]:
        raise ValueError(f"Image/mask size mismatch: {image_path} vs {mask_path}")

    if mask_invert:
        mask = 1.0 - mask
    mask = _apply_morph(mask.clip(0.0, 1.0), morph_kernel_px, morph_mode)
    mask = _apply_feather(mask, feather_radius)

    bbox = _bbox_from_alpha(mask, mask_threshold)
    min_x, min_y, max_x, max_y = bbox
    crop_rgb = rgb[min_y : max_y + 1, min_x : max_x + 1, :]
    crop_alpha = mask[min_y : max_y + 1, min_x : max_x + 1]
    if crop_rgb.shape[0] <= 0 or crop_rgb.shape[1] <= 0:
        raise ValueError(f"Empty crop after masking: {image_path}")

    bg = _load_rgb(background_path)
    bg_h, bg_w = bg.shape[:2]
    crop_h, crop_w = crop_rgb.shape[:2]

    written = 0
    skipped = 0
    for size in sizes:
        out_dir = output_root / str(size)
        out_path = out_dir / f"{size}_{image_path.stem}.png"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        scale = float(size) / 100.0
        target_w = max(1, int(round(crop_w * scale)))
        target_h = max(1, int(round(crop_h * scale)))

        resized_rgb = _resize_array(crop_rgb, (target_w, target_h), is_mask=False)
        resized_alpha = _resize_array(crop_alpha, (target_w, target_h), is_mask=True)

        rng = None
        if placement_mode == "random":
            derived_seed = _stable_seed(seed, image_path.as_posix(), background_path.as_posix(), str(size))
            rng = random.Random(derived_seed)

        paste_xy = _placement(
            (bg_w, bg_h),
            (target_w, target_h),
            placement_mode,
            anchor=anchor_xy,
            bottom_margin_ratio=bottom_margin_ratio,
            rng=rng,
        )
        composed = _composite(bg, resized_rgb, resized_alpha, paste_xy)
        _save_image(out_path, composed)
        written += 1

    return written, skipped


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply size sweep to anchor/angle images using precomputed masks."
    )
    parser.add_argument("--selected-root", default=None)
    parser.add_argument("--target", default="anchors", choices=["anchors", "angles", "angles_colour", "angles_raw"])
    parser.add_argument("--composite-subdir", default="with_composite")
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--masks-root", default=None)
    parser.add_argument("--background-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--background-mode", default="match-model", choices=["match-model", "fixed-model", "path"])
    parser.add_argument("--background-model", default=None)
    parser.add_argument("--background-path", default=None)

    parser.add_argument("--sizes", default="100,80,60,40,20")
    parser.add_argument("--placement", default="center", choices=["bottom_center", "center", "anchor", "random"])
    parser.add_argument("--anchor-x", type=float, default=0.5)
    parser.add_argument("--anchor-y", type=float, default=0.9)
    parser.add_argument("--bottom-margin-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--mask-invert", action="store_true")
    parser.add_argument("--feather-radius", type=float, default=3.0)
    parser.add_argument("--morph-kernel-px", type=int, default=0)
    parser.add_argument("--morph-mode", default="erode", choices=["erode", "dilate", "open", "close"])
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--animals", default=None, help="Comma-separated animal list.")
    parser.add_argument("--angles", default=None, help="Comma-separated angle list (target=angles/angles_raw only).")
    parser.add_argument("--models", default=None, help="Comma-separated model list.")
    parser.add_argument("--colours", "--colors", dest="colours", default=None, help="Comma-separated colour list.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    repo_root = _find_repo_root(Path.cwd())
    if args.selected_root is None:
        if repo_root is None:
            raise ValueError("--selected-root is required when repo root cannot be found.")
        selected_root = (repo_root / "results" / "selected").resolve()
    else:
        selected_root = _resolve_path(args.selected_root)

    target: TargetKind = str(args.target)
    composite_subdir = str(args.composite_subdir)
    if target not in {"anchors", "angles", "angles_colour", "angles_raw"}:
        raise ValueError(f"--target must be anchors/angles/angles_colour/angles_raw: {target}")
    if composite_subdir not in {"with_composite", "without_composite"}:
        raise ValueError(f"--composite-subdir must be with_composite or without_composite: {composite_subdir}")
    angle_colour_target = target in {"angles", "angles_colour"}

    if args.input_root:
        input_root = _resolve_path(args.input_root)
    else:
        if target == "anchors":
            input_root = selected_root / "colour" / composite_subdir
        elif angle_colour_target:
            input_root = selected_root / "colour_angles" / composite_subdir
        else:
            input_root = selected_root / "angles" / composite_subdir

    if args.masks_root:
        masks_root = _resolve_path(args.masks_root)
    else:
        masks_root = selected_root / "masks" / ("anchors" if target == "anchors" else "angles")

    background_root = _resolve_path(args.background_root) if args.background_root else selected_root / "background"

    if args.output_root:
        output_root = _resolve_path(args.output_root)
    else:
        if target == "anchors":
            output_root = selected_root / "colour_size" / composite_subdir
        elif angle_colour_target:
            output_root = selected_root / "colour_size_angles" / composite_subdir
        else:
            output_root = selected_root / "angle_size" / composite_subdir

    sizes = _parse_sizes(args.sizes)
    animals_filter = _parse_list(args.animals)
    angles_filter = _parse_list(args.angles)
    models_filter = _parse_list(args.models)
    colours_filter = _parse_list(args.colours)

    if not selected_root.is_dir():
        raise ValueError(f"Selected root not found: {selected_root}")
    if not input_root.is_dir():
        raise ValueError(f"Input root not found: {input_root}")
    if not masks_root.is_dir():
        raise ValueError(f"Masks root not found: {masks_root}")
    if args.background_mode != "path" and not background_root.is_dir():
        raise ValueError(f"Background root not found: {background_root}")
    if args.background_mode == "fixed-model" and not args.background_model:
        raise ValueError("--background-model is required for fixed-model mode.")
    if args.background_mode == "path" and not args.background_path:
        raise ValueError("--background-path is required for path mode.")
    if not (0.0 <= args.anchor_x <= 1.0 and 0.0 <= args.anchor_y <= 1.0):
        raise ValueError("--anchor-x/--anchor-y must be in [0, 1].")
    if args.bottom_margin_ratio < 0.0:
        raise ValueError("--bottom-margin-ratio must be >= 0.")
    if not (0.0 <= args.mask_threshold <= 1.0):
        raise ValueError("--mask-threshold must be in [0, 1].")

    if target == "anchors":
        items = list(
            _iter_anchor_colour_images(
                input_root,
                animals=animals_filter,
                models=models_filter,
                colours=colours_filter,
            )
        )
    elif angle_colour_target:
        items = list(
            _iter_angle_colour_images(
                input_root,
                animals=animals_filter,
                angles=angles_filter,
                models=models_filter,
                colours=colours_filter,
            )
        )
    else:
        items = list(
            _iter_angle_images_raw(
                input_root,
                animals=animals_filter,
                angles=angles_filter,
                models=models_filter,
            )
        )

    if not items:
        raise ValueError(f"No images found under {input_root}")

    total_images = len(items)
    total_tasks = total_images * len(sizes)
    done_tasks = 0
    written_tasks = 0
    skipped_tasks = 0
    print(
        f"[progress] start target={target} images={total_images} sizes={len(sizes)} "
        f"tasks={total_tasks} overwrite={int(bool(args.overwrite))}",
        flush=True,
    )

    for image_index, item in enumerate(items, start=1):
        if target == "anchors":
            image_path, animal, colour, model = item
            run_id = _extract_run_id(image_path)
            mask_path = masks_root / animal / model / f"mask_run_{run_id}.png"
            output_dir = output_root / animal / colour / model / "images"
        elif angle_colour_target:
            image_path, animal, angle, colour, model = item
            run_id = _extract_run_id(image_path)
            mask_path = masks_root / animal / angle / model / f"mask_run_{run_id}.png"
            output_dir = output_root / animal / angle / colour / model / "images"
        else:
            image_path, animal, angle, model = item
            run_id = _extract_run_id(image_path)
            mask_path = masks_root / animal / angle / model / f"mask_run_{run_id}.png"
            output_dir = output_root / animal / angle / model / "images"

        if not mask_path.is_file():
            raise FileNotFoundError(
                "Mask not found (no fallback): "
                f"{mask_path} for image {image_path}"
            )

        if args.background_mode == "match-model":
            background_path = _build_background_path(background_root, model)
        elif args.background_mode == "fixed-model":
            background_path = _build_background_path(background_root, str(args.background_model))
        else:
            background_path = _resolve_path(args.background_path)
        if not background_path.exists():
            raise ValueError(f"Background not found: {background_path}")

        written, skipped = _compose_sizes_for_image(
            image_path=image_path,
            mask_path=mask_path,
            background_path=background_path,
            output_root=output_dir,
            sizes=sizes,
            placement_mode=str(args.placement),
            anchor_xy=(float(args.anchor_x), float(args.anchor_y)),
            bottom_margin_ratio=float(args.bottom_margin_ratio),
            seed=int(args.seed),
            mask_threshold=float(args.mask_threshold),
            mask_invert=bool(args.mask_invert),
            feather_radius=float(args.feather_radius),
            morph_kernel_px=int(args.morph_kernel_px),
            morph_mode=str(args.morph_mode),
            overwrite=bool(args.overwrite),
        )
        written_tasks += written
        skipped_tasks += skipped
        done_tasks += len(sizes)
        progress = 100.0 * float(done_tasks) / float(total_tasks) if total_tasks > 0 else 100.0
        print(
            f"[progress] image={image_index}/{total_images} "
            f"tasks={done_tasks}/{total_tasks} ({progress:.1f}%) "
            f"written={written_tasks} skipped={skipped_tasks} "
            f"last={image_path}",
            flush=True,
        )

    print(
        f"[progress] done target={target} tasks={done_tasks}/{total_tasks} "
        f"written={written_tasks} skipped={skipped_tasks}",
        flush=True,
    )


if __name__ == "__main__":
    main()
