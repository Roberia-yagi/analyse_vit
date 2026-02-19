from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

from analyse_vit.rare_colour_bias.generation.gen_utils import _find_repo_root, _resolve_path


PlacementMode = str


def _parse_sizes(raw: str) -> List[int]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("--sizes must include at least one value.")
    sizes: List[int] = []
    for item in items:
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid size value: {item}") from exc
        if value <= 0:
            raise ValueError("--sizes values must be > 0.")
        sizes.append(value)
    return sizes


def _parse_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or None


def _load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def _load_mask(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode in {"RGBA", "LA"}:
        img = img.split()[-1]
    else:
        img = img.convert("L")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def _apply_morph(alpha: np.ndarray, kernel_px: int, mode: str) -> np.ndarray:
    if kernel_px <= 0:
        return alpha
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for morph operations.") from exc

    kernel = np.ones((kernel_px, kernel_px), dtype=np.uint8)
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

    if mode == "erode":
        out = cv2.erode(alpha_u8, kernel, iterations=1)
    elif mode == "dilate":
        out = cv2.dilate(alpha_u8, kernel, iterations=1)
    elif mode == "open":
        out = cv2.morphologyEx(alpha_u8, cv2.MORPH_OPEN, kernel)
    elif mode == "close":
        out = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(f"Invalid morph mode: {mode}")

    return out.astype(np.float32) / 255.0


def _apply_feather(alpha: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return alpha
    img = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img).astype(np.float32) / 255.0


def _bbox_from_alpha(alpha: np.ndarray, threshold: float) -> Tuple[int, int, int, int]:
    ys, xs = np.where(alpha > threshold)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Mask has no foreground after thresholding.")
    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())
    return min_x, min_y, max_x, max_y


def _resize_array(arr: np.ndarray, size: Tuple[int, int], *, is_mask: bool) -> np.ndarray:
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("Resize size must be positive.")
    if is_mask:
        mode = "L"
        img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode=mode)
    else:
        mode = "RGB"
        img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode=mode)
    resized = img.resize(size, Image.BICUBIC)
    out = np.asarray(resized).astype(np.float32) / 255.0
    return out


def _placement(
    bg_size: Tuple[int, int],
    obj_size: Tuple[int, int],
    mode: PlacementMode,
    *,
    anchor: Tuple[float, float],
    bottom_margin_ratio: float,
    rng: Optional[random.Random] = None,
) -> Tuple[int, int]:
    bg_w, bg_h = bg_size
    obj_w, obj_h = obj_size
    if obj_w <= 0 or obj_h <= 0:
        raise ValueError("Object size must be positive.")

    if mode == "center":
        x = int(round((bg_w - obj_w) / 2.0))
        y = int(round((bg_h - obj_h) / 2.0))
    elif mode == "bottom_center":
        margin = int(round(bg_h * bottom_margin_ratio))
        x = int(round((bg_w - obj_w) / 2.0))
        y = bg_h - obj_h - margin
    elif mode == "anchor":
        anchor_x, anchor_y = anchor
        x = int(round(anchor_x * bg_w - obj_w / 2.0))
        y = int(round(anchor_y * bg_h - obj_h))
    elif mode == "random":
        if rng is None:
            raise ValueError("Random placement requires RNG.")
        max_x = max(bg_w - obj_w, 0)
        max_y = max(bg_h - obj_h, 0)
        x = rng.randint(0, max_x) if max_x > 0 else 0
        y = rng.randint(0, max_y) if max_y > 0 else 0
    else:
        raise ValueError(f"Invalid placement mode: {mode}")

    x = min(max(x, 0), max(bg_w - obj_w, 0))
    y = min(max(y, 0), max(bg_h - obj_h, 0))
    return x, y


def _composite(
    bg: np.ndarray,
    fg: np.ndarray,
    alpha: np.ndarray,
    paste_xy: Tuple[int, int],
) -> np.ndarray:
    bg_h, bg_w = bg.shape[:2]
    obj_h, obj_w = fg.shape[:2]
    x0, y0 = paste_xy
    x1 = x0 + obj_w
    y1 = y0 + obj_h

    ix0 = max(x0, 0)
    iy0 = max(y0, 0)
    ix1 = min(x1, bg_w)
    iy1 = min(y1, bg_h)
    if ix0 >= ix1 or iy0 >= iy1:
        raise ValueError("Object does not overlap background.")

    sx0 = ix0 - x0
    sy0 = iy0 - y0
    sx1 = sx0 + (ix1 - ix0)
    sy1 = sy0 + (iy1 - iy0)

    out = bg.copy()
    alpha_roi = alpha[sy0:sy1, sx0:sx1][..., None]
    fg_roi = fg[sy0:sy1, sx0:sx1, :]
    bg_roi = out[iy0:iy1, ix0:ix1, :]
    blended = fg_roi * alpha_roi + bg_roi * (1.0 - alpha_roi)
    out[iy0:iy1, ix0:ix1, :] = blended
    return out


def _stable_seed(seed: int, *items: str) -> int:
    payload = "|".join([str(seed), *items]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def _iter_anchor_images(anchors_root: Path) -> Iterable[Path]:
    yield from anchors_root.rglob("images/anchor_run_*.png")


def _parse_anchor_path(anchor_path: Path, anchors_root: Path) -> Tuple[str, str, str]:
    rel = anchor_path.relative_to(anchors_root)
    parts = rel.parts
    if len(parts) != 4 or parts[2] != "images":
        raise ValueError(
            "Expected anchor path as anchors/{animal}/{model}/images/anchor_run_XX.png"
            f" but got: {rel}"
        )
    animal = parts[0]
    model = parts[1]
    stem = anchor_path.stem
    if not stem.startswith("anchor_run_"):
        raise ValueError(f"Unexpected anchor filename: {anchor_path.name}")
    run_id = stem.split("anchor_run_", 1)[1]
    if not run_id:
        raise ValueError(f"Missing run id in anchor filename: {anchor_path.name}")
    return animal, model, run_id


def _build_mask_path(masks_root: Path, animal: str, model: str, run_id: str) -> Path:
    return masks_root / animal / model / f"mask_run_{run_id}.png"


def _build_background_path(background_root: Path, model: str) -> Path:
    return background_root / model / "bg.png"


def _save_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    img.save(path)


def _anchor_run_id(anchor_path: Path) -> str:
    stem = anchor_path.stem
    if "anchor_run_" not in stem:
        raise ValueError(f"Unexpected anchor filename: {anchor_path.name}")
    run_id = stem.split("anchor_run_", 1)[1]
    if not run_id:
        raise ValueError(f"Missing run id in anchor filename: {anchor_path.name}")
    return run_id


def _compose_sizes(
    *,
    anchor_path: Path,
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
) -> None:
    anchor_rgb = _load_rgb(anchor_path)
    mask = _load_mask(mask_path)
    if anchor_rgb.shape[0] != mask.shape[0] or anchor_rgb.shape[1] != mask.shape[1]:
        raise ValueError(f"Anchor/mask size mismatch: {anchor_path} vs {mask_path}")

    if mask_invert:
        mask = 1.0 - mask

    mask = np.clip(mask, 0.0, 1.0)
    mask = _apply_morph(mask, morph_kernel_px, morph_mode)
    mask = _apply_feather(mask, feather_radius)

    bbox = _bbox_from_alpha(mask, mask_threshold)
    min_x, min_y, max_x, max_y = bbox
    crop_rgb = anchor_rgb[min_y : max_y + 1, min_x : max_x + 1, :]
    crop_alpha = mask[min_y : max_y + 1, min_x : max_x + 1]

    bg = _load_rgb(background_path)
    bg_h, bg_w = bg.shape[:2]

    crop_h, crop_w = crop_rgb.shape[:2]
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError("Empty crop after masking.")

    for size in sizes:
        scale = size / 100.0
        target_w = max(1, int(round(crop_w * scale)))
        target_h = max(1, int(round(crop_h * scale)))

        resized_rgb = _resize_array(crop_rgb, (target_w, target_h), is_mask=False)
        resized_alpha = _resize_array(crop_alpha, (target_w, target_h), is_mask=True)

        rng = None
        if placement_mode == "random":
            derived = _stable_seed(seed, anchor_path.as_posix(), background_path.as_posix(), str(size))
            rng = random.Random(derived)

        paste_xy = _placement(
            (bg_w, bg_h),
            (target_w, target_h),
            placement_mode,
            anchor=anchor_xy,
            bottom_margin_ratio=bottom_margin_ratio,
            rng=rng,
        )

        composite = _composite(bg, resized_rgb, resized_alpha, paste_xy)

        size_dir = output_root / str(size)
        out_path = size_dir / f"{size}_run_{_anchor_run_id(anchor_path)}.png"
        _save_image(out_path, composite)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Composite anchors onto backgrounds with size sweep.")
    parser.add_argument(
        "--selected-root",
        default=None,
        help="Root directory that contains anchors, masks/anchors, and background.",
    )
    parser.add_argument("--anchors-root", default=None)
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

    parser.add_argument("--animals", default=None, help="Comma-separated list of animals to process.")
    parser.add_argument("--models", default=None, help="Comma-separated list of gen models to process.")

    args = parser.parse_args(argv)

    repo_root = _find_repo_root(Path.cwd())
    if args.selected_root is None:
        if repo_root is None:
            raise ValueError("--selected-root is required when repo root cannot be found.")
        selected_root = repo_root / "results" / "selected"
    else:
        selected_root = _resolve_path(args.selected_root)

    anchors_root = (
        _resolve_path(args.anchors_root)
        if args.anchors_root
        else selected_root / "anchors" / "without_composite"
    )
    masks_root = _resolve_path(args.masks_root) if args.masks_root else selected_root / "masks" / "anchors"
    background_root = _resolve_path(args.background_root) if args.background_root else selected_root / "background"
    output_root = _resolve_path(args.output_root) if args.output_root else selected_root / "size"

    sizes = _parse_sizes(args.sizes)
    animals_filter = _parse_list(args.animals)
    models_filter = _parse_list(args.models)

    if not (0.0 <= args.anchor_x <= 1.0 and 0.0 <= args.anchor_y <= 1.0):
        raise ValueError("--anchor-x/--anchor-y must be in [0, 1].")
    if args.bottom_margin_ratio < 0.0:
        raise ValueError("--bottom-margin-ratio must be >= 0.")
    if not (0.0 <= args.mask_threshold <= 1.0):
        raise ValueError("--mask-threshold must be in [0, 1].")

    if args.background_mode == "fixed-model" and not args.background_model:
        raise ValueError("--background-model is required for fixed-model mode.")
    if args.background_mode == "path" and not args.background_path:
        raise ValueError("--background-path is required for path mode.")

    if not anchors_root.exists():
        raise ValueError(f"Anchors root not found: {anchors_root}")
    if not masks_root.exists():
        raise ValueError(f"Masks root not found: {masks_root}")
    if args.background_mode != "path" and not background_root.exists():
        raise ValueError(f"Background root not found: {background_root}")

    anchor_paths = list(_iter_anchor_images(anchors_root))
    if not anchor_paths:
        raise ValueError(f"No anchor images found under {anchors_root}")

    for anchor_path in sorted(anchor_paths):
        animal, model, run_id = _parse_anchor_path(anchor_path, anchors_root)
        if animals_filter and animal not in animals_filter:
            continue
        if models_filter and model not in models_filter:
            continue

        mask_path = _build_mask_path(masks_root, animal, model, run_id)
        if not mask_path.exists():
            print(f"[WARN] Mask not found for {anchor_path}: {mask_path}")
            continue

        if args.background_mode == "match-model":
            background_path = _build_background_path(background_root, model)
        elif args.background_mode == "fixed-model":
            background_path = _build_background_path(background_root, args.background_model)
        else:
            background_path = _resolve_path(args.background_path)

        if not background_path.exists():
            raise ValueError(f"Background not found: {background_path}")

        output_dir = output_root / animal / model / "images"

        _compose_sizes(
            anchor_path=anchor_path,
            mask_path=mask_path,
            background_path=background_path,
            output_root=output_dir,
            sizes=sizes,
            placement_mode=args.placement,
            anchor_xy=(args.anchor_x, args.anchor_y),
            bottom_margin_ratio=args.bottom_margin_ratio,
            seed=args.seed,
            mask_threshold=args.mask_threshold,
            mask_invert=args.mask_invert,
            feather_radius=args.feather_radius,
            morph_kernel_px=args.morph_kernel_px,
            morph_mode=args.morph_mode,
        )


if __name__ == "__main__":
    main()
