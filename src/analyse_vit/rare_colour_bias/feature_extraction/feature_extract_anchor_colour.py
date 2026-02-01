from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image

from analyse_vit.rare_colour_bias.feature_extraction.feature_extraction import MODEL_SPECS, _load_model, _poolings_for_spec, _resolve_models
from analyse_vit.rare_colour_bias.generation.gen_utils import _get_hf_token

LOGGER = logging.getLogger("analyse_vit.feature_extract_anchor_colour")

IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
COMPOSITE_SUBDIRS = ("without_composite", "with_composite")
TARGET_KINDS = ("anchors", "colour")
TARGET_ALIASES = {
    "anchors": "anchors",
    "anchor": "anchors",
    "colour": "colour",
    "colours": "colour",
    "color": "colour",
    "colors": "colour",
}


def _default_input_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "results" / "selected").resolve()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_composites(requested: Sequence[str] | None) -> Tuple[str, ...]:
    if not requested:
        return COMPOSITE_SUBDIRS
    invalid = [c for c in requested if c not in COMPOSITE_SUBDIRS]
    if invalid:
        raise ValueError(f"Composite {invalid} not supported. Allowed: {COMPOSITE_SUBDIRS}")
    return tuple(requested)


def _resolve_targets(requested: Sequence[str] | None) -> Tuple[str, ...]:
    if not requested:
        return TARGET_KINDS
    resolved: List[str] = []
    invalid: List[str] = []
    for target in requested:
        key = TARGET_ALIASES.get(target)
        if not key:
            invalid.append(target)
            continue
        if key not in resolved:
            resolved.append(key)
    if invalid:
        raise ValueError(f"Target {invalid} not supported. Allowed: {sorted(TARGET_ALIASES)}")
    return tuple(resolved)


def _find_target_dirs(input_root: Path, composites: Sequence[str], kinds: Sequence[str]) -> List[Path]:
    target_dirs: Dict[str, Path] = {}
    for composite in composites:
        if "anchors" in kinds:
            root = input_root / "anchors" / composite
            if root.is_dir():
                for path in root.rglob("images"):
                    if path.is_dir():
                        target_dirs[str(path.resolve())] = path
        if "colour" in kinds:
            for root in (input_root / "colour" / composite, input_root / "colours" / composite):
                if not root.is_dir():
                    continue
                for path in root.rglob("images"):
                    if path.is_dir():
                        target_dirs[str(path.resolve())] = path
    return sorted(target_dirs.values(), key=lambda p: str(p))


def _iter_image_paths(target_dirs: Sequence[Path]) -> List[Path]:
    images: Dict[str, Path] = {}
    for target_dir in target_dirs:
        for path in target_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images[str(path.resolve())] = path
    return sorted(images.values(), key=lambda p: str(p))


def _load_image(path: Path) -> Image.Image:
    return Image.open(path.resolve()).convert("RGB")


def _structured_output_base(input_path: Path, output_root: Path) -> Path | None:
    parts = input_path.parts
    if "anchors" in parts:
        idx = parts.index("anchors")
        if len(parts) <= idx + 3:
            return None
        composite = parts[idx + 1]
        if composite not in ("without_composite", "with_composite"):
            return None
        animal = parts[idx + 2]
        gen_model = parts[idx + 3]
        return output_root / "anchors" / composite / animal / gen_model / "features"
    if "colour" in parts:
        idx = parts.index("colour")
        if len(parts) <= idx + 4:
            return None
        composite = parts[idx + 1]
        if composite not in ("without_composite", "with_composite"):
            return None
        animal = parts[idx + 2]
        color = parts[idx + 3]
        gen_model = parts[idx + 4]
        return output_root / "colour" / composite / animal / color / gen_model / "features"
    if "colours" in parts:
        idx = parts.index("colours")
        if len(parts) <= idx + 4:
            return None
        composite = parts[idx + 1]
        if composite not in ("without_composite", "with_composite"):
            return None
        animal = parts[idx + 2]
        color = parts[idx + 3]
        gen_model = parts[idx + 4]
        return output_root / "colours" / composite / animal / color / gen_model / "features"
    return None


def _output_path_for(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    model_key: str,
    pooling: str,
) -> Tuple[Path, Path, Path]:
    structured_base = _structured_output_base(input_path, output_root)
    if structured_base is None:
        raise ValueError(
            "Input path is not under anchors/colour(s)/{without,with}_composite. "
            "Features must be saved under selected/anchors/{without_composite,with_composite} or "
            "selected/colour(s)/{without_composite,with_composite}."
        )
    out_path = structured_base / model_key / pooling / f"{input_path.stem}.pt"
    try:
        out_rel = out_path.relative_to(output_root)
    except ValueError:
        out_rel = Path(out_path.name)
    index_path = out_path.parent / "index.jsonl"
    return out_path, out_rel, index_path


def _resolve_poolings(requested: Sequence[str] | None, spec_key: str, allowed: Sequence[str]) -> Tuple[str, ...]:
    if not requested:
        return tuple(allowed)
    if not requested:
        return tuple(allowed)
    invalid = [p for p in requested if p not in allowed]
    if invalid:
        raise ValueError(f"Pooling {invalid} not supported for model '{spec_key}'. Allowed: {allowed}")
    return tuple(requested)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract features for all images under anchor/colour directories.",
    )
    parser.add_argument(
        "--input-root",
        default=str(_default_input_root()),
        help=(
            "Root directory that contains structured anchors/colour(s) dirs "
            "(default: repo_root/results/selected; expects anchors/{without_composite,with_composite} "
            "and colour(s)/{without_composite,with_composite})."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Output root for features. Must be the same as --input-root. "
            "Features are always stored under selected/anchors/{without_composite,with_composite} or "
            "selected/colour(s)/{without_composite,with_composite}."
        ),
    )
    parser.add_argument(
        "--composites",
        nargs="+",
        default=None,
        help=(
            "Composite subdirs to process (default: without_composite with_composite). "
            "Choices: without_composite, with_composite."
        ),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help=(
            "Targets to process (default: anchors colour). "
            "Choices: anchors, colour (aliases: colours, color, colors)."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token (optional; also read from HF_TOKEN/HUGGINGFACE_HUB_TOKEN).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SPECS.keys()),
        help=(
            "Models to run (aliases: pe, siglip2, qwen-embed). You may also pass custom model ids like "
            "'pe:PE-Core-L14-336' or 'siglip2:google/siglip2-giant-opt-patch16-384'."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Single model to run (repeatable). Overrides --models when provided.",
    )
    parser.add_argument(
        "--poolings",
        nargs="+",
        default=None,
        help="Optional pooling list (default: all poolings supported by each model).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing feature files.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    if not input_root.exists():
        raise SystemExit(f"Input root not found: {input_root}")
    if input_root.name != "selected":
        raise SystemExit(
            "Input root must be the selected directory (e.g. /.../results/selected). "
            "Features are saved under selected/anchors/{without_composite,with_composite} or "
            "selected/colour(s)/{without_composite,with_composite}."
        )

    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = input_root
    if output_root != input_root:
        raise SystemExit(
            "Output root must match input root. Features are saved under "
            "selected/anchors/{without_composite,with_composite} or "
            "selected/colour(s)/{without_composite,with_composite}."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    composites = _resolve_composites(args.composites)
    targets = _resolve_targets(args.targets)

    missing_composites = []
    for composite in composites:
        anchors_root = input_root / "anchors" / composite
        colour_root = input_root / "colour" / composite
        colours_root = input_root / "colours" / composite
        if "anchors" in targets and not anchors_root.is_dir():
            missing_composites.append(
                f"Missing anchors directory under input root: {anchors_root}"
            )
        if "colour" in targets and not colour_root.is_dir() and not colours_root.is_dir():
            missing_composites.append(
                "Missing colour(s) directory under input root: "
                f"{colour_root} or {colours_root}"
            )
    if missing_composites:
        raise SystemExit("; ".join(missing_composites))

    target_dirs = _find_target_dirs(input_root, composites, targets)
    if not target_dirs:
        raise SystemExit(f"No anchor/colour directories found under {input_root}")
    image_paths = _iter_image_paths(target_dirs)
    if not image_paths:
        raise SystemExit(f"No images found under anchor/colour directories in {input_root}")

    device = _resolve_device(args.device)
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Found %d images under %d target dirs.", len(image_paths), len(target_dirs))

    hf_token = _get_hf_token(args.hf_token)
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    model_args = args.model if args.model else args.models
    models = _resolve_models(model_args)
    if not models:
        raise SystemExit("No models selected. Please provide at least one model via --models.")

    for spec in models:
        LOGGER.info("Loading model: %s", spec.model_id)
        model, processor, extractor = _load_model(spec, device, hf_token)
        allowed_poolings = _poolings_for_spec(spec)
        poolings = _resolve_poolings(args.poolings, spec.key, allowed_poolings)
        for pooling in poolings:
            LOGGER.info("Pooling: %s", pooling)
            saved = 0
            skipped = 0
            records_by_index: Dict[Path, List[Dict[str, object]]] = {}

            pending: List[Tuple[Path, Path, Path, Path]] = []
            for path in image_paths:
                out_path, out_rel, index_path = _output_path_for(path, input_root, output_root, spec.key, pooling)
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                    records_by_index.setdefault(index_path, []).append(
                        {
                            "status": "skipped_exists",
                            "input_path": str(path),
                            "input_relpath": str(path.relative_to(input_root)),
                            "output_path": str(out_path),
                            "output_relpath": str(out_rel),
                            "model": spec.model_id,
                            "model_key": spec.key,
                            "pooling": pooling,
                        }
                    )
                    continue
                pending.append((path, out_path, out_rel, index_path))

            if not pending:
                LOGGER.info("All features already exist for %s/%s. Skipping.", spec.key, pooling)
                continue

            for start in range(0, len(pending), args.batch_size):
                batch_items = pending[start : start + args.batch_size]
                images = []
                valid_items: List[Tuple[Path, Path, Path, Path]] = []
                for path, out_path, out_rel, index_path in batch_items:
                    try:
                        images.append(_load_image(path))
                        valid_items.append((path, out_path, out_rel, index_path))
                    except Exception as exc:
                        LOGGER.warning("Skipping %s (image load failed: %s)", path, exc)
                if not valid_items:
                    continue

                features = extractor(model, processor, images, device, args.batch_size, pooling)
                for idx, (path, out_path, out_rel, index_path) in enumerate(valid_items):
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(features[idx], out_path)
                    records_by_index.setdefault(index_path, []).append(
                        {
                            "status": "saved",
                            "input_path": str(path),
                            "input_relpath": str(path.relative_to(input_root)),
                            "output_path": str(out_path),
                            "output_relpath": str(out_rel),
                            "model": spec.model_id,
                            "model_key": spec.key,
                            "pooling": pooling,
                            "shape": list(features[idx].shape),
                            "dtype": str(features[idx].dtype),
                        }
                    )
                    saved += 1

            for index_path, records in records_by_index.items():
                if not records:
                    continue
                index_path.parent.mkdir(parents=True, exist_ok=True)
                index_path.write_text(
                    "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
                    encoding="utf-8",
                )
            LOGGER.info(
                "Wrote %d features for %s/%s (skipped %d). Index: %s",
                saved,
                spec.key,
                pooling,
                skipped,
                index_path,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
