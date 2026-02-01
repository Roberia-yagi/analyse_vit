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

LOGGER = logging.getLogger("analyse_vit.feature_extract_size")

IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _default_input_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "results" / "selected" / "size").resolve()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_target_dirs(input_root: Path) -> List[Path]:
    targets: Dict[str, Path] = {}
    for path in input_root.rglob("images"):
        if path.is_dir():
            targets[str(path.resolve())] = path
    return sorted(targets.values(), key=lambda p: str(p))


def _iter_image_paths(target_dirs: Sequence[Path]) -> List[Tuple[Path, Path]]:
    images: Dict[str, Tuple[Path, Path]] = {}
    for target_dir in target_dirs:
        for path in target_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images[str(path.resolve())] = (path, target_dir)
    return [images[key] for key in sorted(images.keys())]


def _load_image(path: Path) -> Image.Image:
    return Image.open(path.resolve()).convert("RGB")


def _output_path_for(
    input_path: Path,
    images_dir: Path,
    output_root: Path,
    model_key: str,
    pooling: str,
) -> Tuple[Path, Path, Path]:
    features_root = images_dir.parent / "features"
    rel_under_images = input_path.relative_to(images_dir)
    out_path = features_root / model_key / pooling / rel_under_images.with_suffix(".pt")
    try:
        out_rel = out_path.relative_to(output_root)
    except ValueError:
        out_rel = Path(out_path.name)
    index_path = out_path.parent / "index.jsonl"
    return out_path, out_rel, index_path


def _resolve_poolings(requested: Sequence[str] | None, spec_key: str, allowed: Sequence[str]) -> Tuple[str, ...]:
    if not requested:
        return tuple(allowed)
    invalid = [p for p in requested if p not in allowed]
    selected = [p for p in requested if p in allowed]
    if invalid:
        LOGGER.warning(
            "Skipping unsupported poolings for model '%s': %s (allowed: %s)",
            spec_key,
            invalid,
            allowed,
        )
    if not selected:
        raise ValueError(
            f"No requested poolings supported for model '{spec_key}'. "
            f"Requested: {list(requested)} Allowed: {allowed}"
        )
    return tuple(selected)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract features for all images under selected/size directories.",
    )
    parser.add_argument(
        "--input-root",
        default=str(_default_input_root()),
        help=(
            "Root directory that contains size directories "
            "(default: repo_root/results/selected/size)."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Output root for features. Must be the same as --input-root. "
            "Features are stored under each images/ sibling features/ directory."
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
    if input_root.name != "size" or input_root.parent.name != "selected":
        raise SystemExit(
            "Input root must be the selected/size directory "
            "(e.g. /.../results/selected/size)."
        )

    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        output_root = input_root
    if output_root != input_root:
        raise SystemExit(
            "Output root must match input root. Features are saved under selected/size/*/features."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    target_dirs = _find_target_dirs(input_root)
    if not target_dirs:
        raise SystemExit(f"No images directories found under {input_root}")
    image_items = _iter_image_paths(target_dirs)
    if not image_items:
        raise SystemExit(f"No images found under size directories in {input_root}")

    device = _resolve_device(args.device)
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Found %d images under %d target dirs.", len(image_items), len(target_dirs))

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
            for path, images_dir in image_items:
                out_path, out_rel, index_path = _output_path_for(
                    path,
                    images_dir,
                    output_root,
                    spec.key,
                    pooling,
                )
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
