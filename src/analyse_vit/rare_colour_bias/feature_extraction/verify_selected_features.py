from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image

from analyse_vit.rare_colour_bias.feature_extraction.feature_extraction import (
    _load_model,
    _poolings_for_spec,
    _resolve_models,
)

LOGGER = logging.getLogger("analyse_vit.verify_selected_features")

COMPOSITE_SUBDIRS = ("without_composite", "with_composite")
GEN_MODEL_ORDER = ("flux", "qwen", "sd3.5")
DEFAULT_MODELS = ("pe", "qwen-embed", "siglip2")


@dataclass(frozen=True)
class FeatureCase:
    kind: str
    composite: str | None
    label: str
    images_dir: Path
    image_path: Path


def _require_absolute(path: Path, label: str) -> None:
    if not str(path).startswith("/"):
        raise SystemExit(f"Error: {label} must be an absolute path (absolute path required): {path}")
    if not path.exists():
        raise SystemExit(f"Error: {label} not found (absolute path required): {path}")


def _sorted_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)


def _pick_gen_model(root: Path, preferred: Sequence[str]) -> Path:
    for name in preferred:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    dirs = _sorted_dirs(root)
    if not dirs:
        raise SystemExit(f"Error: No generation model directory found under {root}")
    return dirs[0]


def _pick_first_dir(root: Path) -> Path:
    dirs = _sorted_dirs(root)
    if not dirs:
        raise SystemExit(f"Error: No subdirectories found under {root}")
    return dirs[0]


def _pick_first_numeric_dir(root: Path) -> Path:
    dirs = _sorted_dirs(root)
    if not dirs:
        raise SystemExit(f"Error: No size subdirectories found under {root}")

    def _sort_key(path: Path) -> Tuple[int, str]:
        try:
            return (0, f"{int(path.name):08d}")
        except ValueError:
            return (1, path.name)

    return sorted(dirs, key=_sort_key)[0]


def _find_run_image(root: Path, run_id: str, recursive: bool = False) -> Path:
    pattern = f"*run_{run_id}.*"
    paths = list(root.rglob(pattern) if recursive else root.glob(pattern))
    if not paths:
        raise SystemExit(f"Error: No images matching {pattern} under {root}")
    return sorted(paths, key=lambda p: str(p))[0]


def _load_image(path: Path) -> Image.Image:
    return Image.open(path.resolve()).convert("RGB")


def _feature_path_for(image_path: Path, images_dir: Path, model_key: str, pooling: str) -> Path:
    features_root = images_dir.parent / "features"
    rel_under_images = image_path.relative_to(images_dir)
    return features_root / model_key / pooling / rel_under_images.with_suffix(".pt")


def _resolve_colour_root(selected_root: Path, composite: str, animal: str) -> Path:
    colour_root = selected_root / "colour" / composite / animal
    if colour_root.is_dir():
        return colour_root
    colours_root = selected_root / "colours" / composite / animal
    if colours_root.is_dir():
        return colours_root
    raise SystemExit(
        "Error: colour root not found under selected/colour(s) "
        f"for composite={composite}, animal={animal}."
    )


def _build_cases(selected_root: Path, animal: str, run_id: str) -> List[FeatureCase]:
    cases: List[FeatureCase] = []

    for composite in COMPOSITE_SUBDIRS:
        anchor_root = selected_root / "anchors" / composite / animal
        if not anchor_root.is_dir():
            raise SystemExit(f"Error: anchors root not found: {anchor_root}")
        anchor_gen_root = _pick_gen_model(anchor_root, GEN_MODEL_ORDER)
        anchor_images = anchor_gen_root / "images"
        anchor_image = _find_run_image(anchor_images, run_id, recursive=False)
        cases.append(
            FeatureCase(
                kind="anchors",
                composite=composite,
                label=f"anchors/{composite} ({anchor_gen_root.name})",
                images_dir=anchor_images,
                image_path=anchor_image,
            )
        )

        angle_root = selected_root / "angles" / composite / animal
        if not angle_root.is_dir():
            raise SystemExit(f"Error: angles root not found: {angle_root}")
        angle_dir = _pick_first_dir(angle_root)
        angle_gen_root = _pick_gen_model(angle_dir, GEN_MODEL_ORDER)
        angle_images = angle_gen_root / "images"
        angle_image = _find_run_image(angle_images, run_id, recursive=False)
        cases.append(
            FeatureCase(
                kind="angles",
                composite=composite,
                label=f"angles/{composite}/{angle_dir.name} ({angle_gen_root.name})",
                images_dir=angle_images,
                image_path=angle_image,
            )
        )

        colour_root = _resolve_colour_root(selected_root, composite, animal)
        colour_dir = _pick_first_dir(colour_root)
        colour_gen_root = _pick_gen_model(colour_dir, GEN_MODEL_ORDER)
        colour_images = colour_gen_root / "images"
        colour_image = _find_run_image(colour_images, run_id, recursive=False)
        cases.append(
            FeatureCase(
                kind="colour",
                composite=composite,
                label=f"colour/{composite}/{colour_dir.name} ({colour_gen_root.name})",
                images_dir=colour_images,
                image_path=colour_image,
            )
        )

    size_root = selected_root / "size" / animal
    if not size_root.is_dir():
        raise SystemExit(f"Error: size root not found: {size_root}")
    size_gen_root = _pick_gen_model(size_root, GEN_MODEL_ORDER)
    size_images = size_gen_root / "images"
    size_bucket = _pick_first_numeric_dir(size_images)
    size_image = _find_run_image(size_bucket, run_id, recursive=False)
    cases.append(
        FeatureCase(
            kind="size",
            composite=None,
            label=f"size ({size_gen_root.name}/{size_bucket.name})",
            images_dir=size_images,
            image_path=size_image,
        )
    )

    return cases


def _extract_and_compare(
    cases: Sequence[FeatureCase],
    model_key: str,
    pooling: str,
    extractor,
    model,
    processor,
    device: str,
    batch_size: int,
    rtol: float,
    atol: float,
    run_id: str,
    diagnose: bool,
    auto_threshold: bool,
) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    mismatched: List[str] = []

    image_items: List[Tuple[FeatureCase, Path]] = []
    for case in cases:
        feature_path = _feature_path_for(case.image_path, case.images_dir, model_key, pooling)
        if not feature_path.exists():
            missing.append(f"{feature_path} (from {case.label})")
            continue
        image_items.append((case, feature_path))

    if not image_items:
        return missing, mismatched

    images = [_load_image(case.image_path) for case, _ in image_items]
    with torch.no_grad():
        features = extractor(
            model,
            processor,
            images,
            device,
            batch_size,
            pooling,
        )

    if features.ndim == 1:
        features = features.unsqueeze(0)

    for (case, feature_path), fresh in zip(image_items, features):
        saved = torch.load(feature_path, map_location="cpu")
        if isinstance(saved, dict):
            if "features" in saved:
                saved = saved["features"]
            elif "embedding" in saved:
                saved = saved["embedding"]
        if not torch.is_tensor(saved):
            mismatched.append(f"{feature_path} (from {case.label}) [saved is not a tensor]")
            continue
        saved = saved.detach().cpu().float()
        fresh = fresh.detach().cpu().float()
        if saved.shape != fresh.shape:
            mismatched.append(
                f"{feature_path} (from {case.label}) [shape {tuple(saved.shape)} != {tuple(fresh.shape)}]"
            )
            continue
        is_allclose = torch.allclose(saved, fresh, rtol=rtol, atol=atol)
        if diagnose or auto_threshold:
            decision = _diagnose_threshold(feature_path, saved, fresh, run_id, case.label)
        else:
            decision = None

        if auto_threshold and decision is not None:
            if decision["is_mismatch"]:
                mismatched.append(
                    f"{feature_path} (from {case.label}) [cosine_dist={decision['fresh_cos']:.6g} > "
                    f"threshold(p90)={decision['threshold']:.6g}]"
                )
            continue

        if not is_allclose:
            diff = (saved - fresh).abs().max().item()
            mismatched.append(
                f"{feature_path} (from {case.label}) [max_abs_diff={diff:.6g}]"
            )
            if diagnose:
                _log_diagnose(feature_path, saved, fresh, run_id, case.label)

    return missing, mismatched


def _diagnose_threshold(
    feature_path: Path,
    saved: torch.Tensor,
    fresh: torch.Tensor,
    run_id: str,
    label: str,
) -> Dict[str, float] | None:
    run_token = f"run_{run_id}"
    name = feature_path.name
    if run_token not in name:
        LOGGER.warning("Diagnose skipped (run token not in filename): %s", feature_path)
        return None

    pattern = name.replace(run_token, "run_*")
    sibling_paths = sorted(feature_path.parent.glob(pattern))
    sibling_paths = [p for p in sibling_paths if p != feature_path]
    if not sibling_paths:
        LOGGER.warning("Diagnose skipped (no sibling runs found): %s", feature_path)
        return None

    saved = saved.detach().cpu().float()
    fresh = fresh.detach().cpu().float()
    fresh_cos = _cosine_distance(saved, fresh)

    dists: List[float] = []
    for path in sibling_paths:
        other = torch.load(path, map_location="cpu")
        if isinstance(other, dict):
            if "features" in other:
                other = other["features"]
            elif "embedding" in other:
                other = other["embedding"]
        if not torch.is_tensor(other):
            continue
        other = other.detach().cpu().float()
        if other.shape != saved.shape:
            continue
        dists.append(_cosine_distance(saved, other))

    if not dists:
        LOGGER.warning("Diagnose skipped (no comparable siblings): %s", feature_path)
        return None

    stats = _distance_stats(dists)
    threshold = stats["p90"]
    is_mismatch = fresh_cos > threshold
    LOGGER.info(
        "Diagnose [%s]: fresh_vs_saved cosine_dist=%.6g | siblings n=%d "
        "min=%.6g p10=%.6g p50=%.6g p90=%.6g max=%.6g",
        label,
        fresh_cos,
        int(stats["n"]),
        stats["min"],
        stats["p10"],
        stats["p50"],
        stats["p90"],
        stats["max"],
    )
    LOGGER.info(
        "Diagnose [%s]: decision=%s (fresh_cos=%.6g, threshold=p90=%.6g)",
        label,
        "mismatch" if is_mismatch else "match",
        fresh_cos,
        threshold,
    )
    return {
        "fresh_cos": fresh_cos,
        "threshold": threshold,
        "is_mismatch": is_mismatch,
    }


def _log_diagnose(
    feature_path: Path,
    saved: torch.Tensor,
    fresh: torch.Tensor,
    run_id: str,
    label: str,
) -> None:
    _ = _diagnose_threshold(feature_path, saved, fresh, run_id, label)


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten()
    b = b.flatten()
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return float("inf")
    return (1.0 - torch.dot(a, b) / denom).item()


def _distance_stats(values: Sequence[float]) -> Dict[str, float]:
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _quantile(q: float) -> float:
        if n == 1:
            return sorted_vals[0]
        idx = (n - 1) * q
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    return {
        "n": float(n),
        "min": sorted_vals[0],
        "p10": _quantile(0.10),
        "p50": _quantile(0.50),
        "p90": _quantile(0.90),
        "max": sorted_vals[-1],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify stored features under results/selected against fresh extraction "
            "for eagle run_00 across anchors/angles/colour/size."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Absolute path to results directory (must contain selected/).",
    )
    parser.add_argument(
        "--animal",
        default="eagle",
        help="Animal to verify (default: eagle).",
    )
    parser.add_argument(
        "--run-id",
        default="00",
        help="Run id to verify (default: 00).",
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
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Models to verify (default: pe qwen-embed siglip2).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for allclose (default: 1e-5).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for allclose (default: 1e-6).",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        default=False,
        help="Log cosine distance of fresh vs saved compared to sibling runs.",
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        default=True,
        help="Use sibling-run cosine distance p90 as an automatic mismatch threshold.",
    )
    return parser


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    _require_absolute(results_dir, "RESULTS_DIR")
    if results_dir.name == "selected":
        selected_root = results_dir
    else:
        selected_root = results_dir / "selected"
    if not selected_root.is_dir():
        raise SystemExit(f"Error: selected root not found under RESULTS_DIR: {selected_root}")

    cases = _build_cases(selected_root, args.animal, args.run_id)
    for case in cases:
        LOGGER.info("Case: %s -> %s", case.label, case.image_path)

    device = _resolve_device(args.device)
    LOGGER.info("Using device: %s", device)

    specs = _resolve_models(args.models)
    if not specs:
        raise SystemExit("Error: No models selected. Provide --models.")

    missing_all: List[str] = []
    mismatched_all: List[str] = []

    for spec in specs:
        LOGGER.info("Loading model: %s", spec.model_id)
        model, processor, extractor = _load_model(spec, device, None)
        poolings = _poolings_for_spec(spec)
        for pooling in poolings:
            missing, mismatched = _extract_and_compare(
                cases,
                spec.key,
                pooling,
                extractor,
                model,
                processor,
                device,
                args.batch_size,
                args.rtol,
                args.atol,
                args.run_id,
                args.diagnose,
                args.auto_threshold,
            )
            missing_all.extend(missing)
            mismatched_all.extend(mismatched)

    if missing_all:
        LOGGER.error("Missing feature files (%d):", len(missing_all))
        for entry in missing_all:
            LOGGER.error("  %s", entry)
    if mismatched_all:
        LOGGER.error("Mismatched feature files (%d):", len(mismatched_all))
        for entry in mismatched_all:
            LOGGER.error("  %s", entry)

    if missing_all or mismatched_all:
        raise SystemExit("Feature verification failed. See missing/mismatched entries above.")

    LOGGER.info("All selected features match for %d cases.", len(cases))


if __name__ == "__main__":
    main()
