from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

LOGGER = logging.getLogger("analyse_vit.feature_anchor_factor_scalar")

DEFAULT_VISION_ORDER = ("pe-core-l14-336", "siglip2-giant-opt-patch16-384", "qwen3-vl-8b-embed")
DEFAULT_GEN_ORDER = ("flux", "qwen", "sd3.5")
DEFAULT_GEN_MODEL = "qwen"
DEFAULT_POOLING = "attention_pooling"
DEFAULT_COMPOSITE_SUBDIR = "with_composite"

RUN_ID_PATTERN = re.compile(r"(?:^|_)run_(\d+)$")
VISION_MODEL_ALIASES: Dict[str, str] = {
    "qwen": "qwen3-vl-8b-embed",
    "siglip": "siglip2-giant-opt-patch16-384",
    "siglip2": "siglip2-giant-opt-patch16-384",
    "pe": "pe-core-l14-336",
}
DEFAULT_VISION_KEYS = ("qwen", "siglip2", "pe")


def _default_selected_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "results" / "selected").resolve()


def _default_output_root(selected_root: Path) -> Path:
    return (selected_root / "analysis" / "anchor_factor_scalar").resolve()


def _ordered(items: Sequence[str], preferred: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    for name in preferred:
        if name in items:
            ordered.append(name)
    for name in sorted(items):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _list_animals(anchors_root: Path) -> List[str]:
    if not anchors_root.is_dir():
        return []
    return sorted([path.name for path in anchors_root.iterdir() if path.is_dir()])


def _list_gen_models(anchors_root: Path, animals: Sequence[str]) -> List[str]:
    models: set[str] = set()
    for animal in animals:
        animal_dir = anchors_root / animal
        if not animal_dir.is_dir():
            continue
        for model_dir in animal_dir.iterdir():
            if model_dir.is_dir():
                models.add(model_dir.name)
    return _ordered(sorted(models), DEFAULT_GEN_ORDER)


def _list_vision_models(anchors_root: Path, animals: Sequence[str], gen_model: str) -> List[str]:
    models: set[str] = set()
    for animal in animals:
        features_dir = anchors_root / animal / gen_model / "features"
        if not features_dir.is_dir():
            continue
        for model_dir in features_dir.iterdir():
            if model_dir.is_dir():
                models.add(model_dir.name)
    return _ordered(sorted(models), DEFAULT_VISION_ORDER)


def _list_poolings(anchors_root: Path, animals: Sequence[str], gen_model: str, vision_model: str) -> List[str]:
    poolings: set[str] = set()
    for animal in animals:
        features_dir = anchors_root / animal / gen_model / "features" / vision_model
        if not features_dir.is_dir():
            continue
        for pooling_dir in features_dir.iterdir():
            if pooling_dir.is_dir():
                poolings.add(pooling_dir.name)
    return sorted(poolings)


def _resolve_colour_root(selected_root: Path, composite_subdir: str) -> Path:
    colour_root = selected_root / "colour" / composite_subdir
    if colour_root.is_dir():
        return colour_root
    colours_root = selected_root / "colours" / composite_subdir
    if colours_root.is_dir():
        return colours_root
    raise SystemExit(
        "Error: colour root not found under selected/colour(s) "
        f"for composite-subdir='{composite_subdir}'."
    )


def _parse_run_id(path: Path) -> str:
    match = RUN_ID_PATTERN.search(path.stem)
    if match is None:
        raise SystemExit(f"Error: failed to parse run_id from feature filename: {path}")
    return match.group(1)


def _build_run_feature_map(feature_dir: Path) -> Dict[str, Path]:
    if not feature_dir.is_dir():
        raise SystemExit(f"Error: feature directory not found: {feature_dir}")
    feature_paths = sorted(feature_dir.glob("*.pt"))
    if not feature_paths:
        raise SystemExit(f"Error: no feature files found in: {feature_dir}")

    run_to_path: Dict[str, Path] = {}
    for feature_path in feature_paths:
        run_id = _parse_run_id(feature_path)
        if run_id in run_to_path:
            raise SystemExit(
                f"Error: duplicate run_id={run_id} in directory: {feature_dir}"
            )
        run_to_path[run_id] = feature_path
    return run_to_path


def _sorted_run_ids(run_ids: Sequence[str]) -> List[str]:
    return sorted(run_ids, key=lambda value: int(value))


def _require_exact_run_match(
    *,
    anchor_runs: Sequence[str],
    variant_runs: Sequence[str],
    context: str,
) -> List[str]:
    anchor_set = set(anchor_runs)
    variant_set = set(variant_runs)
    if anchor_set != variant_set:
        missing_in_variant = sorted(anchor_set - variant_set, key=int)
        extra_in_variant = sorted(variant_set - anchor_set, key=int)
        raise SystemExit(
            f"Error: run_id mismatch for {context}. "
            f"missing_in_variant={missing_in_variant} extra_in_variant={extra_in_variant}"
        )
    return _sorted_run_ids(list(anchor_set))


def _load_feature(path: Path, cache: Dict[Path, torch.Tensor]) -> torch.Tensor:
    cached = cache.get(path)
    if cached is not None:
        return cached
    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, torch.Tensor):
        raise SystemExit(f"Error: feature file is not a torch.Tensor: {path}")
    tensor = loaded.flatten().to(dtype=torch.float32)
    cache[path] = tensor
    return tensor


def _l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise SystemExit(f"Error: feature dimension mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float(torch.norm(a - b, p=2).item())


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise SystemExit("Error: cannot compute mean from empty values.")
    return float(sum(values) / len(values))


def _std(values: Sequence[float], mean_value: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return float(variance**0.5)


def _sample_variance(values: Sequence[float], mean_value: float) -> float:
    if len(values) < 2:
        return 0.0
    return float(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


def _population_variance(values: Sequence[float], mean_value: float) -> float:
    if not values:
        raise SystemExit("Error: cannot compute population variance from empty values.")
    return float(sum((value - mean_value) ** 2 for value in values) / len(values))


def _shannon_entropy_from_probabilities(probabilities: Sequence[float]) -> float:
    if not probabilities:
        raise SystemExit("Error: probabilities is empty; entropy is undefined.")
    if any(prob < 0.0 for prob in probabilities):
        raise SystemExit(f"Error: probabilities must be non-negative: {probabilities}")
    total = float(sum(probabilities))
    if total == 0.0:
        raise SystemExit("Error: probability sum is zero; entropy is undefined.")
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(
            f"Error: probability sum must be 1.0 for entropy calculation (got {total})."
        )
    entropy = 0.0
    for prob in probabilities:
        if prob > 0.0:
            entropy -= float(prob * math.log(prob))
    return float(entropy)


def _normalized_variance_by_squared_mean(variance: float, mean_value: float, context: str) -> float:
    if mean_value == 0.0:
        raise SystemExit(
            f"Error: mean is zero for {context}; normalized variance by squared mean is undefined."
        )
    return float(variance / (mean_value**2))


def _stats(values: Sequence[float]) -> Dict[str, float | int]:
    mean_value = _mean(values)
    variance = _sample_variance(values, mean_value)
    return {
        "count": len(values),
        "mean": mean_value,
        "variance": variance,
        "normalized_variance_by_squared_mean": _normalized_variance_by_squared_mean(
            variance,
            mean_value,
            "factor distances",
        ),
        "std": _std(values, mean_value),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _discover_colours(colour_root: Path, animals: Sequence[str]) -> List[str]:
    if not animals:
        raise SystemExit("Error: no animals available for colour discovery.")
    first_animal_dir = colour_root / animals[0]
    if not first_animal_dir.is_dir():
        raise SystemExit(f"Error: colour root for animal not found: {first_animal_dir}")
    colours = sorted([path.name for path in first_animal_dir.iterdir() if path.is_dir()])
    if not colours:
        raise SystemExit(f"Error: no colour directories found under: {first_animal_dir}")
    return colours


def _discover_angles(angles_root: Path, animals: Sequence[str], gen_model: str) -> List[str]:
    discovered: set[str] = set()
    for animal in animals:
        animal_root = angles_root / animal
        if not animal_root.is_dir():
            continue
        for angle_dir in animal_root.iterdir():
            if not angle_dir.is_dir():
                continue
            if (angle_dir / gen_model).is_dir():
                discovered.add(angle_dir.name)
    angles = sorted(discovered)
    if not angles:
        raise SystemExit(
            f"Error: no angle directories found under {angles_root} for gen_model={gen_model}"
        )
    return angles


def _discover_sizes(
    size_root: Path,
    animals: Sequence[str],
    gen_model: str,
    vision_model: str,
    pooling: str,
    include_size_100: bool,
) -> List[str]:
    if not animals:
        raise SystemExit("Error: no animals available for size discovery.")
    sample_root = size_root / animals[0] / gen_model / "features" / vision_model / pooling
    if not sample_root.is_dir():
        raise SystemExit(f"Error: size features root not found: {sample_root}")
    names = sorted([path.name for path in sample_root.iterdir() if path.is_dir()])
    if not names:
        raise SystemExit(f"Error: no size directories found under: {sample_root}")

    numeric_sizes: List[Tuple[int, str]] = []
    other_sizes: List[str] = []
    for name in names:
        try:
            numeric_sizes.append((int(name), name))
        except ValueError:
            other_sizes.append(name)
    sorted_sizes = [name for _, name in sorted(numeric_sizes)] + sorted(other_sizes)
    if not include_size_100:
        sorted_sizes = [name for name in sorted_sizes if name != "100"]
    if not sorted_sizes:
        raise SystemExit("Error: no size directories selected (check --include-size-100 / dataset).")
    return sorted_sizes


def _compute_for_combo(
    *,
    selected_root: Path,
    output_root: Path,
    composite_subdir: str,
    animals: Sequence[str],
    gen_model: str,
    vision_model: str,
    pooling: str,
    colours: Sequence[str],
    angles: Sequence[str],
    sizes: Sequence[str],
) -> None:
    anchors_root = selected_root / "anchors" / composite_subdir
    angles_root = selected_root / "angles" / composite_subdir
    colour_root = _resolve_colour_root(selected_root, composite_subdir)
    size_root = selected_root / "size"

    feature_cache: Dict[Path, torch.Tensor] = {}
    distances: Dict[str, List[float]] = {"color": [], "orientation": [], "size": []}
    pair_records: List[Dict[str, object]] = []

    for animal in animals:
        anchor_dir = anchors_root / animal / gen_model / "features" / vision_model / pooling
        anchor_map = _build_run_feature_map(anchor_dir)
        anchor_runs = _sorted_run_ids(list(anchor_map.keys()))

        for colour in colours:
            colour_dir = colour_root / animal / colour / gen_model / "features" / vision_model / pooling
            colour_map = _build_run_feature_map(colour_dir)
            run_ids = _require_exact_run_match(
                anchor_runs=anchor_runs,
                variant_runs=list(colour_map.keys()),
                context=f"factor=color animal={animal} colour={colour} gen={gen_model} vision={vision_model} pooling={pooling}",
            )
            for run_id in run_ids:
                anchor_feat = _load_feature(anchor_map[run_id], feature_cache)
                colour_feat = _load_feature(colour_map[run_id], feature_cache)
                value = _l2_distance(colour_feat, anchor_feat)
                distances["color"].append(value)
                pair_records.append(
                    {
                        "factor": "color",
                        "animal": animal,
                        "variant": colour,
                        "run_id": run_id,
                        "distance": value,
                    }
                )

        for angle in angles:
            angle_dir = angles_root / animal / angle / gen_model / "features" / vision_model / pooling
            angle_map = _build_run_feature_map(angle_dir)
            run_ids = _require_exact_run_match(
                anchor_runs=anchor_runs,
                variant_runs=list(angle_map.keys()),
                context=f"factor=orientation animal={animal} angle={angle} gen={gen_model} vision={vision_model} pooling={pooling}",
            )
            for run_id in run_ids:
                anchor_feat = _load_feature(anchor_map[run_id], feature_cache)
                angle_feat = _load_feature(angle_map[run_id], feature_cache)
                value = _l2_distance(angle_feat, anchor_feat)
                distances["orientation"].append(value)
                pair_records.append(
                    {
                        "factor": "orientation",
                        "animal": animal,
                        "variant": angle,
                        "run_id": run_id,
                        "distance": value,
                    }
                )

        for size in sizes:
            size_dir = size_root / animal / gen_model / "features" / vision_model / pooling / size
            size_map = _build_run_feature_map(size_dir)
            run_ids = _require_exact_run_match(
                anchor_runs=anchor_runs,
                variant_runs=list(size_map.keys()),
                context=f"factor=size animal={animal} size={size} gen={gen_model} vision={vision_model} pooling={pooling}",
            )
            for run_id in run_ids:
                anchor_feat = _load_feature(anchor_map[run_id], feature_cache)
                size_feat = _load_feature(size_map[run_id], feature_cache)
                value = _l2_distance(size_feat, anchor_feat)
                distances["size"].append(value)
                pair_records.append(
                    {
                        "factor": "size",
                        "animal": animal,
                        "variant": size,
                        "run_id": run_id,
                        "distance": value,
                    }
                )

    if not distances["color"]:
        raise SystemExit("Error: no valid distance pairs collected for factor=color.")
    if not distances["orientation"]:
        raise SystemExit("Error: no valid distance pairs collected for factor=orientation.")
    if not distances["size"]:
        raise SystemExit("Error: no valid distance pairs collected for factor=size.")

    out_dir = output_root / gen_model / vision_model / pooling
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "selected_root": str(selected_root),
        "composite_subdir": composite_subdir,
        "gen_model": gen_model,
        "vision_model": vision_model,
        "pooling": pooling,
        "animals": list(animals),
        "colours": list(colours),
        "angles": list(angles),
        "sizes": list(sizes),
        "scalars": {
            "color": _stats(distances["color"]),
            "orientation": _stats(distances["orientation"]),
            "size": _stats(distances["size"]),
        },
    }
    factor_means = {
        "color": float(summary["scalars"]["color"]["mean"]),
        "orientation": float(summary["scalars"]["orientation"]["mean"]),
        "size": float(summary["scalars"]["size"]["mean"]),
    }
    means_list = [factor_means["color"], factor_means["orientation"], factor_means["size"]]
    means_mean = _mean(means_list)
    means_variance = _population_variance(means_list, means_mean)
    sum_attribute_means = float(sum(means_list))
    if sum_attribute_means == 0.0:
        raise SystemExit(
            "Error: sum of attribute means is zero; attribute dominance ratio is undefined."
        )
    dominant_attribute, dominant_attribute_mean = max(
        factor_means.items(),
        key=lambda item: item[1],
    )
    attribute_dominance_ratio = float(dominant_attribute_mean / sum_attribute_means)
    attribute_probabilities = {
        name: float(value / sum_attribute_means) for name, value in factor_means.items()
    }
    entropy = _shannon_entropy_from_probabilities(list(attribute_probabilities.values()))
    max_entropy = float(math.log(3.0))
    entropy_normalized_by_log3 = float(entropy / max_entropy)
    entropy_based_dominance = float(1.0 - entropy_normalized_by_log3)
    summary["three_factor_mean_statistics"] = {
        "factor_means": factor_means,
        "mean_of_factor_means": means_mean,
        "variance_of_factor_means": means_variance,
        "normalized_variance_by_squared_mean": _normalized_variance_by_squared_mean(
            means_variance,
            means_mean,
            "three factor means",
        ),
    }
    summary["attribute_dominance"] = {
        "definition": "max_attribute_mean / sum_attribute_means",
        "dominant_attribute": dominant_attribute,
        "dominant_attribute_mean": float(dominant_attribute_mean),
        "sum_attribute_means": sum_attribute_means,
        "attribute_dominance_ratio": attribute_dominance_ratio,
        "attribute_probabilities": attribute_probabilities,
        "entropy_definition": "-sum(p_a * log(p_a))",
        "entropy": entropy,
        "entropy_normalized_by_log3": entropy_normalized_by_log3,
        "entropy_based_dominance_definition": "1 - (entropy / log(3))",
        "entropy_based_dominance": entropy_based_dominance,
    }

    summary_path = out_dir / "anchor_factor_scalar_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    pairs_path = out_dir / "anchor_factor_scalar_pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as fp:
        for record in pair_records:
            fp.write(json.dumps(record, ensure_ascii=True) + "\n")

    LOGGER.info("Saved summary: %s", summary_path)
    LOGGER.info("Saved pair records: %s", pairs_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute scalar mean L2 distance from anchor to element features. "
            "Returns one scalar each for color/orientation/size."
        ),
    )
    parser.add_argument(
        "--selected-root",
        default=str(_default_selected_root()),
        help="Structured root with anchors/angles/colour/size (default: repo_root/results/selected).",
    )
    parser.add_argument(
        "--composite-subdir",
        default=DEFAULT_COMPOSITE_SUBDIR,
        help="Composite subdir under anchors/angles/colour(s) (default: with_composite).",
    )
    parser.add_argument(
        "--gen-model",
        default=DEFAULT_GEN_MODEL,
        help="Generation model (default: qwen).",
    )
    parser.add_argument(
        "--vision-model",
        default=None,
        help="Vision model key (qwen/siglip/siglip2/pe). If omitted, run all: qwen,siglip2,pe.",
    )
    parser.add_argument(
        "--pooling",
        default=DEFAULT_POOLING,
        help="Pooling key under features (default: attention_pooling).",
    )
    parser.add_argument(
        "--colours",
        nargs="+",
        default=None,
        help="Colour variants to include (default: discover from the first animal).",
    )
    parser.add_argument(
        "--angles",
        nargs="+",
        default=None,
        help="Orientation variants to include (default: discover under selected/angles/<composite-subdir>).",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        help="Size variants to include (default: discover from the first animal).",
    )
    parser.add_argument(
        "--include-size-100",
        action="store_true",
        help="Include size=100 in size factor (default: excluded).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root (default: <selected-root>/analysis/anchor_factor_scalar).",
    )
    return parser


def _validate_unique(name: str, values: Sequence[str]) -> None:
    if len(values) != len(set(values)):
        raise SystemExit(f"Error: {name} contains duplicates: {values}")


def _resolve_vision_model_alias(name: str) -> str:
    resolved = VISION_MODEL_ALIASES.get(name)
    if resolved is None:
        allowed = ", ".join(sorted(VISION_MODEL_ALIASES.keys()))
        raise SystemExit(
            f"Error: unsupported vision model key '{name}'. Supported keys: {allowed}"
        )
    return resolved


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    selected_root = Path(args.selected_root).expanduser().resolve()
    if not selected_root.is_dir():
        raise SystemExit(f"Error: selected root not found: {selected_root}")

    composite_subdir = str(args.composite_subdir).strip()
    if composite_subdir not in {"with_composite", "without_composite"}:
        raise SystemExit(
            "Error: composite-subdir must be 'with_composite' or 'without_composite': "
            f"{composite_subdir}"
        )

    anchors_root = selected_root / "anchors" / composite_subdir
    if not anchors_root.is_dir():
        raise SystemExit(f"Error: anchors root not found: {anchors_root}")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else _default_output_root(selected_root)
    )

    animals = _list_animals(anchors_root)
    if not animals:
        raise SystemExit(f"Error: no animals found under {anchors_root}")

    gen_models = [args.gen_model] if args.gen_model else _list_gen_models(anchors_root, animals)
    if not gen_models:
        raise SystemExit("Error: no generation models found under anchors.")

    for gen_model in gen_models:
        if args.vision_model:
            vision_models = [_resolve_vision_model_alias(str(args.vision_model).strip())]
        else:
            vision_models = [_resolve_vision_model_alias(key) for key in DEFAULT_VISION_KEYS]
        if not vision_models:
            raise SystemExit(f"Error: no vision models found for gen_model={gen_model}")

        for vision_model in vision_models:
            poolings = [args.pooling] if args.pooling else _list_poolings(anchors_root, animals, gen_model, vision_model)
            if not poolings:
                raise SystemExit(
                    f"Error: no pooling directories found for gen_model={gen_model}, vision_model={vision_model}"
                )

            for pooling in poolings:
                colour_root = _resolve_colour_root(selected_root, composite_subdir)
                colours = list(args.colours) if args.colours else _discover_colours(colour_root, animals)
                _validate_unique("colours", colours)

                angles_root = selected_root / "angles" / composite_subdir
                if not angles_root.is_dir():
                    raise SystemExit(f"Error: angles root not found: {angles_root}")
                angles = list(args.angles) if args.angles else _discover_angles(angles_root, animals, gen_model)
                _validate_unique("angles", angles)
                if "front" in angles:
                    raise SystemExit(
                        "Error: 'front' is not supported in this script. "
                        "Use only generated orientation variants under selected/angles."
                    )

                size_root = selected_root / "size"
                if not size_root.is_dir():
                    raise SystemExit(f"Error: size root not found: {size_root}")
                sizes = (
                    list(args.sizes)
                    if args.sizes
                    else _discover_sizes(
                        size_root=size_root,
                        animals=animals,
                        gen_model=gen_model,
                        vision_model=vision_model,
                        pooling=pooling,
                        include_size_100=bool(args.include_size_100),
                    )
                )
                if not args.include_size_100:
                    sizes = [size for size in sizes if size != "100"]
                if not sizes:
                    raise SystemExit("Error: sizes is empty after filtering.")
                _validate_unique("sizes", sizes)

                _compute_for_combo(
                    selected_root=selected_root,
                    output_root=output_root,
                    composite_subdir=composite_subdir,
                    animals=animals,
                    gen_model=gen_model,
                    vision_model=vision_model,
                    pooling=pooling,
                    colours=colours,
                    angles=angles,
                    sizes=sizes,
                )


if __name__ == "__main__":
    main()
