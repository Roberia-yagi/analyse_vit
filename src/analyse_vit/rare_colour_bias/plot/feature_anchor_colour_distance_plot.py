from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch

LOGGER = logging.getLogger("analyse_vit.feature_anchor_colour_distance_plot")

DEFAULT_VISION_ORDER = ("pe-core-l14-336", "siglip2-giant-opt-patch16-384", "qwen3-vl-8b-embed")
DEFAULT_GEN_ORDER = ("flux", "qwen", "sd3.5")
COMPOSITE_SUBDIR = "without_composite"


def _default_selected_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "results" / "selected").resolve()


def _default_output_root(selected_root: Path) -> Path:
    return (selected_root / "analysis" / "anchor_colour_distance").resolve()


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
    return sorted([p.name for p in anchors_root.iterdir() if p.is_dir()])


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


def _list_poolings(anchors_root: Path, animals: Sequence[str], gen_model: str, vision: str) -> List[str]:
    poolings: set[str] = set()
    for animal in animals:
        model_dir = anchors_root / animal / gen_model / "features" / vision
        if not model_dir.is_dir():
            continue
        for pooling_dir in model_dir.iterdir():
            if pooling_dir.is_dir():
                poolings.add(pooling_dir.name)
    return sorted(poolings)


def _iter_colours(colour_root: Path, animal: str) -> List[str]:
    root = colour_root / animal
    if not root.is_dir():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def _anchor_center(anchor_dir: Path) -> torch.Tensor:
    anchor_files = sorted(anchor_dir.glob("*.pt"))
    if not anchor_files:
        raise FileNotFoundError(f"No anchor features found in {anchor_dir}")
    tensors = [_load_tensor(path) for path in anchor_files]
    return torch.stack(tensors, dim=0).mean(dim=0)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return torch.norm(x, p=2, dim=-1)


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = (_norm(a) * _norm(b)).clamp_min(eps)
    return (a * b).sum(dim=-1) / denom


def _relative_change(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = _norm(a).clamp_min(eps)
    return _norm(a - b) / denom


def _norm_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (_norm(a) - _norm(b)).abs()


def _anchor_mean_metrics(anchor_dir: Path, center: torch.Tensor) -> Dict[str, float]:
    anchor_files = sorted(anchor_dir.glob("*.pt"))
    if not anchor_files:
        raise FileNotFoundError(f"No anchor features found in {anchor_dir}")
    tensors = torch.stack([_load_tensor(path) for path in anchor_files], dim=0)
    center_batch = center.expand_as(tensors)
    return {
        "distance": float(_norm(tensors - center_batch).mean().item()),
        "norm_diff": float(_norm_diff(tensors, center_batch).mean().item()),
        "cosine": float(_cosine(tensors, center_batch).mean().item()),
        "relative_change": float(_relative_change(center_batch, tensors).mean().item()),
    }


def _collect_colour_metrics(colour_dir: Path, anchor_center: torch.Tensor) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {
        "distance": [],
        "norm_diff": [],
        "cosine": [],
        "relative_change": [],
    }
    for path in sorted(colour_dir.glob("*.pt")):
        feat = _load_tensor(path)
        values["distance"].append(float(torch.norm(feat - anchor_center, p=2).item()))
        values["norm_diff"].append(float(_norm_diff(feat, anchor_center).item()))
        values["cosine"].append(float(_cosine(feat, anchor_center).item()))
        values["relative_change"].append(float(_relative_change(anchor_center, feat).item()))
    return values


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Sequence[float], mean_value: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return float(variance**0.5)


def _summarize(metrics: Dict[str, Dict[str, Dict[str, List[float]]]]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for metric_name, colour_map in metrics.items():
        summary[metric_name] = {}
        for colour, animal_map in colour_map.items():
            summary[metric_name][colour] = {}
            for animal, values in animal_map.items():
                mean_value = _mean(values)
                summary[metric_name][colour][animal] = {
                    "count": len(values),
                    "mean": mean_value,
                    "std": _std(values, mean_value),
                    "min": float(min(values)) if values else 0.0,
                    "max": float(max(values)) if values else 0.0,
                }
    return summary


def _plot_metric_grid(
    *,
    metric_key: str,
    metric_label: str,
    distances: Dict[str, Dict[str, Dict[str, List[float]]]],
    anchor_centers: Dict[str, torch.Tensor],
    anchor_self_means: Dict[str, Dict[str, float]],
    used_animals: List[str],
    colours_sorted: List[str],
    colour_to_marker: Dict[str, str],
    colour_to_color: Dict[str, str],
    out_dir: Path,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    x_positions = list(range(len(used_animals)))
    for colour in colours_sorted:
        marker = colour_to_marker[colour]
        color = colour_to_color.get(colour, "black")
        means = []
        for animal in used_animals:
            values = distances.get(metric_key, {}).get(colour, {}).get(animal, [])
            means.append(_mean(values))
        ax.scatter(
            x_positions,
            means,
            s=90,
            alpha=0.9,
            label=colour,
            marker=marker,
            color=color,
            edgecolors="black" if colour == "white" else "none",
            linewidths=0.8 if colour == "white" else 0.0,
        )

    if anchor_centers:
        other_means = []
        for animal in used_animals:
            center = anchor_centers.get(animal)
            if center is None:
                other_means.append(float("nan"))
                continue
            other_centers = [val for key, val in anchor_centers.items() if key != animal]
            if not other_centers:
                other_means.append(float("nan"))
                continue
            stacked = torch.stack(other_centers, dim=0)
            center_batch = center.expand_as(stacked)
            if metric_key == "distance":
                values = _norm(stacked - center_batch)
            elif metric_key == "norm_diff":
                values = _norm_diff(stacked, center_batch)
            elif metric_key == "cosine":
                values = _cosine(stacked, center_batch)
            elif metric_key == "relative_change":
                values = _relative_change(center_batch, stacked)
            else:
                values = torch.full((stacked.shape[0],), float("nan"))
            other_means.append(float(values.mean().item()))
        ax.scatter(
            x_positions,
            other_means,
            s=140,
            alpha=0.9,
            label="others_mean",
            marker="X",
            color="black",
        )

    if anchor_self_means:
        self_means = [anchor_self_means.get(animal, {}).get(metric_key, float("nan")) for animal in used_animals]
        ax.scatter(
            x_positions,
            self_means,
            s=120,
            alpha=0.9,
            label="self_mean",
            marker="x",
            color="black",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(used_animals, rotation=45, ha="right", fontsize=16)
    ax.set_ylabel(metric_label, fontsize=17)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_title(title, fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            borderaxespad=0.0,
            fontsize=14,
        )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    fig.savefig(out_dir, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot feature change metrics from anchor center to each colour for each animal.",
    )
    parser.add_argument(
        "--features-root",
        default=None,
        help=(
            "Legacy: path to .../features/<model_key>/<pooling>. "
            "If set, structured args are ignored."
        ),
    )
    parser.add_argument(
        "--selected-root",
        default=str(_default_selected_root()),
        help="Structured root with anchors/colour (default: repo_root/results/selected).",
    )
    parser.add_argument("--gen-model", default=None, help="Generation model (flux/qwen/sd3.5).")
    parser.add_argument("--vision-model", default=None, help="Vision encoder key (e.g., pe-core-l14-336).")
    parser.add_argument("--pooling", default=None, help="Pooling/embedding key (e.g., attention_pooling).")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root for plots (default: <selected-root>/analysis/anchor_colour_distance).",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    return parser


def _run_structured(
    selected_root: Path,
    output_root: Path,
    gen_models: Sequence[str],
    vision_models: Sequence[str],
    poolings: Sequence[str],
    dpi: int,
) -> None:
    anchors_root = selected_root / "anchors" / COMPOSITE_SUBDIR
    colour_root = selected_root / "colour" / COMPOSITE_SUBDIR
    animals = _list_animals(anchors_root)
    if not animals:
        raise SystemExit(f"No animals found under {anchors_root}")

    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    colour_to_marker: Dict[str, str] = {}
    colour_to_color: Dict[str, str] = {
        "black": "black",
        "white": "#f0f0f0",
        "gray": "gray",
        "grey": "gray",
        "red": "red",
        "orange": "orange",
        "yellow": "gold",
        "green": "green",
        "cyan": "cyan",
        "blue": "blue",
        "purple": "purple",
        "magenta": "magenta",
        "pink": "hotpink",
        "brown": "sienna",
    }

    for gen_model in gen_models:
        for vision in vision_models:
            for pooling in poolings:
                distances: Dict[str, Dict[str, Dict[str, List[float]]]] = {
                    "distance": {},
                    "norm_diff": {},
                    "cosine": {},
                    "relative_change": {},
                }
                anchor_centers: Dict[str, torch.Tensor] = {}
                anchor_self_means: Dict[str, Dict[str, float]] = {}
                used_animals: List[str] = []
                for animal in animals:
                    anchor_dir = anchors_root / animal / gen_model / "features" / vision / pooling
                    if not anchor_dir.is_dir():
                        continue
                    colour_dirs = _iter_colours(colour_root, animal)
                    if not colour_dirs:
                        continue
                    try:
                        center = _anchor_center(anchor_dir)
                    except FileNotFoundError:
                        continue
                    anchor_centers[animal] = center
                    anchor_self_means[animal] = _anchor_mean_metrics(anchor_dir, center)
                    used_animals.append(animal)
                    for color in colour_dirs:
                        colour_dir = (
                            colour_root
                            / animal
                            / color
                            / gen_model
                            / "features"
                            / vision
                            / pooling
                        )
                        if not colour_dir.is_dir():
                            continue
                        metrics = _collect_colour_metrics(colour_dir, center)
                        if not metrics["distance"]:
                            continue
                        for metric_key, values in metrics.items():
                            distances[metric_key].setdefault(color, {})[animal] = values

                if not used_animals or not distances["distance"]:
                    LOGGER.info(
                        "No distances for gen=%s vision=%s pooling=%s; skipping.",
                        gen_model,
                        vision,
                        pooling,
                    )
                    continue

                used_animals = sorted(set(used_animals))
                colours_sorted = sorted(distances["distance"].keys())
                for colour in colours_sorted:
                    if colour not in colour_to_marker:
                        colour_to_marker[colour] = marker_cycle[len(colour_to_marker) % len(marker_cycle)]

                out_dir = output_root / gen_model / vision / pooling
                out_dir.mkdir(parents=True, exist_ok=True)
                summary = {
                    "selected_root": str(selected_root),
                    "gen_model": gen_model,
                    "vision_model": vision,
                    "pooling": pooling,
                    "animals": used_animals,
                    "summary": _summarize(distances),
                }
                (out_dir / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                title = f"{gen_model} | {vision} | {pooling}"
                _plot_metric_grid(
                    metric_key="distance",
                    metric_label="L2 distance from anchor center",
                    distances=distances,
                    anchor_centers=anchor_centers,
                    anchor_self_means=anchor_self_means,
                    used_animals=used_animals,
                    colours_sorted=colours_sorted,
                    colour_to_marker=colour_to_marker,
                    colour_to_color=colour_to_color,
                    out_dir=out_dir / "anchor_colour_distance_grid.png",
                    title=title,
                    dpi=dpi,
                )
                _plot_metric_grid(
                    metric_key="norm_diff",
                    metric_label="| ||a|| - ||b|| | (norm diff)",
                    distances=distances,
                    anchor_centers=anchor_centers,
                    anchor_self_means=anchor_self_means,
                    used_animals=used_animals,
                    colours_sorted=colours_sorted,
                    colour_to_marker=colour_to_marker,
                    colour_to_color=colour_to_color,
                    out_dir=out_dir / "anchor_colour_norm_diff_grid.png",
                    title=title,
                    dpi=dpi,
                )
                _plot_metric_grid(
                    metric_key="cosine",
                    metric_label="cos(a, b) (angle)",
                    distances=distances,
                    anchor_centers=anchor_centers,
                    anchor_self_means=anchor_self_means,
                    used_animals=used_animals,
                    colours_sorted=colours_sorted,
                    colour_to_marker=colour_to_marker,
                    colour_to_color=colour_to_color,
                    out_dir=out_dir / "anchor_colour_cosine_grid.png",
                    title=title,
                    dpi=dpi,
                )
                _plot_metric_grid(
                    metric_key="relative_change",
                    metric_label="||a-b|| / ||a|| (relative change)",
                    distances=distances,
                    anchor_centers=anchor_centers,
                    anchor_self_means=anchor_self_means,
                    used_animals=used_animals,
                    colours_sorted=colours_sorted,
                    colour_to_marker=colour_to_marker,
                    colour_to_color=colour_to_color,
                    out_dir=out_dir / "anchor_colour_relative_change_grid.png",
                    title=title,
                    dpi=dpi,
                )
                LOGGER.info("Saved plot: %s", out_dir / "anchor_colour_distance_grid.png")
                LOGGER.info("Saved plot: %s", out_dir / "anchor_colour_norm_diff_grid.png")
                LOGGER.info("Saved plot: %s", out_dir / "anchor_colour_cosine_grid.png")
                LOGGER.info("Saved plot: %s", out_dir / "anchor_colour_relative_change_grid.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    if args.features_root:
        raise SystemExit(
            "--features-root legacy mode is no longer supported in this configuration. "
            "Use --selected-root with --gen-model/--vision-model/--pooling."
        )

    selected_root = Path(args.selected_root).expanduser().resolve()
    if not selected_root.is_dir():
        raise SystemExit(f"Selected root not found: {selected_root}")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else _default_output_root(selected_root)
    )

    anchors_root = selected_root / "anchors" / COMPOSITE_SUBDIR
    animals = _list_animals(anchors_root)
    if not animals:
        raise SystemExit(f"No animals found under {anchors_root}")

    gen_models = [args.gen_model] if args.gen_model else _list_gen_models(anchors_root, animals)
    if not gen_models:
        raise SystemExit("No generation models found.")

    vision_models: List[str] = []
    if args.vision_model:
        vision_models = [args.vision_model]
    else:
        for gen_model in gen_models:
            vision_models.extend(_list_vision_models(anchors_root, animals, gen_model))
        vision_models = _ordered(sorted(set(vision_models)), DEFAULT_VISION_ORDER)
    if not vision_models:
        raise SystemExit("No vision models found under anchors.")

    poolings: List[str] = []
    if args.pooling:
        poolings = [args.pooling]
    else:
        for gen_model in gen_models:
            for vision in vision_models:
                poolings.extend(_list_poolings(anchors_root, animals, gen_model, vision))
        poolings = sorted(set(poolings))
    if not poolings:
        raise SystemExit("No pooling directories found under anchors.")

    _run_structured(selected_root, output_root, gen_models, vision_models, poolings, args.dpi)


if __name__ == "__main__":
    main()
