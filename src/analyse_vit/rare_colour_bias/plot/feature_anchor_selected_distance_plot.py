from __future__ import annotations

import argparse
import json
import logging
import os
import time
from math import atan2, pi
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import torch
from PIL import Image


LOGGER = logging.getLogger("analyse_vit.feature_anchor_selected_distance_plot")

DEFAULT_VISION_ORDER = ("pe-core-l14-336", "siglip2-giant-opt-patch16-384", "qwen3-vl-8b-embed")
DEFAULT_GEN_ORDER = ("flux", "qwen", "sd3.5")
COMPOSITE_SUBDIR = "without_composite"


def _default_selected_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "results" / "selected").resolve()


def _default_output_root(selected_root: Path) -> Path:
    return (selected_root / "analysis" / "anchor_selected_distance").resolve()


def _ordered(items: Sequence[str], preferred: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    for name in preferred:
        if name in items:
            ordered.append(name)
    for name in sorted(items):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _list_animals(root: Path) -> List[str]:
    if not root.is_dir():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


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


def _load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def _anchor_center(anchor_dir: Path) -> torch.Tensor:
    anchor_files = sorted(anchor_dir.glob("*.pt"))
    if not anchor_files:
        raise FileNotFoundError(f"No anchor features found in {anchor_dir}")
    tensors = [_load_tensor(path) for path in anchor_files]
    return torch.stack(tensors, dim=0).mean(dim=0)


def _mean_distance(feature_dir: Path, anchor_center: torch.Tensor) -> Optional[float]:
    values: List[float] = []
    for path in sorted(feature_dir.glob("*.pt")):
        feat = _load_tensor(path)
        values.append(float(torch.norm(feat - anchor_center, p=2).item()))
    if not values:
        return None
    return float(sum(values) / len(values))


def _resolve_colour_root(selected_root: Path) -> Path:
    colour_root = selected_root / "colour" / COMPOSITE_SUBDIR
    if colour_root.is_dir():
        return colour_root
    colours_root = selected_root / "colours" / COMPOSITE_SUBDIR
    if colours_root.is_dir():
        return colours_root
    raise SystemExit(
        "Error: colour root not found under selected/colour(s) "
        f"with composite '{COMPOSITE_SUBDIR}': {colour_root}"
    )


def _iter_anchor_image_mask_pairs(anchors_root: Path, animal: str, gen_model: str) -> Iterable[Tuple[Path, Path]]:
    images_dir = anchors_root / animal / gen_model / "images"
    masks_dir = anchors_root / animal / gen_model / "masks"
    if not images_dir.is_dir():
        raise SystemExit(f"Error: anchor images dir not found: {images_dir}")
    if not masks_dir.is_dir():
        raise SystemExit(f"Error: anchor masks dir not found: {masks_dir}")

    image_paths = sorted(images_dir.glob("anchor_run_*.png"))
    if not image_paths:
        image_paths = sorted(images_dir.glob("run_*.png"))
    for image_path in image_paths:
        stem = image_path.stem
        run_id = ""
        if stem.startswith("anchor_run_"):
            run_id = stem.split("anchor_run_", 1)[1]
        elif stem.startswith("run_"):
            run_id = stem.split("run_", 1)[1]
        if not run_id:
            continue
        mask_path = masks_dir / f"mask_run_{run_id}.png"
        if not mask_path.is_file():
            LOGGER.warning("Mask not found for %s: %s", image_path.name, mask_path)
            continue
        yield image_path, mask_path


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    # sRGB (D65) -> CIE Lab
    def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    rgb_lin = _srgb_to_linear(rgb)
    r = rgb_lin[..., 0]
    g = rgb_lin[..., 1]
    b = rgb_lin[..., 2]

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # D65 reference white
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x /= xn
    y /= yn
    z /= zn

    delta = 6.0 / 29.0
    def _f(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta**3, t ** (1.0 / 3.0), (t / (3 * delta**2)) + (4.0 / 29.0))

    fx = _f(x)
    fy = _f(y)
    fz = _f(z)

    L = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def _delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    # Scalar implementation for two Lab vectors.
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    kL = kC = kH = 1.0
    avg_L = 0.5 * (L1 + L2)
    C1 = (a1**2 + b1**2) ** 0.5
    C2 = (a2**2 + b2**2) ** 0.5
    avg_C = 0.5 * (C1 + C2)

    G = 0.5 * (1.0 - (avg_C**7 / (avg_C**7 + 25.0**7)) ** 0.5) if avg_C != 0 else 0.0
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = (a1p**2 + b1**2) ** 0.5
    C2p = (a2p**2 + b2**2) ** 0.5
    avg_Cp = 0.5 * (C1p + C2p)

    def _hp(a: float, b: float) -> float:
        if a == 0 and b == 0:
            return 0.0
        angle = atan2(b, a)
        if angle < 0:
            angle += 2.0 * pi
        return angle

    h1p = _hp(a1p, b1)
    h2p = _hp(a2p, b2)
    dhp = h2p - h1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif dhp > pi:
        dhp -= 2.0 * pi
    elif dhp < -pi:
        dhp += 2.0 * pi

    dLp = L2 - L1
    dCp = C2p - C1p
    dHp = 2.0 * (C1p * C2p) ** 0.5 * np.sin(dhp / 2.0)

    avg_Lp = 0.5 * (L1 + L2)
    avg_hp = h1p + h2p
    if C1p * C2p == 0:
        avg_hp = h1p + h2p
    else:
        if abs(h1p - h2p) > pi:
            avg_hp = (h1p + h2p + 2.0 * pi) / 2.0
        else:
            avg_hp = (h1p + h2p) / 2.0

    T = (
        1.0
        - 0.17 * np.cos(avg_hp - pi / 6.0)
        + 0.24 * np.cos(2.0 * avg_hp)
        + 0.32 * np.cos(3.0 * avg_hp + pi / 30.0)
        - 0.20 * np.cos(4.0 * avg_hp - 63.0 * pi / 180.0)
    )

    delta_ro = 30.0 * pi / 180.0
    Rc = 2.0 * (avg_Cp**7 / (avg_Cp**7 + 25.0**7)) ** 0.5 if avg_Cp != 0 else 0.0
    Sl = 1.0 + (0.015 * (avg_Lp - 50.0) ** 2) / (20.0 + (avg_Lp - 50.0) ** 2) ** 0.5
    Sc = 1.0 + 0.045 * avg_Cp
    Sh = 1.0 + 0.015 * avg_Cp * T
    Rt = -np.sin(2.0 * delta_ro) * Rc * np.exp(-(((avg_hp - 275.0 * pi / 180.0) / (25.0 * pi / 180.0)) ** 2))

    dE = ((dLp / (kL * Sl)) ** 2 + (dCp / (kC * Sc)) ** 2 + (dHp / (kH * Sh)) ** 2 + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))) ** 0.5
    return float(dE)


def _iter_colour_image_mask_pairs(
    colour_images_dir: Path,
    masks_dir: Path,
) -> Iterable[Tuple[Path, Path]]:
    if not colour_images_dir.is_dir():
        raise SystemExit(f"Error: colour images dir not found: {colour_images_dir}")
    if not masks_dir.is_dir():
        raise SystemExit(f"Error: anchor masks dir not found: {masks_dir}")

    image_paths = sorted(colour_images_dir.glob("*_run_*.png"))
    if not image_paths:
        image_paths = sorted(colour_images_dir.glob("run_*.png"))
    for image_path in image_paths:
        stem = image_path.stem
        run_id = ""
        if "_run_" in stem:
            run_id = stem.split("_run_", 1)[1]
        elif stem.startswith("run_"):
            run_id = stem.split("run_", 1)[1]
        if not run_id:
            continue
        mask_path = masks_dir / f"mask_run_{run_id}.png"
        if not mask_path.is_file():
            LOGGER.warning("Mask not found for %s: %s", image_path.name, mask_path)
            continue
        yield image_path, mask_path


def _masked_lab_mean(image_path: Path, mask_path: Path) -> Tuple[np.ndarray, int]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    mask_arr = (np.asarray(mask, dtype=np.float32) / 255.0) > 0.5
    if rgb.shape[:2] != mask_arr.shape[:2]:
        raise SystemExit(f"Error: image/mask size mismatch: {image_path} vs {mask_path}")
    if not np.any(mask_arr):
        return np.zeros(3, dtype=np.float32), 0

    lab = _rgb_to_lab(rgb)
    pixels = lab[mask_arr]
    if pixels.size == 0:
        return np.zeros(3, dtype=np.float32), 0
    return pixels.mean(axis=0), int(mask_arr.sum())


def _masked_lab_mean_from_pair(pair: Tuple[Path, Path]) -> Tuple[np.ndarray, int]:
    image_path, mask_path = pair
    return _masked_lab_mean(image_path, mask_path)


def _parallel_masked_lab_means(pairs: Sequence[Tuple[Path, Path]]) -> List[Tuple[np.ndarray, int]]:
    if not pairs:
        return []
    max_workers = min(32, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_masked_lab_mean_from_pair, pairs))


def _anchor_lab_mean(anchors_root: Path, animal: str, gen_model: str) -> np.ndarray:
    total = 0
    lab_sum = np.zeros(3, dtype=np.float64)
    pairs = list(_iter_anchor_image_mask_pairs(anchors_root, animal, gen_model))
    for (image_path, _), (lab, count) in zip(pairs, _parallel_masked_lab_means(pairs), strict=True):
        if count == 0:
            LOGGER.warning("Empty mask for %s", image_path.name)
            continue
        lab_sum += lab * count
        total += count

    if total == 0:
        raise SystemExit(f"Error: no valid anchor mask pixels for {animal}/{gen_model}")
    return lab_sum / float(total)


def _colour_lab_mean(colour_images_dir: Path, masks_dir: Path) -> Optional[np.ndarray]:
    total = 0
    lab_sum = np.zeros(3, dtype=np.float64)
    pairs = list(_iter_colour_image_mask_pairs(colour_images_dir, masks_dir))
    for (image_path, _), (lab, count) in zip(pairs, _parallel_masked_lab_means(pairs), strict=True):
        if count == 0:
            LOGGER.warning("Empty mask for %s", image_path.name)
            continue
        lab_sum += lab * count
        total += count
    if total == 0:
        return None
    return lab_sum / float(total)


def _pick_colour_ranks(
    *,
    selected_root: Path,
    anchors_root: Path,
    animal: str,
    gen_model: str,
    ranks: Sequence[int],
) -> Dict[int, Dict[str, float | str | None]]:
    candidates = _colour_candidates(
        selected_root=selected_root,
        anchors_root=anchors_root,
        animal=animal,
        gen_model=gen_model,
        cache=None,
    )
    needed = max(ranks)
    if len(candidates) < needed:
        raise SystemExit(
            f"Error: not enough colours to select rank {needed} for animal={animal} "
            f"(available={len(candidates)})"
        )

    selected: Dict[int, Dict[str, float | str | None]] = {}
    for rank in ranks:
        dist, name = candidates[rank - 1]
        selected[rank] = {
            "name": name,
            "delta_e00": dist,
        }
    return selected


def _colour_candidates(
    *,
    selected_root: Path,
    anchors_root: Path,
    animal: str,
    gen_model: str,
    cache: Optional[Dict[Tuple[str, str], List[Tuple[float, str]]]],
) -> List[Tuple[float, str]]:
    cache_key = (animal, gen_model)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    colour_root = _resolve_colour_root(selected_root)
    animal_root = colour_root / animal
    if not animal_root.is_dir():
        raise SystemExit(f"Error: colour root not found for animal: {animal_root}")

    anchor_lab = _anchor_lab_mean(anchors_root, animal, gen_model)
    masks_dir = anchors_root / animal / gen_model / "masks"
    candidates: List[Tuple[float, str]] = []
    for colour_dir in sorted([p for p in animal_root.iterdir() if p.is_dir()]):
        colour_name = colour_dir.name
        colour_images_dir = colour_dir / gen_model / "images"
        colour_lab = _colour_lab_mean(colour_images_dir, masks_dir)
        if colour_lab is None:
            LOGGER.warning("No valid colour mask pixels for %s (animal=%s); skipping", colour_name, animal)
            continue
        dist = _delta_e_ciede2000(anchor_lab, colour_lab)
        candidates.append((dist, colour_name))

    if not candidates:
        raise SystemExit(f"Error: no colour candidates for animal={animal}")

    candidates.sort(key=lambda item: item[0])
    if cache is not None:
        cache[cache_key] = candidates
    return candidates


def _choose_global_colours(
    *,
    selected_root: Path,
    anchors_root: Path,
    animals: Sequence[str],
    gen_model: str,
    ranks: Sequence[int],
    cache: Dict[Tuple[str, str], List[Tuple[float, str]]],
) -> Tuple[Dict[int, str], Dict[int, Dict[str, Dict[str, float]]]]:
    colour_root = _resolve_colour_root(selected_root)
    rank_counts: Dict[int, Dict[str, int]] = {rank: {} for rank in ranks}
    rank_delta: Dict[int, Dict[str, List[float]]] = {rank: {} for rank in ranks}
    for animal in animals:
        if not (colour_root / animal).is_dir():
            continue
        candidates = _colour_candidates(
            selected_root=selected_root,
            anchors_root=anchors_root,
            animal=animal,
            gen_model=gen_model,
            cache=cache,
        )
        for rank in ranks:
            if len(candidates) < rank:
                raise SystemExit(
                    f"Error: not enough colours to select rank {rank} for animal={animal} "
                    f"(available={len(candidates)})"
                )
            delta, name = candidates[rank - 1]
            rank_counts[rank][name] = rank_counts[rank].get(name, 0) + 1
            rank_delta[rank].setdefault(name, []).append(delta)

    chosen: Dict[int, str] = {}
    stats: Dict[int, Dict[str, Dict[str, float]]] = {}
    for rank in ranks:
        if not rank_counts[rank]:
            raise SystemExit(f"Error: no colour candidates found for rank={rank} (gen_model={gen_model})")
        def _key(item: Tuple[str, int]) -> Tuple[int, float, str]:
            name, count = item
            mean_delta = float(sum(rank_delta[rank].get(name, [0.0])) / len(rank_delta[rank].get(name, [1.0])))
            return (-count, mean_delta, name)

        best_name = sorted(rank_counts[rank].items(), key=_key)[0][0]
        chosen[rank] = best_name
        stats[rank] = {}
        for name, count in rank_counts[rank].items():
            mean_delta = float(sum(rank_delta[rank].get(name, [0.0])) / len(rank_delta[rank].get(name, [1.0])))
            stats[rank][name] = {"count": int(count), "mean_delta_e00": mean_delta}
    return chosen, stats


def _plot_all_series(
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    animals: List[str],
    series: Sequence[Tuple[str, Sequence[float], str, str]],
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 8.5))
    x_positions = list(range(len(animals)))
    for label, values, marker, color in series:
        edge_kwargs: Dict[str, float | str] = {}
        if color not in {"black"}:
            try:
                rgba = mcolors.to_rgba(color)
            except ValueError as exc:
                raise SystemExit(
                    f"Error: plot color '{color}' is not a valid Matplotlib color name or hex."
                ) from exc
            if rgba[:3] == (1.0, 1.0, 1.0):
                edge_kwargs = {"edgecolor": "black", "linewidth": 1.2}
        ax.scatter(x_positions, values, s=85, alpha=0.9, label=label, marker=marker, color=color)
        if edge_kwargs:
            ax.scatter(
                x_positions,
                values,
                s=85,
                alpha=0.9,
                label=None,
                marker=marker,
                color=color,
                **edge_kwargs,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(animals, rotation=45, ha="right", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_title(title, fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            borderaxespad=0.0,
            fontsize=12,
            ncol=2,
        )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot anchor L2 distance for size/angle/colour with fixed selections.",
    )
    parser.add_argument(
        "--selected-root",
        default=str(_default_selected_root()),
        help="Structured root with anchors/angles/colour/size (default: repo_root/results/selected).",
    )
    parser.add_argument(
        "--composite-subdir",
        default=COMPOSITE_SUBDIR,
        help="Composite subdir under anchors/angles/colour(s) (default: without_composite).",
    )
    parser.add_argument("--gen-model", default=None, help="Generation model (flux/qwen/sd3.5).")
    parser.add_argument("--vision-model", default=None, help="Vision encoder key (e.g., pe-core-l14-336).")
    parser.add_argument("--pooling", default=None, help="Pooling/embedding key (e.g., attention_pooling).")
    parser.add_argument(
        "--size-values",
        nargs=2,
        default=("30", "60"),
        help="Two size buckets to plot (default: 30 60).",
    )
    parser.add_argument(
        "--angles",
        nargs=2,
        default=("back", "left"),
        help="Two angles to plot (default: back left).",
    )
    parser.add_argument(
        "--colour-ranks",
        nargs=2,
        type=int,
        default=(5, 10),
        help="Two colour ranks to select by hue distance (default: 5 10).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root for plots (default: <selected-root>/analysis/anchor_selected_distance).",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    return parser


def _run_structured(
    selected_root: Path,
    output_root: Path,
    gen_models: Sequence[str],
    vision_models: Sequence[str],
    poolings: Sequence[str],
    sizes: Sequence[str],
    angles: Sequence[str],
    colour_ranks: Sequence[int],
    dpi: int,
) -> None:
    overall_start = time.perf_counter()
    anchors_root = selected_root / "anchors" / COMPOSITE_SUBDIR
    angles_root = selected_root / "angles" / COMPOSITE_SUBDIR
    size_root = selected_root / "size"
    colour_root = _resolve_colour_root(selected_root)
    colour_candidates_cache: Dict[Tuple[str, str], List[Tuple[float, str]]] = {}
    global_colour_cache: Dict[Tuple[str, Tuple[int, ...]], Tuple[Dict[int, str], Dict[int, Dict[str, Dict[str, float]]]]] = {}
    mean_distance_cache: Dict[Path, Optional[float]] = {}
    anchor_center_cache: Dict[Path, torch.Tensor] = {}

    animals = _list_animals(anchors_root)
    if not animals:
        raise SystemExit(f"No animals found under {anchors_root}")
    LOGGER.info(
        "Start plot: animals=%d gen_models=%d vision_models=%d poolings=%d",
        len(animals),
        len(gen_models),
        len(vision_models),
        len(poolings),
    )

    for gen_model in gen_models:
        rank_key = tuple(colour_ranks)
        global_cache_key = (gen_model, rank_key)
        if global_cache_key not in global_colour_cache:
            global_colour_cache[global_cache_key] = _choose_global_colours(
                selected_root=selected_root,
                anchors_root=anchors_root,
                animals=animals,
                gen_model=gen_model,
                ranks=colour_ranks,
                cache=colour_candidates_cache,
            )
        global_colour_names, global_colour_stats = global_colour_cache[global_cache_key]
        for vision in vision_models:
            for pooling in poolings:
                loop_start = time.perf_counter()
                size_values: Dict[str, Tuple[float, float]] = {}
                angle_values: Dict[str, Tuple[float, float]] = {}
                colour_values: Dict[str, Tuple[float, float]] = {}
                anchor_centers: Dict[str, torch.Tensor] = {}
                anchor_self_means: Dict[str, float] = {}

                for animal in animals:
                    animal_start = time.perf_counter()
                    anchor_dir = anchors_root / animal / gen_model / "features" / vision / pooling
                    if not anchor_dir.is_dir():
                        continue
                    center = anchor_center_cache.get(anchor_dir)
                    if center is None:
                        try:
                            center = _anchor_center(anchor_dir)
                        except FileNotFoundError:
                            continue
                        anchor_center_cache[anchor_dir] = center
                    anchor_centers[animal] = center
                    self_mean = mean_distance_cache.get(anchor_dir)
                    if self_mean is None and anchor_dir not in mean_distance_cache:
                        self_mean = _mean_distance(anchor_dir, center)
                        mean_distance_cache[anchor_dir] = self_mean
                    if self_mean is not None:
                        anchor_self_means[animal] = self_mean

                    size_dir_a = size_root / animal / gen_model / "features" / vision / pooling / sizes[0]
                    size_dir_b = size_root / animal / gen_model / "features" / vision / pooling / sizes[1]
                    if size_dir_a.is_dir() and size_dir_b.is_dir():
                        size_mean_a = mean_distance_cache.get(size_dir_a)
                        if size_mean_a is None and size_dir_a not in mean_distance_cache:
                            size_mean_a = _mean_distance(size_dir_a, center)
                            mean_distance_cache[size_dir_a] = size_mean_a
                        size_mean_b = mean_distance_cache.get(size_dir_b)
                        if size_mean_b is None and size_dir_b not in mean_distance_cache:
                            size_mean_b = _mean_distance(size_dir_b, center)
                            mean_distance_cache[size_dir_b] = size_mean_b
                        if size_mean_a is not None and size_mean_b is not None:
                            size_values[animal] = (size_mean_a, size_mean_b)
                        else:
                            LOGGER.warning("No size features for %s", animal)
                    else:
                        LOGGER.warning("Missing size dirs for %s (%s/%s)", animal, sizes[0], sizes[1])

                    angle_dir_a = (
                        angles_root / animal / angles[0] / gen_model / "features" / vision / pooling
                    )
                    angle_dir_b = (
                        angles_root / animal / angles[1] / gen_model / "features" / vision / pooling
                    )
                    if angle_dir_a.is_dir() and angle_dir_b.is_dir():
                        angle_mean_a = mean_distance_cache.get(angle_dir_a)
                        if angle_mean_a is None and angle_dir_a not in mean_distance_cache:
                            angle_mean_a = _mean_distance(angle_dir_a, center)
                            mean_distance_cache[angle_dir_a] = angle_mean_a
                        angle_mean_b = mean_distance_cache.get(angle_dir_b)
                        if angle_mean_b is None and angle_dir_b not in mean_distance_cache:
                            angle_mean_b = _mean_distance(angle_dir_b, center)
                            mean_distance_cache[angle_dir_b] = angle_mean_b
                        if angle_mean_a is not None and angle_mean_b is not None:
                            angle_values[animal] = (angle_mean_a, angle_mean_b)
                        else:
                            LOGGER.warning("No angle features for %s", animal)
                    else:
                        LOGGER.warning("Missing angle dirs for %s (%s/%s)", animal, angles[0], angles[1])

                    if (colour_root / animal).is_dir():
                        colour_dir_a = (
                            colour_root
                            / animal
                            / str(global_colour_names[colour_ranks[0]])
                            / gen_model
                            / "features"
                            / vision
                            / pooling
                        )
                        colour_dir_b = (
                            colour_root
                            / animal
                            / str(global_colour_names[colour_ranks[1]])
                            / gen_model
                            / "features"
                            / vision
                            / pooling
                        )
                        if colour_dir_a.is_dir() and colour_dir_b.is_dir():
                            colour_mean_a = mean_distance_cache.get(colour_dir_a)
                            if colour_mean_a is None and colour_dir_a not in mean_distance_cache:
                                colour_mean_a = _mean_distance(colour_dir_a, center)
                                mean_distance_cache[colour_dir_a] = colour_mean_a
                            colour_mean_b = mean_distance_cache.get(colour_dir_b)
                            if colour_mean_b is None and colour_dir_b not in mean_distance_cache:
                                colour_mean_b = _mean_distance(colour_dir_b, center)
                                mean_distance_cache[colour_dir_b] = colour_mean_b
                            if colour_mean_a is not None and colour_mean_b is not None:
                                colour_values[animal] = (colour_mean_a, colour_mean_b)
                            else:
                                LOGGER.warning("No colour features for %s", animal)
                        else:
                            LOGGER.warning("Missing colour features for %s", animal)
                    else:
                        LOGGER.warning("Missing colour root for %s", animal)

                    LOGGER.info(
                        "Processed animal=%s gen=%s vision=%s pooling=%s in %.2fs",
                        animal,
                        gen_model,
                        vision,
                        pooling,
                        time.perf_counter() - animal_start,
                    )

                common_animals = sorted(
                    set(size_values.keys()) & set(angle_values.keys()) & set(colour_values.keys())
                )
                if not common_animals:
                    LOGGER.info("No data for gen=%s vision=%s pooling=%s; skipping.", gen_model, vision, pooling)
                    continue

                out_dir = output_root / gen_model / vision / pooling
                out_dir.mkdir(parents=True, exist_ok=True)

                other_means: Dict[str, float] = {}
                for animal in common_animals:
                    center = anchor_centers.get(animal)
                    if center is None:
                        continue
                    other_centers = [val for key, val in anchor_centers.items() if key != animal]
                    if not other_centers:
                        continue
                    stacked = torch.stack(other_centers, dim=0)
                    center_batch = center.expand_as(stacked)
                    values = torch.norm(stacked - center_batch, p=2, dim=1)
                    other_means[animal] = float(values.mean().item())

                summary = {
                    "selected_root": str(selected_root),
                    "gen_model": gen_model,
                    "vision_model": vision,
                    "pooling": pooling,
                    "sizes": list(sizes),
                    "angles": list(angles),
                    "colour_ranks": list(colour_ranks),
                    "fixed_colour_names": {str(rank): global_colour_names[rank] for rank in colour_ranks},
                    "fixed_colour_stats": global_colour_stats,
                    "colour_distance_metric": "CIEDE2000",
                    "animals": common_animals,
                    "anchor_self_means": {animal: anchor_self_means.get(animal, float("nan")) for animal in common_animals},
                    "anchor_other_means": {animal: other_means.get(animal, float("nan")) for animal in common_animals},
                }
                (out_dir / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
                    encoding="utf-8",
                )

                title = f"{gen_model} | {vision} | {pooling}"
                series = [
                    (f"size={sizes[0]}", [size_values[a][0] for a in common_animals], "o", "#1f77b4"),
                    (f"size={sizes[1]}", [size_values[a][1] for a in common_animals], "s", "#ff7f0e"),
                    (f"angle={angles[0]}", [angle_values[a][0] for a in common_animals], "D", "#2ca02c"),
                    (f"angle={angles[1]}", [angle_values[a][1] for a in common_animals], "^", "#d62728"),
                    (str(global_colour_names[colour_ranks[0]]), [colour_values[a][0] for a in common_animals], "v", str(global_colour_names[colour_ranks[0]])),
                    (str(global_colour_names[colour_ranks[1]]), [colour_values[a][1] for a in common_animals], "P", str(global_colour_names[colour_ranks[1]])),
                    ("self_mean", [anchor_self_means.get(a, float("nan")) for a in common_animals], "x", "black"),
                    ("others_mean", [other_means.get(a, float("nan")) for a in common_animals], "X", "black"),
                ]
                _plot_all_series(
                    out_path=out_dir / "anchor_selected_distance_all.png",
                    title=title,
                    ylabel="L2 distance from anchor center",
                    animals=common_animals,
                    series=series,
                    dpi=dpi,
                )
                LOGGER.info("Saved plot: %s", out_dir / "anchor_selected_distance_all.png")
                LOGGER.info(
                    "Finished gen=%s vision=%s pooling=%s in %.2fs (size=%d angle=%d colour=%d)",
                    gen_model,
                    vision,
                    pooling,
                    time.perf_counter() - loop_start,
                    len(size_values),
                    len(angle_values),
                    len(colour_values),
                )
    LOGGER.info("All done in %.2fs", time.perf_counter() - overall_start)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    global COMPOSITE_SUBDIR
    COMPOSITE_SUBDIR = args.composite_subdir
    if COMPOSITE_SUBDIR not in {"with_composite", "without_composite"}:
        raise SystemExit(
            "Error: composite-subdir must be 'with_composite' or 'without_composite': "
            f"{COMPOSITE_SUBDIR}"
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
        raise SystemExit("No generation models found under anchors.")

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

    sizes = list(args.size_values)
    if len(sizes) != 2:
        raise SystemExit("size-values must provide exactly 2 entries.")

    angles = list(args.angles)
    if len(angles) != 2:
        raise SystemExit("angles must provide exactly 2 entries.")

    colour_ranks = list(args.colour_ranks)
    if len(colour_ranks) != 2:
        raise SystemExit("colour-ranks must provide exactly 2 entries.")
    if min(colour_ranks) < 1:
        raise SystemExit("colour-ranks must be >= 1.")

    _run_structured(
        selected_root,
        output_root,
        gen_models,
        vision_models,
        poolings,
        sizes,
        angles,
        colour_ranks,
        args.dpi,
    )


if __name__ == "__main__":
    main()
