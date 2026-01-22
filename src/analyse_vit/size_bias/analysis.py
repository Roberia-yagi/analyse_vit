from __future__ import annotations

import argparse
import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw, ImageFont

from analyse_vit.clip_oscope.path_utils import resolve_path


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageEntry:
    category: str
    scale_ratio: float
    path: Path


def _sanitize_name(name: str) -> str:
    stripped = name.strip().lower()
    if not stripped:
        return "item"
    cleaned = []
    for ch in stripped:
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in (" ", "-", "_"):
            cleaned.append("_")
    output = "".join(cleaned).strip("_")
    while "__" in output:
        output = output.replace("__", "_")
    return output or "item"


def _load_meta(run_dir: Path) -> list[ImageEntry]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    images = meta.get("images", [])
    if not isinstance(images, list) or not images:
        raise ValueError(f"meta.json has no images list: {meta_path}")
    entries: list[ImageEntry] = []
    for item in images:
        category = str(item["category"])
        scale_ratio = float(item["scale_ratio"])
        rel_path = Path(item["output_file"])
        img_path = (run_dir / rel_path).resolve()
        entries.append(ImageEntry(category=category, scale_ratio=scale_ratio, path=img_path))
    return entries


def _group_by_category(entries: Iterable[ImageEntry]) -> Dict[str, list[ImageEntry]]:
    grouped: Dict[str, list[ImageEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.category, []).append(entry)
    for category, items in grouped.items():
        grouped[category] = sorted(items, key=lambda e: e.scale_ratio)
    return grouped


def _load_model(model_name: str, pretrained: str, device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    return model, preprocess


def _extract_image_features(
    model,
    preprocess,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    use_amp = device.type == "cuda"
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16) if use_amp else contextlib.nullcontext()
    )
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        images = [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
        inputs = torch.stack(images, dim=0).to(device)
        with torch.no_grad(), autocast_ctx:
            batch_features = model.encode_image(inputs).float()
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        features.append(batch_features.cpu())
    return torch.cat(features, dim=0)


def _plot_heatmap(sim: np.ndarray, scales: list[float], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sim_min = float(np.min(sim))
    sim_max = float(np.max(sim))
    if sim_min == sim_max:
        sim_min = sim_min - 1e-6
        sim_max = sim_max + 1e-6
    im = ax.imshow(sim, vmin=sim_min, vmax=sim_max, cmap="viridis")
    ticks = list(range(len(scales)))
    labels = [f"{scale:.2f}" for scale in scales]
    ax.set_xticks(ticks, labels, rotation=45, ha="right")
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("Scale ratio")
    ax.set_ylabel("Scale ratio")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_reference_curve(
    sim: np.ndarray,
    scales: list[float],
    title: str,
    out_path: Path,
) -> None:
    ref_idx = 0
    ref_sim = sim[ref_idx]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scales, ref_sim, marker="o", linewidth=1.5)
    ax.set_xlabel("Scale ratio")
    ax.set_ylabel("Cosine similarity to smallest scale")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _collect_pairwise(sim: np.ndarray, scales: list[float]) -> Dict[float, list[float]]:
    bucket: Dict[float, list[float]] = {}
    for i in range(len(scales)):
        for j in range(i + 1, len(scales)):
            diff = round(abs(scales[i] - scales[j]), 4)
            bucket.setdefault(diff, []).append(float(sim[i, j]))
    return bucket


def _plot_pairwise_summary(pairwise: Dict[float, list[float]], out_path: Path) -> None:
    diffs = sorted(pairwise.keys())
    means = [float(np.mean(pairwise[diff])) for diff in diffs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(diffs, means, marker="o", linewidth=1.5)
    ax.set_xlabel("Scale ratio difference")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Cosine similarity vs scale difference (avg)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _make_contact_sheet(
    images: list[Image.Image],
    labels: list[str],
    cols: int,
    out_path: Path,
    pad: int = 12,
    label_pad: int = 6,
    bg_color: Tuple[int, int, int] = (248, 248, 248),
) -> None:
    if not images:
        return
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length.")

    widths = [img.width for img in images]
    heights = [img.height for img in images]
    cell_w = max(widths)
    cell_h = max(heights)
    rows = (len(images) + cols - 1) // cols

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    label_heights = []
    if font is not None:
        dummy = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(dummy)
        for label in labels:
            label_heights.append(draw.textbbox((0, 0), label, font=font)[3])
    label_h = max(label_heights) if label_heights else 12

    sheet_w = pad + cols * (cell_w + pad) - pad
    sheet_h = pad + rows * (cell_h + label_h + label_pad + pad) - pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=bg_color)
    draw = ImageDraw.Draw(sheet)

    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (cell_w + pad)
        y0 = pad + row * (cell_h + label_h + label_pad + pad)
        x_img = x0 + (cell_w - img.width) // 2
        y_img = y0 + (cell_h - img.height) // 2
        sheet.paste(img, (x_img, y_img))
        if font is not None:
            text_x = x0
            text_y = y0 + cell_h + label_pad
            draw.text((text_x, text_y), label, fill=(20, 20, 20), font=font)

    sheet.save(out_path)


def _pca_2d(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, keepdims=True)
    centered = features - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    projected = centered @ components.T
    total_var = (s ** 2).sum()
    explained = (s[:2] ** 2) / total_var if total_var > 0 else np.zeros(2)
    return projected, explained


def _plot_pca(
    projected: np.ndarray,
    scales: np.ndarray,
    categories: list[str],
    out_path: Path,
    title: str,
    explained: np.ndarray,
) -> None:
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    unique_categories = sorted(set(categories))
    markers = {cat: marker_cycle[i % len(marker_cycle)] for i, cat in enumerate(unique_categories)}

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    scatter = None
    for cat in unique_categories:
        idx = [i for i, value in enumerate(categories) if value == cat]
        pts = projected[idx]
        scatter = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=scales[idx],
            cmap="viridis",
            marker=markers[cat],
            edgecolors="none",
            alpha=0.85,
            label=cat,
        )
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Scale ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze size-bias composites (cossim + PCA)")
    parser.add_argument("--run-dir", default=None, help="Path to a run directory containing meta.json.")
    parser.add_argument("--results-root", default="results", help="Root results directory.")
    parser.add_argument("--results-date", default=None, help="Run directory name under results-root.")
    parser.add_argument("--output-dir", default=None, help="Output directory for analysis artifacts.")

    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto.")
    return parser


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: Optional[Iterable[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)

    if args.run_dir:
        run_dir = resolve_path(args.run_dir)
    elif args.results_date:
        results_root = resolve_path(args.results_root)
        run_dir = resolve_path(results_root / args.results_date)
    else:
        raise SystemExit("Please provide --run-dir or --results-date.")

    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    output_dir = resolve_path(args.output_dir) if args.output_dir else (run_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = _load_meta(run_dir)
    grouped = _group_by_category(entries)
    LOGGER.info("Found %d categories.", len(grouped))

    device = _resolve_device(args.device)
    LOGGER.info("Using device: %s", device)
    model, preprocess = _load_model(args.model_name, args.pretrained, device)

    global_features: list[np.ndarray] = []
    global_scales: list[float] = []
    global_categories: list[str] = []

    pairwise_bucket: Dict[float, list[float]] = {}
    common_scales: Optional[list[float]] = None
    common_sims: list[np.ndarray] = []

    for category, items in grouped.items():
        scales = [entry.scale_ratio for entry in items]
        paths = [entry.path for entry in items]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

        LOGGER.info("Extracting features for %s (%d images).", category, len(paths))
        features = _extract_image_features(model, preprocess, paths, device, args.batch_size)
        features_np = features.numpy()
        sim = features_np @ features_np.T

        safe_name = _sanitize_name(category)
        _plot_heatmap(
            sim,
            scales,
            f"{category}: cosine similarity by scale",
            output_dir / f"cossim_heatmap_{safe_name}.png",
        )
        _plot_reference_curve(
            sim,
            scales,
            f"{category}: similarity to smallest scale",
            output_dir / f"cossim_ref_{safe_name}.png",
        )

        per_cat = _collect_pairwise(sim, scales)
        for diff, values in per_cat.items():
            pairwise_bucket.setdefault(diff, []).extend(values)

        contact_images = [Image.open(path).convert("RGB") for path in paths]
        contact_labels = [f"scale {scale:.2f}" for scale in scales]
        _make_contact_sheet(
            contact_images,
            contact_labels,
            cols=5,
            out_path=output_dir / f"contact_sheet_{safe_name}.png",
        )
        for img in contact_images:
            img.close()

        if common_scales is None:
            common_scales = scales
            common_sims.append(sim)
        elif common_scales == scales:
            common_sims.append(sim)

        global_features.append(features_np)
        global_scales.extend(scales)
        global_categories.extend([category] * len(scales))

    if pairwise_bucket:
        _plot_pairwise_summary(pairwise_bucket, output_dir / "cossim_vs_scale_diff.png")

    if common_scales and len(common_sims) > 1:
        mean_sim = np.mean(np.stack(common_sims, axis=0), axis=0)
        _plot_heatmap(
            mean_sim,
            common_scales,
            "Average cosine similarity by scale",
            output_dir / "cossim_heatmap_avg.png",
        )

    if global_features:
        all_features = np.concatenate(global_features, axis=0)
        projected, explained = _pca_2d(all_features)
        _plot_pca(
            projected,
            np.array(global_scales),
            global_categories,
            output_dir / "pca_scales.png",
            "PCA of image features (colored by scale)",
            explained,
        )

    LOGGER.info("Analysis complete. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
