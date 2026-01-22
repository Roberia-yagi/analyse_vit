from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from PIL import Image

from analyse_vit.clip_oscope.path_utils import resolve_path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompositeEntry:
    left_scale: float
    right_scale: float
    path: Path
    left_only_path: Path
    right_only_path: Path


@dataclass(frozen=True)
class SingleEntry:
    scale: float
    path: Path


def _scale_key(value: float) -> float:
    return round(float(value), 4)


def _load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


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
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if use_amp else contextlib.nullcontext()
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        images = [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
        inputs = torch.stack(images, dim=0).to(device)
        with torch.no_grad(), autocast_ctx:
            batch_features = model.encode_image(inputs).float()
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        features.append(batch_features.cpu())
    return torch.cat(features, dim=0)


def _plot_heatmap(
    sim: np.ndarray,
    left_scales: list[float],
    right_scales: list[float],
    title: str,
    out_path: Path,
    x_label: str,
    y_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sim_min = float(np.min(sim))
    sim_max = float(np.max(sim))
    if sim_min == sim_max:
        sim_min = sim_min - 1e-6
        sim_max = sim_max + 1e-6
    im = ax.imshow(sim, vmin=sim_min, vmax=sim_max, cmap="viridis")
    ax.set_xticks(list(range(len(right_scales))), [f"{scale:.2f}" for scale in right_scales], rotation=45)
    ax.set_yticks(list(range(len(left_scales))), [f"{scale:.2f}" for scale in left_scales])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _parse_pair_entries(pair_meta: dict, run_dir: Path):
    images = pair_meta.get("images", {})
    single_left = images.get("single_left", [])
    single_right = images.get("single_right", [])
    composites = images.get("composites", [])

    if not single_left or not single_right or not composites:
        raise ValueError("meta.json does not contain expected images entries.")

    left_entries = [
        SingleEntry(scale=float(item["scale_ratio"]), path=(run_dir / item["output_file"]).resolve())
        for item in single_left
    ]
    right_entries = [
        SingleEntry(scale=float(item["scale_ratio"]), path=(run_dir / item["output_file"]).resolve())
        for item in single_right
    ]

    composite_entries = [
        CompositeEntry(
            left_scale=float(item["left_scale_ratio"]),
            right_scale=float(item["right_scale_ratio"]),
            path=(run_dir / item["output_file"]).resolve(),
            left_only_path=(run_dir / item["left_only_file"]).resolve(),
            right_only_path=(run_dir / item["right_only_file"]).resolve(),
        )
        for item in composites
    ]

    return left_entries, right_entries, composite_entries


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_features(
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
    paths: list[Path],
) -> Dict[Path, np.ndarray]:
    features = _extract_image_features(model, preprocess, paths, device, batch_size)
    features_np = features.numpy()
    return {path: features_np[idx] for idx, path in enumerate(paths)}


def _write_similarity_csv(
    rows: list[dict[str, float]],
    out_path: Path,
) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze dual-object size grid (composite vs singles).")
    parser.add_argument("--run-dir", default=None, help="Path to a run directory containing meta.json.")
    parser.add_argument("--results-root", default="results", help="Root results directory.")
    parser.add_argument("--results-date", default=None, help="Run directory name under results-root.")
    parser.add_argument("--output-dir", default=None, help="Output directory for analysis artifacts.")

    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave empty for auto.")
    return parser


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

    output_dir = resolve_path(args.output_dir) if args.output_dir else (run_dir / "analysis_dual")
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_meta(run_dir)
    pairs = meta.get("pairs", [])
    if not pairs:
        raise SystemExit("meta.json has no pairs data to analyze.")

    device = _resolve_device(args.device)
    LOGGER.info("Using device: %s", device)
    model, preprocess = _load_model(args.model_name, args.pretrained, device)

    for pair_meta in pairs:
        left_category = pair_meta.get("left_category", "left")
        right_category = pair_meta.get("right_category", "right")
        pair_key = pair_meta.get("pair_key", f"{left_category}__{right_category}")
        left_entries, right_entries, composite_entries = _parse_pair_entries(pair_meta, run_dir)

        left_entries = sorted(left_entries, key=lambda item: item.scale)
        right_entries = sorted(right_entries, key=lambda item: item.scale)
        composite_entries = sorted(
            composite_entries,
            key=lambda item: (_scale_key(item.left_scale), _scale_key(item.right_scale)),
        )

        left_scales = [_scale_key(item.scale) for item in left_entries]
        right_scales = [_scale_key(item.scale) for item in right_entries]

        left_map = { _scale_key(item.scale): item.path for item in left_entries }
        right_map = { _scale_key(item.scale): item.path for item in right_entries }

        composite_paths = [entry.path for entry in composite_entries]
        single_paths = [entry.path for entry in left_entries] + [entry.path for entry in right_entries]
        all_paths = []
        seen = set()
        for path in composite_paths + single_paths:
            if path not in seen:
                seen.add(path)
                all_paths.append(path)

        for path in all_paths:
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

        LOGGER.info(
            "Extracting features for pair %s (%s, %s): %d images.",
            pair_key,
            left_category,
            right_category,
            len(all_paths),
        )
        features = _collect_features(model, preprocess, device, args.batch_size, all_paths)

        left_sim = np.zeros((len(left_scales), len(right_scales)), dtype=np.float32)
        right_sim = np.zeros((len(left_scales), len(right_scales)), dtype=np.float32)
        rows: list[dict[str, float]] = []

        for entry in composite_entries:
            left_key = _scale_key(entry.left_scale)
            right_key = _scale_key(entry.right_scale)
            left_idx = left_scales.index(left_key)
            right_idx = right_scales.index(right_key)

            composite_feat = features[entry.path]
            left_feat = features[left_map[left_key]]
            right_feat = features[right_map[right_key]]

            left_score = float(np.dot(composite_feat, left_feat))
            right_score = float(np.dot(composite_feat, right_feat))

            left_sim[left_idx, right_idx] = left_score
            right_sim[left_idx, right_idx] = right_score

            rows.append(
                {
                    "left_scale": entry.left_scale,
                    "right_scale": entry.right_scale,
                    "left_similarity": left_score,
                    "right_similarity": right_score,
                }
            )

        pair_output_dir = output_dir / pair_key
        pair_output_dir.mkdir(parents=True, exist_ok=True)

        _plot_heatmap(
            left_sim,
            left_scales,
            right_scales,
            f"{left_category} similarity (composite vs left-only)",
            pair_output_dir / "left_similarity_heatmap.png",
            "Right scale ratio",
            "Left scale ratio",
        )
        _plot_heatmap(
            right_sim,
            left_scales,
            right_scales,
            f"{right_category} similarity (composite vs right-only)",
            pair_output_dir / "right_similarity_heatmap.png",
            "Right scale ratio",
            "Left scale ratio",
        )

        _write_similarity_csv(rows, pair_output_dir / "similarity_table.csv")

    LOGGER.info("Analysis complete. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
