from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor


LOGGER = logging.getLogger("analyse_vit.feature_similarity")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "clip": ModelSpec(key="clip", model_id="openai/clip-vit-large-patch14"),
    "siglip": ModelSpec(key="siglip", model_id="google/siglip-base-patch16-256"),
    "dinov2": ModelSpec(key="dinov2", model_id="facebook/dinov2-base"),
}

IMAGE_KEYS = ("real_dom", "real_rare", "toy_dom", "toy_rare")
ALL_PAIRS: Tuple[Tuple[str, str], ...] = tuple(
    (left, right) for left in IMAGE_KEYS for right in IMAGE_KEYS
)
DEFAULT_PAIRS: Tuple[Tuple[str, str], ...] = ALL_PAIRS
SPECIAL_PAIRS: Dict[str, Tuple[str, str]] = {
    "real dominant vs real rare (same subject, different color)": ("real_dom", "real_rare"),
    "toy dominant vs toy rare (same subject, different color)": ("toy_dom", "toy_rare"),
    "real dominant vs toy dominant (same color, different subject)": ("real_dom", "toy_dom"),
    "real rare vs toy rare (same color, different subject)": ("real_rare", "toy_rare"),
}


@dataclass
class RunFeatures:
    run: str
    features_raw: Dict[str, torch.Tensor]
    features_norm: Dict[str, torch.Tensor]


def _default_results_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "results").resolve()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _collect_run_dirs(results_dir: Path) -> List[Path]:
    run_dirs = [path for path in results_dir.iterdir() if path.is_dir() and path.name.startswith("run_")]
    return sorted(run_dirs, key=lambda path: path.name)


def _load_image(path: Path) -> Image.Image:
    return Image.open(path.resolve()).convert("RGB")


def _find_image_path(run_dir: Path, key: str) -> Path:
    candidates = (
        run_dir / "outputs" / f"scene_{key}.png",
        run_dir / "outputs" / f"{key}.png",
        run_dir / f"scene_{key}.png",
        run_dir / f"{key}.png",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image for '{key}' not found in {run_dir}")


def _load_images_for_run(run_dir: Path, keys: Sequence[str]) -> Dict[str, Image.Image]:
    images: Dict[str, Image.Image] = {}
    for key in keys:
        images[key] = _load_image(_find_image_path(run_dir, key))
    return images


def _extract_features(
    model: torch.nn.Module,
    processor: object,
    images: Sequence[Image.Image],
    device: str,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    features: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            if hasattr(model, "get_image_features"):
                batch_features = model.get_image_features(**inputs)
            else:
                outputs = model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    batch_features = outputs.pooler_output
                else:
                    batch_features = outputs.last_hidden_state[:, 0]
            features.append(batch_features.detach().cpu())
    return torch.cat(features, dim=0)


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=-1)


def _cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    return float(torch.dot(vec_a, vec_b).item())


def _toy_center_vector(run_features: Sequence[RunFeatures]) -> torch.Tensor:
    toy_vectors = []
    for run in run_features:
        toy_vectors.append(run.features_norm["toy_dom"])
        toy_vectors.append(run.features_norm["toy_rare"])
    if not toy_vectors:
        raise ValueError("Toy vectors are empty; cannot compute toy center.")
    stacked = torch.stack(toy_vectors, dim=0)
    center = stacked.mean(dim=0)
    return F.normalize(center, dim=0)


def _real_center_vector(run_features: Sequence[RunFeatures]) -> torch.Tensor:
    real_vectors = []
    for run in run_features:
        real_vectors.append(run.features_norm["real_dom"])
        real_vectors.append(run.features_norm["real_rare"])
    if not real_vectors:
        raise ValueError("Real vectors are empty; cannot compute real center.")
    stacked = torch.stack(real_vectors, dim=0)
    center = stacked.mean(dim=0)
    return F.normalize(center, dim=0)


def _parse_pairs(pairs: Sequence[str]) -> Tuple[Tuple[str, str], ...]:
    parsed: List[Tuple[str, str]] = []
    for pair in pairs:
        if ":" not in pair:
            raise ValueError(f"Pair must be formatted as 'key_a:key_b': {pair}")
        left, right = pair.split(":", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise ValueError(f"Pair must contain non-empty keys: {pair}")
        parsed.append((left, right))
    return tuple(parsed)


def _resolve_models(model_args: Sequence[str]) -> List[ModelSpec]:
    resolved: List[ModelSpec] = []
    for entry in model_args:
        entry = entry.strip()
        if entry in MODEL_SPECS:
            resolved.append(MODEL_SPECS[entry])
            continue
        matches = [spec for spec in MODEL_SPECS.values() if spec.model_id == entry]
        if matches:
            resolved.append(matches[0])
            continue
        valid = ", ".join(sorted(MODEL_SPECS.keys()))
        raise ValueError(f"Unknown model '{entry}'. Use one of: {valid} or the full model id.")
    return resolved


def _format_pair_key(left: str, right: str) -> str:
    return f"{left}__{right}"


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: List[float], mean_value: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return float(variance**0.5)


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _tsne_embeddings(
    embeddings: torch.Tensor,
    labels: List[str],
    runs: List[str],
    output_dir: Path,
    model_key: str,
    seed: int,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
) -> None:
    try:
        from sklearn.manifold import TSNE  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("scikit-learn is required for --tsne.") from exc

    if embeddings.shape[0] < 2:
        LOGGER.warning("Skipping t-SNE because there are fewer than 2 samples.")
        return

    max_perplexity = max(1, embeddings.shape[0] - 1)
    if perplexity >= max_perplexity:
        perplexity = float(max(1, max_perplexity // 3))
        LOGGER.info("Adjusted t-SNE perplexity to %s due to small sample size.", perplexity)

    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=seed,
            init="pca",
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=n_iter,
            random_state=seed,
            init="pca",
        )
    coords = tsne.fit_transform(embeddings.cpu().numpy())

    rows: List[Dict[str, object]] = []
    for idx, (label, run) in enumerate(zip(labels, runs)):
        rows.append(
            {
                "model": model_key,
                "run": run,
                "condition": label,
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]),
            }
        )

    csv_path = output_dir / f"tsne_{model_key}.csv"
    _write_csv(csv_path, rows, fieldnames=["model", "run", "condition", "x", "y"])
    LOGGER.info("Saved t-SNE coordinates: %s", csv_path)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("matplotlib is not installed; skipping t-SNE plot.")
        return

    color_map = {
        "real_dom": "#1f77b4",
        "real_rare": "#ff7f0e",
        "toy_dom": "#2ca02c",
        "toy_rare": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(6, 6))
    for condition in IMAGE_KEYS:
        indices = [i for i, label in enumerate(labels) if label == condition]
        if not indices:
            continue
        xs = [coords[i, 0] for i in indices]
        ys = [coords[i, 1] for i in indices]
        ax.scatter(xs, ys, label=condition, c=color_map.get(condition, "#333333"), alpha=0.75)
    ax.set_title(f"t-SNE ({model_key})")
    ax.legend(loc="best")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plot_path = output_dir / f"tsne_{model_key}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    LOGGER.info("Saved t-SNE plot: %s", plot_path)


def _standardize(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.full_like(std, 1e-6), std)
    normalized = (features - mean) / std
    return normalized, mean, std


def _train_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = torch.nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(train_x).squeeze(1)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()

    return model


def _evaluate_probe(model: torch.nn.Module, features: torch.Tensor, labels: torch.Tensor) -> Tuple[float, torch.Tensor]:
    logits = model(features).squeeze(1)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    accuracy = float((preds == labels).float().mean().item())
    return accuracy, probs


def _probe_probs(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    logits = model(features).squeeze(1)
    return torch.sigmoid(logits)


def _split_probe_runs(
    run_features: Sequence[RunFeatures],
    seed: int,
    train_ratio: float,
) -> Tuple[List[RunFeatures], List[RunFeatures]]:
    if len(run_features) < 2:
        raise ValueError("Need at least 2 runs to split train/test without leakage.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("probe train ratio must be between 0 and 1 (exclusive).")
    rng = random.Random(seed)
    indices = list(range(len(run_features)))
    rng.shuffle(indices)
    train_count = int(round(len(run_features) * train_ratio))
    train_count = max(1, min(train_count, len(run_features) - 1))
    train_idx = set(indices[:train_count])
    train_runs = [run_features[i] for i in range(len(run_features)) if i in train_idx]
    test_runs = [run_features[i] for i in range(len(run_features)) if i not in train_idx]
    return train_runs, test_runs


def _linear_probe_analysis(
    run_features: List[RunFeatures],
    output_dir: Path,
    model_key: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    train_ratio: float,
) -> None:
    train_runs, test_runs = _split_probe_runs(run_features, seed=seed, train_ratio=train_ratio)
    LOGGER.info(
        "Linear probe run split (train=%d, test=%d) with ratio=%.2f",
        len(train_runs),
        len(test_runs),
        train_ratio,
    )
    train_features: List[torch.Tensor] = []
    train_labels: List[float] = []
    test_features: List[torch.Tensor] = []
    test_labels: List[float] = []
    condition_features: Dict[str, List[torch.Tensor]] = {key: [] for key in IMAGE_KEYS}

    for run in train_runs:
        train_features.append(run.features_raw["real_dom"])
        train_labels.append(0.0)
        train_features.append(run.features_raw["toy_dom"])
        train_labels.append(1.0)

    for run in test_runs:
        test_features.append(run.features_raw["real_rare"])
        test_labels.append(0.0)
        test_features.append(run.features_raw["toy_rare"])
        test_labels.append(1.0)

    for run in run_features:
        for key in IMAGE_KEYS:
            condition_features[key].append(run.features_raw[key])

    train_x = torch.stack(train_features, dim=0)
    train_y = torch.tensor(train_labels)
    test_x = torch.stack(test_features, dim=0)
    test_y = torch.tensor(test_labels)

    train_x, mean, std = _standardize(train_x)
    test_x = (test_x - mean) / std

    model = _train_linear_probe(train_x, train_y, epochs=epochs, lr=lr, weight_decay=weight_decay, seed=seed)
    train_acc, train_probs = _evaluate_probe(model, train_x, train_y)
    test_acc, test_probs = _evaluate_probe(model, test_x, test_y)

    condition_stats: Dict[str, Dict[str, float]] = {}
    for key, feats in condition_features.items():
        feats_tensor = torch.stack(feats, dim=0)
        feats_tensor = (feats_tensor - mean) / std
        probs = _probe_probs(model, feats_tensor)
        condition_stats[key] = {
            "mean_prob_nonreal": float(probs.mean().item()),
            "std_prob_nonreal": float(probs.std(unbiased=False).item()),
        }

    summary = {
        "model": model_key,
        "train_runs": [run.run for run in train_runs],
        "test_runs": [run.run for run in test_runs],
        "train_run_count": len(train_runs),
        "test_run_count": len(test_runs),
        "train_ratio": train_ratio,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_mean_prob_nonreal": float(train_probs.mean().item()),
        "test_mean_prob_nonreal": float(test_probs.mean().item()),
        "condition_stats": condition_stats,
        "real_rare_minus_real_dom_mean_prob": condition_stats["real_rare"]["mean_prob_nonreal"]
        - condition_stats["real_dom"]["mean_prob_nonreal"],
    }

    output_path = output_dir / f"linear_probe_{model_key}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    LOGGER.info("Saved linear probe summary: %s", output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse composite image embeddings for cosine similarity, t-SNE, and linear probes.",
    )
    parser.add_argument(
        "--results-date",
        required=True,
        help="Results directory name (e.g., 20260119_105730).",
    )
    parser.add_argument(
        "--results-root",
        default=str(_default_results_root()),
        help="Root directory that contains results/<date> (default: repo_root/results).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for analysis artifacts (default: results/<date>/analysis).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SPECS.keys()),
        help="Models to run: clip siglip dinov2 (or full model ids).",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=[f"{left}:{right}" for left, right in DEFAULT_PAIRS],
        help="Pairs to compare, formatted as key_a:key_b.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional path to write per-run JSONL results.",
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Run t-SNE and save coordinates (and plot if matplotlib is available).",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (auto-adjusted if too large for the sample count).",
    )
    parser.add_argument(
        "--tsne-learning-rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate.",
    )
    parser.add_argument(
        "--tsne-n-iter",
        type=int,
        default=1000,
        help="Number of t-SNE iterations.",
    )
    parser.add_argument(
        "--tsne-seed",
        type=int,
        default=42,
        help="Random seed for t-SNE.",
    )
    parser.add_argument(
        "--linear-probe",
        action="store_true",
        help="Run linear probe (train on dominant, test on rare).",
    )
    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=200,
        help="Epochs for linear probe training.",
    )
    parser.add_argument(
        "--probe-lr",
        type=float,
        default=1e-2,
        help="Learning rate for linear probe training.",
    )
    parser.add_argument(
        "--probe-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for linear probe training.",
    )
    parser.add_argument(
        "--probe-train-ratio",
        type=float,
        default=0.5,
        help="Fraction of runs used to train the linear probe (rest used for test).",
    )
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=7,
        help="Random seed for linear probe training.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    results_dir = (results_root / args.results_date).resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    run_dirs = _collect_run_dirs(results_dir)
    if not run_dirs:
        raise SystemExit(f"No run_* directories found in {results_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (results_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    LOGGER.info("Using device: %s", device)

    pairs = _parse_pairs(args.pairs)
    models = _resolve_models(args.models)

    output_handle = None
    if args.output_jsonl:
        output_path = Path(args.output_jsonl).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")
        LOGGER.info("Writing per-run results to %s", output_path)

    try:
        for spec in models:
            LOGGER.info("Loading model: %s", spec.model_id)
            try:
                processor = AutoProcessor.from_pretrained(spec.model_id)
            except Exception:
                processor = AutoImageProcessor.from_pretrained(spec.model_id)
            model = AutoModel.from_pretrained(spec.model_id).to(device)

            metrics_cos: Dict[str, List[float]] = {_format_pair_key(a, b): [] for a, b in pairs}
            metrics_special: Dict[str, List[float]] = {label: [] for label in SPECIAL_PAIRS}
            metrics_toy_center: Dict[str, List[float]] = {"real_dom": [], "real_rare": []}
            metrics_real_center: Dict[str, List[float]] = {"toy_dom": [], "toy_rare": []}
            run_feature_list: List[RunFeatures] = []
            run_payloads: Dict[str, Dict[str, object]] = {}

            for run_dir in run_dirs:
                images_by_key = _load_images_for_run(run_dir, IMAGE_KEYS)
                ordered_images = [images_by_key[key] for key in IMAGE_KEYS]
                features_raw = _extract_features(model, processor, ordered_images, device, args.batch_size)
                features_norm = _normalize_features(features_raw)

                raw_by_key = {key: features_raw[idx] for idx, key in enumerate(IMAGE_KEYS)}
                norm_by_key = {key: features_norm[idx] for idx, key in enumerate(IMAGE_KEYS)}
                run_feature_list.append(RunFeatures(run=run_dir.name, features_raw=raw_by_key, features_norm=norm_by_key))

                run_result: Dict[str, Dict[str, float]] = {"cosine": {}, "special": {}}
                for left, right in pairs:
                    if left not in norm_by_key or right not in norm_by_key:
                        raise KeyError(f"Missing feature for pair {left}:{right} in {run_dir}")
                    pair_key = _format_pair_key(left, right)
                    cos_value = _cosine_similarity(norm_by_key[left], norm_by_key[right])
                    run_result["cosine"][pair_key] = cos_value
                    metrics_cos[pair_key].append(cos_value)

                for label, (left, right) in SPECIAL_PAIRS.items():
                    value = _cosine_similarity(norm_by_key[left], norm_by_key[right])
                    run_result["special"][label] = value
                    metrics_special[label].append(value)

                payload = {
                    "run": run_dir.name,
                    "model": spec.model_id,
                    "pairs": run_result,
                }
                LOGGER.info("%s %s special=%s", run_dir.name, spec.key, run_result["special"])
                run_payloads[run_dir.name] = payload

            toy_center = _toy_center_vector(run_feature_list)
            real_center = _real_center_vector(run_feature_list)
            for run in run_feature_list:
                real_dom_sim = _cosine_similarity(run.features_norm["real_dom"], toy_center)
                real_rare_sim = _cosine_similarity(run.features_norm["real_rare"], toy_center)
                metrics_toy_center["real_dom"].append(real_dom_sim)
                metrics_toy_center["real_rare"].append(real_rare_sim)
                toy_dom_sim = _cosine_similarity(run.features_norm["toy_dom"], real_center)
                toy_rare_sim = _cosine_similarity(run.features_norm["toy_rare"], real_center)
                metrics_real_center["toy_dom"].append(toy_dom_sim)
                metrics_real_center["toy_rare"].append(toy_rare_sim)
                payload = run_payloads.get(run.run)
                if payload is not None:
                    payload["toy_center_similarity"] = {
                        "real_dom": real_dom_sim,
                        "real_rare": real_rare_sim,
                    }
                    payload["real_center_similarity"] = {
                        "toy_dom": toy_dom_sim,
                        "toy_rare": toy_rare_sim,
                    }

            if output_handle:
                for run in run_feature_list:
                    payload = run_payloads.get(run.run)
                    if payload is not None:
                        output_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

            LOGGER.info("Summary (mean ± std)")
            LOGGER.info("Model: %s", spec.model_id)
            for pair_key, values in metrics_cos.items():
                mean_value = _mean(values)
                std_value = _std(values, mean_value)
                LOGGER.info("  cosine %s: %.6f ± %.6f", pair_key, mean_value, std_value)
            for label, values in metrics_special.items():
                mean_value = _mean(values)
                std_value = _std(values, mean_value)
                LOGGER.info("  special %s: %.6f ± %.6f", label, mean_value, std_value)
            for key, values in metrics_toy_center.items():
                mean_value = _mean(values)
                std_value = _std(values, mean_value)
                LOGGER.info("  toy_center_sim %s: %.6f ± %.6f", key, mean_value, std_value)
            for key, values in metrics_real_center.items():
                mean_value = _mean(values)
                std_value = _std(values, mean_value)
                LOGGER.info("  real_center_sim %s: %.6f ± %.6f", key, mean_value, std_value)

            if args.tsne:
                embeddings = torch.stack(
                    [
                        run.features_norm[key]
                        for run in run_feature_list
                        for key in IMAGE_KEYS
                    ],
                    dim=0,
                )
                labels = [key for _ in run_feature_list for key in IMAGE_KEYS]
                runs = [run.run for run in run_feature_list for _ in IMAGE_KEYS]
                _tsne_embeddings(
                    embeddings,
                    labels,
                    runs,
                    output_dir,
                    spec.key,
                    seed=args.tsne_seed,
                    perplexity=args.tsne_perplexity,
                    learning_rate=args.tsne_learning_rate,
                    n_iter=args.tsne_n_iter,
                )

            if args.linear_probe:
                _linear_probe_analysis(
                    run_feature_list,
                    output_dir,
                    spec.key,
                    epochs=args.probe_epochs,
                    lr=args.probe_lr,
                    weight_decay=args.probe_weight_decay,
                    seed=args.probe_seed,
                    train_ratio=args.probe_train_ratio,
                )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if output_handle:
            output_handle.close()


if __name__ == "__main__":
    main()
