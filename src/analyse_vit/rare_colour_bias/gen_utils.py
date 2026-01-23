from __future__ import annotations

import inspect
import logging
import os
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from .gen_types import RunDirs


_LOGGER_CACHE: Dict[str, logging.Logger] = {}


def _resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _find_repo_root(start: Path) -> Optional[Path]:
    for parent in (start, *start.parents):
        if (parent / "AGENTS.md").exists():
            return parent
    return None


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _get_hf_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token:
        return explicit_token
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        token = os.getenv(env_name)
        if token:
            return token
    return None


def _prepare_run_dirs(output_dir: Path) -> RunDirs:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = output_dir / "inputs"
    masks_dir = output_dir / "masks"
    outputs_dir = output_dir / "outputs"
    meta_dir = output_dir / "meta"
    for path in (inputs_dir, masks_dir, outputs_dir, meta_dir):
        path.mkdir(parents=True, exist_ok=True)
    return RunDirs(root=output_dir, inputs=inputs_dir, masks=masks_dir, outputs=outputs_dir, meta=meta_dir)


def _setup_logger(run_dirs: RunDirs, *, name_suffix: str = "") -> logging.Logger:
    cache_key = f"{run_dirs.root}{name_suffix}"
    cached = _LOGGER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    logger = logging.getLogger(f"flux_sam_composite.{run_dirs.root.name}{name_suffix}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # Defensive cleanup (only for this logger)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    file_handler = logging.FileHandler(run_dirs.meta / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    _LOGGER_CACHE[cache_key] = logger
    return logger


def _get_version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _get_cv2_version() -> Optional[str]:
    try:
        import cv2  # noqa: F401
    except Exception:
        return None
    import cv2

    return getattr(cv2, "__version__", None)


def _load_required_rgb_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing generated image: {path}")
    return Image.open(path).convert("RGB")


def _resolve_run_input(run_dirs: RunDirs, filename: str) -> Path:
    for base_dir in (run_dirs.inputs, run_dirs.outputs, run_dirs.masks, run_dirs.meta):
        candidate = base_dir / filename
        if candidate.exists():
            return candidate
    legacy = run_dirs.root / filename
    if legacy.exists():
        return legacy
    return run_dirs.inputs / filename


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in sig.parameters}


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 30) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 31) & 0xFFFFFFFFFFFFFFFF
    return x


def _derive_run_seed(base_seed: int, run_index: int, salt: int) -> int:
    mixed = (base_seed & 0xFFFFFFFFFFFFFFFF) ^ ((run_index + 1) * 0x9E3779B97F4A7C15) ^ salt
    return int(_splitmix64(mixed) & 0x7FFFFFFFFFFFFFFF)


def _create_timestamp_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = base_dir / timestamp
    suffix = 1
    while run_root.exists():
        run_root = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root
