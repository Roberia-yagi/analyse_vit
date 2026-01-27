from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
from PIL import Image

if TYPE_CHECKING:
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

from .gen_utils import _get_hf_token

LOGGER = logging.getLogger("analyse_vit.feature_similarity")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    backend: str = "hf"


MODEL_SPECS: Dict[str, ModelSpec] = {
    "pe-core-l14-336": ModelSpec(key="pe-core-l14-336", model_id="PE-Core-L14-336", backend="perception"),
    "qwen3-vl-8b": ModelSpec(key="qwen3-vl-8b", model_id="Qwen/Qwen3-VL-8B-Instruct", backend="qwen_vl"),
    "qwen3-vl-8b-embed": ModelSpec(
        key="qwen3-vl-8b-embed",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        backend="qwen_vl_embed",
    ),
}
MODEL_ALIASES: Dict[str, str] = {
    "pe": "pe-core-l14-336",
    "pe-core": "pe-core-l14-336",
    "pe-core-l14": "pe-core-l14-336",
    "pe-core-l14-336": "pe-core-l14-336",
    "core": "pe-core-l14-336",
    "l14": "pe-core-l14-336",
    "qwen": "qwen3-vl-8b",
    "qwen3": "qwen3-vl-8b",
    "qwen3-vl": "qwen3-vl-8b",
    "qwen3-vl-8b": "qwen3-vl-8b",
    "qwen-embed": "qwen3-vl-8b-embed",
    "qwen_embed": "qwen3-vl-8b-embed",
    "qwen3-embed": "qwen3-vl-8b-embed",
    "qwen3-vl-8b-embed": "qwen3-vl-8b-embed",
}

POOLING_CHOICES: Tuple[str, ...] = ("attention_pooling", "mean_patch", "cls")

IMAGE_KEYS = ("real_dom", "real_rare", "toy_dom", "toy_rare")
KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "real_dom": ("real_normal",),
    "real_rare": ("real_atypical",),
    "toy_dom": ("toy_normal",),
    "toy_rare": ("toy_atypical",),
}
KEY_SUBDIR_PRIORITY: Dict[str, Tuple[str, ...]] = {
    "real_dom": ("outputs", ""),
    "toy_dom": ("outputs", ""),
    "real_rare": ("outputs", ""),
    "toy_rare": ("outputs", ""),
}
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


def _token_kwargs(fn: Callable[..., object], token: str | None) -> Dict[str, str]:
    if not token:
        return {}
    sig = inspect.signature(fn)
    if "token" in sig.parameters:
        return {"token": token}
    if "use_auth_token" in sig.parameters:
        return {"use_auth_token": token}
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    return {}


def _is_unknown_architecture_error(exc: Exception) -> bool:
    message = str(exc)
    return "does not recognize this architecture" in message and "model type" in message


def _raise_qwen_architecture_error(spec: ModelSpec, exc: Exception) -> None:
    try:
        import transformers as _transformers  # type: ignore
    except Exception:
        _transformers = None
    version = getattr(_transformers, "__version__", "unknown")
    raise SystemExit(
        "Unable to load the Qwen VL checkpoint. The model reports a `qwen3_vl` architecture that is not "
        f"recognized by the installed Transformers ({version}). Please upgrade Transformers (or install from "
        "source) and retry. Alternatively, please skip Qwen by passing `--models pe`."
    ) from exc


def _collect_run_dirs(results_dir: Path) -> List[Path]:
    run_dirs = [path for path in results_dir.iterdir() if path.is_dir() and path.name.startswith("run_")]
    return sorted(run_dirs, key=lambda path: path.name)


def _load_image(path: Path) -> Image.Image:
    return Image.open(path.resolve()).convert("RGB")


def _candidate_filenames(key: str) -> List[str]:
    candidates = [f"scene_{key}.png", f"{key}.png"]
    for alias in KEY_ALIASES.get(key, ()):
        candidates.append(f"scene_{alias}.png")
        candidates.append(f"{alias}.png")
    seen: set[str] = set()
    ordered: List[str] = []
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _search_dirs_for_key(run_dir: Path, key: str) -> List[Path]:
    order = KEY_SUBDIR_PRIORITY.get(key, ("outputs", "inputs", ""))
    dirs: List[Path] = []
    for entry in order:
        candidate = run_dir if entry == "" else run_dir / entry
        dirs.append(candidate)
    return dirs


def _find_image_path(run_dir: Path, key: str) -> Path:
    for search_dir in _search_dirs_for_key(run_dir, key):
        for filename in _candidate_filenames(key):
            candidate = search_dir / filename
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Image for '{key}' not found in {run_dir}")


def _load_images_for_run(run_dir: Path, keys: Sequence[str]) -> Dict[str, Image.Image]:
    images: Dict[str, Image.Image] = {}
    for key in keys:
        images[key] = _load_image(_find_image_path(run_dir, key))
    return images


def _extract_features_hf(
    model: torch.nn.Module,
    processor: object,
    images: Sequence[Image.Image],
    device: str,
    batch_size: int,
    pooling: str,
) -> torch.Tensor:
    model.eval()
    features: List[torch.Tensor] = []
    logged = False
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            if pooling == "attention_pooling":
                if not hasattr(model, "get_image_features"):
                    raise ValueError("hf: attention_pooling requires model.get_image_features.")
                batch_features = model.get_image_features(**inputs)
                if not logged:
                    LOGGER.info("Feature extraction: hf attention_pooling via model.get_image_features().")
                    logged = True
            else:
                outputs = model(**inputs)
                sequence, source = _select_sequence_from_outputs(outputs)
                batch_features = _pool_from_sequence(sequence, pooling=pooling, label="hf")
                if not logged:
                    LOGGER.info("Feature extraction: hf %s via model.forward -> %s.", pooling, source)
                    logged = True
            features.append(batch_features.detach().cpu())
    return torch.cat(features, dim=0)


def _extract_features_perception(
    model: torch.nn.Module,
    processor: object,
    images: Sequence[Image.Image],
    device: str,
    batch_size: int,
    pooling: str,
) -> torch.Tensor:
    model.eval()
    features: List[torch.Tensor] = []
    use_amp = device == "cuda"
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    logged = False
    forward_features_model, forward_features_label = _resolve_forward_features_model(model)
    with torch.no_grad(), autocast_ctx:
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = torch.stack([processor(image) for image in batch], dim=0).to(device)
            if pooling == "attention_pooling":
                if not hasattr(model, "encode_image"):
                    raise ValueError("perception: attention_pooling requires model.encode_image.")
                batch_features = model.encode_image(inputs)
                if not logged:
                    LOGGER.info("Feature extraction: perception attention_pooling via model.encode_image().")
                    logged = True
            else:
                if forward_features_model is not None:
                    sequence = _call_forward_features(forward_features_model, inputs)
                    if sequence is None:
                        raise ValueError("perception: forward_features returned None.")
                    if sequence.dim() != 3:
                        raise ValueError(
                            f"perception: forward_features must return tokens (3D), got {sequence.dim()}D."
                        )
                    batch_features = _pool_from_sequence(sequence, pooling=pooling, label="perception")
                    if not logged:
                        LOGGER.info(
                            "Feature extraction: perception %s via %s(strip_cls_token=False).",
                            pooling,
                            forward_features_label,
                        )
                        logged = True
                else:
                    outputs = model(inputs)
                    try:
                        sequence, source = _select_sequence_from_outputs(outputs)
                        batch_features = _pool_from_sequence(sequence, pooling=pooling, label="perception")
                        if not logged:
                            LOGGER.info(
                                "Feature extraction: perception %s via model.forward -> %s.",
                                pooling,
                                source,
                            )
                            logged = True
                    except ValueError as exc:
                        if pooling != "cls":
                            raise ValueError(
                                "perception: cls/mean_patch requires forward_features or sequence outputs; "
                                "model.forward did not return tokens."
                            ) from exc
                        pooled, source = _select_pooled_from_outputs(outputs)
                        batch_features = pooled
                        if not logged:
                            LOGGER.warning(
                                "Feature extraction: perception cls via model.forward pooled output (%s); "
                                "no tokens available.",
                                source,
                            )
                            logged = True
            features.append(batch_features.float().detach().cpu())
    return torch.cat(features, dim=0)


def _filter_kwargs_for_callable(func: Callable[..., object], kwargs: Dict[str, object]) -> Dict[str, object]:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in sig.parameters}


def _resolve_forward_features_model(model: torch.nn.Module) -> Tuple[torch.nn.Module | None, str]:
    if hasattr(model, "forward_features"):
        return model, "model.forward_features"
    for attr in ("visual", "vision_model", "image_encoder", "encoder", "backbone", "trunk"):
        candidate = getattr(model, attr, None)
        if isinstance(candidate, torch.nn.Module) and hasattr(candidate, "forward_features"):
            return candidate, f"model.{attr}.forward_features"
    return None, "model.forward_features"


def _unwrap_qwen_vision_encoder(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "get_vision_tower"):
        vision = model.get_vision_tower()
    elif hasattr(model, "vision_tower"):
        vision = getattr(model, "vision_tower")
    elif hasattr(model, "visual"):
        vision = getattr(model, "visual")
    elif hasattr(model, "vision_model"):
        vision = getattr(model, "vision_model")
    else:
        raise ValueError("Qwen VL model does not expose a vision encoder (vision_tower/visual/vision_model).")

    if isinstance(vision, (list, tuple)):
        if not vision:
            raise ValueError("Qwen VL vision_tower is empty.")
        vision = vision[0]
    if not isinstance(vision, torch.nn.Module):
        raise ValueError("Qwen VL vision encoder is not a torch module.")
    return vision


def _resolve_qwen_vision_blocks(model: torch.nn.Module) -> Tuple[Sequence[torch.nn.Module], str] | None:
    def _get_attr(obj: object, name: str) -> object | None:
        value = getattr(obj, name, None)
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        return value

    def _resolve_path(obj: object, path: str) -> object | None:
        current = obj
        for attr in path.split("."):
            current = _get_attr(current, attr)
            if current is None:
                return None
        return current

    for path in ("visual.blocks", "vision_tower.blocks", "vision_model.blocks", "encoder.blocks", "blocks"):
        blocks = _resolve_path(model, path)
        if isinstance(blocks, torch.nn.ModuleList):
            return list(blocks), f"model.{path}"
        if isinstance(blocks, (list, tuple)) and blocks and isinstance(blocks[0], torch.nn.Module):
            return list(blocks), f"model.{path}"
    return None


def _coerce_block_output(output: object) -> torch.Tensor | None:
    def _prefer_sequence(values: Sequence[object]) -> torch.Tensor | None:
        for item in values:
            if isinstance(item, torch.Tensor) and item.dim() == 3:
                return item
        for item in values:
            if isinstance(item, torch.Tensor):
                return item
        return None

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _prefer_sequence(output)
    if isinstance(output, dict):
        sequence = _prefer_sequence(list(output.values()))
        if sequence is not None:
            return sequence
        for key in ("hidden_states", "last_hidden_state", "output"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    for attr in ("hidden_states", "last_hidden_state", "output"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _mean_pool_flat_sequence(
    hidden_2d: torch.Tensor,
    grid_thw: torch.Tensor | None,
    label: str,
) -> torch.Tensor:
    if hidden_2d.dim() != 2:
        raise ValueError(f"{label}: expected 2D hidden states for flat pooling, got {hidden_2d.dim()}D.")
    lengths: List[int] = []
    if grid_thw is not None:
        if grid_thw.dim() != 2 or grid_thw.shape[-1] != 3:
            raise ValueError(f"{label}: grid_thw must be shape [B,3], got {tuple(grid_thw.shape)}.")
        lengths = [int(row[0].item() * row[1].item() * row[2].item()) for row in grid_thw]
    if not lengths:
        raise ValueError(f"{label}: missing grid_thw for flat hidden states.")
    if sum(lengths) != hidden_2d.shape[0]:
        raise ValueError(
            f"{label}: token count mismatch (sum={sum(lengths)}) vs hidden_2d={hidden_2d.shape[0]}."
        )
    chunks = torch.split(hidden_2d, lengths, dim=0)
    pooled = [chunk.mean(dim=0) if chunk.numel() else hidden_2d.new_zeros(hidden_2d.shape[1]) for chunk in chunks]
    return torch.stack(pooled, dim=0)


def _vision_requires_hidden_states(vision: torch.nn.Module) -> bool:
    try:
        sig = inspect.signature(vision.forward)
    except (TypeError, ValueError):
        return False
    params = set(sig.parameters.keys())
    if "hidden_states" in params and "grid_thw" in params:
        return True
    if "pixel_values" in params or "images" in params:
        return False
    return False


def _extract_features_qwen_vl(
    model: torch.nn.Module,
    processor: object,
    images: Sequence[Image.Image],
    device: str,
    batch_size: int,
    pooling: str,
) -> torch.Tensor:
    model.eval()
    features: List[torch.Tensor] = []
    use_amp = device == "cuda"
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    logged = False
    block_capture: List[torch.Tensor] = []
    block_handle = None
    block_label = ""
    block_info = _resolve_qwen_vision_blocks(model)
    if block_info is not None and pooling != "attention_pooling":
        blocks, blocks_path = block_info
        if not blocks:
            raise ValueError(f"Qwen VL vision blocks not found at {blocks_path}.")
        block_index = len(blocks) - 1
        block_label = f"{blocks_path}[{block_index}]"

        def _capture_block_output(_module: torch.nn.Module, _inputs: Tuple[object, ...], output: object) -> None:
            tensor = _coerce_block_output(output)
            if tensor is not None:
                block_capture.append(tensor)

        block_handle = blocks[block_index].register_forward_hook(_capture_block_output)
    with torch.no_grad(), autocast_ctx:
        try:
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                inputs = _prepare_qwen_vl_inputs(processor, batch)
                inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
                call_kwargs = _filter_kwargs_for_callable(model.forward, inputs)
                if block_capture:
                    block_capture.clear()
                outputs = model(**call_kwargs)
                if pooling == "attention_pooling":
                    pooled, source = _select_pooled_from_outputs(outputs)
                    batch_features = pooled
                    if not logged:
                        LOGGER.info("Feature extraction: qwen_vl attention_pooling via model.forward -> %s.", source)
                        logged = True
                else:
                    if block_label:
                        if not block_capture:
                            raise ValueError(f"Qwen VL did not capture block output from {block_label}.")
                        sequence = block_capture[-1]
                        if sequence.dim() == 2:
                            grid_thw = None
                            if isinstance(inputs, dict):
                                grid_thw = inputs.get("image_grid_thw")
                                if grid_thw is None:
                                    grid_thw = inputs.get("video_grid_thw")
                            batch_features = _mean_pool_flat_sequence(
                                sequence,
                                grid_thw if torch.is_tensor(grid_thw) else None,
                                label=f"qwen_vl {block_label}",
                            )
                        elif sequence.dim() == 3:
                            batch_features = _pool_from_sequence(sequence, pooling=pooling, label="qwen_vl")
                        else:
                            raise ValueError(
                                f"Qwen VL {block_label} output must be 2D or 3D for pooling, got {sequence.dim()}D."
                            )
                        if not logged:
                            LOGGER.info(
                                "Feature extraction: qwen_vl %s via %s output.",
                                pooling,
                                block_label,
                            )
                            logged = True
                    else:
                        sequence, source = _select_vision_sequence_from_outputs(outputs)
                        batch_features = _pool_from_sequence(sequence, pooling=pooling, label="qwen_vl")
                        if not logged:
                            LOGGER.info("Feature extraction: qwen_vl %s via model.forward -> %s.", pooling, source)
                            logged = True
                features.append(batch_features.float().detach().cpu())
        finally:
            if block_handle is not None:
                block_handle.remove()
    return torch.cat(features, dim=0)


def _prepare_qwen_vl_inputs(
    processor: object,
    images: Sequence[Image.Image],
    require_input_ids: bool = False,
) -> Dict[str, object]:
    image_token = getattr(processor, "image_token", None)
    if not image_token:
        tokenizer = getattr(processor, "tokenizer", None)
        image_token = getattr(tokenizer, "image_token", None) if tokenizer is not None else None
    if not image_token:
        token_id = getattr(processor, "image_token_id", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if token_id is None and tokenizer is not None:
            token_id = getattr(tokenizer, "image_token_id", None)
        if token_id is not None and tokenizer is not None:
            try:
                image_token = tokenizer.decode([token_id])
            except Exception:
                image_token = None
    if not image_token:
        image_token = "<image>"

    placeholders = [image_token] * len(images)
    empty_texts = [""] * len(images)
    candidates = []
    if not require_input_ids:
        candidates.append({"images": images, "return_tensors": "pt"})
    candidates.extend(
        [
            {"text": placeholders, "images": images, "return_tensors": "pt"},
            {"text": empty_texts, "images": images, "return_tensors": "pt"},
            {"prompt": placeholders, "images": images, "return_tensors": "pt"},
        ]
    )
    for kwargs in candidates:
        try:
            return processor(**kwargs)
        except (TypeError, ValueError):
            continue
    raise ValueError("Failed to prepare inputs for Qwen VL processor.")


def _select_eos_embeddings(
    sequence: torch.Tensor,
    input_ids: torch.Tensor | None,
    eos_token_id: int | None,
) -> torch.Tensor:
    if sequence.dim() != 3:
        raise ValueError(f"Expected sequence tensor with 3 dims for EOS pooling, got {sequence.dim()}.")
    batch_size, seq_len, _ = sequence.shape
    if (
        input_ids is None
        or eos_token_id is None
        or input_ids.shape[0] != batch_size
        or input_ids.shape[1] != seq_len
    ):
        return sequence[:, -1, :]
    indices: List[int] = []
    for row in range(batch_size):
        positions = torch.where(input_ids[row] == eos_token_id)[0]
        indices.append(int(positions[-1].item()) if positions.numel() else seq_len - 1)
    index_tensor = torch.tensor(indices, device=sequence.device)
    batch_indices = torch.arange(batch_size, device=sequence.device)
    return sequence[batch_indices, index_tensor]


def _extract_features_qwen_vl_eos(
    model: torch.nn.Module,
    processor: object,
    images: Sequence[Image.Image],
    device: str,
    batch_size: int,
    pooling: str,
) -> torch.Tensor:
    if pooling != "attention_pooling":
        raise ValueError("qwen_vl_embed supports only attention_pooling (EOS token).")
    model.eval()
    features: List[torch.Tensor] = []
    use_amp = device == "cuda"
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0] if eos_token_id else None
    logged = False
    with torch.no_grad(), autocast_ctx:
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = _prepare_qwen_vl_inputs(processor, batch)
            if "input_ids" not in inputs:
                inputs = _prepare_qwen_vl_inputs(processor, batch, require_input_ids=True)
            inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            call_kwargs = _filter_kwargs_for_callable(model.forward, inputs)
            outputs = model(**call_kwargs)
            sequence, source = _select_sequence_from_outputs(outputs)
            eos_embeddings = _select_eos_embeddings(
                sequence,
                inputs.get("input_ids"),
                eos_token_id,
            )
            if not logged:
                LOGGER.info(
                    "Feature extraction: qwen_vl_embed attention_pooling via model.forward -> %s -> EOS token.",
                    source,
                )
                logged = True
            features.append(eos_embeddings.float().detach().cpu())
    return torch.cat(features, dim=0)


def _poolings_for_spec(spec: ModelSpec) -> Tuple[str, ...]:
    if spec.backend == "perception":
        return ("attention_pooling", "cls", "mean_patch")
    if spec.backend == "qwen_vl":
        return ("mean_patch",)
    if spec.backend == "qwen_vl_embed":
        return ("attention_pooling",)
    return POOLING_CHOICES


def _select_pooled_from_outputs(outputs: object) -> Tuple[torch.Tensor, str]:
    if isinstance(outputs, torch.Tensor):
        if outputs.dim() == 2:
            return outputs, "outputs (tensor)"
        raise ValueError("Outputs is a tensor but not 2D; pooled output required.")
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor) and first.dim() == 2:
            return first, "outputs[0]"
    for attr in ("pooler_output", "pooled_output", "image_embeds", "image_features"):
        value = getattr(outputs, attr, None)
        if isinstance(value, torch.Tensor):
            return value, f"outputs.{attr}"
    vision_out = getattr(outputs, "vision_model_output", None)
    if vision_out is not None:
        for attr in ("pooler_output", "pooled_output"):
            value = getattr(vision_out, attr, None)
            if isinstance(value, torch.Tensor):
                return value, f"outputs.vision_model_output.{attr}"
        if isinstance(vision_out, (tuple, list)) and vision_out:
            first = vision_out[0]
            if isinstance(first, torch.Tensor) and first.dim() == 2:
                return first, "outputs.vision_model_output[0]"
    raise ValueError("No pooled output found for attention_pooling.")


def _select_vision_sequence_from_outputs(outputs: object) -> Tuple[torch.Tensor, str]:
    vision_out = getattr(outputs, "vision_model_output", None)
    if vision_out is not None:
        if hasattr(vision_out, "last_hidden_state") and vision_out.last_hidden_state is not None:
            return vision_out.last_hidden_state, "outputs.vision_model_output.last_hidden_state"
        if isinstance(vision_out, (tuple, list)) and vision_out:
            first = vision_out[0]
            if isinstance(first, torch.Tensor) and first.dim() == 3:
                return first, "outputs.vision_model_output[0]"
    for attr in ("vision_hidden_states", "image_hidden_states"):
        value = getattr(outputs, attr, None)
        if isinstance(value, torch.Tensor) and value.dim() == 3:
            return value, f"outputs.{attr}"
    return _select_sequence_from_outputs(outputs)


def _select_sequence_from_outputs(outputs: object) -> Tuple[torch.Tensor, str]:
    if isinstance(outputs, torch.Tensor):
        if outputs.dim() == 3:
            return outputs, "outputs (tensor)"
        raise ValueError("Outputs is a tensor but not 3D; sequence output required.")
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state, "outputs.last_hidden_state"
    if hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
        return outputs.encoder_last_hidden_state, "outputs.encoder_last_hidden_state"
    vision_out = getattr(outputs, "vision_model_output", None)
    if vision_out is not None:
        if hasattr(vision_out, "last_hidden_state") and vision_out.last_hidden_state is not None:
            return vision_out.last_hidden_state, "outputs.vision_model_output.last_hidden_state"
        if isinstance(vision_out, (tuple, list)) and vision_out:
            first = vision_out[0]
            if isinstance(first, torch.Tensor) and first.dim() == 3:
                return first, "outputs.vision_model_output[0]"
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor) and first.dim() == 3:
            return first, "outputs[0]"
    raise ValueError("No sequence output found for cls/mean_patch.")


def _pool_from_sequence(sequence: torch.Tensor, pooling: str, label: str) -> torch.Tensor:
    if sequence.dim() != 3:
        raise ValueError(f"{label}: pooling='{pooling}' expects a 3D sequence tensor, got {sequence.dim()}D.")
    if pooling == "cls":
        return sequence[:, 0]
    if pooling == "mean_patch":
        if sequence.shape[1] <= 1:
            raise ValueError(f"{label}: mean_patch requires patch tokens but sequence length is {sequence.shape[1]}.")
        return sequence[:, 1:].mean(dim=1)
    raise ValueError(f"{label}: unknown pooling option '{pooling}'.")


def _call_forward_features(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor | None:
    if not hasattr(model, "forward_features"):
        return None
    try:
        sig = inspect.signature(model.forward_features)
    except (TypeError, ValueError):
        return model.forward_features(inputs)
    kwargs: Dict[str, object] = {}
    if "strip_cls_token" in sig.parameters:
        kwargs["strip_cls_token"] = False
    return model.forward_features(inputs, **kwargs)


def _normalize_features(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=-1)


def _cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    return float(torch.dot(vec_a, vec_b).item())


def _l2_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    return float(torch.norm(vec_a - vec_b, p=2).item())


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


def _toy_center_vector_raw(run_features: Sequence[RunFeatures]) -> torch.Tensor:
    toy_vectors = []
    for run in run_features:
        toy_vectors.append(run.features_raw["toy_dom"])
        toy_vectors.append(run.features_raw["toy_rare"])
    if not toy_vectors:
        raise ValueError("Toy vectors are empty; cannot compute toy center.")
    stacked = torch.stack(toy_vectors, dim=0)
    return stacked.mean(dim=0)


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


def _real_center_vector_raw(run_features: Sequence[RunFeatures]) -> torch.Tensor:
    real_vectors = []
    for run in run_features:
        real_vectors.append(run.features_raw["real_dom"])
        real_vectors.append(run.features_raw["real_rare"])
    if not real_vectors:
        raise ValueError("Real vectors are empty; cannot compute real center.")
    stacked = torch.stack(real_vectors, dim=0)
    return stacked.mean(dim=0)


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
        normalized = entry.lower().replace("_", "-")
        if normalized in MODEL_ALIASES:
            entry = MODEL_ALIASES[normalized]
        if entry in MODEL_SPECS:
            resolved.append(MODEL_SPECS[entry])
            continue
        lowered = entry.lower()
        if lowered.startswith("pe:") or lowered.startswith("perception:"):
            model_id = entry.split(":", 1)[1].strip()
            if not model_id:
                raise ValueError("Perception model must be formatted as 'pe:PE-Core-L14-336'.")
            key = f"pe-{model_id}".lower().replace("/", "_").replace(" ", "_")
            resolved.append(ModelSpec(key=key, model_id=model_id, backend="perception"))
            continue
        if lowered.startswith("facebook/pe-"):
            model_id = entry.split("/", 1)[1].strip()
            key = f"pe-{model_id}".lower().replace("/", "_").replace(" ", "_")
            resolved.append(ModelSpec(key=key, model_id=model_id, backend="perception"))
            continue
        for prefix, backend, key_prefix in (
            ("qwen:", "qwen_vl", "qwen-vl"),
            ("qwen-vl:", "qwen_vl", "qwen-vl"),
            ("qwen3:", "qwen_vl", "qwen-vl"),
            ("qwen3-vl:", "qwen_vl", "qwen-vl"),
            ("qwen-embed:", "qwen_vl_embed", "qwen-vl-embed"),
            ("qwen_embed:", "qwen_vl_embed", "qwen-vl-embed"),
            ("qwen3-embed:", "qwen_vl_embed", "qwen-vl-embed"),
            ("qwen3-vl-embed:", "qwen_vl_embed", "qwen-vl-embed"),
        ):
            if lowered.startswith(prefix):
                model_id = entry.split(":", 1)[1].strip()
                if not model_id:
                    raise ValueError(f"{prefix[:-1]} model must be formatted as '{prefix}<model_id>'.")
                slug = model_id.lower().replace("/", "_").replace(" ", "_")
                key = f"{key_prefix}-{slug}"
                resolved.append(ModelSpec(key=key, model_id=model_id, backend=backend))
                break
        else:
            matches = [spec for spec in MODEL_SPECS.values() if spec.model_id == entry]
            if matches:
                resolved.append(matches[0])
                continue
            valid = ", ".join(sorted(MODEL_SPECS.keys()))
            raise ValueError(
                f"Unknown model '{entry}'. Use one of: {valid}, aliases: pe, qwen, qwen-embed, or the full model id."
            )
        continue
    return resolved


def _load_model(
    spec: ModelSpec,
    device: str,
    hf_token: str | None,
) -> Tuple[
    torch.nn.Module,
    object,
    Callable[[torch.nn.Module, object, Sequence[Image.Image], str, int, str], torch.Tensor],
]:
    if spec.backend in ("qwen_vl", "qwen_vl_embed"):
        try:
            from transformers import AutoModel, AutoProcessor
        except Exception as exc:
            raise SystemExit(
                "transformers is required for Qwen models. Install with the analysis-qwen extra. "
                f"(import error: {exc!r})"
            ) from exc
        try:
            from transformers import AutoModelForVision2Seq
        except Exception:
            AutoModelForVision2Seq = None

        try:
            processor = AutoProcessor.from_pretrained(
                spec.model_id,
                trust_remote_code=True,
                **_token_kwargs(AutoProcessor.from_pretrained, hf_token),
            )
            if AutoModelForVision2Seq is not None:
                try:
                    full_model = AutoModelForVision2Seq.from_pretrained(
                        spec.model_id,
                        trust_remote_code=True,
                        **_token_kwargs(AutoModelForVision2Seq.from_pretrained, hf_token),
                    )
                except Exception:
                    full_model = AutoModel.from_pretrained(
                        spec.model_id,
                        trust_remote_code=True,
                        **_token_kwargs(AutoModel.from_pretrained, hf_token),
                    )
            else:
                full_model = AutoModel.from_pretrained(
                    spec.model_id,
                    trust_remote_code=True,
                    **_token_kwargs(AutoModel.from_pretrained, hf_token),
                )
        except ValueError as exc:
            if _is_unknown_architecture_error(exc):
                _raise_qwen_architecture_error(spec, exc)
            raise

        if spec.backend == "qwen_vl":
            vision_encoder = _unwrap_qwen_vision_encoder(full_model).to(device)
            if _vision_requires_hidden_states(vision_encoder):
                full_model = full_model.to(device)
                return full_model, processor, _extract_features_qwen_vl
            return vision_encoder, processor, _extract_features_qwen_vl
        full_model = full_model.to(device)
        return full_model, processor, _extract_features_qwen_vl_eos

    if spec.backend == "perception":
        try:
            import core.vision_encoder.pe as pe  # type: ignore
            import core.vision_encoder.transforms as pe_transforms  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "perception_models is required for PE models. "
                "Install with: pip install git+https://github.com/facebookresearch/perception_models.git"
            ) from exc

        if spec.model_id.startswith("PE-Core"):
            model = pe.CLIP.from_config(spec.model_id, pretrained=True)
        else:
            model = pe.VisionTransformer.from_config(spec.model_id, pretrained=True)
        model = model.to(device)
        processor = pe_transforms.get_image_transform(model.image_size)
        return model, processor, _extract_features_perception

    try:
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor
    except Exception as exc:
        raise SystemExit(
            "transformers is required for HF models. Install with the analysis-qwen extra. "
            f"(import error: {exc!r})"
        ) from exc

    try:
        processor = AutoProcessor.from_pretrained(
            spec.model_id,
            **_token_kwargs(AutoProcessor.from_pretrained, hf_token),
        )
    except Exception:
        processor = AutoImageProcessor.from_pretrained(
            spec.model_id,
            **_token_kwargs(AutoImageProcessor.from_pretrained, hf_token),
        )
    model = AutoModel.from_pretrained(
        spec.model_id,
        **_token_kwargs(AutoModel.from_pretrained, hf_token),
    ).to(device)
    return model, processor, _extract_features_hf


def _format_pair_key(left: str, right: str) -> str:
    return f"{left}__{right}"


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: List[float], mean_value: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return float(variance**0.5)


def _summary_stats(values: List[float]) -> Dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean_value = _mean(values)
    std_value = _std(values, mean_value)
    min_value = float(min(values))
    max_value = float(max(values))
    return {
        "count": len(values),
        "mean": mean_value,
        "std": std_value,
        "min": min_value,
        "max": max_value,
    }


def _summary_map(values_map: Dict[str, List[float]]) -> Dict[str, Dict[str, float | int]]:
    return {key: _summary_stats(values) for key, values in values_map.items()}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse composite image embeddings for cosine similarity.",
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
        "--hf-token",
        default=None,
        help="Hugging Face token (optional; also read from HF_TOKEN/HUGGINGFACE_HUB_TOKEN).",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=[f"{left}:{right}" for left, right in DEFAULT_PAIRS],
        help="Pairs to compare, formatted as key_a:key_b.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SPECS.keys()),
        help=(
            "Models to run (aliases: pe, qwen, qwen-embed). You may also pass custom model ids like "
            "'pe:PE-Core-L14-336' or 'qwen:Qwen/Qwen3-VL-8B-Instruct'."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Single model to run (repeatable). Overrides --models when provided.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Path to write per-run JSONL results (default: <output-dir>/feature_similarity.jsonl).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to write JSON analysis summary (default: <output-dir>/feature_analysis.json).",
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

    hf_token = _get_hf_token(args.hf_token)
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    pairs = _parse_pairs(args.pairs)
    model_args = args.model if args.model else args.models
    models = _resolve_models(model_args)
    if not models:
        raise SystemExit("No models selected. Please provide at least one model via --models.")

    output_handle = None
    output_json_path = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else (output_dir / "feature_analysis.json")
    )
    output_jsonl_path = (
        Path(args.output_jsonl).expanduser().resolve()
        if args.output_jsonl
        else (output_dir / "feature_similarity.jsonl")
    )
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    output_handle = output_jsonl_path.open("w", encoding="utf-8")
    LOGGER.info("Writing per-run results to %s", output_jsonl_path)
    LOGGER.info("Writing analysis summary to %s", output_json_path)

    try:
        analysis_payloads: List[Dict[str, object]] = []
        for spec in models:
            LOGGER.info("Loading model: %s", spec.model_id)
            model, processor, extractor = _load_model(spec, device, hf_token)
            if spec.backend == "qwen_vl":
                LOGGER.info("qwen_vl does not expose pooled outputs; using mean_patch only.")

            for pooling in _poolings_for_spec(spec):
                LOGGER.info("Pooling: %s", pooling)
                metrics_pairs_cos: Dict[str, List[float]] = {_format_pair_key(a, b): [] for a, b in pairs}
                metrics_pairs_l2: Dict[str, List[float]] = {_format_pair_key(a, b): [] for a, b in pairs}
                metrics_special_cos: Dict[str, List[float]] = {label: [] for label in SPECIAL_PAIRS}
                metrics_special_l2: Dict[str, List[float]] = {label: [] for label in SPECIAL_PAIRS}
                metrics_toy_center_cos: Dict[str, List[float]] = {"real_dom": [], "real_rare": []}
                metrics_real_center_cos: Dict[str, List[float]] = {"toy_dom": [], "toy_rare": []}
                metrics_toy_center_l2: Dict[str, List[float]] = {"real_dom": [], "real_rare": []}
                metrics_real_center_l2: Dict[str, List[float]] = {"toy_dom": [], "toy_rare": []}
                run_feature_list: List[RunFeatures] = []
                run_payloads: Dict[str, Dict[str, object]] = {}

                for run_dir in run_dirs:
                    images_by_key = _load_images_for_run(run_dir, IMAGE_KEYS)
                    ordered_images = [images_by_key[key] for key in IMAGE_KEYS]
                    features_raw = extractor(model, processor, ordered_images, device, args.batch_size, pooling)
                    features_norm = _normalize_features(features_raw)

                    raw_by_key = {key: features_raw[idx] for idx, key in enumerate(IMAGE_KEYS)}
                    norm_by_key = {key: features_norm[idx] for idx, key in enumerate(IMAGE_KEYS)}
                    run_feature_list.append(
                        RunFeatures(run=run_dir.name, features_raw=raw_by_key, features_norm=norm_by_key)
                    )

                    run_result = {"cosine": {}, "l2": {}}
                    run_special = {"cosine": {}, "l2": {}}
                    for left, right in pairs:
                        if left not in norm_by_key or right not in norm_by_key:
                            raise KeyError(f"Missing feature for pair {left}:{right} in {run_dir}")
                        pair_key = _format_pair_key(left, right)
                        cos_value = _cosine_similarity(norm_by_key[left], norm_by_key[right])
                        l2_value = _l2_distance(raw_by_key[left], raw_by_key[right])
                        run_result["cosine"][pair_key] = cos_value
                        run_result["l2"][pair_key] = l2_value
                        metrics_pairs_cos[pair_key].append(cos_value)
                        metrics_pairs_l2[pair_key].append(l2_value)

                    for label, (left, right) in SPECIAL_PAIRS.items():
                        cos_value = _cosine_similarity(norm_by_key[left], norm_by_key[right])
                        l2_value = _l2_distance(raw_by_key[left], raw_by_key[right])
                        run_special["cosine"][label] = cos_value
                        run_special["l2"][label] = l2_value
                        metrics_special_cos[label].append(cos_value)
                        metrics_special_l2[label].append(l2_value)

                    payload = {
                        "run": run_dir.name,
                        "model": spec.model_id,
                        "model_key": spec.key,
                        "pooling": pooling,
                        "pairs": run_result,
                        "special": run_special,
                    }
                    run_payloads[run_dir.name] = payload

                toy_center_cos = _toy_center_vector(run_feature_list)
                real_center_cos = _real_center_vector(run_feature_list)
                toy_center_l2 = _toy_center_vector_raw(run_feature_list)
                real_center_l2 = _real_center_vector_raw(run_feature_list)

                raw_norms: List[float] = []
                for run in run_feature_list:
                    for key in IMAGE_KEYS:
                        raw_norms.append(float(torch.norm(run.features_raw[key], p=2).item()))
                raw_norm_stats = _summary_stats(raw_norms)
                max_abs_dev = max((abs(value - 1.0) for value in raw_norms), default=0.0)
                unit_norm = bool(raw_norms) and max_abs_dev <= 1e-3
                raw_norm_stats["max_abs_dev_from_1"] = max_abs_dev

                for run in run_feature_list:
                    cos_features = run.features_norm
                    raw_features = run.features_raw
                    real_dom_cos = _cosine_similarity(cos_features["real_dom"], toy_center_cos)
                    real_rare_cos = _cosine_similarity(cos_features["real_rare"], toy_center_cos)
                    metrics_toy_center_cos["real_dom"].append(real_dom_cos)
                    metrics_toy_center_cos["real_rare"].append(real_rare_cos)
                    toy_dom_cos = _cosine_similarity(cos_features["toy_dom"], real_center_cos)
                    toy_rare_cos = _cosine_similarity(cos_features["toy_rare"], real_center_cos)
                    metrics_real_center_cos["toy_dom"].append(toy_dom_cos)
                    metrics_real_center_cos["toy_rare"].append(toy_rare_cos)

                    real_dom_l2 = _l2_distance(raw_features["real_dom"], toy_center_l2)
                    real_rare_l2 = _l2_distance(raw_features["real_rare"], toy_center_l2)
                    metrics_toy_center_l2["real_dom"].append(real_dom_l2)
                    metrics_toy_center_l2["real_rare"].append(real_rare_l2)
                    toy_dom_l2 = _l2_distance(raw_features["toy_dom"], real_center_l2)
                    toy_rare_l2 = _l2_distance(raw_features["toy_rare"], real_center_l2)
                    metrics_real_center_l2["toy_dom"].append(toy_dom_l2)
                    metrics_real_center_l2["toy_rare"].append(toy_rare_l2)

                    payload = run_payloads.get(run.run)
                    if payload is not None:
                        payload["toy_center_similarity"] = {
                            "real_dom": real_dom_cos,
                            "real_rare": real_rare_cos,
                        }
                        payload["real_center_similarity"] = {
                            "toy_dom": toy_dom_cos,
                            "toy_rare": toy_rare_cos,
                        }
                        payload["toy_center_distance"] = {
                            "real_dom": real_dom_l2,
                            "real_rare": real_rare_l2,
                        }
                        payload["real_center_distance"] = {
                            "toy_dom": toy_dom_l2,
                            "toy_rare": toy_rare_l2,
                        }

                if output_handle:
                    for run in run_feature_list:
                        payload = run_payloads.get(run.run)
                        if payload is not None:
                            output_handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

                diff_cos = [
                    r - d for d, r in zip(metrics_toy_center_cos["real_dom"], metrics_toy_center_cos["real_rare"])
                ]
                diff_l2 = [
                    r - d for d, r in zip(metrics_toy_center_l2["real_dom"], metrics_toy_center_l2["real_rare"])
                ]
                model_summary = {
                    "metrics": ["cosine", "l2"],
                    "pairs": {"cosine": _summary_map(metrics_pairs_cos), "l2": _summary_map(metrics_pairs_l2)},
                    "special": {"cosine": _summary_map(metrics_special_cos), "l2": _summary_map(metrics_special_l2)},
                    "toy_center_similarity": _summary_map(metrics_toy_center_cos),
                    "real_center_similarity": _summary_map(metrics_real_center_cos),
                    "toy_center_distance": _summary_map(metrics_toy_center_l2),
                    "real_center_distance": _summary_map(metrics_real_center_l2),
                    "toy_center_cosine_diff_rare_minus_dom": _summary_stats(diff_cos),
                    "toy_center_l2_diff_rare_minus_dom": _summary_stats(diff_l2),
                    "raw_feature_norms": raw_norm_stats,
                    "raw_features_unit_norm": unit_norm,
                }
                model_payload = {
                    "model": spec.model_id,
                    "model_key": spec.key,
                    "pooling": pooling,
                    "results_date": args.results_date,
                    "results_dir": str(results_dir),
                    "run_count": len(run_feature_list),
                    "pairs": [f"{left}:{right}" for left, right in pairs],
                    "special_pairs": SPECIAL_PAIRS,
                    "runs": [run_payloads[run.run] for run in run_feature_list if run.run in run_payloads],
                    "summary": model_summary,
                }
                analysis_payloads.append(model_payload)

                LOGGER.info("Summary (mean ± std)")
                LOGGER.info("Model: %s", spec.model_id)
                LOGGER.info("Pooling: %s", pooling)
                mean_dom = _mean(metrics_toy_center_cos["real_dom"])
                std_dom = _std(metrics_toy_center_cos["real_dom"], mean_dom)
                mean_rare = _mean(metrics_toy_center_cos["real_rare"])
                std_rare = _std(metrics_toy_center_cos["real_rare"], mean_rare)
                mean_diff = _mean(diff_cos)
                std_diff = _std(diff_cos, mean_diff)
                LOGGER.info("  toy_center_cosine real_dom: %.6f ± %.6f", mean_dom, std_dom)
                LOGGER.info("  toy_center_cosine real_rare: %.6f ± %.6f", mean_rare, std_rare)
                LOGGER.info("  toy_center_cosine (rare - dom): %.6f ± %.6f", mean_diff, std_diff)

                mean_dom_l2 = _mean(metrics_toy_center_l2["real_dom"])
                std_dom_l2 = _std(metrics_toy_center_l2["real_dom"], mean_dom_l2)
                mean_rare_l2 = _mean(metrics_toy_center_l2["real_rare"])
                std_rare_l2 = _std(metrics_toy_center_l2["real_rare"], mean_rare_l2)
                mean_diff_l2 = _mean(diff_l2)
                std_diff_l2 = _std(diff_l2, mean_diff_l2)
                l2_note = " (raw features; unit-norm)" if unit_norm else " (raw features)"
                LOGGER.info("  toy_center_l2%s real_dom: %.6f ± %.6f", l2_note, mean_dom_l2, std_dom_l2)
                LOGGER.info("  toy_center_l2%s real_rare: %.6f ± %.6f", l2_note, mean_rare_l2, std_rare_l2)
                LOGGER.info("  toy_center_l2%s (rare - dom): %.6f ± %.6f", l2_note, mean_diff_l2, std_diff_l2)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if analysis_payloads:
            analysis_payload = {
                "results_date": args.results_date,
                "results_dir": str(results_dir),
                "output_dir": str(output_dir),
                "device": device,
                "metrics": ["cosine", "l2"],
                "models": analysis_payloads,
            }
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            output_json_path.write_text(
                json.dumps(analysis_payload, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        if output_handle:
            output_handle.close()


if __name__ == "__main__":
    main()
