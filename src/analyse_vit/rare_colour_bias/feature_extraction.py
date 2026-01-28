from __future__ import annotations

import contextlib
import inspect
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple, TYPE_CHECKING

import torch
from PIL import Image
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

if TYPE_CHECKING:
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

LOGGER = logging.getLogger("analyse_vit.feature_extraction")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    backend: str = "hf"


@dataclass(frozen=True)
class VllmEmbedder:
    llm: object
    tokenizer: object


MODEL_SPECS: Dict[str, ModelSpec] = {
    "pe-core-l14-336": ModelSpec(key="pe-core-l14-336", model_id="PE-Core-L14-336", backend="perception"),
    "qwen3-vl-8b": ModelSpec(key="qwen3-vl-8b", model_id="Qwen/Qwen3-VL-8B-Instruct", backend="qwen_vl"),
    "qwen3-vl-8b-embed": ModelSpec(
        key="qwen3-vl-8b-embed",
        model_id="Qwen/Qwen3-VL-Embedding-8B",
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
EMBED_INSTRUCTION = "Represent the user's input."


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


def _extract_features_qwen_vl(
    model: torch.nn.Module,
    processor: object,
    images: Sequence[Image.Image],
    device: str,
    batch_size: int,
    pooling: str,
) -> torch.Tensor:
    if pooling != "attention_pooling":
        raise ValueError("qwen_vl supports only attention_pooling.")
    if not hasattr(model, "get_image_features"):
        raise ValueError("qwen_vl requires model.get_image_features.")
    model.eval()
    features: List[torch.Tensor] = []
    logged = False
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = _prepare_qwen_vl_inputs(processor, batch)
            inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            call_kwargs = _filter_kwargs_for_callable(model.get_image_features, inputs)
            outputs = model.get_image_features(**call_kwargs)
            pooled, source = _select_pooled_from_outputs(outputs)
            if pooled.dim() != 2:
                raise ValueError(
                    f"qwen_vl pooler_output must be 2D [B,D], got {pooled.dim()}D."
                )
            if not logged:
                LOGGER.info(
                    "Feature extraction: qwen_vl attention_pooling via model.get_image_features (%s).",
                    source,
                )
                logged = True
            features.append(pooled.float().detach().cpu())
    return torch.cat(features, dim=0)


def _prepare_qwen_vl_inputs(
    processor: object,
    images: Sequence[Image.Image],
    require_input_ids: bool = False,
) -> Dict[str, object]:
    if require_input_ids:
        raise ValueError("qwen_vl does not support require_input_ids input preparation.")
    image_list = list(images)
    # Qwen3VLProcessor expects text entries; provide empty strings to avoid None handling errors.
    try:
        inputs = processor(text=[""] * len(image_list), images=image_list, return_tensors="pt")
    except TypeError:
        inputs = processor(images=image_list, return_tensors="pt")
    if not isinstance(inputs, Mapping):
        raise ValueError("qwen_vl processor must return a dict of tensors.")
    inputs = dict(inputs)
    # Keep only vision-relevant keys to avoid text attention_mask collisions inside Qwen3-VL.
    inputs = {
        key: value
        for key, value in inputs.items()
        if key in {"pixel_values", "image_grid_thw"} or key.startswith("pixel_values_")
    }
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise ValueError("qwen_vl processor must return pixel_values and image_grid_thw.")
    return inputs


def _format_input_to_conversation(
    image: Image.Image,
    text: str | None = None,
    instruction: str = EMBED_INSTRUCTION,
) -> List[Dict[str, object]]:
    content: List[Dict[str, object]] = [{"type": "image", "image": image}]
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": ""})
    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]


def _prepare_vllm_inputs_for_images(
    images: Sequence[Image.Image],
    tokenizer: object,
    instruction: str = EMBED_INSTRUCTION,
) -> List[Dict[str, object]]:
    conversations = [_format_input_to_conversation(image, instruction=instruction) for image in images]
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("vllm tokenizer must support apply_chat_template for qwen_vl_embed.")
    prompts = [
        tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        for conversation in conversations
    ]
    return [
        {"prompt": prompt, "multi_modal_data": {"image": image}}
        for prompt, image in zip(prompts, images, strict=True)
    ]


def _extract_features_qwen_vl_embed(
    model: object,
    processor: object,
    images: Sequence[Image.Image],
    device: str,
    batch_size: int,
    pooling: str,
) -> torch.Tensor:
    if pooling != "attention_pooling":
        raise ValueError("qwen_vl_embed supports only attention_pooling.")
    if not isinstance(model, VllmEmbedder):
        raise ValueError("qwen_vl_embed expects a VLLM embedder. Please use the vLLM-based loader.")
    llm = model.llm
    tokenizer = model.tokenizer
    vllm_inputs = _prepare_vllm_inputs_for_images(images, tokenizer)
    outputs = llm.embed(vllm_inputs)
    if len(outputs) != len(images):
        raise ValueError(
            f"qwen_vl_embed vLLM returned {len(outputs)} embeddings for {len(images)} inputs."
        )
    embeddings: List[torch.Tensor] = []
    for output in outputs:
        embedding = getattr(output.outputs, "embedding", None)
        if embedding is None:
            raise ValueError("qwen_vl_embed vLLM outputs missing embedding.")
        embeddings.append(torch.as_tensor(embedding, dtype=torch.float32))
    if not embeddings:
        raise ValueError("qwen_vl_embed produced no embeddings.")
    stacked = torch.stack(embeddings, dim=0)
    LOGGER.info("Feature extraction: qwen_vl_embed attention_pooling via vLLM embeddings.")
    return stacked


def _poolings_for_spec(spec: ModelSpec) -> Tuple[str, ...]:
    if spec.backend == "perception":
        return ("attention_pooling", "cls", "mean_patch")
    if spec.backend == "qwen_vl":
        return ("attention_pooling",)
    if spec.backend == "qwen_vl_embed":
        return ("attention_pooling",)
    return POOLING_CHOICES


def _select_pooled_from_outputs(outputs: object) -> Tuple[torch.Tensor, str]:
    if isinstance(outputs, torch.Tensor):
        if outputs.dim() == 2:
            return outputs, "outputs (tensor)"
        raise ValueError("Outputs is a tensor but not 2D; pooled output required.")
    if isinstance(outputs, Mapping):
        for key in ("pooler_output", "pooled_output", "image_embeds", "image_features"):
            value = outputs.get(key)
            if isinstance(value, torch.Tensor):
                return value, f"outputs['{key}']"
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


def _resolve_models(model_args: Sequence[str]) -> List[ModelSpec]:
    resolved: List[ModelSpec] = []
    embed_model_id = MODEL_SPECS["qwen3-vl-8b-embed"].model_id
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
                if backend == "qwen_vl_embed" and model_id != embed_model_id:
                    raise ValueError(
                        f"qwen_vl_embed only supports model_id='{embed_model_id}', got '{model_id}'."
                    )
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
    object,
    object,
    Callable[[object, object, Sequence[Image.Image], str, int, str], torch.Tensor],
]:
    if spec.backend in ("qwen_vl", "qwen_vl_embed"):
        if spec.backend == "qwen_vl_embed":
            embed_model_id = MODEL_SPECS["qwen3-vl-8b-embed"].model_id
            if spec.model_id != embed_model_id:
                raise ValueError(
                    f"qwen_vl_embed only supports model_id='{embed_model_id}', got '{spec.model_id}'."
                )
            try:
                from vllm import LLM, EngineArgs
            except Exception as exc:
                raise ImportError(
                    "qwen_vl_embed requires vLLM. Please install vllm in the analysis-qwen extra."
                ) from exc
            engine_args = EngineArgs(
                model=spec.model_id,
                runner="pooling",
                dtype="bfloat16",
                trust_remote_code=True,
            )
            llm = LLM(**vars(engine_args))
            tokenizer = llm.llm_engine.tokenizer
            return VllmEmbedder(llm=llm, tokenizer=tokenizer), None, _extract_features_qwen_vl_embed

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
            raise SystemExit(
                "qwen_vl requires AutoModelForVision2Seq. Please install a compatible transformers version."
            )

        try:
            processor = AutoProcessor.from_pretrained(
                spec.model_id,
                trust_remote_code=True,
                **_token_kwargs(AutoProcessor.from_pretrained, hf_token),
            )
            full_model = AutoModelForVision2Seq.from_pretrained(
                spec.model_id,
                trust_remote_code=True,
                **_token_kwargs(AutoModelForVision2Seq.from_pretrained, hf_token),
            )
        except ValueError as exc:
            if _is_unknown_architecture_error(exc):
                _raise_qwen_architecture_error(spec, exc)
            raise

        full_model = full_model.to(device)
        return full_model, processor, _extract_features_qwen_vl

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
        from transformers import AutoModel, AutoProcessor
    except Exception as exc:
        raise SystemExit(
            "transformers is required for HF models. Install with the analysis-qwen extra. "
            f"(import error: {exc!r})"
        ) from exc

    processor = AutoProcessor.from_pretrained(
        spec.model_id,
        **_token_kwargs(AutoProcessor.from_pretrained, hf_token),
    )
    model = AutoModel.from_pretrained(
        spec.model_id,
        **_token_kwargs(AutoModel.from_pretrained, hf_token),
    ).to(device)
    return model, processor, _extract_features_hf
