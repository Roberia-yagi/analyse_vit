# pe.py
from __future__ import annotations

import contextlib
import inspect
import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image

LOGGER = logging.getLogger("analyse_vit.backends.pe")


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


def _resolve_forward_features_model(model: torch.nn.Module) -> Tuple[torch.nn.Module | None, str]:
    if hasattr(model, "forward_features"):
        return model, "model.forward_features"
    for attr in ("visual", "vision_model", "image_encoder", "encoder", "backbone", "trunk"):
        candidate = getattr(model, attr, None)
        if isinstance(candidate, torch.nn.Module) and hasattr(candidate, "forward_features"):
            return candidate, f"model.{attr}.forward_features"
    return None, "model.forward_features"


def extract_features_pe(
    model: torch.nn.Module,
    processor: object,
    images: Sequence[Image.Image],
    device: str,
    batch_size: int,
    pooling: str,
) -> torch.Tensor:
    model.eval()
    features: List[torch.Tensor] = []
    use_amp = device == "cuda" or str(device).startswith("cuda")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    logged = False
    forward_features_model, forward_features_label = _resolve_forward_features_model(model)

    transform = processor  # perception の processor は callable transform を想定

    with torch.no_grad(), autocast_ctx:
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = torch.stack([transform(image) for image in batch], dim=0).to(device)

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


def load_pe_model(
    model_id: str,
    device: str,
) -> tuple[torch.nn.Module, object, object]:
    try:
        import core.vision_encoder.pe as pe  # type: ignore
        import core.vision_encoder.transforms as pe_transforms  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "perception_models is required for PE models. "
            "Install with: pip install git+https://github.com/facebookresearch/perception_models.git"
        ) from exc

    if model_id.startswith("PE-Core"):
        model = pe.CLIP.from_config(model_id, pretrained=True)
    else:
        model = pe.VisionTransformer.from_config(model_id, pretrained=True)
    model = model.to(device)
    processor = pe_transforms.get_image_transform(model.image_size)

    return model, processor, extract_features_pe
