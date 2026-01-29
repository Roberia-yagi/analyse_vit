# siglip2.py
from __future__ import annotations

import contextlib
import logging
from typing import List, Mapping, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

LOGGER = logging.getLogger("analyse_vit.backends.siglip2")


def _select_pooled_from_outputs(outputs: object) -> Tuple[torch.Tensor, str]:
    if isinstance(outputs, torch.Tensor):
        if outputs.dim() == 2:
            return outputs, "outputs (tensor)"
        raise ValueError("Outputs is a tensor but not 2D; pooled output required.")

    if isinstance(outputs, Mapping):
        for key in ("pooler_output", "pooled_output", "image_embeds", "image_features"):
            value = outputs.get(key)
            if isinstance(value, torch.Tensor) and value.dim() == 2:
                return value, f"outputs['{key}']"

    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor) and first.dim() == 2:
            return first, "outputs[0]"

    for attr in ("pooler_output", "pooled_output", "image_embeds", "image_features"):
        value = getattr(outputs, attr, None)
        if isinstance(value, torch.Tensor) and value.dim() == 2:
            return value, f"outputs.{attr}"

    vision_out = getattr(outputs, "vision_model_output", None)
    if vision_out is not None:
        for attr in ("pooler_output", "pooled_output"):
            value = getattr(vision_out, attr, None)
            if isinstance(value, torch.Tensor) and value.dim() == 2:
                return value, f"outputs.vision_model_output.{attr}"
        if isinstance(vision_out, (tuple, list)) and vision_out:
            first = vision_out[0]
            if isinstance(first, torch.Tensor) and first.dim() == 2:
                return first, "outputs.vision_model_output[0]"

    raise ValueError("No pooled output found.")


def _select_sequence_from_outputs(outputs: object) -> Tuple[torch.Tensor, str]:
    if isinstance(outputs, torch.Tensor):
        if outputs.dim() == 3:
            return outputs, "outputs (tensor)"
        raise ValueError("Outputs is a tensor but not 3D; sequence output required.")

    for attr in ("last_hidden_state", "encoder_last_hidden_state"):
        value = getattr(outputs, attr, None)
        if isinstance(value, torch.Tensor) and value.dim() == 3:
            return value, f"outputs.{attr}"

    vision_out = getattr(outputs, "vision_model_output", None)
    if vision_out is not None:
        v = getattr(vision_out, "last_hidden_state", None)
        if isinstance(v, torch.Tensor) and v.dim() == 3:
            return v, "outputs.vision_model_output.last_hidden_state"
        if isinstance(vision_out, (tuple, list)) and vision_out:
            first = vision_out[0]
            if isinstance(first, torch.Tensor) and first.dim() == 3:
                return first, "outputs.vision_model_output[0]"

    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor) and first.dim() == 3:
            return first, "outputs[0]"

    raise ValueError("No sequence output found.")


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


def _model_is_dispatched(model: torch.nn.Module) -> bool:
    return bool(getattr(model, "hf_device_map", None))


def _infer_vision_input_device(model: torch.nn.Module, fallback_device: str) -> torch.device:
    # SigLIP/SigLIP2: vision_model.embeddings.patch_embedding が Conv2d
    try:
        vision_model = getattr(model, "vision_model", None)
        if vision_model is not None:
            embeddings = getattr(vision_model, "embeddings", None)
            if embeddings is not None:
                patch = getattr(embeddings, "patch_embedding", None)
                if patch is not None and hasattr(patch, "weight"):
                    w = patch.weight
                    if isinstance(w, torch.Tensor):
                        return w.device
    except Exception:
        pass

    # それ以外: meta 以外の最初のパラメータのデバイス
    try:
        for p in model.parameters():
            if isinstance(p, torch.Tensor) and p.device.type != "meta":
                return p.device
    except Exception:
        pass

    # 最終フォールバック
    return torch.device(fallback_device)


def extract_features_siglip2(
    model: torch.nn.Module,
    processor: object,
    images: Sequence[Image.Image],
    device: str,
    batch_size: int,
    pooling: str,
) -> torch.Tensor:
    """
    PE と同一インターフェース:
      - pooling: "attention_pooling" | "cls" | "mean_patch"
      - 戻り値: (N, D) の torch.Tensor (CPU, float32)
    """
    model.eval()
    features: List[torch.Tensor] = []

    dispatched = _model_is_dispatched(model)
    target_device = _infer_vision_input_device(model, fallback_device=device)

    use_amp = target_device.type == "cuda"
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

    logged = False

    with torch.no_grad(), autocast_ctx:
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]

            inputs = processor(images=list(batch), return_tensors="pt")
            if not isinstance(inputs, Mapping) and not hasattr(inputs, "to"):
                raise ValueError("siglip2: processor(...) must return a mapping-like object or BatchFeature with .to().")

            # device_map="auto" の場合でも、pixel_values は vision の最初の層があるデバイスへ送る必要がある
            if hasattr(inputs, "to"):
                inputs = inputs.to(target_device)
            else:
                inputs = {k: (v.to(target_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

            if pooling == "attention_pooling":
                if not hasattr(model, "get_image_features"):
                    raise ValueError("siglip2: attention_pooling requires model.get_image_features.")
                batch_features = model.get_image_features(**inputs)  # (B, D)
                if not logged:
                    if dispatched:
                        LOGGER.info(
                            "Feature extraction: SigLIP2 attention_pooling via model.get_image_features() "
                            "(device_map enabled; inputs -> %s).",
                            str(target_device),
                        )
                    else:
                        LOGGER.info("Feature extraction: SigLIP2 attention_pooling via model.get_image_features().")
                    logged = True
            else:
                outputs = model(**inputs)
                try:
                    sequence, source = _select_sequence_from_outputs(outputs)
                    batch_features = _pool_from_sequence(sequence, pooling=pooling, label="siglip2")
                    if not logged:
                        LOGGER.info("Feature extraction: SigLIP2 %s via model.forward -> %s.", pooling, source)
                        logged = True
                except ValueError as exc:
                    if pooling != "cls":
                        raise ValueError(
                            "siglip2: cls/mean_patch requires sequence outputs; model.forward did not return tokens."
                        ) from exc

                    pooled, source = _select_pooled_from_outputs(outputs)
                    batch_features = pooled
                    if not logged:
                        LOGGER.warning(
                            "Feature extraction: SigLIP2 cls via pooled output (%s); no tokens available.", source
                        )
                        logged = True

            features.append(batch_features.float().detach().cpu())

    return torch.cat(features, dim=0)


def load_siglip2_model(
    model_id: str,
    device: str,
) -> tuple[torch.nn.Module, object, object]:
    """
    PE と同一インターフェース:
      return (model, processor, extract_features_fn)
    """
    want_cuda = (device == "cuda") or str(device).startswith("cuda")
    torch_dtype = torch.bfloat16 if want_cuda else None

    # 可能なら device_map="auto"（巨大モデルを想定）。失敗時は通常ロードへフォールバック。
    try:
        if want_cuda:
            model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype, device_map="auto").eval()
        else:
            model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype).to(device).eval()
    except Exception:
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype).to(device).eval()

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, extract_features_siglip2
