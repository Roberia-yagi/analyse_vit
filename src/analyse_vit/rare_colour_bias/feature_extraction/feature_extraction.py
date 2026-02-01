# feature_extraction.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from PIL import Image

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

LOGGER = logging.getLogger("analyse_vit.feature_extraction")

if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    backend: str


@dataclass(frozen=True)
class LoadedModel:
    spec: ModelSpec
    model: object
    processor: object
    extractor: Callable[[object, object, Sequence[Image.Image], str, int, str], torch.Tensor]

    # 既存コードとの互換性確保:
    # model, processor, extractor = _load_model(...) を成立させる
    def __iter__(self):
        yield self.model
        yield self.processor
        yield self.extractor


MODEL_SPECS: Dict[str, ModelSpec] = {
    "pe-core-l14-336": ModelSpec(key="pe-core-l14-336", model_id="PE-Core-L14-336", backend="pe"),
    "qwen3-vl-8b-embed": ModelSpec(
        key="qwen3-vl-8b-embed",
        model_id="Qwen/Qwen3-VL-Embedding-8B",
        backend="qwen3_vl_embedding",
    ),
    "siglip2-giant-opt-patch16-384": ModelSpec(
        key="siglip2-giant-opt-patch16-384",
        model_id="google/siglip2-giant-opt-patch16-384",
        backend="siglip2",
    ),
}

MODEL_ALIASES: Dict[str, str] = {
    "pe": "pe-core-l14-336",
    "pe-core": "pe-core-l14-336",
    "pe-core-l14": "pe-core-l14-336",
    "pe-core-l14-336": "pe-core-l14-336",
    "core": "pe-core-l14-336",
    "l14": "pe-core-l14-336",
    "qwen-embed": "qwen3-vl-8b-embed",
    "qwen_embed": "qwen3-vl-8b-embed",
    "qwen3-embed": "qwen3-vl-8b-embed",
    "qwen3-vl-8b-embed": "qwen3-vl-8b-embed",
    "siglip2": "siglip2-giant-opt-patch16-384",
    "siglip2-giant": "siglip2-giant-opt-patch16-384",
    "siglip2-giant-384": "siglip2-giant-opt-patch16-384",
    "siglip2-giant-opt-patch16-384": "siglip2-giant-opt-patch16-384",
}

POOLING_CHOICES: Tuple[str, ...] = ("attention_pooling", "mean_patch", "cls")


def _poolings_for_spec(spec: ModelSpec) -> Tuple[str, ...]:
    if spec.backend == "pe":
        return ("attention_pooling", "cls", "mean_patch")
    if spec.backend == "qwen3_vl_embedding":
        return ("attention_pooling",)
    if spec.backend == "siglip2":
        return ("attention_pooling", "cls", "mean_patch")
    return POOLING_CHOICES


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

        if lowered.startswith("pe:"):
            model_id = entry.split(":", 1)[1].strip()
            if not model_id:
                raise ValueError("PE model must be formatted as 'pe:PE-Core-L14-336'.")
            key = f"pe-{model_id}".lower().replace("/", "_").replace(" ", "_")
            resolved.append(ModelSpec(key=key, model_id=model_id, backend="pe"))
            continue

        for prefix, backend, key_prefix in (
            ("qwen-embed:", "qwen3_vl_embedding", "qwen-vl-embed"),
            ("qwen_embed:", "qwen3_vl_embedding", "qwen-vl-embed"),
            ("qwen3-embed:", "qwen3_vl_embedding", "qwen-vl-embed"),
            ("qwen3-vl-embed:", "qwen3_vl_embedding", "qwen-vl-embed"),
        ):
            if lowered.startswith(prefix):
                model_id = entry.split(":", 1)[1].strip()
                if not model_id:
                    raise ValueError(f"{prefix[:-1]} model must be formatted as '{prefix}<model_id>'.")
                if model_id != embed_model_id:
                    raise ValueError(
                        f"qwen3_vl_embedding only supports model_id='{embed_model_id}', got '{model_id}'."
                    )
                slug = model_id.lower().replace("/", "_").replace(" ", "_")
                key = f"{key_prefix}-{slug}"
                resolved.append(ModelSpec(key=key, model_id=model_id, backend=backend))
                break
        else:
            if lowered.startswith("siglip2:"):
                model_id = entry.split(":", 1)[1].strip()
                if not model_id:
                    raise ValueError("SigLIP2 model must be formatted as 'siglip2:<model_id>'.")
                slug = model_id.lower().replace("/", "_").replace(" ", "_")
                key = f"siglip2-{slug}"
                resolved.append(ModelSpec(key=key, model_id=model_id, backend="siglip2"))
                continue

            valid = ", ".join(sorted(MODEL_SPECS.keys()))
            raise ValueError(
                f"Unknown model '{entry}'. Use one of: {valid}, aliases: pe, qwen-embed, siglip2, "
                "or explicit ids: pe:<model_id>, qwen-embed:<model_id>, siglip2:<model_id>."
            )

    return resolved


def _load_model(
    spec: ModelSpec,
    device: str,
    hf_token: str | None,
) -> LoadedModel:
    from analyse_vit.rare_colour_bias.feature_extraction.pe import load_pe_model
    from analyse_vit.rare_colour_bias.feature_extraction.qwen3_vl_embedding import (
        load_qwen3_vl_embedding_model,
    )
    from analyse_vit.rare_colour_bias.feature_extraction.siglip2 import load_siglip2_model

    if spec.backend == "pe":
        model, processor, extractor = load_pe_model(spec.model_id, device)
        return LoadedModel(spec=spec, model=model, processor=processor, extractor=extractor)

    if spec.backend == "qwen3_vl_embedding":
        model, processor, extractor = load_qwen3_vl_embedding_model(spec.model_id, device, hf_token)
        return LoadedModel(spec=spec, model=model, processor=processor, extractor=extractor)

    if spec.backend == "siglip2":
        model, processor, extractor = load_siglip2_model(spec.model_id, device)
        return LoadedModel(spec=spec, model=model, processor=processor, extractor=extractor)

    raise ValueError(f"Unsupported backend: {spec.backend}")


class FeatureExtractor:
    def __init__(
        self,
        models: Sequence[str] = ("pe",),
        device: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token
        specs = _resolve_models(models)
        self.models: Dict[str, LoadedModel] = {}
        for spec in specs:
            self.models[spec.key] = _load_model(spec, self.device, self.hf_token)

    def available_models(self) -> List[str]:
        return list(self.models.keys())

    def extract(
        self,
        images: Sequence[Image.Image],
        model_key: str,
        pooling: str = "attention_pooling",
        batch_size: int = 8,
    ) -> torch.Tensor:
        if model_key not in self.models:
            raise ValueError(f"Unknown loaded model_key='{model_key}'. Available: {sorted(self.models.keys())}")

        loaded = self.models[model_key]
        allowed = _poolings_for_spec(loaded.spec)
        if pooling not in allowed:
            raise ValueError(f"Invalid pooling='{pooling}' for {model_key}. Allowed: {allowed}")

        return loaded.extractor(
            loaded.model,
            loaded.processor,
            images,
            self.device,
            batch_size,
            pooling,
        )

    def extract_all(
        self,
        images: Sequence[Image.Image],
        pooling: str = "attention_pooling",
        batch_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for key, loaded in self.models.items():
            allowed = _poolings_for_spec(loaded.spec)
            if pooling not in allowed:
                raise ValueError(f"Invalid pooling='{pooling}' for {key}. Allowed: {allowed}")
            outputs[key] = loaded.extractor(
                loaded.model,
                loaded.processor,
                images,
                self.device,
                batch_size,
                pooling,
            )
        return outputs
