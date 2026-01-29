# qwen3_vl_embedding.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from PIL import Image

LOGGER = logging.getLogger("analyse_vit.backends.qwen3_vl_embedding")

EMBED_INSTRUCTION = "Represent the user's input."


@dataclass(frozen=True)
class VllmEmbedder:
    llm: object
    tokenizer: object


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
        raise ValueError("vLLM tokenizer must support apply_chat_template for qwen_vl_embed.")
    prompts = [
        tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        for conversation in conversations
    ]
    return [
        {"prompt": prompt, "multi_modal_data": {"image": image}}
        for prompt, image in zip(prompts, images, strict=True)
    ]


def extract_features_qwen3_vl_embedding(
    model: object,
    processor: object,  # 未使用（互換のため残す）
    images: Sequence[Image.Image],
    device: str,  # 未使用（vLLM 側で管理）
    batch_size: int,  # 未使用（vLLM.embed が内部で処理）
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
        raise ValueError(f"qwen_vl_embed vLLM returned {len(outputs)} embeddings for {len(images)} inputs.")

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


def load_qwen3_vl_embedding_model(
    model_id: str,
    device: str,  # 未使用（互換のため残す）
    hf_token: str | None,  # 未使用（vLLM が別経路のため）
) -> tuple[VllmEmbedder, None, object]:
    try:
        from vllm import LLM, EngineArgs
    except Exception as exc:
        raise ImportError("qwen_vl_embed requires vLLM. Please install vllm.") from exc

    engine_args = EngineArgs(
        model=model_id,
        runner="pooling",
        dtype="bfloat16",
        trust_remote_code=True,
    )
    llm = LLM(**vars(engine_args))
    tokenizer = llm.llm_engine.tokenizer
    return VllmEmbedder(llm=llm, tokenizer=tokenizer), None, extract_features_qwen3_vl_embedding
