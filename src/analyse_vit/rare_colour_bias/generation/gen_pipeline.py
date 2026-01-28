from __future__ import annotations

import inspect
import logging
import os
import time
from typing import Any, Dict, Iterable, Optional, Tuple, Sequence

from PIL import Image

from .gen_types import _PIPELINES
from .gen_utils import _filter_kwargs_for_callable


def _get_pipeline_spec(pipeline: str) -> Dict[str, Any]:
    spec = _PIPELINES.get(pipeline)
    if spec is None:
        raise ValueError(f"Unknown pipeline type: {pipeline}")
    return spec


def _resolve_generation_defaults(pipeline: str, model_id: str) -> Tuple[int, float]:
    spec = _get_pipeline_spec(pipeline)
    overrides = spec.get("model_overrides", {}).get(model_id)
    if overrides:
        return int(overrides["num_inference_steps"]), float(overrides["guidance_scale"])
    defaults = spec["defaults"]
    return int(defaults["num_inference_steps"]), float(defaults["guidance_scale"])


def _resolve_generation_resolution(pipeline: str, model_id: str) -> Tuple[int, int]:
    spec = _get_pipeline_spec(pipeline)
    overrides = spec.get("model_overrides", {}).get(model_id)
    if overrides and "resolution" in overrides:
        w, h = overrides["resolution"]
        return int(w), int(h)
    w, h = spec["defaults"]["resolution"]
    return int(w), int(h)


def _default_model_id_for(pipeline: str) -> str:
    return str(_get_pipeline_spec(pipeline)["default_model_id"])


def _resolve_torch_dtype(name: str):
    import torch

    if name == "bf8":
        for attr in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"):
            dtype = getattr(torch, attr, None)
            if dtype is None:
                continue
            orig_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(dtype)
            except Exception:
                try:
                    torch.set_default_dtype(orig_dtype)
                except Exception:
                    pass
                continue
            else:
                try:
                    torch.set_default_dtype(orig_dtype)
                except Exception:
                    pass
            try:
                torch.empty(1, dtype=dtype)
            except Exception:
                continue
            return dtype

        logging.getLogger(__name__).warning(
            "bf8 requested but float8 is unsupported in this torch build; falling back to bfloat16."
        )
        return torch.bfloat16

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return dtype_map.get(name, torch.bfloat16)


def _release_torch_cuda() -> None:
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _configure_qwen_pipeline(pipe, logger: logging.Logger) -> None:
    """
    Qwen-Image は double-stream attention（image/text）を前提としており、
    attention の戻り値が (img_attn_output, txt_attn_output) のタプルになる必要があります。
    これが崩れると `img_attn_output, txt_attn_output = attn_output` で失敗します。

    ここでは xFormers には一切触れず、Qwen 用の double-stream AttentionProcessor を明示設定します。
    """
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return

    # diffusers のバージョン差に備えて import 経路を複数試す
    ProcessorCls = None
    try:
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenDoubleStreamAttnProcessor2_0 as _Proc,
        )
        ProcessorCls = _Proc
    except Exception:
        try:
            from diffusers.models.attention_processor import (
                QwenDoubleStreamAttnProcessor2_0 as _Proc,
            )
            ProcessorCls = _Proc
        except Exception as exc:
            raise RuntimeError(
                "QwenDoubleStreamAttnProcessor2_0 を import できませんでした。"
                "diffusers のバージョン不整合の可能性があります。"
                "Qwen/Qwen-Image-2512 を torch.compile と併用する場合、"
                "Qwen double-stream attention processor が必要です。"
                f" original_error={type(exc).__name__}: {exc}"
            ) from exc

    if hasattr(transformer, "set_attn_processor"):
        transformer.set_attn_processor(ProcessorCls())
        logger.info("Set Qwen attention processor to %s.", ProcessorCls.__name__)
    else:
        raise RuntimeError("Qwen transformer does not expose set_attn_processor().")

    # 可能なら SDPA を有効化（xFormers には触れない）
    if hasattr(transformer, "set_use_sdpa"):
        try:
            transformer.set_use_sdpa(True)
            logger.info("Enabled SDPA on Qwen transformer.")
        except Exception as exc:
            logger.warning("Failed to enable SDPA on Qwen transformer: %s", exc)



def _maybe_torch_compile(
    pipe,
    logger: logging.Logger,
    enabled: bool,
    mode: str,
    backend: Optional[str],
    *,
    targets: Sequence[str] = ("transformer", "unet", "text_encoder", "text_encoder_2", "vae"),
    matmul_precision: Optional[str] = "high",  # "high" / "medium" / "highest" / None
    suppress_dynamo_errors: bool = False,
    fullgraph: bool = False,
    dynamic: Optional[bool] = None,
) -> None:
    if enabled:
        logger.info("torch.compile is disabled; skipping compilation.")
    return


# ----------------------------
# Diffusion pipeline load & generation (single/batch unified)
# ----------------------------


def _load_text2image_pipeline(
    pipeline_type: str,
    model_id: str,
    torch_dtype: str,
    device: str,
    token: Optional[str],
    local_files_only: bool,
):
    import torch

    logger = logging.getLogger(__name__)
    logger.info(
        "Preparing text-to-image pipeline: type=%s model_id=%s device=%s dtype=%s local_files_only=%s",
        pipeline_type,
        model_id,
        device,
        torch_dtype,
        local_files_only,
    )
    start_time = time.perf_counter()

    if pipeline_type == "flux":
        from diffusers import FluxPipeline as PipelineClass
    elif pipeline_type == "sd3":
        from diffusers import StableDiffusion3Pipeline as PipelineClass
    elif pipeline_type == "qwen":
        from diffusers import DiffusionPipeline as PipelineClass
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    if pipeline_type == "qwen" and torch_dtype == "bf8":
        raise ValueError("Internal configuration error: qwen pipeline must not use bf8.")

    dtype = _resolve_torch_dtype(torch_dtype)
    if device == "cpu":
        dtype = torch.float32
    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    use_device_map = pipeline_type == "qwen" and device in {"cuda", "auto"}

    sig = inspect.signature(PipelineClass.from_pretrained)
    kwargs: Dict[str, Any] = {}
    if "dtype" in sig.parameters:
        kwargs["dtype"] = dtype
    else:
        kwargs["torch_dtype"] = dtype
    kwargs["local_files_only"] = local_files_only

    device_map = None
    if use_device_map:
        device_map = "balanced"
        kwargs["device_map"] = device_map
        logger.info("Using device_map=%s for qwen to reduce GPU memory usage during load.", device_map)

    if token:
        if "token" in sig.parameters:
            kwargs["token"] = token
        elif "use_auth_token" in sig.parameters:
            kwargs["use_auth_token"] = token
        else:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    pipe = None
    try:
        pipe = PipelineClass.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        retry_exc = exc
        err_text = str(exc).lower()
        if device_map is not None and ("device_map" in err_text or "supported strategies" in err_text):
            logger.warning("Device map load failed (%s). Retrying without device_map.", exc)
            kwargs.pop("device_map", None)
            device_map = None
            try:
                pipe = PipelineClass.from_pretrained(model_id, **kwargs)
            except Exception as exc_retry:
                retry_exc = exc_retry
        if pipe is None:
            hints = [f"Failed to load pipeline ({pipeline_type})."]
            hints.append(f"Original error: {type(retry_exc).__name__}: {retry_exc}")
            if token is None:
                hints.append("If the model is gated, pass --hf-token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN.")
            if local_files_only:
                hints.append("local_files_only is enabled; ensure the model is cached or omit --local-files-only.")
            if torch_dtype == "bf8":
                hints.append("bf8 was requested; if float8 is unsupported, try --torch-dtype bfloat16.")
            hints.append("You can also point --model-id to a local path or a public model.")
            raise RuntimeError(" ".join(hints)) from retry_exc

    if device_map is None:
        pipe.to(resolved_device)
    pipe.set_progress_bar_config(disable=True)

    if pipeline_type == "qwen":
        _configure_qwen_pipeline(pipe, logger)

    _maybe_torch_compile(
        pipe,
        logger,
        enabled=False,
        mode="default",
        backend=None,
    )

    logger.info("Pipeline ready (%s). init_time=%.2fs", pipeline_type, time.perf_counter() - start_time)
    return pipe


def _generate_images(
    pipe,
    prompts: Iterable[str],
    seeds: Iterable[int],
    width: int,
    height: int,
    steps: int,
    guidance_scale: Optional[float],
    max_sequence_length: int,
    device: str,
    negative_prompt: Optional[str] = None,
) -> list[Image.Image]:
    import torch

    prompt_list = list(prompts)
    seed_list = list(seeds)
    if len(prompt_list) != len(seed_list):
        raise ValueError("Prompt and seed counts must match for generation.")
    if not prompt_list:
        return []

    sig = inspect.signature(pipe.__call__)
    has_true_cfg = "true_cfg_scale" in sig.parameters
    has_negative = "negative_prompt" in sig.parameters
    neg = negative_prompt.strip() if isinstance(negative_prompt, str) else None
    if neg == "":
        neg = None

    if len(prompt_list) == 1:
        generator = torch.Generator(device=device).manual_seed(seed_list[0])
        call_kwargs: Dict[str, Any] = {
            "prompt": prompt_list[0],
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "max_sequence_length": max_sequence_length,
            "generator": generator,
        }
        if guidance_scale is not None:
            call_kwargs["guidance_scale"] = guidance_scale
            if has_true_cfg:
                call_kwargs["true_cfg_scale"] = guidance_scale
        if guidance_scale is not None and has_negative and neg is not None:
            call_kwargs["negative_prompt"] = neg
        call_kwargs = _filter_kwargs_for_callable(pipe.__call__, call_kwargs)
        result = pipe(**call_kwargs)
        return [result.images[0]]

    generators = [torch.Generator(device=device).manual_seed(s) for s in seed_list]
    call_kwargs = {
        "prompt": prompt_list,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "max_sequence_length": max_sequence_length,
        "generator": generators,
    }
    if guidance_scale is not None:
        call_kwargs["guidance_scale"] = guidance_scale
        if has_true_cfg:
            call_kwargs["true_cfg_scale"] = guidance_scale
    if guidance_scale is not None and has_negative and neg is not None:
        call_kwargs["negative_prompt"] = [neg] * len(prompt_list)

    call_kwargs = _filter_kwargs_for_callable(pipe.__call__, call_kwargs)
    result = pipe(**call_kwargs)
    images = list(result.images)
    if len(images) != len(prompt_list):
        raise RuntimeError("Generation returned an unexpected number of images.")
    return images
