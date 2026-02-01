from __future__ import annotations

import argparse
import gc
import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional, Tuple

from analyse_vit.rare_colour_bias.composite.gen_color import _resolve_color_transform
from analyse_vit.rare_colour_bias.generation.gen_pipeline import (
    _default_model_id_for,
    _load_text2image_pipeline,
    _release_torch_cuda,
    _resolve_generation_defaults,
    _resolve_generation_resolution,
)
from analyse_vit.rare_colour_bias.generation.gen_prompts import _generate_prompt_set, _load_prompt_elements
from analyse_vit.rare_colour_bias.composite.gen_sam import _load_lang_sam_model
from analyse_vit.rare_colour_bias.composite.gen_stages import _run_generation, _run_sam_stage
from analyse_vit.rare_colour_bias.generation.gen_types import (
    ColorParams,
    CompositeParams,
    FluxParams,
    RunConfig,
    SamParams,
    _PIPELINES,
)
from analyse_vit.rare_colour_bias.generation.gen_utils import (
    _create_timestamp_dir,
    _find_repo_root,
    _get_hf_token,
    _load_dotenv,
    _resolve_path,
    _derive_run_seed,
)


def _parse_args(
    argv: Optional[Iterable[str]] = None,
    mode: str = "full",
) -> Tuple[RunConfig, int, Optional[Path], int]:
    parser = argparse.ArgumentParser(description="Text-to-image + SAM composite pipeline")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root directory (required for generation runs; optional for composite-only runs).",
    )
    parser.add_argument("--run-root", default=None, help="Existing run root directory for composite-only runs.")
    parser.add_argument(
        "--background-prompt",
        default=None,
        help="Background prompt (optional; if omitted, background image is not generated).",
    )
    parser.add_argument(
        "--background-only",
        action="store_true",
        help="Generate only the background image (skip anchor generation).",
    )
    parser.add_argument(
        "--background-grayscale",
        action="store_true",
        help="Convert generated background to grayscale before saving.",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt (only used to enable guidance_scale for flux/sd3).",
    )
    parser.add_argument(
        "--object-name",
        default=None,
        help="Object name to use for grounding SAM masks.",
    )
    parser.add_argument(
        "--prompt-elements-json",
        default=None,
        help="JSON file of prompt elements used to build a unique prompt per run.",
    )
    parser.add_argument("--color", default=None)
    parser.add_argument("--seed-bg", type=int, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for non-qwen generation.")

    parser.add_argument("--pipeline", default="flux", choices=list(_PIPELINES.keys()))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--flux-model-id", default=None)  # backward-compat alias
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--torch-dtype", default=None, choices=["bfloat16", "float16", "float32", "bf8"])
    parser.add_argument("--hf-token", default=None, help="Hugging Face token.")
    parser.add_argument("--env-file", default=None, help="Optional .env file to load.")
    parser.add_argument("--local-files-only", action="store_true", help="Load weights only from local cache (no network).")
    parser.add_argument("--flux-local-files-only", action="store_true", help="(deprecated) Use --local-files-only instead.")

    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--sam-prompt-mode", default="grounding", choices=["points", "box", "grounding"])
    parser.add_argument("--sam-box-ratio-w", type=float, default=0.8)
    parser.add_argument("--sam-box-ratio-h", type=float, default=0.85)
    parser.add_argument("--feather-radius-px", type=int, default=3)
    parser.add_argument(
        "--mask-selection-rule",
        default="center_included_max_area",
        choices=["center_included_max_area", "center_included_max_score", "all_masks"],
    )
    parser.add_argument("--morph-kernel-px", type=int, default=3)
    parser.add_argument("--mask-min-area-frac", type=float, default=0.01)
    parser.add_argument("--mask-max-area-frac", type=float, default=0.9)
    parser.add_argument("--lang-sam-box-threshold", type=float, default=0.25)
    parser.add_argument("--lang-sam-text-threshold", type=float, default=0.25)

    parser.add_argument("--anchor-x", type=float, default=0.5)
    parser.add_argument("--anchor-y", type=float, default=0.9)
    parser.add_argument("--target-obj-height-ratio", type=float, default=0.65)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    parser.add_argument("--placement-mode", default="center", choices=["random", "anchor", "center"])

    parser.add_argument("--color-hue-deg", type=float, default=None)
    parser.add_argument(
        "--target-hue-deg",
        type=float,
        default=None,
        help="Deprecated: use --color-hue-deg.",
    )
    parser.add_argument("--min-saturation", type=float, default=0.25)

    parser.add_argument("--device", default="cuda")

    args = parser.parse_args(argv)

    mode = mode.lower()
    if mode not in {"generate", "composite", "full"}:
        raise ValueError(f"Invalid mode: {mode}")

    output_dir = _resolve_path(args.output_dir) if args.output_dir else None
    run_root_override = _resolve_path(args.run_root) if args.run_root else None
    sam_checkpoint = _resolve_path(args.sam_checkpoint) if args.sam_checkpoint else None

    prompt_elements_path = _resolve_path(args.prompt_elements_json) if args.prompt_elements_json else None
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    if negative_prompt is not None and not negative_prompt.strip():
        negative_prompt = None

    require_generation = mode in {"generate", "full"}
    require_composite = mode in {"composite", "full"}

    background_prompt = args.background_prompt
    if background_prompt is not None and not background_prompt.strip():
        background_prompt = None

    if require_generation:
        if output_dir is None:
            raise ValueError("--output-dir is required for generation runs.")
        if args.background_only:
            if background_prompt is None:
                raise ValueError("--background-only requires --background-prompt.")
            if args.seed_bg is None:
                raise ValueError("--seed-bg is required when --background-only is used.")
            if prompt_elements_path is not None or prompt is not None:
                raise ValueError("--background-only does not use --prompt or --prompt-elements-json.")
        else:
            if prompt_elements_path is None and prompt is None:
                raise ValueError("--prompt or --prompt-elements-json must be provided.")
            if background_prompt is not None and args.seed_bg is None:
                raise ValueError("--seed-bg is required when --background-prompt is provided.")

    if require_composite:
        if args.color is None:
            raise ValueError("--color is required for composite runs.")
        if args.sam_checkpoint is None:
            raise ValueError("--sam-checkpoint is required for composite runs.")

    if run_root_override is None and mode == "composite":
        raise ValueError("--run-root is required for composite-only runs.")
    if args.background_only and mode != "generate":
        raise ValueError("--background-only is supported only in generate mode.")
    if output_dir is None and run_root_override is not None:
        output_dir = run_root_override
    prompt = prompt or ""
    if not (0.0 <= args.anchor_x <= 1.0 and 0.0 <= args.anchor_y <= 1.0):
        raise ValueError("--anchor-x/--anchor-y must be in [0, 1].")
    if args.target_obj_height_ratio <= 0.0:
        raise ValueError("--target-obj-height-ratio must be > 0.")
    if args.scale_multiplier <= 0.0:
        raise ValueError("--scale-multiplier must be > 0.")
    if args.mask_selection_rule not in {"center_included_max_area", "center_included_max_score", "all_masks"}:
        raise ValueError(
            "--mask-selection-rule must be center_included_max_area, center_included_max_score, or all_masks."
        )
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be > 0.")
    if args.model_id and args.flux_model_id and args.model_id != args.flux_model_id:
        raise ValueError("--model-id and --flux-model-id must match if both are set.")
    object_name = args.object_name
    if require_composite and args.sam_prompt_mode == "grounding" and not object_name:
        raise ValueError("--object-name is required when --sam-prompt-mode grounding is used.")
    if require_composite:
        if args.color_hue_deg is not None and args.target_hue_deg is not None:
            if args.color_hue_deg != args.target_hue_deg:
                raise ValueError("--color-hue-deg and --target-hue-deg must match when both are set.")
        _resolve_color_transform(args.color or "", args.color_hue_deg or args.target_hue_deg)

    env_file = _resolve_path(args.env_file) if args.env_file else None
    if env_file is None:
        repo_root = _find_repo_root(Path(__file__).resolve())
        if repo_root is not None:
            env_file = repo_root / ".env"
    if env_file is not None:
        _load_dotenv(env_file)

    hf_token = _get_hf_token(args.hf_token)

    model_id = args.model_id or args.flux_model_id or _default_model_id_for(args.pipeline)
    local_files_only = bool(args.local_files_only or args.flux_local_files_only)

    default_width, default_height = _resolve_generation_resolution(args.pipeline, model_id)
    width = args.width if args.width is not None else default_width
    height = args.height if args.height is not None else default_height

    default_steps, default_guidance = _resolve_generation_defaults(args.pipeline, model_id)
    num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else default_steps
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else default_guidance

    torch_dtype = args.torch_dtype or "bfloat16"
    if args.pipeline == "qwen" and torch_dtype == "bf8":
        logging.getLogger(__name__).warning("Qwen pipeline does not use bf8; falling back to bfloat16.")
        torch_dtype = "bfloat16"

    flux = FluxParams(
        pipeline=args.pipeline,
        model_id=model_id,
        width=int(width),
        height=int(height),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        max_sequence_length=int(args.max_sequence_length),
        torch_dtype=str(torch_dtype),
        local_files_only=local_files_only,
    )

    sam = SamParams(
        checkpoint=sam_checkpoint or Path("."),
        model_type=str(args.sam_model_type),
        prompt_mode=str(args.sam_prompt_mode),
        box_ratio_w=float(args.sam_box_ratio_w),
        box_ratio_h=float(args.sam_box_ratio_h),
        feather_radius_px=int(args.feather_radius_px),
        mask_selection_rule=str(args.mask_selection_rule),
        morph_kernel_px=int(args.morph_kernel_px),
        min_area_frac=float(args.mask_min_area_frac),
        max_area_frac=float(args.mask_max_area_frac),
        lang_sam_box_threshold=float(args.lang_sam_box_threshold),
        lang_sam_text_threshold=float(args.lang_sam_text_threshold),
    )

    composite = CompositeParams(
        anchor_x=float(args.anchor_x),
        anchor_y=float(args.anchor_y),
        target_obj_height_ratio=float(args.target_obj_height_ratio),
        scale_multiplier=float(args.scale_multiplier),
        placement_mode=str(args.placement_mode),
    )

    color_params = ColorParams(
        target_hue_deg=float(args.color_hue_deg)
        if args.color_hue_deg is not None
        else (float(args.target_hue_deg) if args.target_hue_deg is not None else None),
        min_saturation=float(args.min_saturation),
    )

    return (
        RunConfig(
            output_dir=output_dir,
            background_prompt=background_prompt,
            background_only=bool(args.background_only),
            background_grayscale=bool(args.background_grayscale),
            prompt=prompt,
            negative_prompt=negative_prompt,
            object_name=object_name,
            prompt_elements_path=prompt_elements_path,
            prompt_elements=None,
            color_name=str(args.color or ""),
            seed_bg=int(args.seed_bg) if args.seed_bg is not None else 0,
            seed=int(args.seed),
            run_index=0,
            seed_offset=0,
            flux=flux,
            sam=sam,
            composite=composite,
            color_params=color_params,
            device=str(args.device),
            hf_token=hf_token,
        ),
        int(args.num_runs),
        run_root_override,
        int(args.batch_size),
    )


def _prepare_prompt_sets(
    config: RunConfig,
    num_runs: int,
) -> Tuple[
    Optional[list[str]],
    Optional[list[dict[str, str]]],
]:
    joiner = ", "
    prompts: Optional[list[str]] = None
    prompt_elements: Optional[list[dict[str, str]]] = None
    if config.prompt_elements_path is not None:
        keys, groups = _load_prompt_elements(config.prompt_elements_path)
        prompts, prompt_elements = _generate_prompt_set(
            keys, groups, num_runs, config.seed, 0xBADDCAFE, joiner, "Prompt"
        )

    return prompts, prompt_elements


def _build_run_configs(
    config: RunConfig,
    num_runs: int,
    run_root: Path,
) -> list[RunConfig]:
    prompts, prompt_elements = _prepare_prompt_sets(config, num_runs)

    run_configs: list[RunConfig] = []
    for run_index in range(num_runs):
        run_dir = run_root / f"run_{run_index:02d}"
        seed_bg = _derive_run_seed(config.seed_bg, run_index, 0xA5A5A5A5)
        seed = _derive_run_seed(config.seed, run_index, 0x5A5A5A5A)

        prompt = prompts[run_index] if prompts is not None else config.prompt
        elements = prompt_elements[run_index] if prompt_elements is not None else None

        run_config = replace(
            config,
            output_dir=run_dir,
            seed_bg=seed_bg,
            seed=seed,
            prompt=prompt,
            prompt_elements=elements,
            run_index=run_index,
            seed_offset=run_index,
        )
        run_configs.append(run_config)

    return run_configs


# ----------------------------
# main
# ----------------------------


def main_generate(argv: Optional[Iterable[str]] = None) -> None:
    config, num_runs, run_root_override, batch_size = _parse_args(argv, mode="generate")
    run_root = run_root_override or _create_timestamp_dir(config.output_dir)
    if run_root_override is not None and not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=True)

    run_configs = _build_run_configs(config, num_runs, run_root)

    pipe = _load_text2image_pipeline(
        config.flux.pipeline,
        config.flux.model_id,
        config.flux.torch_dtype,
        config.device,
        config.hf_token,
        config.flux.local_files_only,
    )

    _run_generation(run_configs, pipe=pipe, flat_outputs=True, flat_root=run_root, batch_size=batch_size)

    try:
        pipe.to("cpu")
    except Exception:
        pass
    del pipe
    gc.collect()
    if config.device != "cpu":
        _release_torch_cuda()


def main_composite(argv: Optional[Iterable[str]] = None) -> None:
    config, num_runs, run_root_override, _ = _parse_args(argv, mode="composite")
    run_root = run_root_override

    run_configs = _build_run_configs(config, num_runs, run_root)

    if config.sam.prompt_mode == "grounding":
        lang_sam_model = _load_lang_sam_model(config.device)
        for rc in run_configs:
            _run_sam_stage(rc, predictor=None, lang_sam_model=lang_sam_model)
    else:
        raise ValueError("grounding 以外の SAM プロンプトは現在無効です。")


def main(argv: Optional[Iterable[str]] = None) -> None:
    config, num_runs, run_root_override, batch_size = _parse_args(argv, mode="full")
    run_root = run_root_override or _create_timestamp_dir(config.output_dir)
    if run_root_override is not None and not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=True)

    run_configs = _build_run_configs(config, num_runs, run_root)

    pipe = _load_text2image_pipeline(
        config.flux.pipeline,
        config.flux.model_id,
        config.flux.torch_dtype,
        config.device,
        config.hf_token,
        config.flux.local_files_only,
    )

    _run_generation(run_configs, pipe=pipe, batch_size=batch_size)

    try:
        pipe.to("cpu")
    except Exception:
        pass
    del pipe
    gc.collect()
    if config.device != "cpu":
        _release_torch_cuda()

    if config.sam.prompt_mode == "grounding":
        lang_sam_model = _load_lang_sam_model(config.device)
        for rc in run_configs:
            _run_sam_stage(rc, predictor=None, lang_sam_model=lang_sam_model)
    else:
        raise ValueError("grounding 以外の SAM プロンプトは現在無効です。")
