from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Iterable, Optional

from ..composite.gen_stages import _run_generation
from .gen_pipeline import (
    _default_model_id_for,
    _load_text2image_pipeline,
    _release_torch_cuda,
    _resolve_generation_defaults,
    _resolve_generation_resolution,
)
from .gen_prompts import _generate_prompt_set, _load_prompt_elements
from .gen_types import ColorParams, CompositeParams, FluxParams, RunConfig, SamParams, _PIPELINES
from .gen_utils import (
    _derive_run_seed,
    _find_repo_root,
    _get_hf_token,
    _load_dotenv,
    _resolve_path,
)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate angle prompts (images + logs).")
    parser.add_argument("--angles-root", default=None, help="Root directory containing angle subfolders.")
    parser.add_argument("--output-root", required=True, help="Output root (raw/angles).")
    parser.add_argument("--models", default="flux,qwen", help="Comma-separated model list (flux,qwen,sd3).")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for non-qwen generation.")

    parser.add_argument("--model-id", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--torch-dtype", default=None, choices=["bfloat16", "float16", "float32", "bf8"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--local-files-only", action="store_true")

    args = parser.parse_args(argv)
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be > 0.")
    return args


def _resolve_angles_root(arg_root: Optional[str]) -> Path:
    if arg_root:
        return _resolve_path(arg_root)
    repo_root = _find_repo_root(Path(__file__).resolve())
    if repo_root is None:
        raise ValueError("--angles-root is required when repo root cannot be resolved.")
    return repo_root / "data" / "angles"


def _iter_prompt_jsons(angles_root: Path) -> list[Path]:
    prompt_jsons = sorted(angles_root.glob("*/*.json"))
    if not prompt_jsons:
        raise ValueError(f"No prompt JSON files found under {angles_root}.")
    return prompt_jsons


def _parse_animal_angle(path: Path) -> tuple[str, str]:
    angle = path.parent.name
    stem = path.stem
    suffix = f"_{angle}"
    if not stem.endswith(suffix):
        raise ValueError(f"Prompt JSON name must end with '{suffix}': {path}")
    animal = stem[: -len(suffix)]
    if not animal:
        raise ValueError(f"Prompt JSON name missing animal prefix: {path}")
    return animal, angle


def _build_run_configs(
    base_config: RunConfig,
    prompt_path: Path,
    num_runs: int,
    seed: int,
) -> list[RunConfig]:
    keys, groups = _load_prompt_elements(prompt_path)
    prompts, prompt_elements = _generate_prompt_set(
        keys, groups, num_runs, seed, 0xBADDCAFE, ", ", "Prompt"
    )

    run_configs: list[RunConfig] = []
    for run_index in range(num_runs):
        seed_bg = _derive_run_seed(0, run_index, 0xA5A5A5A5)
        seed_primary = _derive_run_seed(seed, run_index, 0x5A5A5A5A)
        run_configs.append(
            RunConfig(
                output_dir=base_config.output_dir,
                background_prompt=None,
                background_only=False,
                background_grayscale=False,
                prompt=prompts[run_index],
                negative_prompt=base_config.negative_prompt,
                object_name=None,
                prompt_elements_path=prompt_path,
                prompt_elements=prompt_elements[run_index],
                color_name="",
                seed_bg=seed_bg,
                seed=seed_primary,
                run_index=run_index,
                seed_offset=run_index,
                flux=base_config.flux,
                sam=base_config.sam,
                composite=base_config.composite,
                color_params=base_config.color_params,
                device=base_config.device,
                hf_token=base_config.hf_token,
            )
        )
    return run_configs


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    output_root = _resolve_path(args.output_root)
    angles_root = _resolve_angles_root(args.angles_root)

    env_file = _resolve_path(args.env_file) if args.env_file else None
    if env_file is not None:
        _load_dotenv(env_file)

    hf_token = _get_hf_token(args.hf_token)
    prompt_jsons = _iter_prompt_jsons(angles_root)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("--models must include at least one entry.")
    for model in models:
        if model not in _PIPELINES:
            raise ValueError(f"Unknown pipeline model: {model}")

    for model in models:
        model_id = args.model_id or _default_model_id_for(model)
        width, height = _resolve_generation_resolution(model, model_id)
        if args.width is not None:
            width = args.width
        if args.height is not None:
            height = args.height

        num_steps, guidance_scale = _resolve_generation_defaults(model, model_id)
        if args.num_inference_steps is not None:
            num_steps = args.num_inference_steps
        if args.guidance_scale is not None:
            guidance_scale = args.guidance_scale

        torch_dtype = args.torch_dtype or "bfloat16"
        if model == "qwen" and torch_dtype == "bf8":
            logging.getLogger(__name__).warning("Qwen pipeline does not use bf8; falling back to bfloat16.")
            torch_dtype = "bfloat16"

        flux = FluxParams(
            pipeline=model,
            model_id=model_id,
            width=int(width),
            height=int(height),
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            max_sequence_length=int(args.max_sequence_length),
            torch_dtype=str(torch_dtype),
            local_files_only=bool(args.local_files_only),
        )

        sam = SamParams(
            checkpoint=Path("."),
            model_type="vit_h",
            prompt_mode="grounding",
            box_ratio_w=0.8,
            box_ratio_h=0.85,
            feather_radius_px=3,
            mask_selection_rule="center_included_max_area",
            morph_kernel_px=3,
            min_area_frac=0.01,
            max_area_frac=0.9,
            lang_sam_box_threshold=0.25,
            lang_sam_text_threshold=0.25,
        )

        composite = CompositeParams(
            anchor_x=0.5,
            anchor_y=0.9,
            target_obj_height_ratio=0.65,
            scale_multiplier=1.0,
            placement_mode="center",
        )

        color_params = ColorParams(
            target_hue_deg=None,
            min_saturation=0.25,
        )

        pipe = _load_text2image_pipeline(
            flux.pipeline,
            flux.model_id,
            flux.torch_dtype,
            args.device,
            hf_token,
            flux.local_files_only,
        )

        try:
            for prompt_path in prompt_jsons:
                animal, angle = _parse_animal_angle(prompt_path)
                images_dir = output_root / animal / angle / model / "images"
                logs_dir = output_root / animal / angle / model / "logs"
                images_dir.mkdir(parents=True, exist_ok=True)
                logs_dir.mkdir(parents=True, exist_ok=True)

                base_config = RunConfig(
                    output_dir=images_dir,
                    background_prompt=None,
                    background_only=False,
                    background_grayscale=False,
                    prompt="",
                    negative_prompt=args.negative_prompt,
                    object_name=None,
                    prompt_elements_path=prompt_path,
                    prompt_elements=None,
                    color_name="",
                    seed_bg=0,
                    seed=args.seed,
                    run_index=0,
                    seed_offset=0,
                    flux=flux,
                    sam=sam,
                    composite=composite,
                    color_params=color_params,
                    device=str(args.device),
                    hf_token=hf_token,
                )

                run_configs = _build_run_configs(base_config, prompt_path, args.num_runs, args.seed)
                _run_generation(
                    run_configs,
                    pipe=pipe,
                    flat_outputs=True,
                    flat_root=images_dir,
                    flat_logs_root=logs_dir,
                    batch_size=int(args.batch_size),
                )
        finally:
            try:
                pipe.to("cpu")
            except Exception:
                pass
            del pipe
            gc.collect()
            if args.device != "cpu":
                _release_torch_cuda()


if __name__ == "__main__":
    main()
