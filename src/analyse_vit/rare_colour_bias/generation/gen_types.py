from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class FluxParams:
    pipeline: str
    model_id: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    max_sequence_length: int
    torch_dtype: str
    local_files_only: bool


@dataclass
class SamParams:
    checkpoint: Path
    model_type: str
    prompt_mode: str
    box_ratio_w: float
    box_ratio_h: float
    feather_radius_px: int
    mask_selection_rule: str
    morph_kernel_px: int
    min_area_frac: float
    max_area_frac: float
    lang_sam_box_threshold: float
    lang_sam_text_threshold: float


@dataclass
class CompositeParams:
    anchor_x: float
    anchor_y: float
    target_obj_height_ratio: float
    scale_multiplier: float
    placement_mode: str


@dataclass
class ColorParams:
    target_hue_deg: Optional[float]
    min_saturation: float


@dataclass
class RunConfig:
    output_dir: Path
    background_prompt: Optional[str]
    prompt: str
    negative_prompt: Optional[str]
    object_name: Optional[str]
    prompt_elements_path: Optional[Path]
    prompt_elements: Optional[Dict[str, str]]
    color_name: str
    seed_bg: int
    seed: int
    run_index: int
    seed_offset: int
    flux: FluxParams
    sam: SamParams
    composite: CompositeParams
    color_params: ColorParams
    device: str
    hf_token: Optional[str]


@dataclass(frozen=True)
class RunDirs:
    root: Path
    inputs: Path
    masks: Path
    outputs: Path
    meta: Path


@dataclass
class TransformInfo:
    bbox: Tuple[int, int, int, int]
    scale: float
    anchor_x: float
    anchor_y: float
    paste_x: int
    paste_y: int
    scaled_size: Tuple[int, int]


@dataclass(frozen=True)
class ColorTransform:
    hue_deg: Optional[float]
    desaturate: bool
    value_scale: Optional[float]
    value_lift: Optional[float]

_PIPELINES: Dict[str, Dict[str, Any]] = {
    "flux": {
        "default_model_id": "black-forest-labs/FLUX.1-dev",
        "defaults": {"num_inference_steps": 50, "guidance_scale": 3.5, "resolution": (1024, 1024)},
        "model_overrides": {
            # Intentionally empty: default model matches pipeline defaults.
        },
    },
    "sd3": {
        "default_model_id": "stabilityai/stable-diffusion-3.5-large",
        "defaults": {"num_inference_steps": 28, "guidance_scale": 4.5, "resolution": (1024, 1024)},
        "model_overrides": {
            "stabilityai/stable-diffusion-3.5-large": {
                "num_inference_steps": 30,
                "guidance_scale": 5.0,
                "resolution": (1024, 1024),
            }
        },
    },
    "qwen": {
        "default_model_id": "Qwen/Qwen-Image-2512",
        "defaults": {"num_inference_steps": 50, "guidance_scale": 4.0, "resolution": (1328, 1328)},
        "model_overrides": {
            # Intentionally empty: default model matches pipeline defaults.
        },
    },
}


_COLOR_TRANSFORMS_CANONICAL: Dict[str, ColorTransform] = {
    "red": ColorTransform(hue_deg=0.0, desaturate=False, value_scale=None, value_lift=None),
    "orange": ColorTransform(hue_deg=30.0, desaturate=False, value_scale=None, value_lift=None),
    "yellow": ColorTransform(hue_deg=60.0, desaturate=False, value_scale=None, value_lift=None),
    "green": ColorTransform(hue_deg=120.0, desaturate=False, value_scale=None, value_lift=None),
    "cyan": ColorTransform(hue_deg=180.0, desaturate=False, value_scale=None, value_lift=None),
    "blue": ColorTransform(hue_deg=210.0, desaturate=False, value_scale=None, value_lift=None),
    "purple": ColorTransform(hue_deg=270.0, desaturate=False, value_scale=None, value_lift=None),
    "magenta": ColorTransform(hue_deg=300.0, desaturate=False, value_scale=None, value_lift=None),
    "pink": ColorTransform(hue_deg=330.0, desaturate=False, value_scale=None, value_lift=None),
    "gray": ColorTransform(hue_deg=None, desaturate=True, value_scale=0.6, value_lift=None),
    "black": ColorTransform(hue_deg=None, desaturate=True, value_scale=0.25, value_lift=None),
    "white": ColorTransform(hue_deg=None, desaturate=True, value_scale=None, value_lift=0.25),
}


_COLOR_ALIASES: Dict[str, str] = {
    "grey": "gray",
    "brown": "orange",
}
