from __future__ import annotations

import inspect
import os
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

from .gen_mask import _select_mask, _select_mask_by_score
from ..generation.gen_utils import _get_hf_token


def _maybe_login_hf(token: Optional[str]) -> None:
    if not token:
        return
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    os.environ.setdefault("HF_TOKEN", token)
    try:
        from huggingface_hub import login
    except Exception:
        return
    try:
        login(token=token, add_to_git_credential=False)
    except Exception:
        pass


def _load_lang_sam_model(device: str, hf_token: Optional[str] = None):
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    token = _get_hf_token(hf_token)
    _maybe_login_hf(token)
    model = build_sam3_image_model(device=device)
    return Sam3Processor(model, device=device)


def _predict_sam3_output(
    processor,
    image: Image.Image,
    text_prompt: str,
    box_threshold: float,
) -> Optional[dict[str, Any]]:
    if not (hasattr(processor, "set_image") and hasattr(processor, "set_text_prompt")):
        return None
    state = processor.set_image(image)
    if hasattr(processor, "set_confidence_threshold"):
        processor.set_confidence_threshold(float(box_threshold), state)
    output = processor.set_text_prompt(prompt=text_prompt, state=state)
    if isinstance(output, dict):
        return output
    return None


def _coerce_numpy(array_like) -> Optional[np.ndarray]:
    if array_like is None:
        return None
    if hasattr(array_like, "detach"):
        array_like = array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _normalize_masks_array(masks: np.ndarray, image: Image.Image) -> Optional[np.ndarray]:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        return None
    if masks.shape[-2:] != (image.height, image.width):
        return None
    return masks


def _select_lang_sam_mask(
    masks: np.ndarray,
    scores: Optional[np.ndarray],
    center_xy: Tuple[int, int],
    mask_selection_rule: str,
) -> Optional[np.ndarray]:
    if masks.ndim == 3 and masks.shape[0] == 0:
        return None
    if scores is not None:
        scores = np.asarray(scores)
        if scores.ndim == 0:
            scores = None

    if mask_selection_rule == "center_included_max_score":
        return _select_mask_by_score(masks, scores, center_xy)

    return _select_mask(masks, center_xy)


def _select_best_score_mask(
    masks: np.ndarray,
    scores: Optional[np.ndarray],
    center_xy: Tuple[int, int],
) -> Optional[np.ndarray]:
    if masks.ndim == 2:
        return masks
    if masks.shape[0] == 0:
        return None
    if scores is None:
        return _select_mask(masks, center_xy)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 0 or scores.shape[0] != masks.shape[0]:
        return _select_mask(masks, center_xy)
    finite = np.isfinite(scores)
    if not np.any(finite):
        return _select_mask(masks, center_xy)
    best_idx = int(np.nanargmax(scores))
    return masks[best_idx]


def _predict_masks_lang_sam_with_scores(
    model,
    image: Image.Image,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    output = _predict_sam3_output(model, image, text_prompt, box_threshold)
    if output is None:
        if not hasattr(model, "predict"):
            return None
        sig = inspect.signature(model.predict)
        kwargs: dict[str, Any] = {}
        if "box_threshold" in sig.parameters:
            kwargs["box_threshold"] = box_threshold
        if "text_threshold" in sig.parameters:
            kwargs["text_threshold"] = text_threshold
        if "multimask_output" in sig.parameters:
            kwargs["multimask_output"] = True
        if "return_all_masks" in sig.parameters:
            kwargs["return_all_masks"] = True
        if "return_all" in sig.parameters:
            kwargs["return_all"] = True

        images_arg: Any = image
        texts_arg: Any = text_prompt
        if "images_pil" in sig.parameters or "texts_prompt" in sig.parameters:
            images_arg = [image]
            texts_arg = [text_prompt]

        output = model.predict(images_arg, texts_arg, **kwargs)

    def _extract_from_dict(d: dict) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        m = d.get("masks") if d.get("masks") is not None else d.get("mask")
        s = (
            d.get("mask_scores")
            if d.get("mask_scores") is not None
            else d.get("scores")
            if d.get("scores") is not None
            else d.get("logits")
        )
        m_np = _coerce_numpy(m)
        s_np = _coerce_numpy(s)
        if m_np is None:
            return None, None
        m_np = _normalize_masks_array(m_np, image)
        if m_np is None:
            return None, None

        # スコアがスカラーなら、マスク枚数分に拡張して整合を取る
        if s_np is not None and np.asarray(s_np).ndim == 0:
            s_np = np.full((m_np.shape[0],), float(np.asarray(s_np)))

        return m_np, s_np

    masks_np = None
    scores_np = None

    if isinstance(output, dict):
        masks_np, scores_np = _extract_from_dict(output)
        if masks_np is None:
            return None
        return masks_np, scores_np

    if isinstance(output, list) and output and isinstance(output[0], dict):
        masks_list: list[np.ndarray] = []
        scores_list: list[np.ndarray] = []
        any_scores = False

        for d in output:
            m_np, s_np = _extract_from_dict(d)
            if m_np is None:
                continue
            masks_list.append(m_np)
            if s_np is not None:
                any_scores = True
                scores_list.append(np.asarray(s_np, dtype=np.float32))
            else:
                scores_list.append(None)  # 後で整合処理

        if not masks_list:
            return None

        masks_np = np.concatenate(masks_list, axis=0)

        if any_scores:
            # scores_list の None を NaN 埋め（枚数不一致もここで吸収）
            flat_scores: list[float] = []
            for m_np, s_np in zip(masks_list, scores_list):
                if s_np is None:
                    flat_scores.extend([float("nan")] * m_np.shape[0])
                else:
                    s_arr = np.asarray(s_np).reshape(-1)
                    if s_arr.shape[0] == 1 and m_np.shape[0] > 1:
                        flat_scores.extend([float(s_arr[0])] * m_np.shape[0])
                    elif s_arr.shape[0] != m_np.shape[0]:
                        # どうしても不一致なら最小長に揃える（または NaN 埋め）
                        k = min(s_arr.shape[0], m_np.shape[0])
                        flat_scores.extend([float(x) for x in s_arr[:k]])
                        flat_scores.extend([float("nan")] * (m_np.shape[0] - k))
                    else:
                        flat_scores.extend([float(x) for x in s_arr])
            scores_np = np.asarray(flat_scores, dtype=np.float32)
        else:
            scores_np = None

        return masks_np, scores_np

    # それ以外の list/tuple 出力は従来ロジックを維持（ただし誤判定しやすいので必要なら後で強化）
    masks = None
    scores = None
    if isinstance(output, (list, tuple)):
        for item in output:
            if masks is None:
                arr = _coerce_numpy(item)
                if arr is not None and arr.ndim >= 2:
                    masks = arr
            if scores is None and isinstance(item, (list, tuple)):
                scores = item

    masks_np = _coerce_numpy(masks)
    scores_np = _coerce_numpy(scores)
    if masks_np is None:
        return None
    masks_np = _normalize_masks_array(masks_np, image)
    if masks_np is None:
        return None
    return masks_np, scores_np


def _predict_boxes_lang_sam_with_scores(
    model,
    image: Image.Image,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    output = _predict_sam3_output(model, image, text_prompt, box_threshold)
    if output is None:
        if not hasattr(model, "predict"):
            return None
        sig = inspect.signature(model.predict)
        kwargs: dict[str, Any] = {}
        if "box_threshold" in sig.parameters:
            kwargs["box_threshold"] = box_threshold
        if "text_threshold" in sig.parameters:
            kwargs["text_threshold"] = text_threshold
        if "return_all_boxes" in sig.parameters:
            kwargs["return_all_boxes"] = True
        if "return_all" in sig.parameters:
            kwargs["return_all"] = True

        images_arg: Any = image
        texts_arg: Any = text_prompt
        if "images_pil" in sig.parameters or "texts_prompt" in sig.parameters:
            images_arg = [image]
            texts_arg = [text_prompt]

        output = model.predict(images_arg, texts_arg, **kwargs)

    def _extract_from_dict(d: dict) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        b = d.get("boxes") if d.get("boxes") is not None else d.get("box")
        if b is None and d.get("bboxes") is not None:
            b = d.get("bboxes")
        s = (
            d.get("box_scores")
            if d.get("box_scores") is not None
            else d.get("scores")
            if d.get("scores") is not None
            else d.get("logits")
        )
        b_np = _coerce_numpy(b)
        s_np = _coerce_numpy(s)
        if b_np is None:
            return None, None
        b_np = np.asarray(b_np)
        if b_np.ndim == 1 and b_np.shape[0] == 4:
            b_np = b_np[None, :]
        if b_np.ndim != 2 or b_np.shape[1] != 4:
            return None, None

        if s_np is not None and np.asarray(s_np).ndim == 0:
            s_np = np.full((b_np.shape[0],), float(np.asarray(s_np)))

        return b_np, s_np

    if isinstance(output, dict):
        boxes_np, scores_np = _extract_from_dict(output)
        if boxes_np is None:
            return None
        return boxes_np, scores_np

    if isinstance(output, list) and output and isinstance(output[0], dict):
        boxes_list: list[np.ndarray] = []
        scores_list: list[np.ndarray] = []
        any_scores = False

        for d in output:
            b_np, s_np = _extract_from_dict(d)
            if b_np is None:
                continue
            boxes_list.append(b_np)
            if s_np is not None:
                any_scores = True
                scores_list.append(np.asarray(s_np, dtype=np.float32))
            else:
                scores_list.append(None)

        if not boxes_list:
            return None

        boxes_np = np.concatenate(boxes_list, axis=0)
        if any_scores:
            flat_scores: list[float] = []
            for b_np, s_np in zip(boxes_list, scores_list):
                if s_np is None:
                    flat_scores.extend([float("nan")] * b_np.shape[0])
                else:
                    s_arr = np.asarray(s_np).reshape(-1)
                    if s_arr.shape[0] == 1 and b_np.shape[0] > 1:
                        flat_scores.extend([float(s_arr[0])] * b_np.shape[0])
                    elif s_arr.shape[0] != b_np.shape[0]:
                        k = min(s_arr.shape[0], b_np.shape[0])
                        flat_scores.extend([float(x) for x in s_arr[:k]])
                        flat_scores.extend([float("nan")] * (b_np.shape[0] - k))
                    else:
                        flat_scores.extend([float(x) for x in s_arr])
            scores_np = np.asarray(flat_scores, dtype=np.float32)
        else:
            scores_np = None

        return boxes_np, scores_np

    return None



def _predict_mask_lang_sam(
    model,
    image: Image.Image,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    mask_selection_rule: str,
) -> Optional[np.ndarray]:
    result = _predict_masks_lang_sam_with_scores(model, image, text_prompt, box_threshold, text_threshold)
    if result is None:
        return None
    masks_np, scores_np = result
    if mask_selection_rule == "all_masks":
        center_xy = (image.width // 2, image.height // 2)
        return _select_best_score_mask(masks_np, scores_np, center_xy)
    center_xy = (image.width // 2, image.height // 2)
    return _select_lang_sam_mask(masks_np, scores_np, center_xy, mask_selection_rule)

