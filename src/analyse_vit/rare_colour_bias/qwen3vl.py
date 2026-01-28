#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import contextlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw

LOGGER = logging.getLogger("analyse_vit.qwen3vl.raw_vision_embedding")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _make_dummy_image(size: int = 448) -> Image.Image:
    img = Image.new("RGB", (size, size), (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([size // 8, size // 8, size * 7 // 8, size * 7 // 8], outline=(0, 0, 0), width=3)
    d.line([0, 0, size, size], fill=(0, 0, 0), width=2)
    d.line([0, size, size, 0], fill=(0, 0, 0), width=2)
    d.text((size // 10, size // 10), "dummy", fill=(0, 0, 0))
    return img


def _coerce_tensor(output: object) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        for item in output:
            if isinstance(item, torch.Tensor) and item.dim() == 3:
                return item
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
        return None
    if isinstance(output, dict):
        vals = list(output.values())
        for item in vals:
            if isinstance(item, torch.Tensor) and item.dim() == 3:
                return item
        for item in vals:
            if isinstance(item, torch.Tensor):
                return item
        return None
    for attr in ("last_hidden_state", "hidden_states", "output"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _get_by_path(root: object, path: str) -> object:
    cur = root
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _collect_modules_by_paths(root: torch.nn.Module, paths: Sequence[str]) -> List[Tuple[str, torch.nn.Module]]:
    found: List[Tuple[str, torch.nn.Module]] = []
    for p in paths:
        try:
            obj = _get_by_path(root, p)
        except Exception:
            continue
        if isinstance(obj, torch.nn.Module):
            found.append((p, obj))
    return found


def _resolve_unique_module(root: torch.nn.Module, label: str, paths: Sequence[str]) -> Tuple[str, torch.nn.Module]:
    found = _collect_modules_by_paths(root, paths)
    _require(len(found) >= 1, f"{label} が見つかりません: tried={list(paths)}")

    # 同一オブジェクト（別名）を潰して一意性を判定
    unique_by_id: Dict[int, Tuple[str, torch.nn.Module]] = {}
    for p, m in found:
        unique_by_id.setdefault(id(m), (p, m))

    if len(unique_by_id) == 1:
        return next(iter(unique_by_id.values()))

    raise RuntimeError(
        f"{label} を一意に特定できません: found={len(found)} unique_objects={len(unique_by_id)} "
        f"paths={[p for p, _ in found]}"
    )


def _resolve_unique_blocks(vision_tower: torch.nn.Module) -> Tuple[str, torch.nn.ModuleList]:
    candidates: List[Tuple[str, torch.nn.ModuleList]] = []

    def _try(path: str) -> None:
        try:
            obj = _get_by_path(vision_tower, path)
        except Exception:
            return
        if isinstance(obj, torch.nn.ModuleList):
            candidates.append((path, obj))

    _try("blocks")
    _try("vision_model.blocks")
    _try("visual.blocks")
    _try("encoder.blocks")

    _require(
        len(candidates) == 1,
        f"vision tower の blocks を一意に特定できません: found={len(candidates)} paths={[p for p, _ in candidates]}",
    )
    return candidates[0]


def _filter_kwargs_for_callable(func: object, kwargs: Dict[str, object]) -> Dict[str, object]:
    try:
        sig = inspect.signature(func)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _first_model_device(model: torch.nn.Module) -> torch.device:
    dev_map = getattr(model, "hf_device_map", None)
    if isinstance(dev_map, dict):
        for v in dev_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                return torch.device(v)
    return next(model.parameters()).device


def _token_lengths_from_grid_thw(grid_thw: torch.Tensor, merge_size: int) -> List[int]:
    _require(grid_thw.dim() == 2 and grid_thw.shape[-1] == 3, f"grid_thw の形状が不正です: {tuple(grid_thw.shape)}")
    _require(merge_size >= 1, f"merge_size が不正です: {merge_size}")

    ms2 = merge_size * merge_size
    lengths: List[int] = []
    for row in grid_thw:
        t = int(row[0].item())
        h = int(row[1].item())
        w = int(row[2].item())
        _require(t >= 1 and h >= 1 and w >= 1, f"grid_thw の値が不正です: (t,h,w)=({t},{h},{w})")
        _require((h * w) % ms2 == 0, f"(h*w) が merge_size^2 で割り切れません: h={h} w={w} merge_size={merge_size}")
        lengths.append(t * (h * w // ms2))
    return lengths


def _resolve_image_token(processor: object) -> str:
    image_token = getattr(processor, "image_token", None)
    if image_token:
        return str(image_token)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        image_token = getattr(tokenizer, "image_token", None)
        if image_token:
            return str(image_token)
    raise RuntimeError("processor/tokenizer から image_token を取得できません。")


@dataclass(frozen=True)
class ExtractResult:
    embedding: torch.Tensor  # [B, 1152]
    token_tensor_shape: Tuple[int, ...]
    blocks_path: str
    tower_path: str
    merge_size: int


def extract_raw_vision_embedding_meanpool_1152(
    model_id: str,
    images: Sequence[Image.Image],
    device: str = "cuda",
    batch_size: int = 4,
    local_files_only: bool = False,
) -> ExtractResult:
    _require(len(images) > 0, "images が空です。")

    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(f"transformers の import に失敗しました: {exc!r}") from exc

    if device not in ("cuda", "cpu"):
        raise RuntimeError(f"device は cuda/cpu のみ対応です: got={device!r}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    image_token = _resolve_image_token(processor)

    vision_cfg = getattr(getattr(model, "config", None), "vision_config", None)
    _require(vision_cfg is not None, "model.config.vision_config が見つかりません。")
    expected_dim = int(getattr(vision_cfg, "hidden_size", -1))
    merge_size = int(getattr(vision_cfg, "merge_size", 1))

    _require(expected_dim == 1152, f"vision hidden_size が想定(1152)と一致しません: {expected_dim}")
    _require(merge_size >= 1, f"merge_size が不正です: {merge_size}")

    tower_path, vision_tower = _resolve_unique_module(
        model,
        "vision tower",
        paths=[
            "model.vision_tower",
            "vision_tower",
            "model.visual",
            "visual",
            "model.vision_model",
            "vision_model",
        ],
    )

    blocks_path, blocks = _resolve_unique_blocks(vision_tower)
    _require(len(blocks) >= 1, "vision tower blocks が空です。")
    last_block = blocks[-1]
    last_block_label = f"{tower_path}.{blocks_path}[{len(blocks)-1}]"

    captured: List[torch.Tensor] = []

    def _hook(_m: torch.nn.Module, _inp: Tuple[object, ...], out: object) -> None:
        t = _coerce_tensor(out)
        if t is not None:
            captured.append(t)

    handle = last_block.register_forward_hook(_hook)

    use_amp = (device == "cuda")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

    try:
        with torch.no_grad(), autocast_ctx:
            outs: List[torch.Tensor] = []
            token_shape: Optional[Tuple[int, ...]] = None
            dev = _first_model_device(model)

            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]

                # Qwen3VLProcessor は text 前提のため、必ず text を渡す
                texts = [image_token] * len(batch)
                inputs = processor(text=texts, images=batch, return_tensors="pt")

                _require("pixel_values" in inputs, "processor 出力に pixel_values がありません。")
                _require("image_grid_thw" in inputs, "processor 出力に image_grid_thw がありません。")

                pixel_values = inputs["pixel_values"]
                image_grid_thw = inputs["image_grid_thw"]

                _require(hasattr(model, "get_image_features"), "モデルに get_image_features がありません。")

                captured.clear()
                call_kwargs: Dict[str, object] = {
                    "pixel_values": pixel_values.to(dev) if torch.is_tensor(pixel_values) else pixel_values,
                    "image_grid_thw": image_grid_thw.to(dev) if torch.is_tensor(image_grid_thw) else image_grid_thw,
                }
                call_kwargs = _filter_kwargs_for_callable(model.get_image_features, call_kwargs)  # type: ignore[arg-type]
                _ = model.get_image_features(**call_kwargs)  # type: ignore[misc]

                _require(len(captured) >= 1, f"vision 最終 block 出力を捕捉できませんでした: {last_block_label}")
                seq = captured[-1]

                if token_shape is None:
                    token_shape = tuple(seq.shape)

                # ここで採用する特徴量は 1 種類のみ（mean pooling）。成立しなければ例外。
                if seq.dim() == 3:
                    _require(seq.shape[0] == len(batch), f"3D tokens の batch 次元が不正です: {tuple(seq.shape)}")
                    emb = seq.mean(dim=1)
                elif seq.dim() == 2:
                    _require(torch.is_tensor(image_grid_thw), "2D tokens なのに image_grid_thw が Tensor ではありません。")
                    lengths = _token_lengths_from_grid_thw(image_grid_thw, merge_size=merge_size)
                    total = int(sum(lengths))
                    _require(
                        seq.size(0) == total,
                        f"トークン数が一致しません: seq_tokens={seq.size(0)}, expected={total} "
                        f"(merge_size={merge_size}, grid_thw={image_grid_thw.cpu().tolist()})",
                    )
                    chunks = torch.split(seq, lengths, dim=0)
                    emb = torch.stack([c.mean(dim=0) for c in chunks], dim=0)
                else:
                    raise RuntimeError(f"tokens 次元が不正です: dim={seq.dim()} shape={tuple(seq.shape)}")

                _require(
                    emb.dim() == 2 and emb.shape[1] == expected_dim,
                    f"取得 embedding 次元が期待と違います: {tuple(emb.shape)} expected_dim={expected_dim}",
                )
                outs.append(emb.float().detach().cpu())

            _require(token_shape is not None, "token_shape の取得に失敗しました。")
            return ExtractResult(
                embedding=torch.cat(outs, dim=0),
                token_tensor_shape=token_shape,
                blocks_path=last_block_label,
                tower_path=tower_path,
                merge_size=merge_size,
            )
    finally:
        handle.remove()


def _load_images(paths: Sequence[Path]) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(img)
    return imgs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")

    # 画像は任意。未指定ならダミーを生成する。
    parser.add_argument("--image", type=str, nargs="*", default=None)
    parser.add_argument("--num-dummy", type=int, default=1)
    parser.add_argument("--dummy-size", type=int, default=448)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.image and len(args.image) > 0:
        images = _load_images([Path(x) for x in args.image])
    else:
        _require(args.num_dummy >= 1, f"--num-dummy は 1 以上にしてください: {args.num_dummy}")
        _require(args.dummy_size >= 32, f"--dummy-size が小さすぎます: {args.dummy_size}")
        images = [_make_dummy_image(size=args.dummy_size) for _ in range(args.num_dummy)]

    result = extract_raw_vision_embedding_meanpool_1152(
        model_id=args.model_id,
        images=images,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )

    emb = result.embedding
    LOGGER.info("Captured from: %s", result.blocks_path)
    LOGGER.info("merge_size: %d", result.merge_size)
    LOGGER.info("Token tensor shape (captured): %s", result.token_tensor_shape)
    LOGGER.info("Embedding shape: %s", tuple(emb.shape))
    with torch.no_grad():
        norms = emb.float().norm(dim=-1)
    LOGGER.info("Embedding norms: %s", norms.tolist())

    print("ok")
    print(f"emb.shape={tuple(emb.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
