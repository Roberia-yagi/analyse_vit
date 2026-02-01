from __future__ import annotations

import json
import random
from typing import Dict, Iterable, Tuple

from analyse_vit.rare_colour_bias.generation.gen_utils import _derive_run_seed


def _load_prompt_elements(path) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Prompt elements JSON must be a non-empty object: {path}")
    keys: list[str] = []
    groups: list[Tuple[str, ...]] = []
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"Prompt elements JSON keys must be strings: {path}")
        if not isinstance(value, list) or not value:
            raise ValueError(f"Prompt elements JSON values must be non-empty lists: {path}")
        if not all(isinstance(item, str) and item for item in value):
            raise ValueError(f"Prompt elements JSON lists must contain non-empty strings: {path}")
        keys.append(key)
        groups.append(tuple(value))
    return tuple(keys), tuple(groups)


def _count_combinations(groups: Tuple[Tuple[str, ...], ...]) -> int:
    total = 1
    for group in groups:
        total *= len(group)
    return total


def _prompt_from_index(
    keys: Tuple[str, ...],
    groups: Tuple[Tuple[str, ...], ...],
    index: int,
    joiner: str,
) -> Tuple[str, Dict[str, str]]:
    lengths = [len(group) for group in groups]
    selections: list[str] = [""] * len(lengths)
    for pos in range(len(groups) - 1, -1, -1):
        size = lengths[pos]
        selections[pos] = groups[pos][index % size]
        index //= size
    prompt = joiner.join(selections)
    chosen = {keys[i]: selections[i] for i in range(len(keys))}
    return prompt, chosen


def _generate_prompt_set(
    keys: Tuple[str, ...],
    groups: Tuple[Tuple[str, ...], ...],
    num_runs: int,
    base_seed: int,
    salt: int,
    joiner: str,
    label: str,
) -> Tuple[list[str], list[Dict[str, str]]]:
    total = _count_combinations(groups)
    if num_runs > total:
        raise ValueError(f"{label} prompts require {num_runs} unique combinations, but only {total} are available.")
    rng = random.Random(_derive_run_seed(base_seed, 0, salt))
    indices = rng.sample(range(total), num_runs)
    prompts: list[str] = []
    selections: list[Dict[str, str]] = []
    for index in indices:
        prompt, chosen = _prompt_from_index(keys, groups, index, joiner)
        prompts.append(prompt)
        selections.append(chosen)
    return prompts, selections
