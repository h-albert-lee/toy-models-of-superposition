from __future__ import annotations
from typing import List, Tuple, Dict, Any
from pathlib import Path
import json
from datasets import load_dataset
from PIL import Image


def load_pairs(source: str) -> Tuple[List[str], List[str]]:
    """Load paired toxic/neutral prompts from a local file or HF dataset.

    The ``source`` argument accepts either a path to a JSONL file with
    ``{"toxic": ..., "neutral": ...}`` entries or a Hugging Face dataset
    identifier optionally suffixed with ``:split``.
    """
    path = Path(source)
    toxic: List[str] = []
    neutral: List[str] = []
    if path.exists():
        with path.open("r") as f:
            for line in f:
                item = json.loads(line)
                toxic.append(item["toxic"])
                neutral.append(item["neutral"])
        return toxic, neutral

    if ":" in source:
        dataset_name, split = source.split(":", 1)
    else:
        dataset_name, split = source, "train"
    ds = load_dataset(dataset_name, split=split)
    for ex in ds:
        toxic.append(ex["toxic"])
        neutral.append(ex["neutral"])
    return toxic, neutral


def load_cases(source: str) -> List[Dict[str, Any]]:
    """Load multimodal test cases from a JSONL file or HF dataset.

    Each returned dict will contain at least a ``prompt`` key and optional
    ``image`` entry containing a :class:`PIL.Image.Image`.
    """
    path = Path(source)
    cases: List[Dict[str, Any]] = []
    if path.exists():
        with path.open("r") as f:
            for line in f:
                item = json.loads(line)
                img_entry = item.get("image") or item.get("image_path")
                if isinstance(img_entry, str):
                    item["image"] = Image.open(img_entry).convert("RGB")
                else:
                    item["image"] = img_entry
                cases.append(item)
        return cases

    if ":" in source:
        dataset_name, split = source.split(":", 1)
    else:
        dataset_name, split = source, "train"
    ds = load_dataset(dataset_name, split=split)
    for ex in ds:
        item = dict(ex)
        img_entry = item.get("image") or item.get("image_path")
        if isinstance(img_entry, str):
            img_entry = Image.open(img_entry).convert("RGB")
        item["image"] = img_entry
        cases.append(item)
    return cases
