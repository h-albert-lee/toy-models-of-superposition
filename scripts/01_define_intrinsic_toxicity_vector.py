"""Compute and save the intrinsic toxicity vector (ITV)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from src.vlm_wrapper import VLM_Wrapper
from src.feature_extractor import VectorExtractor


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    wrapper = VLM_Wrapper("llava-hf/llava-1.5-7b-hf")
    extractor = VectorExtractor(wrapper, layer="language_model.layers.0")

    raw = load_jsonl("data/intrinsic_toxicity_cases.jsonl")
    data = []
    for item in raw:
        img = Image.open(item["image_path"]).convert("RGB")
        data.append({"text": item["text"], "image": img})

    itv = extractor.compute_itv(data)

    Path("results").mkdir(exist_ok=True)
    torch.save(itv, "results/intrinsic_toxicity_vector.pt")


if __name__ == "__main__":
    main()

