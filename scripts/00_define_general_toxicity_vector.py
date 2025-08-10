"""Compute and save the general toxicity vector (GTV)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.vlm_wrapper import VLM_Wrapper
from src.feature_extractor import VectorExtractor


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    wrapper = VLM_Wrapper("llava-hf/llava-1.5-7b-hf")
    extractor = VectorExtractor(wrapper, layer="language_model.layers.0")

    data = load_jsonl("data/general_toxic_neutral_text_pairs.jsonl")
    gtv = extractor.compute_gtv(data)

    Path("results").mkdir(exist_ok=True)
    torch.save(gtv, "results/general_toxicity_vector.pt")


if __name__ == "__main__":
    main()

