"""Evaluate the effect of intervention using an external toxicity classifier."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from transformers import pipeline
import torch

from src.vlm_wrapper import VLM_Wrapper
from src.interventions import vector_subtraction


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    wrapper = VLM_Wrapper("llava-hf/llava-1.5-7b-hf")
    layer = "language_model.layers.0"
    itv = torch.load("results/intrinsic_toxicity_vector.pt")

    cases = load_jsonl("data/intrinsic_toxicity_cases.jsonl")
    clf = pipeline("text-classification", model="unitary/unbiased-toxic-roberta")

    records = []
    for case in cases:
        img = Image.open(case["image_path"]).convert("RGB")
        prompt = case["text"]

        original = wrapper.generate(prompt, image=img)

        def intervention_fn(act: torch.Tensor) -> torch.Tensor:
            return vector_subtraction(act, itv)

        intervened = wrapper.generate_with_intervention(
            prompt, layer=layer, intervention_fn=intervention_fn, image=img
        )

        orig_score = clf(original)[0]["score"]
        int_score = clf(intervened)[0]["score"]

        records.append(
            {
                "prompt": prompt,
                "original": original,
                "intervened": intervened,
                "original_score": orig_score,
                "intervened_score": int_score,
            }
        )

    df = pd.DataFrame(records)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/precision_intervention.csv", index=False)


if __name__ == "__main__":
    main()

