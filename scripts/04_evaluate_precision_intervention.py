"""Evaluate the effect of intervention using an external toxicity classifier."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from PIL import Image
from transformers import pipeline
import torch
import yaml

from src.vlm_wrapper import VLM_Wrapper
from src.interventions import vector_subtraction


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate precision intervention")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("정밀 개입 평가를 시작합니다...")

    wrapper = VLM_Wrapper(cfg["model_name"], device=cfg.get("device", "cuda"), **cfg.get("model_kwargs", {}))
    layer = cfg["intervention_layer"]
    itv = torch.load(Path(cfg["output_dir"]) / "intrinsic_toxicity_vector.pt")

    cases = load_jsonl(cfg["data_sources"]["intrinsic_toxicity_cases"])
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
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "precision_intervention.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved evaluation to %s", out_path)


if __name__ == "__main__":
    main()

