"""Compute and save the intrinsic toxicity vector (ITV)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from PIL import Image

from src.vlm_wrapper import VLM_Wrapper
from src.feature_extractor import VectorExtractor


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the intrinsic toxicity vector")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    wrapper = VLM_Wrapper(cfg["model_name"], device=cfg.get("device", "cuda"), **cfg.get("model_kwargs", {}))
    extractor = VectorExtractor(wrapper, layer=cfg["extraction_layer"], batch_size=cfg.get("batch_size", 8))

    raw = load_jsonl(cfg["data_sources"]["intrinsic_toxicity_cases"])
    data = []
    for item in raw:
        img = Image.open(item[cfg.get("image_key", "image_path")]).convert("RGB")
        data.append({"text": item["text"], "image": img})

    itv = extractor.compute_itv(data)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "intrinsic_toxicity_vector.pt"
    torch.save(itv, out_path)
    logger.info("Saved ITV to %s", out_path)


if __name__ == "__main__":
    main()

