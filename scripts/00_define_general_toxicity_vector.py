"""Compute and save the general toxicity vector (GTV)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml

from src.vlm_wrapper import VLM_Wrapper
from src.feature_extractor import VectorExtractor


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the general toxicity vector")
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

    data = load_jsonl(cfg["data_sources"]["toxic_neutral_pairs"])
    gtv = extractor.compute_gtv(data)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "general_toxicity_vector.pt"
    torch.save(gtv, out_path)
    logger.info("Saved GTV to %s", out_path)


if __name__ == "__main__":
    main()

