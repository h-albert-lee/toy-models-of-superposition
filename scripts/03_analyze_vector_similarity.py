"""Analyze similarity between GTV and ITV vectors."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze vector similarity")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    out_dir = Path(cfg["output_dir"])
    gtv = torch.load(out_dir / "general_toxicity_vector.pt")
    itv = torch.load(out_dir / "intrinsic_toxicity_vector.pt")

    sim = F.cosine_similarity(gtv.flatten(), itv.flatten(), dim=0)
    logger.info("Cosine similarity: %.4f", sim.item())


if __name__ == "__main__":
    main()

