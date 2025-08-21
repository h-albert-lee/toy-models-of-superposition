"""Run toxicity vector analysis across multiple layers and evaluate the best."""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-layer experiments")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    return parser.parse_args()


def run_script(script: str, config_path: Path) -> None:
    subprocess.run([
        "python",
        script,
        "--config",
        str(config_path),
    ], check=True)


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    layers = cfg.get("layers", [cfg["extraction_layer"]])
    base_out_dir = Path(cfg["output_dir"])
    base_out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for layer in layers:
        logger.info("Processing layer %s", layer)
        layer_cfg = deepcopy(cfg)
        layer_cfg["extraction_layer"] = layer
        layer_cfg["intervention_layer"] = layer
        layer_dir = base_out_dir / layer.replace(".", "_")
        layer_dir.mkdir(parents=True, exist_ok=True)
        layer_cfg["output_dir"] = str(layer_dir)
        temp_cfg_path = layer_dir / "config.yaml"
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(layer_cfg, f)

        run_script("scripts/00_define_general_toxicity_vector.py", temp_cfg_path)
        run_script("scripts/01_define_intrinsic_toxicity_vector.py", temp_cfg_path)

        gtv = torch.load(layer_dir / "general_toxicity_vector.pt")
        itv = torch.load(layer_dir / "intrinsic_toxicity_vector.pt")
        sim = F.cosine_similarity(gtv.flatten(), itv.flatten(), dim=0).item()
        results.append({"layer": layer, "cosine_similarity": sim})
        logger.info("Layer %s cosine similarity %.4f", layer, sim)

    result_path = base_out_dir / "layer_cosine_results.csv"
    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "cosine_similarity"])
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved results to %s", result_path)

    best = max(results, key=lambda x: x["cosine_similarity"])
    logger.info(
        "Best layer %s with cosine similarity %.4f",
        best["layer"],
        best["cosine_similarity"],
    )

    best_cfg = deepcopy(cfg)
    best_layer_dir = base_out_dir / best["layer"].replace(".", "_")
    best_cfg["intervention_layer"] = best["layer"]
    best_cfg["output_dir"] = str(best_layer_dir)
    best_cfg_path = best_layer_dir / "best_config.yaml"
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f)

    logger.info("정밀 개입 평가를 시작합니다...")
    run_script("scripts/04_evaluate_precision_intervention.py", best_cfg_path)


if __name__ == "__main__":
    main()
