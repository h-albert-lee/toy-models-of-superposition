#!/usr/bin/env python3
"""
Prepare benchmark datasets using the recommended configurations.

This script uses the standardized benchmark datasets:
- VLM benchmark: oneonlee/Meme-Safety-Bench (multimodal)
- Text benchmark: textdetox/multilingual_toxicity_dataset (English subset)
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.data_loaders import registry


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets")
    parser.add_argument("--config", type=str, default="configs/benchmark_config.yaml",
                       help="Benchmark configuration file")
    parser.add_argument("--strategy", type=str, default="benchmark_only",
                       choices=["benchmark_only", "comprehensive", "text_only", "multimodal_only"],
                       help="Dataset combination strategy")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Maximum samples per dataset")
    parser.add_argument("--output-dir", type=str, default="data/benchmark",
                       help="Output directory for prepared datasets")
    parser.add_argument("--language", type=str, default="en",
                       help="Language for multilingual datasets")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load benchmark configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_text_benchmark(max_samples: int, language: str, output_dir: Path):
    """Prepare text-only benchmark using MultilingualToxicityLoader."""
    print("Preparing text benchmark...")
    
    try:
        loader = registry.get_loader("MultilingualToxicityLoader")
        samples = loader.load(max_samples=max_samples, language=language)
        
        if samples:
            output_file = output_dir / "text_benchmark.jsonl"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"✓ Saved {len(samples)} text benchmark samples to {output_file}")
            return len(samples)
        else:
            print("✗ No text samples loaded")
            return 0
            
    except Exception as e:
        print(f"✗ Error preparing text benchmark: {e}")
        return 0


def prepare_vlm_benchmark(max_samples: int, output_dir: Path):
    """Prepare VLM benchmark using MemeSafetyBenchLoader."""
    print("Preparing VLM benchmark...")
    
    try:
        loader = registry.get_loader("MemeSafetyBenchLoader")
        samples = loader.load(max_samples=max_samples)
        
        if samples:
            output_file = output_dir / "vlm_benchmark.jsonl"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    # Handle PIL images by converting to path or description
                    sample_copy = sample.copy()
                    if 'image' in sample_copy:
                        sample_copy['image'] = "<PIL.Image>"  # Placeholder
                    f.write(json.dumps(sample_copy, ensure_ascii=False) + '\n')
            
            print(f"✓ Saved {len(samples)} VLM benchmark samples to {output_file}")
            return len(samples)
        else:
            print("✗ No VLM samples loaded (dataset may be gated)")
            return 0
            
    except Exception as e:
        print(f"✗ Error preparing VLM benchmark: {e}")
        return 0


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("Benchmark Dataset Preparation")
    print("="*60)
    print(f"Strategy: {args.strategy}")
    print(f"Max samples: {args.max_samples}")
    print(f"Language: {args.language}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    output_dir = Path(args.output_dir)
    total_samples = 0
    
    # Prepare datasets based on strategy
    if args.strategy in ["benchmark_only", "comprehensive", "text_only"]:
        text_samples = prepare_text_benchmark(args.max_samples, args.language, output_dir)
        total_samples += text_samples
    
    if args.strategy in ["benchmark_only", "comprehensive", "multimodal_only"]:
        vlm_samples = prepare_vlm_benchmark(args.max_samples, output_dir)
        total_samples += vlm_samples
    
    # Create summary
    summary = {
        "strategy": args.strategy,
        "max_samples_per_dataset": args.max_samples,
        "language": args.language,
        "total_samples": total_samples,
        "datasets_prepared": []
    }
    
    if args.strategy in ["benchmark_only", "comprehensive", "text_only"]:
        summary["datasets_prepared"].append({
            "name": "text_benchmark",
            "loader": "MultilingualToxicityLoader",
            "source": "textdetox/multilingual_toxicity_dataset",
            "samples": text_samples if 'text_samples' in locals() else 0
        })
    
    if args.strategy in ["benchmark_only", "comprehensive", "multimodal_only"]:
        summary["datasets_prepared"].append({
            "name": "vlm_benchmark", 
            "loader": "MemeSafetyBenchLoader",
            "source": "oneonlee/Meme-Safety-Bench",
            "samples": vlm_samples if 'vlm_samples' in locals() else 0
        })
    
    # Save summary
    summary_file = output_dir / "benchmark_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print()
    print("="*60)
    print("Preparation Summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Datasets prepared: {len(summary['datasets_prepared'])}")
    for dataset in summary["datasets_prepared"]:
        print(f"    - {dataset['name']}: {dataset['samples']} samples")
    print(f"  Summary saved to: {summary_file}")
    print("="*60)


if __name__ == "__main__":
    main()