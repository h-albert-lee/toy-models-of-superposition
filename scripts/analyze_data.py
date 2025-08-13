"""Script to analyze dataset quality and characteristics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data_analysis import DataQualityAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze toxicity dataset quality")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset JSONL file")
    parser.add_argument("--output_dir", type=str, default="analysis", help="Output directory for analysis")
    parser.add_argument("--compare", nargs="+", help="Additional datasets to compare with")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    analyzer = DataQualityAnalyzer()
    
    # Single dataset analysis
    logger.info(f"Analyzing dataset: {args.data_path}")
    results = analyzer.analyze_dataset(args.data_path, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET ANALYSIS SUMMARY")
    print("="*50)
    
    basic_stats = results["basic_stats"]
    print(f"Total samples: {basic_stats['total_samples']}")
    print(f"Toxic-neutral pairs: {basic_stats['has_toxic_text']}")
    print(f"Multimodal cases: {basic_stats['has_image_path']}")
    
    if "quality_issues" in results:
        issues = results["quality_issues"]
        print(f"\nQuality Issues:")
        for issue, count in issues.items():
            if count > 0:
                print(f"  - {issue.replace('_', ' ').title()}: {count}")
    
    if "similarity_stats" in results and "mean_similarity" in results["similarity_stats"]:
        sim_stats = results["similarity_stats"]
        print(f"\nSimilarity Analysis:")
        print(f"  - Mean similarity: {sim_stats['mean_similarity']:.3f}")
        print(f"  - High similarity pairs (>0.8): {sim_stats['high_similarity_pairs']}")
        print(f"  - Low similarity pairs (<0.2): {sim_stats['low_similarity_pairs']}")
    
    # Comparison analysis if requested
    if args.compare:
        logger.info("Performing comparison analysis...")
        all_datasets = [args.data_path] + args.compare
        comparison_results = analyzer.compare_datasets(all_datasets, args.output_dir + "_comparison")
        
        print(f"\n" + "="*50)
        print("DATASET COMPARISON")
        print("="*50)
        
        for name, stats in comparison_results["basic_stats"].items():
            print(f"{name}: {stats['total_samples']} samples")
    
    print(f"\nDetailed analysis saved to: {args.output_dir}")
    print("Check the generated plots and JSON report for more insights.")


if __name__ == "__main__":
    main()