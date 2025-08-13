"""Enhanced dataset preparation script using the flexible data loader framework."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import yaml

from src.data_loaders import registry, combiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare toxicity datasets with flexible loaders")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--loader", type=str, help="Single loader to use")
    parser.add_argument("--list-loaders", action="store_true", help="List available loaders")
    parser.add_argument("--loader-info", type=str, help="Get info about a specific loader")
    parser.add_argument("--output", type=str, default="data/prepared_dataset.jsonl", help="Output file path")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to load")
    parser.add_argument("--strategy", type=str, default="balanced", 
                       choices=["balanced", "weighted", "sequential"], 
                       help="Combination strategy for multiple datasets")
    
    # Loader-specific arguments
    parser.add_argument("--file-path", type=str, help="File path for file-based loaders")
    parser.add_argument("--api-url", type=str, help="API URL for API loader")
    parser.add_argument("--language", type=str, default="en", help="Language for multilingual datasets")
    parser.add_argument("--toxicity-threshold", type=float, default=0.5, help="Toxicity threshold")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_samples(samples: List[Dict], output_path: str):
    """Save samples to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # List available loaders
    if args.list_loaders:
        loaders = registry.list_loaders()
        print("Available loaders:")
        for loader_name in sorted(loaders):
            try:
                info = registry.get_loader_info(loader_name)
                print(f"  {loader_name}: {info.get('description', 'No description')}")
            except:
                print(f"  {loader_name}: (info unavailable)")
        return
    
    # Get loader info
    if args.loader_info:
        try:
            info = registry.get_loader_info(args.loader_info)
            print(f"Loader: {args.loader_info}")
            print(f"Name: {info.get('name', 'Unknown')}")
            print(f"Description: {info.get('description', 'No description')}")
            print(f"Type: {info.get('type', 'Unknown')}")
            if 'source' in info:
                print(f"Source: {info['source']}")
            if 'languages' in info:
                print(f"Languages: {', '.join(info['languages'])}")
            if 'supported_formats' in info:
                print(f"Supported formats: {', '.join(info['supported_formats'])}")
            return
        except Exception as e:
            print(f"Error getting loader info: {e}")
            return
    
    # Load data
    all_samples = []
    
    if args.config:
        # Load from configuration file
        config = load_config(args.config)
        dataset_configs = []
        
        for loader_name, loader_config in config.get('loaders', {}).items():
            dataset_configs.append({
                'loader': loader_name,
                'params': loader_config,
                'max_samples': loader_config.get('max_samples', args.max_samples),
                'weight': loader_config.get('weight', 1.0)
            })
        
        if dataset_configs:
            all_samples = combiner.combine_datasets(
                dataset_configs,
                strategy=args.strategy,
                max_total_samples=args.max_samples
            )
        else:
            logger.error("No loaders configured in config file")
            return
    
    elif args.loader:
        # Load from single loader
        try:
            loader_params = {}
            
            # Add common parameters
            if args.max_samples:
                loader_params['max_samples'] = args.max_samples
            if args.file_path:
                loader_params['file_path'] = args.file_path
            if args.api_url:
                loader_params['api_url'] = args.api_url
            if args.language:
                loader_params['language'] = args.language
            if args.toxicity_threshold:
                loader_params['toxicity_threshold'] = args.toxicity_threshold
            
            loader = registry.get_loader(args.loader, **loader_params)
            all_samples = loader.load(**loader_params)
            
        except Exception as e:
            logger.error(f"Failed to load data with {args.loader}: {e}")
            return
    
    else:
        logger.error("Either --config or --loader must be specified")
        return
    
    # Filter samples
    if all_samples:
        # Apply additional filtering
        filtered_samples = []
        for sample in all_samples:
            # Basic validation
            if 'toxic_text' in sample and 'neutral_text' in sample:
                if len(sample['toxic_text'].strip()) >= 10 and len(sample['neutral_text'].strip()) >= 10:
                    filtered_samples.append(sample)
            elif 'text' in sample:
                if len(sample['text'].strip()) >= 10:
                    filtered_samples.append(sample)
        
        logger.info(f"Filtered {len(all_samples)} -> {len(filtered_samples)} samples")
        all_samples = filtered_samples
    
    # Save results
    if all_samples:
        save_samples(all_samples, args.output)
        logger.info(f"Saved {len(all_samples)} samples to {args.output}")
        
        # Print summary
        print(f"\nDataset Summary:")
        print(f"Total samples: {len(all_samples)}")
        
        # Check sample types
        pair_samples = sum(1 for s in all_samples if 'toxic_text' in s and 'neutral_text' in s)
        multimodal_samples = sum(1 for s in all_samples if 'image_path' in s or 'image' in s)
        
        if pair_samples > 0:
            print(f"Toxic-neutral pairs: {pair_samples}")
        if multimodal_samples > 0:
            print(f"Multimodal samples: {multimodal_samples}")
        
        # Show sample
        if all_samples:
            print(f"\nSample data:")
            sample = all_samples[0]
            for key, value in sample.items():
                if key == 'image':
                    print(f"  {key}: <PIL.Image>")
                else:
                    value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"  {key}: {value_str}")
    
    else:
        logger.warning("No samples loaded")


if __name__ == "__main__":
    main()