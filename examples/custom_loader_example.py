"""Example of creating and using a custom data loader."""

from src.data_loaders import registry, BaseDataLoader
from typing import Dict, List, Any
import random


class SyntheticToxicityLoader(BaseDataLoader):
    """Example custom loader that generates synthetic toxicity data."""
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Synthetic Toxicity Generator",
            "description": "Generate synthetic toxic-neutral pairs for testing",
            "type": "synthetic",
            "languages": ["en"],
            "features": ["configurable_templates", "random_generation"]
        }
    
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        max_samples = kwargs.get('max_samples', 100)
        
        # Synthetic toxic templates
        toxic_templates = [
            "You are such a {adjective} {noun}!",
            "I hate {target} like you",
            "Go {action} yourself",
            "This is {adjective} {noun}",
            "{target} are so {adjective}"
        ]
        
        # Neutral counterparts
        neutral_templates = [
            "You are a wonderful {noun}!",
            "I appreciate {target} like you",
            "Have a great day",
            "This is excellent {noun}",
            "{target} are very {adjective}"
        ]
        
        # Word lists
        adjectives_toxic = ["stupid", "horrible", "disgusting", "worthless", "pathetic"]
        adjectives_neutral = ["smart", "wonderful", "amazing", "valuable", "impressive"]
        nouns = ["person", "individual", "human", "being", "soul"]
        targets = ["people", "individuals", "humans", "folks", "everyone"]
        actions = ["hurt", "harm", "leave", "abandon", "ignore"]
        
        samples = []
        
        for i in range(max_samples):
            # Pick random templates
            toxic_template = random.choice(toxic_templates)
            neutral_template = random.choice(neutral_templates)
            
            # Fill in templates
            toxic_text = toxic_template.format(
                adjective=random.choice(adjectives_toxic),
                noun=random.choice(nouns),
                target=random.choice(targets),
                action=random.choice(actions)
            )
            
            neutral_text = neutral_template.format(
                adjective=random.choice(adjectives_neutral),
                noun=random.choice(nouns),
                target=random.choice(targets),
                action=random.choice(actions)
            )
            
            samples.append({
                "toxic_text": toxic_text,
                "neutral_text": neutral_text,
                "category": "synthetic",
                "template_id": i % len(toxic_templates)
            })
        
        self.logger.info(f"Generated {len(samples)} synthetic samples")
        return samples


def main():
    """Example usage of custom loader."""
    
    # Register the custom loader
    registry.register("synthetic", SyntheticToxicityLoader)
    
    # List all loaders (should include our new one)
    print("Available loaders:")
    for loader_name in sorted(registry.list_loaders()):
        print(f"  - {loader_name}")
    
    print("\n" + "="*50)
    
    # Get info about our custom loader
    info = registry.get_loader_info("synthetic")
    print(f"Custom Loader Info:")
    print(f"  Name: {info['name']}")
    print(f"  Description: {info['description']}")
    print(f"  Type: {info['type']}")
    
    print("\n" + "="*50)
    
    # Use the custom loader
    loader = registry.get_loader("synthetic")
    samples = loader.load(max_samples=5)
    
    print(f"Generated {len(samples)} samples:")
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"  Toxic: {sample['toxic_text']}")
        print(f"  Neutral: {sample['neutral_text']}")
        print(f"  Category: {sample['category']}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()