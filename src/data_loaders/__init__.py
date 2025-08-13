"""Data loading framework for toxicity datasets."""

from .base import BaseDataLoader, DataLoaderRegistry, DatasetCombiner
from .huggingface_loaders import (
    JigsawToxicityLoader,
    HateSpeechOffensiveLoader,
    RealToxicityPromptsLoader,
    MultilingualToxicityLoader
)
from .custom_loaders import (
    CSVLoader,
    JSONLLoader,
    LocalFileLoader,
    MultimodalLoader,
    APILoader
)

# Create and configure the global registry
registry = DataLoaderRegistry()

# Register Hugging Face loaders
registry.register("jigsaw", JigsawToxicityLoader)
registry.register("hate_speech", HateSpeechOffensiveLoader)
registry.register("real_toxicity", RealToxicityPromptsLoader)
registry.register("multilingual", MultilingualToxicityLoader)

# Register custom loaders
registry.register("csv", CSVLoader)
registry.register("jsonl", JSONLLoader)
registry.register("local", LocalFileLoader)
registry.register("multimodal", MultimodalLoader)
registry.register("api", APILoader)

# Create combiner instance
combiner = DatasetCombiner(registry)

__all__ = [
    "BaseDataLoader",
    "DataLoaderRegistry", 
    "DatasetCombiner",
    "registry",
    "combiner",
    # Hugging Face loaders
    "JigsawToxicityLoader",
    "HateSpeechOffensiveLoader",
    "RealToxicityPromptsLoader",
    "MultilingualToxicityLoader",
    # Custom loaders
    "CSVLoader",
    "JSONLLoader",
    "LocalFileLoader",
    "MultimodalLoader",
    "APILoader"
]