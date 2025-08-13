"""Base classes for data loaders."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union


class BaseDataLoader(ABC):
    """Base class for all data loaders."""
    
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = kwargs
    
    @abstractmethod
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        """Load data and return list of samples.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of data samples, each containing at minimum:
            - 'toxic_text': str (for GTV computation)
            - 'neutral_text': str (for GTV computation)
            Or for multimodal:
            - 'text': str
            - 'image': PIL.Image or image path
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get information about this data loader."""
        pass
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate a single sample."""
        # Basic validation - can be overridden
        if 'toxic_text' in sample and 'neutral_text' in sample:
            return bool(sample['toxic_text'].strip() and sample['neutral_text'].strip())
        elif 'text' in sample:
            return bool(sample['text'].strip())
        return False
    
    def filter_samples(self, samples: List[Dict[str, Any]], **filter_kwargs) -> List[Dict[str, Any]]:
        """Filter samples based on criteria."""
        filtered = []
        
        min_length = filter_kwargs.get('min_length', 10)
        max_length = filter_kwargs.get('max_length', 1000)
        remove_duplicates = filter_kwargs.get('remove_duplicates', True)
        
        seen = set() if remove_duplicates else None
        
        for sample in samples:
            if not self.validate_sample(sample):
                continue
            
            # Length filtering
            texts_to_check = []
            if 'toxic_text' in sample:
                texts_to_check.extend([sample['toxic_text'], sample['neutral_text']])
            elif 'text' in sample:
                texts_to_check.append(sample['text'])
            
            if any(len(text) < min_length or len(text) > max_length for text in texts_to_check):
                continue
            
            # Duplicate filtering
            if remove_duplicates:
                sample_key = str(sorted(sample.items()))
                if sample_key in seen:
                    continue
                seen.add(sample_key)
            
            filtered.append(sample)
        
        self.logger.info(f"Filtered {len(samples)} -> {len(filtered)} samples")
        return filtered


class DataLoaderRegistry:
    """Registry for data loaders."""
    
    def __init__(self):
        self._loaders: Dict[str, Type[BaseDataLoader]] = {}
    
    def register(self, name: str, loader_class: Type[BaseDataLoader]):
        """Register a data loader."""
        if not issubclass(loader_class, BaseDataLoader):
            raise ValueError(f"Loader class must inherit from BaseDataLoader")
        
        self._loaders[name] = loader_class
        logging.getLogger(__name__).debug(f"Registered loader: {name}")
    
    def get_loader(self, name: str, **kwargs) -> BaseDataLoader:
        """Get a data loader instance."""
        if name not in self._loaders:
            available = list(self._loaders.keys())
            raise ValueError(f"Unknown loader '{name}'. Available: {available}")
        
        return self._loaders[name](**kwargs)
    
    def list_loaders(self) -> List[str]:
        """List available loaders."""
        return list(self._loaders.keys())
    
    def get_loader_info(self, name: str) -> Dict[str, Any]:
        """Get information about a loader."""
        if name not in self._loaders:
            raise ValueError(f"Unknown loader '{name}'")
        
        # Create temporary instance to get info
        temp_loader = self._loaders[name]()
        return temp_loader.get_info()


class DatasetCombiner:
    """Combine multiple datasets with different strategies."""
    
    def __init__(self, registry: DataLoaderRegistry):
        self.registry = registry
        self.logger = logging.getLogger(__name__)
    
    def combine_datasets(
        self,
        dataset_configs: List[Dict[str, Any]],
        strategy: str = "balanced",
        max_total_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Combine multiple datasets.
        
        Parameters
        ----------
        dataset_configs : List[Dict[str, Any]]
            List of dataset configurations, each containing:
            - 'loader': str (loader name)
            - 'params': Dict (loader parameters)
            - 'max_samples': int (optional)
            - 'weight': float (optional, for weighted strategy)
        strategy : str
            Combination strategy: 'balanced', 'weighted', 'sequential'
        max_total_samples : Optional[int]
            Maximum total samples to return
        
        Returns
        -------
        List[Dict[str, Any]]
            Combined dataset
        """
        all_samples = []
        dataset_samples = []
        
        # Load all datasets
        for config in dataset_configs:
            loader_name = config['loader']
            loader_params = config.get('params', {})
            max_samples = config.get('max_samples')
            
            try:
                loader = self.registry.get_loader(loader_name, **loader_params)
                samples = loader.load(**loader_params)
                
                if max_samples and len(samples) > max_samples:
                    import random
                    samples = random.sample(samples, max_samples)
                
                dataset_samples.append({
                    'name': loader_name,
                    'samples': samples,
                    'weight': config.get('weight', 1.0)
                })
                
                self.logger.info(f"Loaded {len(samples)} samples from {loader_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {loader_name}: {e}")
                continue
        
        # Combine based on strategy
        if strategy == "balanced":
            all_samples = self._combine_balanced(dataset_samples, max_total_samples)
        elif strategy == "weighted":
            all_samples = self._combine_weighted(dataset_samples, max_total_samples)
        elif strategy == "sequential":
            all_samples = self._combine_sequential(dataset_samples, max_total_samples)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Shuffle final result
        import random
        random.shuffle(all_samples)
        
        self.logger.info(f"Combined dataset: {len(all_samples)} total samples")
        return all_samples
    
    def _combine_balanced(self, dataset_samples: List[Dict], max_total: Optional[int]) -> List[Dict]:
        """Combine datasets with equal representation."""
        if not dataset_samples:
            return []
        
        # Calculate samples per dataset
        if max_total:
            samples_per_dataset = max_total // len(dataset_samples)
        else:
            samples_per_dataset = min(len(ds['samples']) for ds in dataset_samples)
        
        combined = []
        for ds in dataset_samples:
            samples = ds['samples'][:samples_per_dataset]
            combined.extend(samples)
        
        return combined
    
    def _combine_weighted(self, dataset_samples: List[Dict], max_total: Optional[int]) -> List[Dict]:
        """Combine datasets based on weights."""
        if not dataset_samples:
            return []
        
        total_weight = sum(ds['weight'] for ds in dataset_samples)
        combined = []
        
        for ds in dataset_samples:
            if max_total:
                target_samples = int((ds['weight'] / total_weight) * max_total)
            else:
                target_samples = len(ds['samples'])
            
            samples = ds['samples'][:target_samples]
            combined.extend(samples)
        
        return combined
    
    def _combine_sequential(self, dataset_samples: List[Dict], max_total: Optional[int]) -> List[Dict]:
        """Combine datasets sequentially."""
        combined = []
        remaining = max_total
        
        for ds in dataset_samples:
            if remaining is not None and remaining <= 0:
                break
            
            samples = ds['samples']
            if remaining is not None:
                samples = samples[:remaining]
                remaining -= len(samples)
            
            combined.extend(samples)
        
        return combined


__all__ = ["BaseDataLoader", "DataLoaderRegistry", "DatasetCombiner"]