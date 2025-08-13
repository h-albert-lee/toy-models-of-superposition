"""Custom data loaders for various file formats."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from PIL import Image

from .base import BaseDataLoader


class CSVLoader(BaseDataLoader):
    """Load data from CSV files."""
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "CSV Loader",
            "description": "Load toxicity data from CSV files",
            "type": "file_loader",
            "supported_formats": [".csv"],
            "required_columns": ["toxic_text", "neutral_text"] or ["text", "label"]
        }
    
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        file_path = kwargs.get('file_path')
        if not file_path:
            raise ValueError("file_path is required for CSVLoader")
        
        toxic_column = kwargs.get('toxic_column', 'toxic_text')
        neutral_column = kwargs.get('neutral_column', 'neutral_text')
        text_column = kwargs.get('text_column', 'text')
        label_column = kwargs.get('label_column', 'label')
        max_samples = kwargs.get('max_samples')
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            self.logger.error(f"Failed to load CSV file {file_path}: {e}")
            return []
        
        samples = []
        
        # Check if we have toxic/neutral columns or text/label columns
        if toxic_column in df.columns and neutral_column in df.columns:
            # Direct toxic-neutral pairs
            for _, row in df.iterrows():
                toxic_text = str(row[toxic_column]).strip()
                neutral_text = str(row[neutral_column]).strip()
                
                if toxic_text and neutral_text:
                    samples.append({
                        "toxic_text": toxic_text,
                        "neutral_text": neutral_text
                    })
        
        elif text_column in df.columns and label_column in df.columns:
            # Text with labels - need to create pairs
            toxic_texts = []
            neutral_texts = []
            
            for _, row in df.iterrows():
                text = str(row[text_column]).strip()
                label = row[label_column]
                
                if not text:
                    continue
                
                # Assume label 1 = toxic, 0 = neutral (can be configured)
                toxic_labels = kwargs.get('toxic_labels', [1, 'toxic', 'hate', 'offensive'])
                neutral_labels = kwargs.get('neutral_labels', [0, 'neutral', 'clean', 'safe'])
                
                if label in toxic_labels:
                    toxic_texts.append(text)
                elif label in neutral_labels:
                    neutral_texts.append(text)
            
            # Create pairs
            min_samples = min(len(toxic_texts), len(neutral_texts))
            if max_samples:
                min_samples = min(min_samples, max_samples)
            
            if min_samples > 0:
                sampled_toxic = random.sample(toxic_texts, min_samples)
                sampled_neutral = random.sample(neutral_texts, min_samples)
                
                for toxic, neutral in zip(sampled_toxic, sampled_neutral):
                    samples.append({
                        "toxic_text": toxic,
                        "neutral_text": neutral
                    })
        
        else:
            self.logger.error(f"CSV file must contain either ({toxic_column}, {neutral_column}) or ({text_column}, {label_column}) columns")
            return []
        
        if max_samples and len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
        
        self.logger.info(f"Loaded {len(samples)} samples from CSV")
        return samples


class JSONLLoader(BaseDataLoader):
    """Load data from JSONL files."""
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "JSONL Loader",
            "description": "Load toxicity data from JSONL files",
            "type": "file_loader",
            "supported_formats": [".jsonl", ".json"],
            "required_fields": ["toxic_text", "neutral_text"] or ["text", "label"]
        }
    
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        file_path = kwargs.get('file_path')
        if not file_path:
            raise ValueError("file_path is required for JSONLLoader")
        
        max_samples = kwargs.get('max_samples')
        
        try:
            samples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        samples.append(data)
        except Exception as e:
            self.logger.error(f"Failed to load JSONL file {file_path}: {e}")
            return []
        
        # Filter and validate samples
        valid_samples = []
        for sample in samples:
            if self.validate_sample(sample):
                valid_samples.append(sample)
        
        if max_samples and len(valid_samples) > max_samples:
            valid_samples = random.sample(valid_samples, max_samples)
        
        self.logger.info(f"Loaded {len(valid_samples)} samples from JSONL")
        return valid_samples


class LocalFileLoader(BaseDataLoader):
    """Load data from local files with automatic format detection."""
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Local File Loader",
            "description": "Load toxicity data from local files with auto-detection",
            "type": "file_loader",
            "supported_formats": [".csv", ".jsonl", ".json", ".txt"],
            "features": ["auto_format_detection", "recursive_directory_loading"]
        }
    
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        path = kwargs.get('path')
        if not path:
            raise ValueError("path is required for LocalFileLoader")
        
        path = Path(path)
        max_samples = kwargs.get('max_samples')
        recursive = kwargs.get('recursive', False)
        
        if path.is_file():
            files = [path]
        elif path.is_dir():
            pattern = kwargs.get('pattern', '*')
            if recursive:
                files = list(path.rglob(pattern))
            else:
                files = list(path.glob(pattern))
            files = [f for f in files if f.is_file()]
        else:
            self.logger.error(f"Path does not exist: {path}")
            return []
        
        all_samples = []
        
        for file_path in files:
            try:
                file_samples = self._load_single_file(file_path, **kwargs)
                all_samples.extend(file_samples)
                self.logger.info(f"Loaded {len(file_samples)} samples from {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if max_samples and len(all_samples) > max_samples:
            all_samples = random.sample(all_samples, max_samples)
        
        self.logger.info(f"Total loaded: {len(all_samples)} samples from {len(files)} files")
        return all_samples
    
    def _load_single_file(self, file_path: Path, **kwargs) -> List[Dict[str, Any]]:
        """Load a single file based on its extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            loader = CSVLoader(**self.config)
            return loader.load(file_path=str(file_path), **kwargs)
        
        elif suffix in ['.jsonl', '.json']:
            loader = JSONLLoader(**self.config)
            return loader.load(file_path=str(file_path), **kwargs)
        
        elif suffix == '.txt':
            return self._load_txt_file(file_path, **kwargs)
        
        else:
            self.logger.warning(f"Unsupported file format: {suffix}")
            return []
    
    def _load_txt_file(self, file_path: Path, **kwargs) -> List[Dict[str, Any]]:
        """Load plain text file (one sample per line)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            # Assume alternating toxic/neutral lines or use labels
            samples = []
            label_pattern = kwargs.get('label_pattern')  # e.g., "TOXIC:", "NEUTRAL:"
            
            if label_pattern:
                current_text = ""
                current_label = None
                
                for line in lines:
                    if line.startswith("TOXIC:"):
                        current_text = line[6:].strip()
                        current_label = "toxic"
                    elif line.startswith("NEUTRAL:"):
                        if current_label == "toxic" and current_text:
                            samples.append({
                                "toxic_text": current_text,
                                "neutral_text": line[8:].strip()
                            })
                        current_text = ""
                        current_label = None
            else:
                # Assume alternating lines
                for i in range(0, len(lines) - 1, 2):
                    samples.append({
                        "toxic_text": lines[i],
                        "neutral_text": lines[i + 1]
                    })
            
            return samples
            
        except Exception as e:
            self.logger.error(f"Failed to load text file {file_path}: {e}")
            return []


class MultimodalLoader(BaseDataLoader):
    """Load multimodal data (text + images)."""
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Multimodal Loader",
            "description": "Load multimodal toxicity data (text + images)",
            "type": "multimodal_loader",
            "supported_formats": [".jsonl", ".csv"],
            "required_fields": ["text", "image_path"]
        }
    
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        file_path = kwargs.get('file_path')
        if not file_path:
            raise ValueError("file_path is required for MultimodalLoader")
        
        image_base_path = kwargs.get('image_base_path', '')
        max_samples = kwargs.get('max_samples')
        load_images = kwargs.get('load_images', False)  # Whether to load actual PIL images
        
        # Load the metadata file
        file_path = Path(file_path)
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        else:  # assume JSONL
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        
        samples = []
        for item in data:
            text = item.get('text', '').strip()
            image_path = item.get('image_path', '')
            
            if not text or not image_path:
                continue
            
            # Resolve image path
            if image_base_path:
                full_image_path = Path(image_base_path) / image_path
            else:
                full_image_path = Path(image_path)
            
            sample = {
                'text': text,
                'image_path': str(full_image_path),
                'category': item.get('category', 'general')
            }
            
            # Optionally load the actual image
            if load_images:
                try:
                    if full_image_path.exists():
                        sample['image'] = Image.open(full_image_path).convert('RGB')
                    else:
                        self.logger.warning(f"Image not found: {full_image_path}")
                        continue
                except Exception as e:
                    self.logger.error(f"Failed to load image {full_image_path}: {e}")
                    continue
            
            samples.append(sample)
        
        if max_samples and len(samples) > max_samples:
            samples = random.sample(samples, max_samples)
        
        self.logger.info(f"Loaded {len(samples)} multimodal samples")
        return samples


class APILoader(BaseDataLoader):
    """Load data from APIs."""
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "API Loader",
            "description": "Load toxicity data from REST APIs",
            "type": "api_loader",
            "supported_apis": ["custom", "huggingface_inference"],
            "features": ["authentication", "pagination", "rate_limiting"]
        }
    
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        api_url = kwargs.get('api_url')
        if not api_url:
            raise ValueError("api_url is required for APILoader")
        
        headers = kwargs.get('headers', {})
        params = kwargs.get('params', {})
        max_samples = kwargs.get('max_samples')
        
        import requests
        
        try:
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different API response formats
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict):
                # Common patterns
                samples = data.get('data', data.get('results', data.get('items', [])))
            else:
                self.logger.error(f"Unexpected API response format: {type(data)}")
                return []
            
            # Validate and filter samples
            valid_samples = [s for s in samples if self.validate_sample(s)]
            
            if max_samples and len(valid_samples) > max_samples:
                valid_samples = random.sample(valid_samples, max_samples)
            
            self.logger.info(f"Loaded {len(valid_samples)} samples from API")
            return valid_samples
            
        except Exception as e:
            self.logger.error(f"Failed to load from API {api_url}: {e}")
            return []


__all__ = [
    "CSVLoader",
    "JSONLLoader", 
    "LocalFileLoader",
    "MultimodalLoader",
    "APILoader"
]