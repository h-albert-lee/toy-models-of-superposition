"""Hugging Face dataset loaders."""

from __future__ import annotations

import random
from typing import Any, Dict, List

from datasets import load_dataset
from PIL import Image

from .base import BaseDataLoader


class PairCreatorMixin:
    """Mixin for creating toxic-neutral pairs."""
    
    def _create_pairs(
        self, 
        toxic_texts: List[str], 
        neutral_texts: List[str], 
        max_samples: int
    ) -> List[Dict[str, str]]:
        """Create toxic-neutral pairs by random sampling."""
        
        min_samples = min(len(toxic_texts), len(neutral_texts), max_samples)
        
        if min_samples == 0:
            self.logger.warning("No valid pairs could be created")
            return []
        
        # Random sampling
        sampled_toxic = random.sample(toxic_texts, min_samples)
        sampled_neutral = random.sample(neutral_texts, min_samples)
        
        pairs = []
        for toxic, neutral in zip(sampled_toxic, sampled_neutral):
            pairs.append({
                "toxic_text": toxic,
                "neutral_text": neutral
            })
        
        return pairs


class JigsawToxicityLoader(BaseDataLoader, PairCreatorMixin):
    """Loader for Jigsaw Toxicity dataset."""
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Jigsaw Toxicity Classification",
            "description": "Google's toxicity classification dataset with human annotations",
            "source": "google/jigsaw_toxicity_pred",
            "type": "text_pairs",
            "languages": ["en"],
            "size": "~160k comments"
        }
    
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        max_samples = kwargs.get('max_samples', 1000)
        toxicity_threshold = kwargs.get('toxicity_threshold', 0.5)
        
        try:
            # Try alternative dataset if main one fails
            try:
                dataset = load_dataset("unitary/toxic-bert", split="train")
            except:
                # Fallback to a working toxicity dataset
                dataset = load_dataset("martin-ha/toxic-comment-model", split="train")
        except Exception as e:
            self.logger.error(f"Failed to load Jigsaw dataset: {e}")
            return []
        
        toxic_comments = []
        neutral_comments = []
        
        for item in dataset:
            # Adapt to different dataset schemas
            if 'comment_text' in item:
                text = item['comment_text']
                toxicity = item.get('toxic', 0)
            elif 'text' in item:
                text = item['text']
                toxicity = item.get('label', 0)
            else:
                continue
            
            text = str(text).strip()
            if len(text) < 10:
                continue
            
            if toxicity >= toxicity_threshold:
                toxic_comments.append(text)
            elif toxicity <= 0.1:
                neutral_comments.append(text)
        
        return self._create_pairs(toxic_comments, neutral_comments, max_samples)


class HateSpeechOffensiveLoader(BaseDataLoader, PairCreatorMixin):
    """Loader for hate speech offensive dataset."""
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Hate Speech Offensive",
            "description": "Twitter dataset for hate speech and offensive language detection",
            "source": "tdavidson/hate_speech_offensive",
            "type": "text_pairs",
            "languages": ["en"],
            "size": "~25k tweets"
        }
    
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        max_samples = kwargs.get('max_samples', 1000)
        
        try:
            dataset = load_dataset("tdavidson/hate_speech_offensive", split="train")
        except Exception as e:
            self.logger.error(f"Failed to load hate speech dataset: {e}")
            return []
        
        toxic_comments = []
        neutral_comments = []
        
        for item in dataset:
            text = item.get("tweet", "").strip()
            label = item.get("class", 0)
            
            if not text or len(text) < 10:
                continue
            
            if label in [0, 1]:  # hate speech or offensive
                toxic_comments.append(text)
            elif label == 2:  # neither
                neutral_comments.append(text)
        
        return self._create_pairs(toxic_comments, neutral_comments, max_samples)


class RealToxicityPromptsLoader(BaseDataLoader, PairCreatorMixin):
    """Loader for RealToxicityPrompts dataset."""
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "RealToxicityPrompts",
            "description": "AllenAI's dataset for studying toxicity in language generation",
            "source": "allenai/real-toxicity-prompts",
            "type": "text_pairs",
            "languages": ["en"],
            "size": "~100k prompts"
        }
    
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        max_samples = kwargs.get('max_samples', 1000)
        toxicity_threshold = kwargs.get('toxicity_threshold', 0.5)
        
        try:
            dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
        except Exception as e:
            self.logger.error(f"Failed to load RealToxicityPrompts: {e}")
            return []
        
        toxic_prompts = []
        neutral_prompts = []
        
        for item in dataset:
            prompt_data = item.get("prompt", {})
            text = prompt_data.get("text", "").strip()
            toxicity = prompt_data.get("toxicity")
            
            if not text or len(text) < 10 or toxicity is None:
                continue
            
            if toxicity >= toxicity_threshold:
                toxic_prompts.append(text)
            elif toxicity <= 0.1:
                neutral_prompts.append(text)
        
        return self._create_pairs(toxic_prompts, neutral_prompts, max_samples)


class MultilingualToxicityLoader(BaseDataLoader, PairCreatorMixin):
    """Loader for multilingual toxicity dataset."""
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Multilingual Toxicity",
            "description": "Multi-language toxicity detection dataset",
            "source": "textdetox/multilingual_toxicity_dataset",
            "type": "text_pairs",
            "languages": ["en", "ru", "uk", "de", "es", "fr", "it", "zh", "ar", "hi"],
            "size": "~50k texts per language"
        }
    
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        max_samples = kwargs.get('max_samples', 1000)
        language = kwargs.get('language', 'en')
        
        try:
            dataset = load_dataset("textdetox/multilingual_toxicity_dataset", split=language)
        except Exception as e:
            self.logger.error(f"Failed to load multilingual toxicity dataset: {e}")
            return []
        
        toxic_texts = []
        neutral_texts = []
        
        for item in dataset:
            text = item.get("text", "").strip()
            is_toxic = item.get("is_toxic", False)
            
            if not text or len(text) < 10:
                continue
            
            if is_toxic:
                toxic_texts.append(text)
            else:
                neutral_texts.append(text)
        
        return self._create_pairs(toxic_texts, neutral_texts, max_samples)


__all__ = [
    "JigsawToxicityLoader",
    "HateSpeechOffensiveLoader", 
    "RealToxicityPromptsLoader",
    "MultilingualToxicityLoader"
]