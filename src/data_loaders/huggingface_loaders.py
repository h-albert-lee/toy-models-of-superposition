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


class MultilingualToxicityLoader(BaseDataLoader):
    """Loader for multilingual toxicity dataset (toxic samples only for superposition analysis)."""
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Multilingual Toxicity Dataset (Toxic Only)",
            "description": "Multi-language toxicity detection dataset - toxic samples only for superposition analysis",
            "source": "textdetox/multilingual_toxicity_dataset",
            "type": "toxic_text",
            "languages": ["en", "ru", "uk", "de", "es", "fr", "it", "zh", "ar", "hi", "am", "he", "hin", "tt", "ja"],
            "size": "~2.5k toxic samples per language",
            "default_language": "en",
            "default_max_samples": 200
        }
    
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        max_samples = kwargs.get('max_samples', 200)
        language = kwargs.get('language', 'en')
        
        try:
            dataset = load_dataset("textdetox/multilingual_toxicity_dataset", split=language)
        except Exception as e:
            self.logger.error(f"Failed to load multilingual toxicity dataset: {e}")
            return []
        
        toxic_samples = []
        
        # Extract texts and labels
        texts = dataset['text']
        labels = dataset['toxic']
        
        for text, is_toxic in zip(texts, labels):
            text = text.strip()
            
            if not text or len(text) < 10:
                continue
            
            # Only collect toxic samples (label == 1)
            if is_toxic == 1:
                toxic_samples.append({
                    "text": text,
                    "is_toxic": True,
                    "language": language,
                    "source": "multilingual_toxicity_dataset"
                })
                
                # Stop if we've reached the max samples
                if len(toxic_samples) >= max_samples:
                    break
        
        self.logger.info(f"Loaded {len(toxic_samples)} toxic samples (max_samples={max_samples})")
        return toxic_samples


class MemeSafetyBenchLoader(BaseDataLoader):
    """Loader for Meme-Safety-Bench dataset (negative/unsafe samples only for superposition analysis)."""
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Meme-Safety-Bench (Unsafe Only)",
            "description": "VLM safety benchmark - unsafe/negative meme samples only for superposition analysis",
            "source": "oneonlee/Meme-Safety-Bench",
            "type": "multimodal_unsafe",
            "languages": ["en"],
            "size": "~500 unsafe meme samples",
            "modalities": ["text", "image"],
            "default_max_samples": 200
        }
    
    def load(self, **kwargs) -> List[Dict[str, Any]]:
        max_samples = kwargs.get('max_samples', 200)
        
        try:
            # Try to load the dataset
            dataset = load_dataset("oneonlee/Meme-Safety-Bench", split="train")
        except Exception as e:
            self.logger.error(f"Failed to load Meme-Safety-Bench dataset: {e}")
            self.logger.info("This dataset may require authentication or may not be publicly accessible")
            return []
        
        unsafe_samples = []
        
        for item in dataset:
            # Extract relevant fields (adapt based on actual dataset structure)
            text = item.get("text", "").strip()
            image = item.get("image")  # PIL Image
            label = item.get("label", 0)  # Safety label
            sentiment = item.get("sentiment", "neutral")  # Sentiment field
            
            if not text and not image:
                continue
            
            # Only collect unsafe/negative samples
            # Check multiple possible indicators for unsafe content
            is_unsafe = (
                label == 1 or  # Assume 1 = unsafe, 0 = safe
                sentiment == "negative" or
                item.get("is_safe", True) == False or
                item.get("toxic", False) == True
            )
            
            if is_unsafe:
                sample = {
                    "text": text,
                    "image": image,
                    "is_safe": False,
                    "sentiment": sentiment,
                    "label": label,
                    "category": item.get("category", "meme"),
                    "source": "meme_safety_bench"
                }
                
                unsafe_samples.append(sample)
                
                # Stop if we've reached the max samples
                if len(unsafe_samples) >= max_samples:
                    break
        
        self.logger.info(f"Loaded {len(unsafe_samples)} unsafe multimodal samples (max_samples={max_samples})")
        return unsafe_samples


__all__ = [
    "JigsawToxicityLoader",
    "HateSpeechOffensiveLoader", 
    "RealToxicityPromptsLoader",
    "MultilingualToxicityLoader",
    "MemeSafetyBenchLoader"
]