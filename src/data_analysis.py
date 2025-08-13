"""Data analysis and quality assessment utilities."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DataQualityAnalyzer:
    """Analyze quality and characteristics of toxicity datasets."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_jsonl(self, path: str) -> List[Dict]:
        """Load JSONL file."""
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    
    def analyze_dataset(self, data_path: str, output_dir: str = "analysis") -> Dict:
        """Comprehensive analysis of a toxicity dataset.
        
        Parameters
        ----------
        data_path : str
            Path to the JSONL dataset file
        output_dir : str
            Directory to save analysis results
            
        Returns
        -------
        Dict
            Analysis results
        """
        data = self.load_jsonl(data_path)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        results = {}
        
        # Basic statistics
        results["basic_stats"] = self._basic_statistics(data)
        
        # Text length analysis
        results["length_stats"] = self._analyze_text_lengths(data, output_path)
        
        # Vocabulary analysis
        results["vocab_stats"] = self._analyze_vocabulary(data, output_path)
        
        # Similarity analysis
        results["similarity_stats"] = self._analyze_similarity(data, output_path)
        
        # Quality issues
        results["quality_issues"] = self._detect_quality_issues(data)
        
        # Save comprehensive report
        self._save_analysis_report(results, output_path / "analysis_report.json")
        
        return results
    
    def _basic_statistics(self, data: List[Dict]) -> Dict:
        """Calculate basic dataset statistics."""
        stats = {
            "total_samples": len(data),
            "has_toxic_text": sum(1 for item in data if "toxic_text" in item),
            "has_neutral_text": sum(1 for item in data if "neutral_text" in item),
            "has_image_path": sum(1 for item in data if "image_path" in item),
            "unique_categories": len(set(item.get("category", "unknown") for item in data))
        }
        
        self.logger.info(f"Dataset contains {stats['total_samples']} samples")
        return stats
    
    def _analyze_text_lengths(self, data: List[Dict], output_path: Path) -> Dict:
        """Analyze text length distributions."""
        toxic_lengths = []
        neutral_lengths = []
        
        for item in data:
            if "toxic_text" in item:
                toxic_lengths.append(len(item["toxic_text"]))
            if "neutral_text" in item:
                neutral_lengths.append(len(item["neutral_text"]))
        
        # Create visualization
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        if toxic_lengths:
            plt.hist(toxic_lengths, bins=30, alpha=0.7, label="Toxic", color="red")
        if neutral_lengths:
            plt.hist(neutral_lengths, bins=30, alpha=0.7, label="Neutral", color="blue")
        plt.xlabel("Text Length (characters)")
        plt.ylabel("Frequency")
        plt.title("Text Length Distribution")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        if toxic_lengths and neutral_lengths:
            plt.boxplot([toxic_lengths, neutral_lengths], labels=["Toxic", "Neutral"])
            plt.ylabel("Text Length (characters)")
            plt.title("Text Length Comparison")
        
        plt.tight_layout()
        plt.savefig(output_path / "text_length_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        stats = {
            "toxic_length_stats": self._calculate_length_stats(toxic_lengths),
            "neutral_length_stats": self._calculate_length_stats(neutral_lengths)
        }
        
        return stats
    
    def _calculate_length_stats(self, lengths: List[int]) -> Dict:
        """Calculate length statistics."""
        if not lengths:
            return {}
        
        return {
            "mean": sum(lengths) / len(lengths),
            "median": sorted(lengths)[len(lengths) // 2],
            "min": min(lengths),
            "max": max(lengths),
            "std": (sum((x - sum(lengths) / len(lengths)) ** 2 for x in lengths) / len(lengths)) ** 0.5
        }
    
    def _analyze_vocabulary(self, data: List[Dict], output_path: Path) -> Dict:
        """Analyze vocabulary characteristics."""
        toxic_texts = [item.get("toxic_text", "") for item in data if "toxic_text" in item]
        neutral_texts = [item.get("neutral_text", "") for item in data if "neutral_text" in item]
        
        # Word frequency analysis
        toxic_words = self._extract_words(toxic_texts)
        neutral_words = self._extract_words(neutral_texts)
        
        toxic_counter = Counter(toxic_words)
        neutral_counter = Counter(neutral_words)
        
        # Find distinctive words
        toxic_distinctive = self._find_distinctive_words(toxic_counter, neutral_counter)
        neutral_distinctive = self._find_distinctive_words(neutral_counter, toxic_counter)
        
        # Create word frequency plots
        self._plot_word_frequencies(toxic_counter, neutral_counter, output_path)
        
        stats = {
            "toxic_vocab_size": len(toxic_counter),
            "neutral_vocab_size": len(neutral_counter),
            "toxic_distinctive_words": toxic_distinctive[:20],
            "neutral_distinctive_words": neutral_distinctive[:20],
            "shared_vocab_size": len(set(toxic_counter.keys()) & set(neutral_counter.keys()))
        }
        
        return stats
    
    def _extract_words(self, texts: List[str]) -> List[str]:
        """Extract words from texts."""
        words = []
        for text in texts:
            # Simple tokenization (consider using proper tokenizer for production)
            text_words = re.findall(r'\b\w+\b', text.lower())
            words.extend(text_words)
        return words
    
    def _find_distinctive_words(self, target_counter: Counter, reference_counter: Counter) -> List[Tuple[str, float]]:
        """Find words that are distinctive to target corpus."""
        distinctive = []
        total_target = sum(target_counter.values())
        total_reference = sum(reference_counter.values())
        
        for word, count in target_counter.most_common(100):
            target_freq = count / total_target
            reference_freq = reference_counter.get(word, 0) / total_reference if total_reference > 0 else 0
            
            if reference_freq == 0:
                ratio = float('inf')
            else:
                ratio = target_freq / reference_freq
            
            if ratio > 2.0:  # Word is at least 2x more frequent in target
                distinctive.append((word, ratio))
        
        return sorted(distinctive, key=lambda x: x[1], reverse=True)
    
    def _plot_word_frequencies(self, toxic_counter: Counter, neutral_counter: Counter, output_path: Path):
        """Plot word frequency comparisons."""
        plt.figure(figsize=(15, 10))
        
        # Top toxic words
        plt.subplot(2, 2, 1)
        toxic_words, toxic_counts = zip(*toxic_counter.most_common(20))
        plt.barh(range(len(toxic_words)), toxic_counts, color="red", alpha=0.7)
        plt.yticks(range(len(toxic_words)), toxic_words)
        plt.xlabel("Frequency")
        plt.title("Most Common Words in Toxic Texts")
        plt.gca().invert_yaxis()
        
        # Top neutral words
        plt.subplot(2, 2, 2)
        neutral_words, neutral_counts = zip(*neutral_counter.most_common(20))
        plt.barh(range(len(neutral_words)), neutral_counts, color="blue", alpha=0.7)
        plt.yticks(range(len(neutral_words)), neutral_words)
        plt.xlabel("Frequency")
        plt.title("Most Common Words in Neutral Texts")
        plt.gca().invert_yaxis()
        
        # Word frequency comparison
        plt.subplot(2, 1, 2)
        common_words = set(toxic_counter.keys()) & set(neutral_counter.keys())
        common_words = list(common_words)[:30]  # Top 30 common words
        
        toxic_freqs = [toxic_counter[word] for word in common_words]
        neutral_freqs = [neutral_counter[word] for word in common_words]
        
        x = range(len(common_words))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], toxic_freqs, width, label="Toxic", color="red", alpha=0.7)
        plt.bar([i + width/2 for i in x], neutral_freqs, width, label="Neutral", color="blue", alpha=0.7)
        
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.title("Word Frequency Comparison")
        plt.xticks(x, common_words, rotation=45, ha="right")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / "vocabulary_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _analyze_similarity(self, data: List[Dict], output_path: Path) -> Dict:
        """Analyze similarity between toxic and neutral pairs."""
        pairs = [(item.get("toxic_text", ""), item.get("neutral_text", "")) 
                for item in data if "toxic_text" in item and "neutral_text" in item]
        
        if not pairs:
            return {"error": "No toxic-neutral pairs found"}
        
        toxic_texts, neutral_texts = zip(*pairs)
        
        # Calculate TF-IDF similarity
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        all_texts = list(toxic_texts) + list(neutral_texts)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(pairs)):
                toxic_vec = tfidf_matrix[i]
                neutral_vec = tfidf_matrix[i + len(pairs)]
                sim = cosine_similarity(toxic_vec, neutral_vec)[0, 0]
                similarities.append(sim)
            
            # Plot similarity distribution
            plt.figure(figsize=(10, 6))
            plt.hist(similarities, bins=30, alpha=0.7, color="green")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Frequency")
            plt.title("Similarity Distribution between Toxic-Neutral Pairs")
            plt.axvline(sum(similarities) / len(similarities), color="red", linestyle="--", 
                       label=f"Mean: {sum(similarities) / len(similarities):.3f}")
            plt.legend()
            plt.savefig(output_path / "similarity_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            stats = {
                "mean_similarity": sum(similarities) / len(similarities),
                "median_similarity": sorted(similarities)[len(similarities) // 2],
                "min_similarity": min(similarities),
                "max_similarity": max(similarities),
                "high_similarity_pairs": sum(1 for s in similarities if s > 0.8),
                "low_similarity_pairs": sum(1 for s in similarities if s < 0.2)
            }
            
        except Exception as e:
            self.logger.error(f"Error in similarity analysis: {e}")
            stats = {"error": str(e)}
        
        return stats
    
    def _detect_quality_issues(self, data: List[Dict]) -> Dict:
        """Detect potential quality issues in the dataset."""
        issues = {
            "empty_texts": 0,
            "very_short_texts": 0,
            "very_long_texts": 0,
            "duplicate_pairs": 0,
            "identical_pairs": 0,
            "missing_fields": 0
        }
        
        seen_pairs = set()
        
        for item in data:
            # Check for missing fields
            if "toxic_text" not in item or "neutral_text" not in item:
                issues["missing_fields"] += 1
                continue
            
            toxic_text = item["toxic_text"]
            neutral_text = item["neutral_text"]
            
            # Empty texts
            if not toxic_text.strip() or not neutral_text.strip():
                issues["empty_texts"] += 1
            
            # Very short texts
            if len(toxic_text) < 10 or len(neutral_text) < 10:
                issues["very_short_texts"] += 1
            
            # Very long texts
            if len(toxic_text) > 500 or len(neutral_text) > 500:
                issues["very_long_texts"] += 1
            
            # Identical pairs
            if toxic_text.strip().lower() == neutral_text.strip().lower():
                issues["identical_pairs"] += 1
            
            # Duplicate pairs
            pair_key = (toxic_text.strip().lower(), neutral_text.strip().lower())
            if pair_key in seen_pairs:
                issues["duplicate_pairs"] += 1
            else:
                seen_pairs.add(pair_key)
        
        return issues
    
    def _save_analysis_report(self, results: Dict, output_path: Path):
        """Save comprehensive analysis report."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Analysis report saved to {output_path}")
    
    def compare_datasets(self, dataset_paths: List[str], output_dir: str = "comparison") -> Dict:
        """Compare multiple datasets."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        datasets = {}
        for path in dataset_paths:
            name = Path(path).stem
            datasets[name] = self.load_jsonl(path)
        
        comparison = {}
        
        # Basic statistics comparison
        comparison["basic_stats"] = {}
        for name, data in datasets.items():
            comparison["basic_stats"][name] = self._basic_statistics(data)
        
        # Create comparison plots
        self._plot_dataset_comparison(datasets, output_path)
        
        return comparison
    
    def _plot_dataset_comparison(self, datasets: Dict[str, List[Dict]], output_path: Path):
        """Create comparison plots for multiple datasets."""
        plt.figure(figsize=(15, 10))
        
        # Sample counts
        plt.subplot(2, 2, 1)
        names = list(datasets.keys())
        counts = [len(data) for data in datasets.values()]
        plt.bar(names, counts, alpha=0.7)
        plt.xlabel("Dataset")
        plt.ylabel("Sample Count")
        plt.title("Dataset Size Comparison")
        plt.xticks(rotation=45)
        
        # Text length comparison
        plt.subplot(2, 2, 2)
        all_lengths = []
        labels = []
        
        for name, data in datasets.items():
            lengths = []
            for item in data:
                if "toxic_text" in item:
                    lengths.append(len(item["toxic_text"]))
                if "neutral_text" in item:
                    lengths.append(len(item["neutral_text"]))
            all_lengths.append(lengths)
            labels.append(name)
        
        plt.boxplot(all_lengths, labels=labels)
        plt.ylabel("Text Length")
        plt.title("Text Length Distribution Comparison")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / "dataset_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()


__all__ = ["DataQualityAnalyzer"]