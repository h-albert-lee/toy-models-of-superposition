# Toy Models of Superposition

This repository hosts a lightweight research toolkit for investigating how toxic and benign concepts coexist or interfere within the latent representations of vision–language models (VLMs). It provides utilities to build, analyze, and evaluate "toxicity" vectors derived from model activations.

## Features

- **Configuration-driven experiments** – Adjust model name, extraction layers, batch size, and data paths through `configs/base_config.yaml`.
- **Automated data loading** – Load and convert toxicity datasets from Hugging Face Hub with `ToxicityDataLoader`.
- **Data quality analysis** – Comprehensive dataset analysis including vocabulary, similarity, and quality metrics.
- **Batch activation extraction** – `VLM_Wrapper.get_activations` and `VectorExtractor` process text or image inputs in batches to maximize GPU utilization.
- **Structured logging** – All scripts log progress with Python's `logging` module.
- **Test coverage** – Core vector operations and feature extraction logic are validated via the `tests/` pytest suite.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

#### Flexible Data Loading System

The toolkit includes a flexible, plugin-based data loading system that supports multiple data sources:

1. **List available loaders**
   ```bash
   python scripts/prepare_datasets.py --list-loaders
   ```

2. **Get information about a specific loader**
   ```bash
   python scripts/prepare_datasets.py --loader-info hate_speech
   ```

3. **Load from a single source**
   ```bash
   # Hugging Face datasets
   python scripts/prepare_datasets.py --loader hate_speech --max-samples 1000
   python scripts/prepare_datasets.py --loader real_toxicity --toxicity-threshold 0.7
   
   # Custom CSV files
   python scripts/prepare_datasets.py --loader csv --file-path your_data.csv --max-samples 500
   
   # JSONL files
   python scripts/prepare_datasets.py --loader jsonl --file-path your_data.jsonl
   
   # Local directory (auto-detect formats)
   python scripts/prepare_datasets.py --loader local --file-path data/directory --pattern "*.csv"
   ```

4. **Combine multiple datasets using configuration**
   ```bash
   python scripts/prepare_datasets.py --config configs/flexible_data_config.yaml --strategy balanced
   ```

5. **Analyze dataset quality**
   ```bash
   python scripts/analyze_data.py --data_path data/prepared_dataset.jsonl --output_dir analysis
   ```

### Vector Computation

3. **Compute the general toxicity vector (GTV)**
   ```bash
   python scripts/00_define_general_toxicity_vector.py --config configs/base_config.yaml
   ```
4. **Compute the intrinsic toxicity vector (ITV)**
   ```bash
   python scripts/01_define_intrinsic_toxicity_vector.py --config configs/base_config.yaml
   ```
5. **Analyze vector similarity**
   ```bash
   python scripts/03_analyze_vector_similarity.py --config configs/base_config.yaml
   ```
6. **Evaluate precision interventions**
   ```bash
   python scripts/04_evaluate_precision_intervention.py --config configs/base_config.yaml
   ```

Each script reads model and data settings from the configuration file and writes outputs to the directory specified by `output_dir`.

## Example configuration

```yaml
model_name: "llava-hf/llava-1.5-7b-hf"
device: "cuda"
model_kwargs: {}
extraction_layer: "language_model.layers.0"
batch_size: 8
data_sources:
  toxic_neutral_pairs: "data/general_toxic_neutral_text_pairs.jsonl"
  intrinsic_toxicity_cases: "data/intrinsic_toxicity_cases.jsonl"
intervention_layer: "language_model.layers.0"
intervention_multiplier: 1.0
output_dir: "results"
```

## Data Sources

The toolkit supports loading data from multiple sources through a flexible plugin system:

### Hugging Face Datasets
- **Jigsaw Toxicity** (`jigsaw`): Google's toxicity classification dataset with human annotations
- **Hate Speech Offensive** (`hate_speech`): Twitter dataset for hate speech and offensive language detection  
- **RealToxicityPrompts** (`real_toxicity`): AllenAI's dataset for studying toxicity in language generation
- **Multilingual Toxicity** (`multilingual`): Multi-language toxicity detection dataset

### Custom File Formats
- **CSV Loader** (`csv`): Load from CSV files with configurable column mapping
- **JSONL Loader** (`jsonl`): Load from JSONL files with automatic validation
- **Local File Loader** (`local`): Auto-detect and load from local directories
- **Multimodal Loader** (`multimodal`): Load text-image pairs for VLM research

### External Sources
- **API Loader** (`api`): Load from REST APIs with authentication and pagination support

### Adding Custom Loaders

To add a new data loader:

1. Create a class inheriting from `BaseDataLoader`
2. Implement `load()` and `get_info()` methods
3. Register it with the registry:

```python
from src.data_loaders import registry, BaseDataLoader

class MyCustomLoader(BaseDataLoader):
    def load(self, **kwargs):
        # Your loading logic here
        return samples
    
    def get_info(self):
        return {"name": "My Loader", "description": "..."}

# Register the loader
registry.register("my_loader", MyCustomLoader)
```

### Data Quality Analysis

The `DataQualityAnalyzer` provides comprehensive analysis including:

- **Basic statistics**: Sample counts, field coverage, category distribution
- **Text length analysis**: Distribution and comparison of text lengths
- **Vocabulary analysis**: Word frequency, distinctive terms, shared vocabulary
- **Similarity analysis**: Cosine similarity between toxic-neutral pairs
- **Quality issues detection**: Empty texts, duplicates, identical pairs

Analysis results include visualizations and detailed JSON reports for further investigation.

## Development & testing

```bash
pytest
```

## License

[MIT](LICENSE)
