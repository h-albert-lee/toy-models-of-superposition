# Toy Models of Superposition

This repository hosts a lightweight research toolkit for investigating how toxic and benign concepts coexist or interfere within the latent representations of vision–language models (VLMs). It provides utilities to build, analyze, and evaluate "toxicity" vectors derived from model activations.

## Features

- **Configuration-driven experiments** – Adjust model name, extraction layers, batch size, and data paths through `configs/base_config.yaml`.
- **Batch activation extraction** – `VLM_Wrapper.get_activations` and `VectorExtractor` process text or image inputs in batches to maximize GPU utilization.
- **Structured logging** – All scripts log progress with Python's `logging` module.
- **Test coverage** – Core vector operations and feature extraction logic are validated via the `tests/` pytest suite.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Compute the general toxicity vector (GTV)**
   ```bash
   python scripts/00_define_general_toxicity_vector.py --config configs/base_config.yaml
   ```
2. **Compute the intrinsic toxicity vector (ITV)**
   ```bash
   python scripts/01_define_intrinsic_toxicity_vector.py --config configs/base_config.yaml
   ```
3. **Analyze vector similarity**
   ```bash
   python scripts/03_analyze_vector_similarity.py --config configs/base_config.yaml
   ```
4. **Evaluate precision interventions**
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

## Development & testing

```bash
pytest
```

## License

[MIT](LICENSE)
