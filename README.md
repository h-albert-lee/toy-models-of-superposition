# VLM Toxicity Superposition Analysis

This project provides a research toolkit for investigating the hypothesis that toxic features are superposed with benign concepts within the latent space of vision-language models (VLMs).

## Project Structure

```
configs/                  # Experiment configuration files
  base_config.yaml
data/                     # Example JSONL datasets used for experiments
  toxic_neutral_pairs.jsonl
  multimodal_test_cases.jsonl
notebooks/
  analysis.ipynb          # Notebook for exploratory analysis
scripts/                  # Command line scripts for experiments
  01_extract_direction.py
  02_analyze_superposition.py
  03_evaluate_intervention.py
src/                      # Core source code
  vlm_wrapper.py
  feature_extractor.py
  analysis.py
  interventions.py
requirements.txt          # Python dependencies
```

## Usage

1. **Extract direction vector**
   ```bash
   python scripts/01_extract_direction.py --config configs/base_config.yaml
   ```
2. **Analyze superposition**
   ```bash
   python scripts/02_analyze_superposition.py --config configs/base_config.yaml
   ```
3. **Evaluate interventions**
   ```bash
   python scripts/03_evaluate_intervention.py --config configs/base_config.yaml
   ```

All scripts expect the model specified in the configuration to be available through the Hugging Face `transformers` library.

### Hugging Face models and datasets

- **Models:** set `model_name` in the config to any Hugging Face VLM ID (e.g. `llava-hf/llava-1.5-7b-hf`).
  Additional arguments for `from_pretrained` may be supplied via `model_kwargs` in the config and will be forwarded by `VLM_Wrapper`.
- **Datasets:** entries under `data_sources` accept either local JSONL paths or Hugging Face dataset identifiers
  (optionally suffixed with `:split`).
  When a dataset ID is provided, the requisite data will be downloaded automatically using the `datasets` library.
