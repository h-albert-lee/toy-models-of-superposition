import yaml
import torch
import pandas as pd
from pathlib import Path
from src.vlm_wrapper import VLMWrapper
from src.analysis import analyze_concept_overlap, analyze_cross_modal_amplification
from src.data_utils import load_cases

def main(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    wrapper = VLMWrapper(cfg['model_name'], cfg['device'], **cfg.get('model_kwargs', {}))

    direction = torch.load(Path(cfg.get('output_dir', 'results')) / 'direction_vector.pt')

    cases = load_cases(cfg['data_sources']['multimodal_test_cases'])

    prompts = [c['prompt'] for c in cases]
    concept_df = analyze_concept_overlap(wrapper, prompts, cfg['extraction_layer'], direction)

    amp_df = analyze_cross_modal_amplification(wrapper, cases, cfg['extraction_layer'], direction)

    out_dir = Path(cfg.get('output_dir', 'results'))
    out_dir.mkdir(parents=True, exist_ok=True)
    concept_df.to_csv(out_dir / 'concept_overlap.csv', index=False)
    amp_df.to_csv(out_dir / 'cross_modal_amplification.csv', index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base_config.yaml')
    args = parser.parse_args()
    main(args.config)
