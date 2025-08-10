import yaml
import torch
from pathlib import Path
from src.vlm_wrapper import VLMWrapper
from src.feature_extractor import FeatureExtractor
from src.data_utils import load_pairs

def main(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    wrapper = VLMWrapper(cfg['model_name'], cfg['device'], **cfg.get('model_kwargs', {}))
    extractor = FeatureExtractor(wrapper)

    data_source = cfg['data_sources']['toxic_neutral_pairs']
    toxic_prompts, neutral_prompts = load_pairs(data_source)
    direction = extractor.compute_direction_vector(toxic_prompts, neutral_prompts, cfg['extraction_layer'])

    output_dir = Path(cfg.get('output_dir', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(direction, output_dir / 'direction_vector.pt')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base_config.yaml')
    args = parser.parse_args()
    main(args.config)
