import json
import yaml
import torch
from pathlib import Path
from transformers import pipeline
from src.vlm_wrapper import VLMWrapper
from src import interventions
from src.data_utils import load_cases

def main(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    wrapper = VLMWrapper(cfg['model_name'], cfg['device'], **cfg.get('model_kwargs', {}))
    direction = torch.load(Path(cfg.get('output_dir', 'results')) / 'direction_vector.pt')

    cases = load_cases(cfg['data_sources']['multimodal_test_cases'])

    classifier = pipeline('text-classification', model='unitary/unbiased-toxic-roberta')

    records = []
    for case in cases:
        prompt = case['prompt']
        image = case.get('image')
        normal = wrapper.generate_with_intervention(prompt, cfg['intervention_layer'], lambda x: x, image=image)
        def fn(act):
            return interventions.activation_subtraction(act, direction, cfg['intervention_multiplier'])
        intervened = wrapper.generate_with_intervention(prompt, cfg['intervention_layer'], fn, image=image)
        score_normal = classifier(normal)[0]['score']
        score_intervened = classifier(intervened)[0]['score']
        records.append({
            'prompt': prompt,
            'normal_output': normal,
            'intervened_output': intervened,
            'normal_score': score_normal,
            'intervened_score': score_intervened
        })

    out_dir = Path(cfg.get('output_dir', 'results'))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'intervention_evaluation.json', 'w') as f:
        json.dump(records, f, indent=2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base_config.yaml')
    args = parser.parse_args()
    main(args.config)
