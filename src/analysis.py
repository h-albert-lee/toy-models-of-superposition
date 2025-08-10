from typing import List, Dict
import torch
import pandas as pd
from .vlm_wrapper import VLMWrapper


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def analyze_concept_overlap(wrapper: VLMWrapper, prompts: List[str], layer: str, direction: torch.Tensor) -> pd.DataFrame:
    acts = wrapper.get_activations(prompts, [layer])[layer]
    sims = _cosine_similarity(acts, direction)
    return pd.DataFrame({"prompt": prompts, "cosine_similarity": sims.cpu().tolist()})


def analyze_cross_modal_amplification(wrapper: VLMWrapper, cases: List[Dict], layer: str, direction: torch.Tensor) -> pd.DataFrame:
    records = []
    for case in cases:
        text = case["prompt"]
        image = case.get("image")
        act_text = wrapper.get_activations([text], [layer])[layer][0]
        act_multi = wrapper.get_activations([text], [layer], image=image)[layer][0]
        sim_text = _cosine_similarity(act_text, direction)
        sim_multi = _cosine_similarity(act_multi, direction)
        records.append({
            "prompt": text,
            "text_only": sim_text.item(),
            "image_text": sim_multi.item()
        })
    return pd.DataFrame(records)
