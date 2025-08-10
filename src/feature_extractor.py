from typing import List
import torch
from .vlm_wrapper import VLMWrapper

class FeatureExtractor:
    def __init__(self, wrapper: VLMWrapper) -> None:
        self.wrapper = wrapper

    def compute_direction_vector(self, toxic_prompts: List[str], neutral_prompts: List[str], layer: str) -> torch.Tensor:
        toxic_acts = self.wrapper.get_activations(toxic_prompts, [layer])[layer]
        neutral_acts = self.wrapper.get_activations(neutral_prompts, [layer])[layer]
        diffs = toxic_acts - neutral_acts
        direction = diffs.mean(dim=0)
        direction = direction / direction.norm()
        return direction
