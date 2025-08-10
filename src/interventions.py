import torch

def activation_subtraction(activation: torch.Tensor, direction: torch.Tensor, multiplier: float = 1.0) -> torch.Tensor:
    projection = (activation @ direction) / (direction @ direction) * direction
    return activation - multiplier * projection
