import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.interventions import vector_subtraction


def test_vector_subtraction_basic():
    activation = torch.tensor([[3.0, 4.0]])
    direction = torch.tensor([3.0, 0.0])
    result = vector_subtraction(activation, direction)
    expected = torch.tensor([[0.0, 4.0]])
    assert torch.allclose(result, expected)
