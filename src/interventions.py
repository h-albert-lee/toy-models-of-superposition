"""Mathematical operations used for activation interventions."""

from __future__ import annotations

import torch


def vector_subtraction(
    activation: torch.Tensor, direction: torch.Tensor, multiplier: float = 1.0
) -> torch.Tensor:
    """Remove the component of ``activation`` along ``direction``.

    The function projects ``activation`` onto ``direction`` and subtracts the
    scaled projection.  Broadcasting is used so that ``activation`` may have an
    arbitrary leading shape, e.g. ``(batch, seq_len, hidden)``.
    """

    proj = torch.nn.functional.linear(activation, direction.unsqueeze(0))
    proj = proj / (direction.norm() ** 2)
    return activation - multiplier * proj * direction


__all__ = ["vector_subtraction"]

