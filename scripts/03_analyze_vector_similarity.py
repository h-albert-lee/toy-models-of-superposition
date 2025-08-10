"""Analyze similarity between GTV and ITV vectors."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def main() -> None:
    gtv = torch.load("results/general_toxicity_vector.pt")
    itv = torch.load("results/intrinsic_toxicity_vector.pt")

    sim = F.cosine_similarity(gtv.flatten(), itv.flatten(), dim=0)
    print(f"Cosine similarity: {sim.item():.4f}")


if __name__ == "__main__":
    main()

