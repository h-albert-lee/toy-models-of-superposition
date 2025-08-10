"""High level utilities for extracting characteristic vectors from VLMs."""

from __future__ import annotations

from typing import List, Dict

import torch
from PIL import Image

from .vlm_wrapper import VLM_Wrapper


class VectorExtractor:
    """Compute general and intrinsic toxicity vectors using a VLM."""

    def __init__(self, wrapper: VLM_Wrapper, layer: str) -> None:
        self.wrapper = wrapper
        self.layer = layer

    # ------------------------------------------------------------------
    # General toxicity vector
    # ------------------------------------------------------------------
    def compute_gtv(self, data: List[Dict[str, str]]) -> torch.Tensor:
        """Compute the general toxicity vector (GTV).

        Parameters
        ----------
        data:
            Sequence of dictionaries with keys ``toxic_text`` and
            ``neutral_text``.
        """

        diffs = []
        for item in data:
            toxic_act = self.wrapper.get_activations(
                item["toxic_text"], None, [self.layer]
            )[self.layer]
            neutral_act = self.wrapper.get_activations(
                item["neutral_text"], None, [self.layer]
            )[self.layer]
            diffs.append(toxic_act - neutral_act)

        return torch.stack(diffs).mean(dim=0)

    # ------------------------------------------------------------------
    # Intrinsic toxicity vector
    # ------------------------------------------------------------------
    def compute_itv(self, data: List[Dict[str, object]]) -> torch.Tensor:
        """Compute the intrinsic toxicity vector (ITV).

        Each item in ``data`` should contain keys ``text`` and ``image``
        where ``image`` is a :class:`~PIL.Image.Image` instance.
        """

        residuals = []
        for item in data:
            text = str(item["text"])
            image = item["image"]
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")

            text_act = self.wrapper.get_activations(text, None, [self.layer])[
                self.layer
            ]
            image_act = self.wrapper.get_activations("", image, [self.layer])[
                self.layer
            ]
            fused_act = self.wrapper.get_activations(text, image, [self.layer])[
                self.layer
            ]

            residuals.append(fused_act - (text_act + image_act))

        return torch.stack(residuals).mean(dim=0)


__all__ = ["VectorExtractor"]

