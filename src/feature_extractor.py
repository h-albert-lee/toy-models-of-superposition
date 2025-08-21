"""High level utilities for extracting characteristic vectors from VLMs."""

from __future__ import annotations

from typing import Dict, Sequence

import logging
import torch
from PIL import Image

from .vlm_wrapper import VLM_Wrapper


class VectorExtractor:
    """Compute general and intrinsic toxicity vectors using a VLM."""

    def __init__(self, wrapper: VLM_Wrapper, layer: str, batch_size: int = 8) -> None:
        self.wrapper = wrapper
        self.layer = layer
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # General toxicity vector
    # ------------------------------------------------------------------
    def compute_gtv(self, data: Sequence[Dict[str, str]]) -> torch.Tensor:
        """Compute the general toxicity vector (GTV).

        Parameters
        ----------
        data:
            Sequence of dictionaries with keys ``toxic_text`` and
            ``neutral_text``.
        """

        diffs = []
        toxic_texts = [item["toxic_text"] for item in data]
        neutral_texts = [item["neutral_text"] for item in data]
        for i in range(0, len(data), self.batch_size):
            t_batch = toxic_texts[i : i + self.batch_size]
            n_batch = neutral_texts[i : i + self.batch_size]
            toxic_act = self.wrapper.get_activations(t_batch, None, [self.layer])[self.layer]
            neutral_act = self.wrapper.get_activations(n_batch, None, [self.layer])[self.layer]
            diffs.append(toxic_act - neutral_act)
            self.logger.debug("Processed GTV batch %d-%d", i, i + len(t_batch))

        return torch.cat(diffs, dim=0).mean(dim=0)

    # ------------------------------------------------------------------
    # Intrinsic toxicity vector
    # ------------------------------------------------------------------
    def compute_itv(self, data: Sequence[Dict[str, object]]) -> torch.Tensor:
        """Compute the intrinsic toxicity vector (ITV).

        Each item in ``data`` should contain keys ``text`` and ``image``
        where ``image`` is a :class:`~PIL.Image.Image` instance.
        """

        residuals = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i : i + self.batch_size]
            texts = [str(item["text"]) for item in batch]
            images = []
            for item in batch:
                img = item["image"]
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                images.append(img)

            text_act = self.wrapper.get_activations(texts, None, [self.layer])[self.layer]
            image_act = self.wrapper.get_activations(["" for _ in images], images, [self.layer])[self.layer]
            fused_act = self.wrapper.get_activations(texts, images, [self.layer])[self.layer]

            # --- 🔥 여기부터 수정 시작 🔥 ---

            # 각 텐서의 시퀀스 길이 차원(dim=1)에 대해 평균을 계산합니다.
            text_act_mean = text_act.mean(dim=1)
            image_act_mean = image_act.mean(dim=1)
            fused_act_mean = fused_act.mean(dim=1)

            # 평균을 낸 벡터들로 연산을 수행합니다.
            residuals.append(fused_act_mean - (text_act_mean + image_act_mean))

            # --- 🔥 여기까지 수정 끝 🔥 ---
            
            self.logger.debug("Processed ITV batch %d-%d", i, i + len(batch))

            # Free image resources to avoid file handles accumulation
            for img in images:
                if isinstance(img, Image.Image):
                    img.close()

        return torch.cat(residuals, dim=0).mean(dim=0)


__all__ = ["VectorExtractor"]

