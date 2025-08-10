import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.feature_extractor import VectorExtractor


class WrapperForGTV:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_activations(self, texts, images, layers):
        layer = layers[0]
        if isinstance(texts, str):
            texts = [texts]
        acts = torch.stack([self.mapping[t] for t in texts])
        return {layer: acts}


def test_compute_gtv_batch():
    mapping = {
        "t1": torch.tensor([1.0, 1.0]),
        "t2": torch.tensor([2.0, 2.0]),
        "n1": torch.tensor([0.5, 0.5]),
        "n2": torch.tensor([1.0, 1.0]),
    }
    extractor = VectorExtractor(WrapperForGTV(mapping), layer="layer", batch_size=2)
    data = [
        {"toxic_text": "t1", "neutral_text": "n1"},
        {"toxic_text": "t2", "neutral_text": "n2"},
    ]
    gtv = extractor.compute_gtv(data)
    assert torch.allclose(gtv, torch.tensor([0.75, 0.75]))


class WrapperForITV:
    def get_activations(self, texts, images, layers):
        layer = layers[0]
        if images is None:
            act = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        elif all(t == "" for t in texts):
            act = torch.tensor([[0.5, 0.25], [0.6, 0.2]])
        else:
            act = torch.tensor([[1.6, 2.15], [3.4, 4.5]])
        return {layer: act}


def test_compute_itv_batch():
    extractor = VectorExtractor(WrapperForITV(), layer="layer", batch_size=2)
    img1 = Image.new("RGB", (1, 1))
    img2 = Image.new("RGB", (1, 1))
    data = [{"text": "a", "image": img1}, {"text": "b", "image": img2}]
    itv = extractor.compute_itv(data)
    expected = torch.tensor([-0.05, 0.1])
    assert torch.allclose(itv, expected)
