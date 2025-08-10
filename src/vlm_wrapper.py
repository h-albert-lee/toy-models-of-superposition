"""Utilities for accessing and intervening on pretrained VLMs.

This module provides a :class:`VLM_Wrapper` class that loads a
Hugging Face vision-language model and exposes helper methods for
extracting internal activations and performing interventions during
generation.  The implementation follows the design laid out in the
development specification, using PyTorch forward hooks to record or
modify activations at arbitrary layers.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

import logging

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


class VLM_Wrapper:
    """Thin wrapper around a Hugging Face VLM model.

    Parameters
    ----------
    model_name:
        Identifier of the pretrained model on the Hugging Face hub.
    device:
        Torch device onto which the model and inputs will be moved.
    model_kwargs:
        Additional keyword arguments passed to
        :func:`transformers.AutoModelForCausalLM.from_pretrained`.
    """

    def __init__(self, model_name: str, device: str = "cuda", **model_kwargs) -> None:
        self.logger = logging.getLogger(__name__)
        self.device = device
        model_kwargs.setdefault("trust_remote_code", True)

        # Load model and accompanying processors/tokenizers.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        ).to(device)

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=model_kwargs["trust_remote_code"]
            )
        except Exception:
            self.processor = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=model_kwargs["trust_remote_code"]
        )

        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Hook management helpers
    # ------------------------------------------------------------------
    def _register_hooks(self, layers_to_hook: List[str]) -> Dict[str, torch.Tensor]:
        """Register forward hooks on the specified layers.

        Returns a dictionary that will be populated with activations during
        the subsequent forward pass.
        """

        activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

        def get_activation_hook(name: str):
            def hook(_module, _inp, output):
                tensor = output[0] if isinstance(output, tuple) else output
                activations[name] = tensor.detach()

            return hook

        for layer_name in layers_to_hook:
            layer = self.model.get_submodule(layer_name)
            handle = layer.register_forward_hook(get_activation_hook(layer_name))
            self.hooks.append(handle)

        return activations

    def _clear_hooks(self) -> None:
        """Remove all previously registered hooks."""

        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------
    def _prepare_inputs(
        self,
        text_prompts: Sequence[str],
        images: Optional[Sequence[Optional[Image.Image]]] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.processor is not None:
            inputs = self.processor(
                text=list(text_prompts),
                images=list(images) if images is not None else None,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        else:
            inputs = self.tokenizer(list(text_prompts), return_tensors="pt", padding=True).to(
                self.device
            )
        return inputs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_activations(
        self,
        text_prompts: Union[str, Sequence[str]],
        images: Optional[Union[Image.Image, Sequence[Optional[Image.Image]]]],
        layers: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Extract activations from specific layers in batch."""

        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        if images is None:
            images_seq: Optional[Sequence[Optional[Image.Image]]] = None
        elif isinstance(images, Image.Image):
            images_seq = [images]
        else:
            images_seq = images

        inputs = self._prepare_inputs(text_prompts, images_seq)
        activations = self._register_hooks(layers)

        with torch.no_grad():
            self.model(**inputs)

        self._clear_hooks()
        return activations

    def generate(
        self, text_prompt: str, image: Optional[Image.Image] = None, **gen_kwargs
    ) -> str:
        """Generate text from the model without interventions."""

        inputs = self._prepare_inputs(text_prompt, image)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_with_intervention(
        self,
        text_prompt: str,
        layer: str,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor],
        image: Optional[Image.Image] = None,
        **gen_kwargs,
    ) -> str:
        """Generate text while applying a hook-based intervention."""

        module = self.model.get_submodule(layer)

        def hook(_module, _inp, output):
            return intervention_fn(output)

        handle = module.register_forward_hook(hook)
        inputs = self._prepare_inputs(text_prompt, image)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        handle.remove()
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


__all__ = ["VLM_Wrapper"]

